"""El mundo compilado, cargado en memoria y consultable.

Los chunks se cargan la primera vez que una consulta los toca. Sus columnas se guardan en
un conjunto común de huecos (arrays compartidos por todos los chunks cargados) y una
rejilla dice en qué hueco está cada chunk, así que cualquier consulta, toque los chunks
que toque, se resuelve con unas pocas indexaciones, sin bucles por chunk ni por punto.

Regla de las columnas:
    por debajo del tramo guardado -> macizo (CELL_SOLID)
    por encima del tramo          -> aire (0)
    fuera de los límites del mapa -> macizo, a cualquier altura
    chunk no guardado (vacío)     -> columnas vacías que empiezan en el nivel más bajo
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .cell import CELL_SOLID
from .chunk import OBSTACLE_PARTS, PARTS, ChunkData
from .store import FORMAT_VERSION, WorldStore

NOT_LOADED, ABSENT = -1, -2          # valores especiales de la rejilla de chunks


class WorldView:
    def __init__(self, store: WorldStore, with_visual: bool = True):
        self.store = store
        meta = store.meta()
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Formato {meta.get('format_version')}, se esperaba {FORMAT_VERSION}")
        self.meta = meta
        self.chunk_size = n = int(meta["chunk_size"])
        self.shift = n.bit_length() - 1                          # potencia de dos
        b = meta["bounds"]
        self.x0, self.y0, self.width, self.depth = b["x0"], b["y0"], b["width"], b["depth"]
        self.level_min, self.level_max = b["level_min"], b["level_max"]
        self.with_visual = with_visual
        self.parts = PARTS if with_visual else OBSTACLE_PARTS

        # Rejilla de chunks que cubre el mapa: hueco de cada chunk, NOT_LOADED o ABSENT
        self.cx0, self.cy0 = self.x0 >> self.shift, self.y0 >> self.shift
        cx1, cy1 = (self.x0 + self.width - 1) >> self.shift, (self.y0 + self.depth - 1) >> self.shift
        self.grid = np.full((cy1 - self.cy0 + 1, cx1 - self.cx0 + 1), NOT_LOADED, np.int32)
        self._reset_pool()

    # --- Conjunto de huecos ---

    def _reset_pool(self):
        cols = self.chunk_size * self.chunk_size
        self.chunks: Dict[Tuple[int, int], ChunkData] = {}
        self._slot_keys = []
        self.pool_start = np.zeros((0, cols), np.int64)
        self.pool_len = np.zeros((0, cols), np.int64)
        self.pool_offset = np.zeros((0, cols), np.int64)     # posición en los búferes comunes
        self.obstacles = np.zeros(0, np.uint32)
        self.visual_cells = np.zeros((0, 4), np.uint32)

    def _add_to_pool(self, chunk: ChunkData):
        base = len(self.obstacles)
        self.pool_start = np.vstack([self.pool_start, chunk.col_start.astype(np.int64)[None]])
        self.pool_len = np.vstack([self.pool_len, chunk.col_len.astype(np.int64)[None]])
        self.pool_offset = np.vstack([self.pool_offset, (chunk.offset + base)[None]])
        self.obstacles = np.concatenate([self.obstacles, chunk.obstacles])
        if self.with_visual:
            self.visual_cells = np.concatenate([self.visual_cells, chunk.visual])
        self._slot_keys.append((chunk.cx, chunk.cy))
        self.grid[chunk.cy - self.cy0, chunk.cx - self.cx0] = len(self._slot_keys) - 1

    def load(self, cx: int, cy: int):
        gy, gx = cy - self.cy0, cx - self.cx0
        if not (0 <= gy < self.grid.shape[0] and 0 <= gx < self.grid.shape[1]):
            return
        if self.grid[gy, gx] != NOT_LOADED:
            return
        chunk = self.store.read_chunk(cx, cy, self.parts)
        if chunk is None:
            self.grid[gy, gx] = ABSENT
        else:
            self.chunks[cx, cy] = chunk
            self._add_to_pool(chunk)

    def chunk(self, cx: int, cy: int) -> Optional[ChunkData]:
        self.load(cx, cy)
        return self.chunks.get((cx, cy))

    def evict(self, keep: Sequence[Tuple[int, int]] = ()):
        """Descarga todos los chunks salvo ``keep`` y compacta el conjunto de huecos."""
        kept = [self.chunks[k] for k in keep if k in self.chunks]
        self.grid[:] = NOT_LOADED
        self._reset_pool()
        for chunk in kept:
            self.chunks[chunk.cx, chunk.cy] = chunk
            self._add_to_pool(chunk)

    @property
    def loaded(self):
        return list(self._slot_keys)

    # --- Consultas ---

    def _locate(self, x, y, z):
        """Para cada consulta: ¿celda guardada?, ¿macizo por debajo o fuera?, posición en los búferes."""
        x, y, z = (np.asarray(a, dtype=np.int64) for a in np.broadcast_arrays(x, y, z))
        shape = x.shape
        x, y, z = x.ravel(), y.ravel(), z.ravel()
        inside = (x >= self.x0) & (x < self.x0 + self.width) & (y >= self.y0) & (y < self.y0 + self.depth)
        gx = np.where(inside, (x >> self.shift) - self.cx0, 0)
        gy = np.where(inside, (y >> self.shift) - self.cy0, 0)
        slot = self.grid[gy, gx]

        missing = inside & (slot == NOT_LOADED)
        if missing.any():                                           # cargar lo que falte
            for key in np.unique(np.stack([gx[missing], gy[missing]], 1), axis=0):
                self.load(int(key[0]) + self.cx0, int(key[1]) + self.cy0)
            slot = self.grid[gy, gx]

        has_chunk = inside & (slot >= 0)
        s = np.where(has_chunk, slot, 0)
        col = ((y & (self.chunk_size - 1)) << self.shift) | (x & (self.chunk_size - 1))
        if len(self._slot_keys):
            start = np.where(has_chunk, self.pool_start[s, col], self.level_min)
            length = np.where(has_chunk, self.pool_len[s, col], 0)
            base = np.where(has_chunk, self.pool_offset[s, col], 0)
        else:
            start = np.full(x.shape, self.level_min, np.int64)
            length = base = np.zeros(x.shape, np.int64)
        rel = z - start
        stored = inside & (rel >= 0) & (rel < length)
        below = ~inside | (rel < 0)
        return stored, below, np.where(stored, base + rel, 0), shape

    def cells(self, x, y, z) -> np.ndarray:
        """Valor de 32 bits de las celdas (x, y, nivel). Acepta escalares o arrays."""
        stored, below, pos, shape = self._locate(x, y, z)
        out = np.where(below, np.uint32(CELL_SOLID), np.uint32(0)).astype(np.uint32)
        out[stored] = self.obstacles[pos[stored]]
        return out.reshape(shape)

    def visual(self, x, y, z) -> np.ndarray:
        """Huecos visuales (…, 4) de las celdas; ceros donde no hay nada guardado."""
        if not self.with_visual:
            raise ValueError("Mundo abierto sin datos visuales (with_visual=False)")
        stored, _below, pos, shape = self._locate(x, y, z)
        out = np.zeros((len(stored), 4), np.uint32)
        out[stored] = self.visual_cells[pos[stored]]
        return out.reshape(shape + (4,))

    def column(self, x: int, y: int) -> Tuple[int, int]:
        """(nivel de inicio, número de celdas guardadas) de una columna."""
        if not (self.x0 <= x < self.x0 + self.width and self.y0 <= y < self.y0 + self.depth):
            return self.level_min, 0
        chunk = self.chunk(x >> self.shift, y >> self.shift)
        if chunk is None:
            return self.level_min, 0
        c = ((y & (self.chunk_size - 1)) << self.shift) | (x & (self.chunk_size - 1))
        return int(chunk.col_start[c]), int(chunk.col_len[c])
