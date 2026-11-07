"""Un chunk: N × N columnas compactas, con sus celdas de obstáculos y visuales.

Orden de las columnas: fila a fila dentro del chunk (índice = ly · N + lx).
Orden de las celdas: columna tras columna, de abajo arriba dentro de cada una.
La celda de la columna c a la altura z está en ``offset[c] + (z − start[c])``.

En disco cada array es un bloque binario (little-endian):
    col_start     int16  por columna    nivel de la primera celda guardada
    col_len       uint8  por columna    número de celdas guardadas (0 = columna vacía)
    obst_palette  uint32                valores de celda distintos del chunk
    obst_index    uint8 o uint16        por celda (el tamaño del bloque dice cuál)
    vis_palette   uint32 × 4            combinaciones de 4 huecos visuales distintas
    vis_index     uint8 o uint16        por celda
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

PARTS = ("col_start", "col_len", "obst_palette", "obst_index", "vis_palette", "vis_index")
OBSTACLE_PARTS = ("col_start", "col_len", "obst_palette", "obst_index")


@dataclass
class ChunkData:
    cx: int
    cy: int
    size: int
    col_start: np.ndarray       # (size²,) int16
    col_len: np.ndarray         # (size²,) uint8
    obstacles: np.ndarray       # (celdas,) uint32
    visual: np.ndarray = None   # (celdas, 4) uint32, o None si solo se cargaron obstáculos

    def __post_init__(self):
        self.offset = np.concatenate([[0], np.cumsum(self.col_len, dtype=np.int64)[:-1]])

    @property
    def cell_count(self) -> int:
        return int(self.col_len.sum(dtype=np.int64))

    # --- Bloques binarios ---

    def to_blobs(self) -> Dict[str, bytes]:
        obst_palette, obst_index = np.unique(self.obstacles, return_inverse=True)
        blobs = {
            "col_start": self.col_start.astype("<i2").tobytes(),
            "col_len": self.col_len.astype(np.uint8).tobytes(),
            "obst_palette": obst_palette.astype("<u4").tobytes(),
            "obst_index": _index_bytes(obst_index, len(obst_palette)),
        }
        if self.visual is not None:
            vis_palette, vis_index = np.unique(self.visual, axis=0, return_inverse=True)
            blobs["vis_palette"] = vis_palette.astype("<u4").tobytes()
            blobs["vis_index"] = _index_bytes(vis_index.ravel(), len(vis_palette))
        return blobs

    @classmethod
    def from_blobs(cls, cx: int, cy: int, size: int, blobs: Dict[str, bytes]) -> "ChunkData":
        col_start = np.frombuffer(blobs["col_start"], "<i2").astype(np.int16)
        col_len = np.frombuffer(blobs["col_len"], np.uint8)
        cells = int(col_len.sum(dtype=np.int64))
        obstacles = (np.frombuffer(blobs["obst_palette"], "<u4")
                     [_read_index(blobs["obst_index"], cells)].astype(np.uint32))
        visual = None
        if blobs.get("vis_palette") is not None:
            palette = np.frombuffer(blobs["vis_palette"], "<u4").reshape(-1, 4)
            visual = palette[_read_index(blobs["vis_index"], cells)].astype(np.uint32)
        return cls(cx, cy, size, col_start, col_len, obstacles, visual)


def _index_bytes(index: np.ndarray, palette_size: int) -> bytes:
    """Índices de 1 byte si la paleta cabe en 256 valores; si no, de 2."""
    if palette_size > 65536:
        raise ValueError(f"Paleta de {palette_size} valores: no cabe en índices de 2 bytes")
    return index.astype(np.uint8 if palette_size <= 256 else "<u2").tobytes()


def _read_index(data: bytes, cells: int) -> np.ndarray:
    if len(data) == cells:
        return np.frombuffer(data, np.uint8)
    if len(data) == 2 * cells:
        return np.frombuffer(data, "<u2")
    raise ValueError(f"Índice de {len(data)} bytes para {cells} celdas")
