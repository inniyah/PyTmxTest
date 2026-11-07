"""Geometría para dibujar el mundo compilado (sin OpenGL: numpy puro).

Cada hueco visual de cada celda es un quad: la imagen de su región del atlas, anclada por
abajo al borde sur de la celda y desplazada hacia arriba según su planta. Los vértices
llevan, además de la posición y las UV, el origen de la celda en metros, que el shader
suma a la posición del mapa de posición para obtener la profundidad por píxel.

Vértice: [x_px, y_px, u, v, origen_x_m, origen_y_m, origen_z_m, alfa, volteado] (9 floats).
"""

import io
from dataclasses import dataclass
from typing import Dict

import numpy as np
from PIL import Image

from engine.world import visual as V
from engine.world.chunk import ChunkData

FLOATS_PER_VERTEX = 9


@dataclass(frozen=True)
class WorldGeometry:
    """Escala y proyección del mundo, leídas de los parámetros del mundo compilado."""
    tile_w: int
    row_h: int
    level_h: int
    cell_w: float
    cell_h: float
    pos_min: tuple
    pos_max: tuple
    closeness_min: float
    closeness_max: float

    @property
    def sz(self) -> float:          # píxeles por metro de fondo (hacia el sur)
        return self.row_h / self.cell_w

    @property
    def sy(self) -> float:          # píxeles por metro de altura
        return self.level_h / self.cell_h

    @classmethod
    def from_meta(cls, meta: dict) -> "WorldGeometry":
        ranges = {}
        for part in str(meta.get("position_range_m") or "x 0 2; y 0 3.75; z 0 2").split(";"):
            axis, lo, hi = part.split()
            ranges[axis] = (float(lo), float(hi))
        b = meta["bounds"]
        cell_w, cell_h = float(meta["cell_width_m"]), float(meta["cell_height_m"])
        row_h, level_h = int(meta["row_height_px"]), int(meta["level_height_px"])
        sz, sy = row_h / cell_w, level_h / cell_h
        # Cercanía posible en todo el mundo, con margen de una planta y una fila
        y_lo, y_hi = (b["level_min"] - 1) * cell_h, (b["level_max"] + 2.5) * cell_h
        z_lo, z_hi = (b["y0"] - 1) * cell_w, (b["y0"] + b["depth"] + 1) * cell_w
        return cls(int(meta["tile_width_px"]), row_h, level_h, cell_w, cell_h,
                   tuple(ranges[a][0] for a in "xyz"), tuple(ranges[a][1] for a in "xyz"),
                   y_lo * sz + z_lo * sy, y_hi * sz + z_hi * sy)

    def closeness(self, y_m, z_m):
        return np.asarray(y_m) * self.sz + np.asarray(z_m) * self.sy


@dataclass
class AtlasLayout:
    """Regiones del atlas como arrays indexados por id de región (0 = sin región)."""
    page: np.ndarray
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    h: np.ndarray
    page_sizes: Dict[int, tuple]          # página -> (ancho, alto)

    @classmethod
    def from_store(cls, store) -> "AtlasLayout":
        rows = store.atlas_regions()
        n = (max(r[0] for r in rows) + 1) if rows else 1
        arrays = {k: np.zeros(n, np.int64) for k in ("page", "x", "y", "w", "h")}
        for rid, page, x, y, w, h, _border in rows:
            for k, v in zip(("page", "x", "y", "w", "h"), (page, x, y, w, h)):
                arrays[k][rid] = v
        sizes = {}
        for page in sorted(set(int(p) for p in arrays["page"][1:])) if rows else []:
            with Image.open(io.BytesIO(store.atlas_page(page, "color"))) as im:
                sizes[page] = im.size
        return cls(page_sizes=sizes, **arrays)


def chunk_quads(chunk: ChunkData, geo: WorldGeometry, atlas: AtlasLayout) -> Dict[int, np.ndarray]:
    """{página del atlas: vértices (N*4, 8)} de todos los huecos visuales del chunk.

    Los quads van ordenados de lejos a cerca (por la cercanía del pie de su celda), para que
    las partes semitransparentes se mezclen bien; la oclusión la resuelve el z-buffer.
    """
    n = chunk.size
    lens = chunk.col_len.astype(np.int64)
    col = np.repeat(np.arange(n * n), lens)
    if col.size == 0:
        return {}
    level = chunk.col_start.astype(np.int64)[col] + (np.arange(col.size) - chunk.offset[col])
    cx, cy = chunk.cx * n + (col % n), chunk.cy * n + (col // n)

    cell, slot = np.nonzero(V.region(chunk.visual))
    if cell.size == 0:
        return {}
    values = chunk.visual[cell, slot]
    rid = V.region(values).astype(np.int64)
    x, y, lv = cx[cell], cy[cell], level[cell]
    w, h = atlas.w[rid].astype(float), atlas.h[rid].astype(float)

    left = x * geo.tile_w
    top = (y + 1) * geo.row_h - h - lv * geo.level_h
    origin = np.stack([x * geo.cell_w, lv * geo.cell_h, y * geo.cell_w], axis=1).astype(float)
    alpha = V.opacity(values)
    flip = V.flipped(values)

    order = np.lexsort((slot, geo.closeness(lv * geo.cell_h, (y + 1) * geo.cell_w)))
    out = {}
    for page in np.unique(atlas.page[rid]):
        sel = order[atlas.page[rid][order] == page]
        pw, ph = atlas.page_sizes[int(page)]
        u0 = atlas.x[rid[sel]] / pw
        u1 = (atlas.x[rid[sel]] + atlas.w[rid[sel]]) / pw
        v0 = atlas.y[rid[sel]] / ph
        v1 = (atlas.y[rid[sel]] + atlas.h[rid[sel]]) / ph
        u0, u1 = np.where(flip[sel], u1, u0), np.where(flip[sel], u0, u1)    # volteo horizontal
        l, t, ww, hh = left[sel], top[sel], w[sel], h[sel]
        verts = np.zeros((sel.size, 4, FLOATS_PER_VERTEX), np.float32)
        # Esquinas: arriba-izq, arriba-der, abajo-der, abajo-izq. La página no se voltea al
        # subirla, así que v crece hacia abajo igual que la imagen.
        verts[:, :, 0] = np.stack([l, l + ww, l + ww, l], axis=1)
        verts[:, :, 1] = np.stack([t, t, t + hh, t + hh], axis=1)
        verts[:, :, 2] = np.stack([u0, u1, u1, u0], axis=1)
        verts[:, :, 3] = np.stack([v0, v0, v1, v1], axis=1)
        verts[:, :, 4:7] = origin[sel][:, None, :]
        verts[:, :, 7] = alpha[sel][:, None]
        verts[:, :, 8] = flip[sel][:, None]
        out[int(page)] = verts.reshape(-1, FLOATS_PER_VERTEX)
    return out
