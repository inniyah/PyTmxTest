"""Render oblicuo de una pieza: color, normales y posición por píxel.

Proyección (la de los mapas existentes): la cara superior de una celda ocupa en pantalla
``row_h`` píxeles por ``tile_w`` de ancho (23/32), y cada planta sube ``level_h`` píxeles.

    px    = x · sx                       sx = tile_w / ancho de celda
    py_up = (D − z) · sz + y · sy        sz = row_h / ancho de celda,  sy = level_h / alto de planta

La imagen tiene ``tile_w`` de ancho y ``row_h + ceil(1.25 · level_h)`` de alto (las rampas
diagonales pueden sobresalir ¼ de planta). Su borde inferior es el borde sur del suelo de
la celda: así se ancla en Tiled y en el motor.

Mapas que se generan para cada tile:
    color     RGBA 8 bits, sombreado plano con una luz fija y el contorno de las caras
    normal    RGBA 8 bits: (n · 0.5 + 0.5) · 255, ejes del mundo (X este, Y arriba, Z sur)
    posición  RGBA 16 bits: (x, y, z) en metros dentro de la celda, escalados a 0..65535
              con los rangos de ``Projection.position_range``; alfa = cobertura
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .mesh import CellSize

LIGHT = np.array([-0.45, 1.0, 0.65]) / np.linalg.norm([-0.45, 1.0, 0.65])
AMBIENT = 0.38
EDGE_DARKEN = 0.72


@dataclass(frozen=True)
class Projection:
    tile_w: int = 128
    row_h: int = 92
    level_h: int = 133
    cell: CellSize = CellSize()  # noqa: B008  (inmutable)

    @property
    def sx(self):
        return self.tile_w / self.cell.width

    @property
    def sz(self):
        return self.row_h / self.cell.width

    @property
    def sy(self):
        return self.level_h / self.cell.height

    @property
    def image_size(self) -> Tuple[int, int]:
        return self.tile_w, self.row_h + math.ceil(1.25 * self.level_h)

    @property
    def position_range(self):
        """(mínimo, máximo) en metros de cada eje del mapa de posición."""
        return ((0.0, self.cell.width), (0.0, 1.25 * self.cell.height), (0.0, self.cell.width))

    def to_screen(self, points: np.ndarray) -> np.ndarray:
        """(n, 3) metros -> (n, 2) píxeles (x a la derecha, y hacia arriba desde el borde inferior)."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        return np.stack([x * self.sx, (self.cell.width - z) * self.sz + y * self.sy], axis=1)

    def closeness(self, points: np.ndarray) -> np.ndarray:
        """Cuanto mayor, más cerca de la cámara (dirección de proyección (0, sz, sy))."""
        return points[:, 1] * self.sz + points[:, 2] * self.sy


DEFAULT_PROJECTION = Projection()


def render(faces: List[Tuple[np.ndarray, Tuple[float, float, float]]], proj: Projection = DEFAULT_PROJECTION):
    """Devuelve (color uint8 (h, w, 4), normal uint8 (h, w, 4), posición uint16 (h, w, 4))."""
    w, h = proj.image_size
    depth = np.full((h, w), -np.inf)
    face_id = np.full((h, w), -1, dtype=np.int32)
    color = np.zeros((h, w, 3))
    normal = np.zeros((h, w, 3))
    position = np.zeros((h, w, 3))

    # Centros de píxel en coordenadas de pantalla con y hacia arriba
    gx, gy = np.meshgrid(np.arange(w) + 0.5, h - (np.arange(h) + 0.5))

    for fid, (points, kd) in enumerate(faces):
        n = np.cross(points[1] - points[0], points[2] - points[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n = n / np.linalg.norm(n)
        shade = np.asarray(kd) * (AMBIENT + (1 - AMBIENT) * max(0.0, float(n @ LIGHT)))
        for k in range(1, len(points) - 1):          # abanico de triángulos
            tri = points[[0, k, k + 1]]
            s = proj.to_screen(tri)
            (x0, y0), (x1, y1), (x2, y2) = s
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(area) < 1e-9:
                continue                              # visto de canto
            i0, i1 = max(0, int(math.floor(s[:, 0].min()))), min(w, int(math.ceil(s[:, 0].max())))
            j_top, j_bot = h - int(math.ceil(s[:, 1].max())), h - int(math.floor(s[:, 1].min()))
            j0, j1 = max(0, j_top), min(h, j_bot)
            if i0 >= i1 or j0 >= j1:
                continue
            px, py = gx[j0:j1, i0:i1], gy[j0:j1, i0:i1]
            l1 = ((x2 - x0) * (py - y0) - (y2 - y0) * (px - x0)) / -area
            l2 = ((x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)) / area
            l0 = 1 - l1 - l2
            eps = -1e-7
            inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
            if not inside.any():
                continue
            p3 = l0[..., None] * tri[0] + l1[..., None] * tri[1] + l2[..., None] * tri[2]
            near = proj.closeness(p3.reshape(-1, 3)).reshape(px.shape)
            win = inside & (near > depth[j0:j1, i0:i1] + 1e-9)
            depth[j0:j1, i0:i1][win] = near[win]
            face_id[j0:j1, i0:i1][win] = fid
            color[j0:j1, i0:i1][win] = shade
            normal[j0:j1, i0:i1][win] = n
            position[j0:j1, i0:i1][win] = p3[win]

    covered = face_id >= 0
    # Contorno: píxeles cuya cara cambia respecto a algún vecino; fuera de la imagen
    # cuenta como vacío, así que la silueta queda cerrada también en los bordes del tile
    padded = np.pad(face_id, 1, constant_values=-1)
    centre = padded[1:-1, 1:-1]
    edge = covered & ((centre != padded[1:-1, 2:]) | (centre != padded[1:-1, :-2])
                      | (centre != padded[2:, 1:-1]) | (centre != padded[:-2, 1:-1]))
    color[edge] *= EDGE_DARKEN

    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = np.clip(color * 255 + 0.5, 0, 255)
    rgba[..., 3] = covered * 255

    nrm = np.zeros((h, w, 4), np.uint8)
    nrm[..., :3] = np.clip((normal * 0.5 + 0.5) * 255 + 0.5, 0, 255)
    nrm[..., 3] = covered * 255

    pos = np.zeros((h, w, 4), np.uint16)
    for axis, (lo, hi) in enumerate(proj.position_range):
        pos[..., axis] = np.clip((position[..., axis] - lo) / (hi - lo) * 65535 + 0.5, 0, 65535)
    pos[..., 3] = covered * 65535
    pos[~covered] = 0
    return rgba, nrm, pos
