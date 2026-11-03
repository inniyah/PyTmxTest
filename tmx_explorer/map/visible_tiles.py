"""Recolección vectorizada de los tiles visibles, lista para dibujar.

Sustituye al bucle Python tile a tile: para cada (nivel, capa) se toma la ventana
visible del array del mapa con numpy y se generan todos sus quads de golpe.
No necesita OpenGL.
"""

from typing import Dict, Sequence

import numpy as np

from ..config import DEPTH_BASE, DEPTH_LAYER_STEP
from ..renderer.sprite_batch import FLIP_D, quad_vertices
from .structure import Map3DStructure
from .tileset_renderer import TileLookup


def collect_visible_tiles(map_3d: Map3DStructure, lookup: TileLookup,
                          x0: int, x1: int, y0: int, y1: int,
                          max_z: int, level_height_offset: float,
                          layer_visibility: Sequence[bool]) -> Dict[int, np.ndarray]:
    """Vértices de los tiles en la ventana [y0:y1, x0:x1], niveles 0..max_z.

    Devuelve ``{página del atlas: vértices (N*4, 9)}`` en coordenadas de mundo.

    Posición: los tiles más altos que la rejilla se anclan por abajo, y cada nivel
    sube ``z * level_height_offset`` píxeles.
    Profundidad: ``DEPTH_BASE + fila + z + capa * DEPTH_LAYER_STEP``.
    Los quads salen en orden fila, columna, nivel, capa: con el depth test en GL_LESS,
    a igual profundidad gana el primero dibujado.
    """
    if x0 >= x1 or y0 >= y1:
        return {}

    tw, th = map_3d.tile_width, map_3d.tile_height
    parts = {k: [] for k in ("y", "x", "z", "n", "gid", "flip")}

    for z in range(min(max_z, map_3d.H - 1) + 1):
        level = map_3d.get_level_value(z)
        for n in range(map_3d.N):
            if not layer_visibility[n] or map_3d.layer_levels[n] != level:
                continue
            window = map_3d.mapa[z, y0:y1, x0:x1, n]
            ys, xs = np.nonzero(window)
            if ys.size == 0:
                continue
            parts["y"].append(ys + y0)
            parts["x"].append(xs + x0)
            parts["z"].append(np.full(ys.size, z, dtype=np.int32))
            parts["n"].append(np.full(ys.size, n, dtype=np.int32))
            parts["gid"].append(window[ys, xs])
            parts["flip"].append(map_3d.flips[z, y0:y1, x0:x1, n][ys, xs])

    if not parts["y"]:
        return {}
    ty, tx, tz, tn, gid, flip = (np.concatenate(parts[k])
                                 for k in ("y", "x", "z", "n", "gid", "flip"))

    page, uv, size = lookup.get(gid)
    keep = page >= 0
    order = np.lexsort((tn, tz, tx, ty))           # fila, columna, nivel, capa
    order = order[keep[order]]
    ty, tx, tz, tn, flip, page, uv, size = (a[order] for a in (ty, tx, tz, tn, flip, page, uv, size))

    w, h = size[:, 0].copy(), size[:, 1].copy()
    diagonal = (flip & FLIP_D) != 0
    w[diagonal], h[diagonal] = size[diagonal, 1], size[diagonal, 0]   # giro de 90º

    world_x = tx * tw
    world_y = (ty + 1) * th - h - tz * level_height_offset
    depth = DEPTH_BASE + ty + tz + tn * DEPTH_LAYER_STEP

    vertices = quad_vertices(world_x, world_y, w, h, depth,
                             uv[:, 0], uv[:, 1], uv[:, 2], uv[:, 3], flips=flip)
    vertices = vertices.reshape(-1, 4, vertices.shape[-1])

    return {int(p): vertices[page == p].reshape(-1, vertices.shape[-1])
            for p in np.unique(page)}
