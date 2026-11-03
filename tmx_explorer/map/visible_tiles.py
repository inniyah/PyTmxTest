"""Recolección vectorizada de los tiles visibles del mundo, lista para dibujar.

Para cada nivel se toma la ventana visible de ``world.tiles`` (todos sus huecos a
la vez) y se generan los quads con numpy. No necesita OpenGL.
"""

from typing import Dict, Optional

import numpy as np

from ..config import DEPTH_BASE, DEPTH_LAYER_STEP, GID_MASK
from ..renderer.sprite_batch import FLIP_D, quad_vertices
from .tileset_renderer import TileLookup
from .world import EMPTY_SOURCE, SLOTS, LayerRole, World

# Profundidad relativa de cada rol respecto a la fila del tile (los personajes de la
# misma fila están en +0.5): los suelos quedan debajo y lo superior, encima.
ROLE_DEPTH = np.array([
    -1.0,   # FLOOR
    0.0,    # OBJECT
    2.0,    # OVERHEAD
    2.0,    # ROOF
], dtype=np.float32)
assert len(ROLE_DEPTH) == len(LayerRole)
assert DEPTH_LAYER_STEP * (SLOTS - 1) < 0.5, "los huecos no deben alcanzar a los personajes"


def collect_visible_tiles(world: World, lookup: TileLookup,
                          x0: int, x1: int, y0: int, y1: int,
                          max_z: int, level_height: float,
                          layer_visible: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
    """Vértices de los tiles en la ventana [y0:y1, x0:x1], niveles 0..max_z.

    Devuelve ``{página del atlas: vértices (N*4, 9)}`` en coordenadas de mundo,
    ordenados de atrás adelante (profundidad creciente) para que las transparencias
    se mezclen bien.

    ``layer_visible`` es un array bool indexado por ``LayerInfo.index`` (None = todas).
    Profundidad: ``DEPTH_BASE + fila + z + ROLE_DEPTH[rol] + hueco * DEPTH_LAYER_STEP``.
    """
    if x0 >= x1 or y0 >= y1:
        return {}

    visible = np.zeros(256, dtype=bool)
    visible[:len(world.layers)] = True if layer_visible is None else layer_visible[:len(world.layers)]
    opacity = np.ones(256, dtype=np.float32)
    opacity[:len(world.layers)] = [l.opacity for l in world.layers]
    visible[EMPTY_SOURCE] = False

    parts = []
    for z in range(min(max_z, world.H - 1) + 1):
        used = int(world.slots_used[z])                 # solo los huecos que usa el nivel
        if used == 0:
            continue
        win = world.tiles[z, y0:y1, x0:x1, :used]       # (h, w, used)
        src = world.source[z, y0:y1, x0:x1, :used]
        ys, xs, ks = np.nonzero(((win & GID_MASK) != 0) & visible[src])
        if ys.size == 0:
            continue
        roles = (world.roles[z, y0:y1, x0:x1][ys, xs] >> (2 * ks).astype(np.uint8)) & 3
        parts.append((ys + y0, xs + x0, np.full(ys.size, z), ks, win[ys, xs, ks],
                      roles, src[ys, xs, ks]))
    if not parts:
        return {}
    ty, tx, tz, tk, gid, role, src = (np.concatenate(c) for c in zip(*parts))

    page, uv, size = lookup.get(gid & GID_MASK)
    depth = DEPTH_BASE + ty + tz + ROLE_DEPTH[role] + tk * DEPTH_LAYER_STEP

    # Una sola ordenación: atrás -> adelante y, a igual profundidad, fila, columna,
    # nivel, hueco (con el depth test en GL_LESS gana el primero dibujado).
    # Los tiles sin imagen en el atlas se descartan en el mismo paso.
    order = np.lexsort((tk, tz, tx, ty, depth))
    order = order[page[order] >= 0]
    if order.size == 0:
        return {}
    ty, tz, tx, gid, src, page, uv, size, depth = (
        a[order] for a in (ty, tz, tx, gid, src, page, uv, size, depth))

    flip = (gid >> 28).astype(np.uint8)
    w, h = size[:, 0], size[:, 1]
    diagonal = (flip & FLIP_D) != 0
    if diagonal.any():                                   # giro de 90º: se intercambian
        w, h = np.where(diagonal, h, w), np.where(diagonal, w, h)

    world_x = tx * world.tile_width
    world_y = (ty + 1) * world.tile_height - h - tz * level_height
    alpha = opacity[src]
    if (alpha == 1.0).all():
        color = (1.0, 1.0, 1.0, 1.0)                     # caso común: sin array por quad
    else:
        color = np.ones((len(ty), 4), dtype=np.float32)
        color[:, 3] = alpha

    vertices = quad_vertices(world_x, world_y, w, h, depth,
                             uv[:, 0], uv[:, 1], uv[:, 2], uv[:, 3], flips=flip, color=color)
    pages = np.unique(page)
    if pages.size == 1:                                  # caso común: atlas de una página
        return {int(pages[0]): vertices}
    vertices = vertices.reshape(-1, 4, vertices.shape[-1])
    return {int(p): vertices[page == p].reshape(-1, vertices.shape[-1]) for p in pages}
