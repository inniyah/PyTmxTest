"""Grafo de navegación: qué superficies se pueden pisar y entre cuáles se puede andar.

Se construye a partir del mundo compilado, con la misma física que mueve a los personajes, para
que el grafo diga lo mismo que el personaje puede hacer de verdad:

- **Nodos**: en el centro de cada celda, cada superficie de suelo en la que se puede estar de pie
  (nada macizo en los pies, ``height`` metros libres encima). Una columna puede tener varios: la
  calle bajo un puente y el tablero, las dos plantas de una casa.
- **Aristas** hacia los 8 vecinos. El segmento entre los dos centros se recorre en pasos de
  ``sample`` metros con la búsqueda de suelo de la física: cada paso sube como mucho ``step`` y
  baja como mucho ``drop`` (más sería caerse: posible, pero no "transitable"), con altura libre
  en cada punto. Cortan el paso las paredes del borde que se cruza (en las plantas que ocupa el
  cuerpo) y las paredes diagonales finas. Una diagonal solo vale si también se puede ir por los
  dos lados (sin cortar esquinas) y llegan al mismo nodo.

Además se mide la **holgura** (distancia al obstáculo más cercano, con los mismos segmentos que
usa la física): la de cada nodo y la mínima a lo largo de cada arista. Cada clase de tamaño usa
las aristas en las que cabe su radio; los nodos en los que no cabe ni el más pequeño se
descartan (por ejemplo, el centro de una celda medio maciza).

Se guarda por chunks (tabla ``nav`` del mundo compilado): para cada nodo, su columna dentro del
chunk, su altura, su material, su holgura (cm) y, para cada dirección, el índice del nodo de
llegada dentro de su columna (-1 si no hay paso) y la holgura de la arista (cm).
"""

import io
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from engine.physics import WalkerPhysics
from engine.physics.walker import FILL_EDGES
from engine.world import cell as C

# Direcciones: E, S, O, N y las diagonales SE, SO, NO, NE (X este, Y sur)
DIRS = np.array([(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)], np.int64)
ORTHO = 4
# Cada diagonal, por sus dos lados: (dirección X, dirección Y)
DIAG_SIDES = {4: (0, 1), 5: (2, 1), 6: (2, 3), 7: (0, 3)}


@dataclass(frozen=True)
class NavParams:
    step: float = 0.3          # lo que se puede subir de un paso a otro (m)
    drop: float = 0.3          # lo que se puede bajar sin caerse (m)
    height: float = 1.75       # altura libre necesaria (m)
    sample: float = 0.25       # distancia entre puntos al recorrer un segmento (m)
    match: float = 0.05        # tolerancia para reconocer el nodo de llegada (m)
    min_radius: float = 0.2    # radio del personaje más pequeño: holgura mínima de nodos y aristas
    reach: float = 1.0         # hasta dónde se mide la holgura (m); más lejos cuenta como libre


@dataclass
class NavNodes:
    """Nodos de todo el mundo (arrays planos) y sus aristas."""
    x: np.ndarray              # (n,) celda
    y: np.ndarray
    h: np.ndarray              # (n,) altura de la superficie (m)
    k: np.ndarray              # (n,) índice dentro de su columna (de abajo arriba)
    material: np.ndarray       # (n,) código de material
    target: np.ndarray         # (n, 8) nodo de llegada (índice global), -1 sin paso
    clear: np.ndarray = None   # (n,) holgura en el centro del nodo (m)
    edge_clear: np.ndarray = None   # (n, 8) holgura mínima a lo largo de cada arista (m)


def clearance(view, physics, px, pz, h, p: NavParams):
    """Distancia (m) desde cada punto al obstáculo más cercano para un cuerpo de pie a altura
    ``h``: paredes y bordes de los macizos de las plantas que ocupa (los mismos segmentos que usa la
    física), y 0 si el punto está dentro de un macizo. Más allá de ``p.reach`` cuenta como libre."""
    cw, chh = physics.p.cell_w, physics.p.cell_h
    px, pz, h = (np.atleast_1d(np.asarray(a, float)) for a in (px, pz, h))
    n = len(px)
    c0x = np.floor((px - p.reach) / cw).astype(np.int64)
    c0y = np.floor((pz - p.reach) / cw).astype(np.int64)
    lo = np.floor((h + p.step) / chh).astype(np.int64)
    hi = np.floor((h + p.height - 1e-3) / chh).astype(np.int64)
    span = int(np.ceil(2 * p.reach / cw)) + 1
    best = np.full(n, p.reach)
    for dl in range(int((hi - lo).max(initial=0)) + 1):
        level = lo + dl
        use = level <= hi
        for i in range(span):
            for j in range(span):
                cx, cy = c0x + i, c0y + j
                cells = view.cells(cx, cy, level)
                fill, wn, ww = C.fill(cells), C.wall_n(cells), C.wall_w(cells)
                # Solo los puntos con algún obstáculo en esta celda (en el terreno, casi ninguno)
                rows = np.nonzero(use & ((fill != 0) | (wn != 0) | (ww != 0)))[0]
                if rows.size == 0:
                    continue
                m = rows.size
                segs = np.concatenate([FILL_EDGES[fill[rows]], np.full((m, 2, 4), np.nan)], axis=1)
                segs[:, 4, :] = np.where(wn[rows][:, None], [0, 0, 1, 0], np.nan)
                segs[:, 5, :] = np.where(ww[rows][:, None], [0, 0, 0, 1], np.nan)
                ox, oy = cx[rows][:, None], cy[rows][:, None]
                ax, az = (segs[..., 0] + ox) * cw, (segs[..., 1] + oy) * cw
                bx, bz = (segs[..., 2] + ox) * cw, (segs[..., 3] + oy) * cw
                dx, dz = bx - ax, bz - az
                qx, qz = px[rows][:, None], pz[rows][:, None]
                t = np.clip(((qx - ax) * dx + (qz - az) * dz) / np.maximum(dx * dx + dz * dz, 1e-12), 0, 1)
                d = np.hypot(qx - (ax + t * dx), qz - (az + t * dz))
                d = np.where(np.isfinite(d), d, np.inf).min(axis=1)
                best[rows] = np.minimum(best[rows], d)
    # Dentro de un macizo (la parte de la celda en la que están los pies): sin holgura
    gx, gz = px / cw, pz / cw
    cx, cz = np.floor(gx).astype(np.int64), np.floor(gz).astype(np.int64)
    feet = view.cells(cx, cz, np.floor((h + 0.1) / chh).astype(np.int64))
    inside = C.fill_contains(C.fill(feet), gx - cx, gz - cz)
    return np.where(inside, 0.0, best)


def _candidate_cells(view):
    """Celdas con suelo en su centro: (x, y, planta, valor)."""
    xs, ys, ls, vs = [], [], [], []
    n = view.chunk_size
    for cx, cy in view.store.chunks():
        ch = view.chunk(cx, cy)
        lens = ch.col_len.astype(np.int64)
        col = np.repeat(np.arange(n * n), lens)
        if col.size == 0:
            continue
        level = ch.col_start.astype(np.int64)[col] + (np.arange(col.size) - ch.offset[col])
        values = ch.obstacles
        has = np.isfinite(C.surface_at(values, 0.5, 0.5))
        xs.append(cx * n + (col % n)[has])
        ys.append(cy * n + (col // n)[has])
        ls.append(level[has])
        vs.append(values[has])
    if not xs:
        e = np.zeros(0, np.int64)
        return e, e, e, np.zeros(0, np.uint32)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ls), np.concatenate(vs)


def find_nodes(view, physics: WalkerPhysics, p: NavParams):
    """Nodos: superficies en el centro de las celdas en las que se puede estar de pie."""
    cw, chh = physics.p.cell_w, physics.p.cell_h
    x, y, level, values = _candidate_cells(view)
    h = (level + C.surface_at(values, 0.5, 0.5)) * chh
    px, pz = (x + 0.5) * cw, (y + 0.5) * cw
    # Que sea la superficie de verdad en ese punto (nada por encima a menos de un escalón) y libre
    top = physics.support(px, pz, h + 0.01, search=2)
    ok = np.abs(top - h) < 1e-3
    ok &= physics.ceiling(px, pz, h) - h >= p.height
    clear = clearance(view, physics, px, pz, h, p)
    ok &= clear >= p.min_radius                      # que quepa al menos el más pequeño
    x, y, h, material = x[ok], y[ok], h[ok], ((values[ok] >> C.MATERIAL_SHIFT) & C.MATERIAL_MASK)
    clear = clear[ok]
    # Una superficie por altura en cada columna (la de una celda partida puede salir dos veces)
    key = np.stack([x, y, np.round(h * 100).astype(np.int64)], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    x, y, h, material, clear = x[first], y[first], h[first], material[first], clear[first]
    order = np.lexsort((h, x, y))
    x, y, h, material, clear = x[order], y[order], h[order], material[order], clear[order]
    new_col = np.ones(len(x), bool)
    new_col[1:] = (x[1:] != x[:-1]) | (y[1:] != y[:-1])
    starts = np.maximum.accumulate(np.where(new_col, np.arange(len(x)), 0))
    k = np.arange(len(x)) - starts
    return x, y, h, k, material.astype(np.uint8), clear


class _Columns:
    """Búsqueda de nodos por columna: (x, y) -> rango [inicio, fin) en los arrays ordenados."""

    def __init__(self, x, y):
        self.key = (y.astype(np.int64) << 32) | (x.astype(np.int64) & 0xFFFFFFFF)
        self.order = np.argsort(self.key, kind="stable")
        self.sorted = self.key[self.order]

    def ranges(self, x, y):
        key = (np.asarray(y, np.int64) << 32) | (np.asarray(x, np.int64) & 0xFFFFFFFF)
        return np.searchsorted(self.sorted, key, "left"), np.searchsorted(self.sorted, key, "right")


def _match(columns: _Columns, h_nodes, x, y, h_end, tol):
    """Nodo de la columna (x, y) a la altura ``h_end`` (± tol), o -1."""
    lo, hi = columns.ranges(x, y)
    out = np.full(len(x), -1, np.int64)
    width = hi - lo
    for j in range(int(width.max(initial=0))):
        has = j < width
        idx = columns.order[np.minimum(lo + j, len(columns.order) - 1)]
        hit = has & (out < 0) & (np.abs(h_nodes[idx] - h_end) <= tol)
        out[hit] = idx[hit]
    return out


def _walk(view, physics, p: NavParams, x, y, h, d):
    """Recorre el segmento desde el centro de (x, y) en la dirección ``d``. Devuelve (posible,
    altura de llegada)."""
    cw, chh = physics.p.cell_w, physics.p.cell_h
    dx, dy = d
    length = cw * float(np.hypot(dx, dy))
    steps = int(np.ceil(length / p.sample))
    x0, z0 = (x + 0.5) * cw, (y + 0.5) * cw
    ok = np.ones(len(x), bool)
    prev = h.copy()
    prev_cell = np.stack([x, y], axis=1)
    prev_side = np.zeros(len(x))
    min_clear = np.full(len(x), p.reach)
    for i in range(1, steps + 1):
        t = i / steps
        px, pz = x0 + dx * cw * t, z0 + dy * cw * t
        hi = physics.support(px, pz, prev + p.step + 0.01, search=2)
        ok &= (hi > prev - p.drop - 1e-6) & (hi > -1e8)
        hi = np.where(ok, hi, prev)
        ok &= physics.ceiling(px, pz, hi) - hi >= p.height
        # Paredes diagonales finas: cambiar de lado de la diagonal dentro de la misma celda
        gx, gz = px / cw, pz / cw
        # en el borde exacto entre celdas, se cuenta la de llegada
        cx = np.floor(gx - 1e-9 * dx).astype(np.int64) if i == steps else np.floor(gx).astype(np.int64)
        cz = np.floor(gz - 1e-9 * dy).astype(np.int64) if i == steps else np.floor(gz).astype(np.int64)
        u, v = gx - cx, gz - cz
        cell = view.cells(cx, cz, np.floor((hi + 0.1) / chh).astype(np.int64))
        fill = C.fill(cell)
        side = np.where(fill == C.FILLS.index("thin_slash"), np.sign(u + v - 1.0),
                        np.where(fill == C.FILLS.index("thin_backslash"), np.sign(u - v), 0.0))
        same = (cx == prev_cell[:, 0]) & (cz == prev_cell[:, 1])
        ok &= ~(same & (side * prev_side < 0))
        prev_side = side
        prev_cell = np.stack([cx, cz], axis=1)
        prev = hi
        min_clear = np.minimum(min_clear, clearance(view, physics, px, pz, hi, p))
    ok &= min_clear >= p.min_radius
    return ok, prev, min_clear


def _wall_between(view, physics, p: NavParams, x, y, h, d):
    """¿Hay pared en el borde que se cruza al ir de (x, y) en la dirección ortogonal ``d``, en
    alguna de las plantas que ocupa el cuerpo?"""
    chh = physics.p.cell_h
    dx, dy = d
    blocked = np.zeros(len(x), bool)
    lo = np.floor((h + 0.01) / chh).astype(np.int64)
    hi = np.floor((h + p.height - 0.01) / chh).astype(np.int64)
    for k in range(int((hi - lo).max(initial=0)) + 1):
        lv = lo + k
        inside = lv <= hi
        if dx == 1:
            cell = view.cells(x + 1, y, lv)
            wall = C.wall_w(cell)
        elif dx == -1:
            cell = view.cells(x, y, lv)
            wall = C.wall_w(cell)
        elif dy == 1:
            cell = view.cells(x, y + 1, lv)
            wall = C.wall_n(cell)
        else:
            cell = view.cells(x, y, lv)
            wall = C.wall_n(cell)
        blocked |= inside & (wall != 0)
    return blocked


def build_nav(view, params: NavParams = None) -> NavNodes:
    physics = WalkerPhysics(view)
    p = params or NavParams()
    x, y, h, k, material, clear = find_nodes(view, physics, p)
    columns = _Columns(x, y)
    target = np.full((len(x), 8), -1, np.int64)
    edge_clear = np.zeros((len(x), 8))
    for di in range(ORTHO):
        d = tuple(DIRS[di])
        ok, h_end, along = _walk(view, physics, p, x, y, h, d)
        ok &= ~_wall_between(view, physics, p, x, y, h, d)
        t = _match(columns, h, x + d[0], y + d[1], h_end, p.match)
        target[:, di] = np.where(ok, t, -1)
        edge_clear[:, di] = np.where(ok, along, 0.0)
    for di in range(ORTHO, 8):
        d = tuple(DIRS[di])
        ok, h_end, along = _walk(view, physics, p, x, y, h, d)
        t = _match(columns, h, x + d[0], y + d[1], h_end, p.match)
        a, b = DIAG_SIDES[di]
        idx = np.arange(len(x))
        via_a = target[idx, a]
        via_b = target[idx, b]
        end_a = np.where(via_a >= 0, target[np.maximum(via_a, 0), b], -1)
        end_b = np.where(via_b >= 0, target[np.maximum(via_b, 0), a], -1)
        ok &= (t >= 0) & (end_a == t) & (end_b == t)
        target[:, di] = np.where(ok, t, -1)
        edge_clear[:, di] = np.where(ok, along, 0.0)
    return NavNodes(x, y, h, k, material, target, clear, edge_clear)


# --- Guardar y leer por chunks -----------------------------------------------------------

def chunk_blobs(nodes: NavNodes, chunk_size: int) -> Dict[Tuple[int, int], bytes]:
    """Para cada chunk: columna local (uint16), altura (float32), material (uint8) y, por
    dirección, el índice del nodo de llegada dentro de SU columna (int8, -1 sin paso)."""
    shift = chunk_size.bit_length() - 1
    cx, cy = nodes.x >> shift, nodes.y >> shift
    local = ((nodes.y & (chunk_size - 1)) * chunk_size + (nodes.x & (chunk_size - 1))).astype(np.uint16)
    k_target = np.where(nodes.target >= 0, nodes.k[np.maximum(nodes.target, 0)], -1).astype(np.int8)
    out = {}
    keys = np.stack([cx, cy], axis=1)
    for key in np.unique(keys, axis=0):
        sel = (cx == key[0]) & (cy == key[1])
        order = np.lexsort((nodes.h[sel], local[sel]))
        buf = io.BytesIO()
        cm = lambda a: np.clip(np.rint(a * 100), 0, 255).astype(np.uint8)    # noqa: E731
        np.savez_compressed(buf, col=local[sel][order], h=nodes.h[sel][order].astype(np.float32),
                            material=nodes.material[sel][order], target=k_target[sel][order],
                            clear=cm(nodes.clear[sel][order]), edge_clear=cm(nodes.edge_clear[sel][order]))
        out[int(key[0]), int(key[1])] = buf.getvalue()
    return out


def write_nav(store, nodes: NavNodes, chunk_size: int):
    for (cx, cy), data in chunk_blobs(nodes, chunk_size).items():
        store.write_nav(cx, cy, data)
    store.set_meta({"nav": {"nodes": int(len(nodes.x)), "params": NavParams().__dict__}})
