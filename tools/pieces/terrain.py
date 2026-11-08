"""Mundo grande: terreno de alturas en las esquinas (tipo Populous) con casas encima.

1. Alturas en las esquinas de las casillas, en cuartos de planta (0.75 m): ruido suave de
   varias escalas, llanuras aplanadas y solares para las casas a una planta exacta. Una
   relajación garantiza que dos esquinas vecinas difieran como mucho en un cuarto.
2. Cada casilla se convierte en piezas: si sus cuatro esquinas están en un plano, una pieza
   entera (llano, pendiente suave o diagonal); si no, se parte por una diagonal y cada mitad es
   un medio suelo. Si las dos mitades caen en plantas distintas, la de abajo lleva una media
   celda maciza bajo la otra mitad.
3. Encima: casas de dos plantas (puerta al sur, escalera interior) y farolas.
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import tmx_manager as tm

from .catalog import build_catalog
from .render import DEFAULT_PROJECTION, Projection

QUARTERS_PER_LEVEL = 4
LAYER_CLASS = {"suelo": "floor", "piezas": "object", "techos": "overhead"}


# --- 1. Alturas -------------------------------------------------------------------------

def _smooth_noise(shape, cell, rng):
    """Ruido de valor: una rejilla aleatoria de paso ``cell`` interpolada (bilineal suavizada)."""
    h, w = shape
    gh, gw = h // cell + 2, w // cell + 2
    grid = rng.random((gh, gw))
    y = np.arange(h) / cell
    x = np.arange(w) / cell
    y0, x0 = y.astype(int), x.astype(int)
    ty, tx = y - y0, x - x0
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)            # suavizado
    a = grid[y0][:, x0]
    b = grid[y0][:, x0 + 1]
    c = grid[y0 + 1][:, x0]
    d = grid[y0 + 1][:, x0 + 1]
    top = a + (b - a) * tx[None, :]
    bottom = c + (d - c) * tx[None, :]
    return top + (bottom - top) * ty[:, None]


def relax(heights: np.ndarray, pinned: np.ndarray = None, max_iter: int = 400) -> np.ndarray:
    """Ajusta las alturas (enteras) para que dos esquinas vecinas difieran como mucho en 1,
    rebajando las que sobran y subiendo las que faltan junto a las fijadas (``pinned``)."""
    H = heights.astype(np.int64).copy()
    fixed = np.zeros_like(H, bool) if pinned is None else pinned
    target = H.copy()
    for _ in range(max_iter):
        old = H.copy()
        p = np.pad(H, 1, mode="edge")
        nb_min = np.minimum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        nb_max = np.maximum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        H = np.minimum(H, nb_min + 1)
        H = np.maximum(H, nb_max - 1)
        H[fixed] = target[fixed]
        if np.array_equal(H, old):
            break
    return H


def make_heights(width: int, depth: int, rng, max_levels: int = 6, plains: float = 0.22):
    """Alturas en cuartos de planta en las (depth+1) × (width+1) esquinas: colinas de laderas
    con la pendiente máxima y valles llanos (como en Populous)."""
    shape = (depth + 1, width + 1)
    n = sum(_smooth_noise(shape, s, rng) * a for s, a in ((64, 1.0), (32, 0.5), (16, 0.25), (8, 0.12)))
    n = (n - n.min()) / (n.max() - n.min())
    n = np.clip((n - plains) / (1 - plains), 0, 1)                          # llanuras abajo
    q = np.rint(n ** 1.15 * max_levels * QUARTERS_PER_LEVEL)
    return relax(q)


def _propagate(values: np.ndarray, op, step: int, clamp: int) -> np.ndarray:
    """Cota que crece (op = max, step = -1) o decrece (op = min, step = +1) en 1 por vecino,
    desde los valores dados, hasta estabilizarse (``clamp`` limita lo que no importa)."""
    V = values.copy()
    while True:
        p = np.pad(V, 1, mode="edge")
        nb = op.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        new = op(V, nb + step)
        new = np.maximum(new, clamp) if op is np.maximum else np.minimum(new, clamp)
        if np.array_equal(new, V):
            return V
        V = new


def flatten_plots(H: np.ndarray, plots: List[Tuple[int, int, int, int]]):
    """Aplana cada solar (x, y, ancho, fondo, en casillas) a una planta exacta y ajusta el
    terreno de alrededor: desde cada solar se propagan una cota inferior y otra superior (un
    cuarto por casilla de distancia) y el terreno se recorta entre ellas. Como todo cumple la
    regla de vecinas, el resultado también, y los solares conservan su altura."""
    lo_c, hi_c = int(H.min()) - 1, int(H.max()) + 1
    lower = np.full_like(H, lo_c)
    upper = np.full_like(H, hi_c)
    for x, y, w, d in plots:
        area = H[y:y + d + 1, x:x + w + 1]
        level_q = int(np.rint(area.mean() / QUARTERS_PER_LEVEL)) * QUARTERS_PER_LEVEL
        lower[y:y + d + 1, x:x + w + 1] = level_q
        upper[y:y + d + 1, x:x + w + 1] = level_q
    lower = _propagate(lower, np.maximum, -1, lo_c)
    upper = _propagate(upper, np.minimum, +1, hi_c)
    if (lower > upper).any():
        raise ValueError("Solares incompatibles: demasiado cerca para su diferencia de altura")
    return np.clip(H, lower, upper)


def choose_plots(H: np.ndarray, count: int, size=(6, 5), margin=4, rng=None, max_spread=3):
    """Solares en zonas poco inclinadas, separados entre sí."""
    rng = rng or np.random.default_rng()
    w, d = size
    depth, width = H.shape[0] - 1, H.shape[1] - 1
    plots = []
    for _ in range(count * 60):
        if len(plots) == count:
            break
        x = int(rng.integers(margin, width - w - margin))
        y = int(rng.integers(margin, depth - d - margin))
        area = H[y - 1:y + d + 2, x - 1:x + w + 2]
        if area.max() - area.min() > max_spread:
            continue
        if any(abs(x - px) < w + 2 * margin and abs(y - py) < d + 2 * margin for px, py, _, _ in plots):
            continue
        plots.append((x, y, w, d))
    return plots


# --- 2. Casillas -> piezas -----------------------------------------------------------------

SLOPE_DIR = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}
DIAG_DIR = {(1, -1): "NE", (1, 1): "SE", (-1, 1): "SW", (-1, -1): "NW"}
OTHER_HALF = {"nw": "se", "se": "nw", "ne": "sw", "sw": "ne"}


def plane_piece(part: str, a: int, b: int, c: int) -> Tuple[str, int]:
    """(nombre de pieza, planta) del plano h = a + b·u + c·v (en cuartos) sobre esa parte."""
    base = a + min(0, b) + min(0, c)                     # el punto más bajo de la celda
    level, frac = divmod(base, QUARTERS_PER_LEVEL)
    tag = f"{frac * 25:03d}"
    prefix = "floor" if part == "all" else f"floor_{part}"
    if b == 0 and c == 0:
        return f"{prefix}_flat_{tag}", level
    if (b, c) in SLOPE_DIR:
        return f"{prefix}_slope_{SLOPE_DIR[(b, c)]}_{tag}", level
    return f"{prefix}_ramp_{DIAG_DIR[(b, c)]}_{tag}", level


def tile_pieces(nw: int, ne: int, sw: int, se: int):
    """Piezas de una casilla: lista de (planta, nombre)."""
    if nw + se == ne + sw:                               # las cuatro esquinas en un plano
        name, level = plane_piece("all", nw, ne - nw, sw - nw)
        return [(level, name)]
    if abs(nw - se) <= abs(ne - sw):                     # partir por la diagonal más "plana"
        halves = [("ne", nw, ne - nw, se - ne), ("sw", nw, se - sw, sw - nw)]
    else:
        halves = [("nw", nw, ne - nw, sw - nw), ("se", se - (se - sw) - (se - ne), se - sw, se - ne)]
    placed = [(part, *plane_piece(part, a, b, c)) for part, a, b, c in halves]
    out = [(level, name) for _part, name, level in placed]
    (p1, _n1, l1), (p2, _n2, l2) = placed
    if l1 != l2:                                         # la mitad de abajo, maciza bajo la otra
        low_level, high_part = (l1, p2) if l1 < l2 else (l2, p1)
        out.append((low_level, f"fill_half_{high_part}"))
    return out


# --- Construcción del mapa con arrays ---------------------------------------------------

class GridBuilder:
    """Como CourseBuilder, pero con arrays: (planta, capa) -> array de GIDs."""

    def __init__(self, width: int, depth: int, pieces=None):
        self.width, self.depth = width, depth
        self.pieces = {p.name: p for p in (pieces or build_catalog())}
        self.gid = {name: i + 1 for i, name in enumerate(self.pieces)}
        self.layers: Dict[Tuple[int, str], List[np.ndarray]] = {}
        self.lights: Dict[int, list] = {}

    def put(self, level: int, x: int, y: int, name: str):
        base = "piezas" if name.startswith(("fill_", "wall_")) else \
            {"floor": "suelo", "object": "piezas", "overhead": "techos"}[self.pieces[name].layer]
        stack = self.layers.setdefault((level, base), [])
        for arr in stack:
            if arr[y, x] == 0:
                arr[y, x] = self.gid[name]
                return
        arr = np.zeros((self.depth, self.width), np.uint32)
        arr[y, x] = self.gid[name]
        stack.append(arr)

    def light(self, level, x, y, color="#ffd9a0", radius=6.0, intensity=1.5, height=2.2):
        self.lights.setdefault(level, []).append((x, y, color, radius, intensity, height))

    def to_map(self, tsx_path: Path, tmx_path: Path, proj: Projection) -> tm.TiledMap:
        m = tm.create_empty_map(self.width, self.depth, proj.tile_w, proj.row_h)
        m.tiledversion = "1.11.0"
        m.properties["level_height"] = tm.Property("level_height", "int", proj.level_h)
        m.tilesets.append(tm.Tileset(firstgid=1, name="pieces", tilewidth=proj.image_size[0],
                                     tileheight=proj.image_size[1],
                                     source=os.path.relpath(tsx_path, tmx_path.parent)))
        next_id, next_obj = 1, 1
        for level in sorted({lv for lv, _ in self.layers} | set(self.lights)):
            group = tm.LayerGroup(name=f"Planta {level}", id=next_id,
                                  properties={"Z": tm.Property("Z", "int", level)},
                                  offsety=-level * proj.level_h)
            next_id += 1
            for base in ("suelo", "piezas", "techos"):
                for k, arr in enumerate(self.layers.get((level, base), [])):
                    layer = tm.create_layer(base if k == 0 else f"{base}{k + 1}", self.width, self.depth)
                    layer.id, next_id = next_id, next_id + 1
                    layer.klass = LAYER_CLASS[base]
                    layer.data.tiles = tm.array.array("I", arr.ravel().tolist())
                    group.layers.append(layer)
            if level in self.lights:
                og = tm.ObjectGroup(name="luces", id=next_id)
                next_id += 1
                for x, y, color, radius, intensity, height in self.lights[level]:
                    obj = tm.MapObject(id=next_obj, name="luz", type="light", x=x * proj.tile_w,
                                       y=y * proj.row_h, properties={
                                           "color": tm.Property("color", "color", color),
                                           "radius": tm.Property("radius", "float", radius),
                                           "intensity": tm.Property("intensity", "float", intensity),
                                           "height": tm.Property("height", "float", height)})
                    obj.extra_children.append(ET.Element("point"))
                    og.objects.append(obj)
                    next_obj += 1
                group.layers.append(og)
            m.layers.append(group)
        m.extra_attrib.update({"nextlayerid": str(next_id), "nextobjectid": str(next_obj)})
        return m


# --- 3. Casas ---------------------------------------------------------------------------

def place_house(b: GridBuilder, x0: int, y0: int, w: int, d: int, level: int, rng):
    """Casa de dos plantas sobre un solar llano a la planta ``level``: paredes finas, puerta al
    sur, escalera interior contra la pared este (de sur a norte; la pared hace de barandilla por
    ese lado) y forjado con su hueco."""
    x1, y1 = x0 + w - 1, y0 + d - 1
    door = x0 + int(rng.integers(1, w - 2))
    stairs = [(x1, y1 - 1), (x1, y1 - 2)]                   # rampa N 0 y rampa N 0.5
    for lv in (level, level + 1):
        for x in range(x0, x1 + 1):
            b.put(lv, x, y0, "wall_N")
            if not (lv == level and x == door):
                b.put(lv, x, y1, "wall_S")
        for y in range(y0, y1 + 1):
            b.put(lv, x0, y, "wall_W")
            b.put(lv, x1, y, "wall_E")
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x, y) not in stairs:
                b.put(level + 1, x, y, "floor_flat_000")
    for sx, sy in stairs:
        b.put(level + 1, sx, sy, "wall_W")
    b.put(level + 1, stairs[0][0], stairs[0][1] + 1, "wall_N")
    b.light(level, x0 + w / 2, y0 + d / 2, "#ffe2b8", radius=5.0, intensity=1.4)
    b.light(level + 1, x0 + w / 2 - 1, y0 + d / 2, "#ffb070", radius=5.0, intensity=1.2)
    b.light(level, door + 0.5, y1 + 1.6, "#ffcf8a", radius=6.0, intensity=1.5)
    return stairs


@dataclass
class TerrainInfo:
    heights: np.ndarray
    plots: list


def build_terrain(tmx_path: Path, tsx_path: Path, width: int = 256, depth: int = 256,
                  houses: int = 24, seed: int = 1, proj: Projection = DEFAULT_PROJECTION,
                  pieces=None) -> TerrainInfo:
    rng = np.random.default_rng(seed)
    H = make_heights(width, depth, rng)
    plots = choose_plots(H, houses, rng=rng)
    H = flatten_plots(H, plots)
    b = GridBuilder(width, depth, pieces)

    stairs_cells = {}
    for x, y, w, d in plots:
        level = int(H[y, x]) // QUARTERS_PER_LEVEL
        for cell in place_house(b, x + 1, y + 1, w - 2, d - 2, level, rng):
            stairs_cells[cell] = level

    # Terreno: cada casilla, sus piezas (las de la escalera de las casas la sustituyen)
    for y in range(depth):
        for x in range(width):
            if (x, y) in stairs_cells:
                continue
            for level, name in tile_pieces(int(H[y, x]), int(H[y, x + 1]), int(H[y + 1, x]), int(H[y + 1, x + 1])):
                b.put(level, x, y, name)
    for (x, y), lv in stairs_cells.items():
        name = "floor_ramp_N_000" if (x, y + 1) not in stairs_cells else "floor_ramp_N_050"
        b.put(lv, x, y, name)

    # Farolas sueltas por el terreno
    for _ in range(width * depth // 1200):
        x, y = rng.uniform(2, width - 2), rng.uniform(2, depth - 2)
        level = int(H[int(y), int(x)]) // QUARTERS_PER_LEVEL
        b.light(level, x, y, "#ffcf8a", radius=7.0, intensity=1.5, height=2.5 + (H[int(y), int(x)] % 4) * 0.75)

    m = b.to_map(tsx_path, tmx_path, proj)
    tmx_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(tmx_path, encoding="base64", compression="zlib")
    return TerrainInfo(H, plots)
