"""Compilador: un mapa de Tiled (.tmx) -> mundo compilado (SQLite, ver engine.world.store).

Pasos:
 1. Integrar las capas en huecos por celda (World de tmx_explorer: plantas, roles, opacidad).
 2. Parte de celda de cada tile (propiedades del tileset -> CellSpec; reflejada si el tile
    está volteado en horizontal) y combinación de las partes de cada celda. Cada combinación
    distinta de tiles se resuelve una sola vez.
 3. Paredes sur y este -> norte y oeste de la vecina (fuera del mapa todo es macizo).
 4. Atlas: un hueco por tile usado, con su borde; color y, si el tileset los tiene, mapas de
    normales y de posición con la misma disposición.
 5. Columnas compactas por chunk, con paletas.
 6. Informe: errores (conflictos), avisos y estadísticas, también guardado en ``meta``.

El compilador es una herramienta: puede depender de Tiled y del visor antiguo. El motor
solo lee el resultado.
"""

import copy
import io
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

import tmx_manager as tm
from tmx_explorer.map.tileset_renderer import load_tileset_images
from tmx_explorer.map.world import World
from tmx_explorer.renderer.atlas import pack_shelves

from engine import png16
from engine.world import cell as C
from engine.world import visual as V
from engine.world.cellspec import CellConflict, CellSpec
from engine.world.chunk import ChunkData
from engine.world.materials import Materials, load_materials
from engine.world.store import FORMAT_VERSION, SQLiteWorldStore

COMPILER_VERSION = "0.1"
GID_MASK, FLIP_H_BIT, FLIP_OTHER_BITS = 0x0FFFFFFF, 0x80000000, 0x70000000
SPEC_KEYS = {"walls", "fill", "floor", "floor_part", "ceiling", "ceiling_part", "flags"}
ATLAS_MAX = 4096
# Borde de cada región del atlas (píxeles del borde replicados) y alineación de las regiones:
# 2^MIP_LEVELS, para que los niveles reducidos (mipmaps) hasta 1/2^MIP_LEVELS no mezclen tiles
# vecinos y cada región empiece en un texel exacto en todos los niveles.
MIP_LEVELS = 4
ATLAS_BORDER = 1 << MIP_LEVELS
MAX_LISTED = 20          # posiciones listadas por cada error en el informe


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self):
        return {"errors": self.errors, "warnings": self.warnings, "stats": self.stats}

    def summary(self) -> str:
        lines = [f"{len(self.errors)} errores, {len(self.warnings)} avisos"]
        lines += [f"  ERROR {e}" for e in self.errors] + [f"  aviso {w}" for w in self.warnings]
        lines += [f"  {k}: {v}" for k, v in self.stats.items()]
        return "\n".join(lines)


# --- 2. Partes de celda por tile -----------------------------------------------------

def tile_specs(tmx: tm.TiledMap, report: Report) -> Dict[int, CellSpec]:
    specs: Dict[int, CellSpec] = {}
    for ts in tmx.tilesets:
        for tile_id, tile in ts.tiles.items():
            props = {k: str(p.value) for k, p in tile.properties.items() if k in SPEC_KEYS}
            if not props:
                continue
            try:
                specs[ts.firstgid + tile_id] = CellSpec.from_properties(props)
            except (ValueError, KeyError) as e:
                report.errors.append(f"Tileset '{ts.name}', tile {tile_id}: {e}")
    return specs


# --- Materiales del suelo ------------------------------------------------------------

def tile_materials(tmx: tm.TiledMap, materials: Materials, report: Report):
    """(GID -> código de material, GIDs que SOLO llevan material: marcadores sin geometría)."""
    codes, markers = {}, set()
    for ts in tmx.tilesets:
        for tile_id, tile in ts.tiles.items():
            prop = tile.properties.get("material")
            if prop is None:
                continue
            gid = ts.firstgid + tile_id
            try:
                codes[gid] = materials.code(str(prop.value))
            except KeyError as e:
                report.errors.append(f"Tileset '{ts.name}', tile {tile_id}: {e}")
                continue
            if not any(k in tile.properties for k in SPEC_KEYS):
                markers.add(gid)
    return codes, markers


def _tile_layers(tmx: tm.TiledMap):
    """Todas las capas de tiles (también dentro de grupos)."""
    def walk(layers):
        for layer in layers:
            if isinstance(layer, tm.LayerGroup):
                yield from walk(layer.layers)
            elif isinstance(layer, tm.TileLayer):
                yield layer
    return list(walk(tmx.layers))


def cell_materials(tmx: tm.TiledMap, world: World, codes: Dict[int, int]) -> np.ndarray:
    """(H, D, W) uint8: el PEOR material (código menor) de los tiles de cada celda, 0 si ninguno.
    Se lee de las capas tal cual, antes de juntar los tiles en huecos: los marcadores no ocupan
    ninguno."""
    out = np.zeros((world.H, world.D, world.W), np.uint8)
    if not codes:
        return out
    table = np.zeros(max(codes) + 1, np.uint8)
    for gid, code in codes.items():
        table[gid] = code
    entries = World._collect_layers(world, tmx)
    for (layer, level, _role, _opacity, _name) in entries:
        z = level + world.level_offset
        gids = np.frombuffer(layer.dense_tiles(world.origin_x, world.origin_y, world.W, world.D),
                             np.uint32).reshape(world.D, world.W) & GID_MASK
        code = table[np.minimum(gids, len(table) - 1)] * (gids < len(table))
        mine = out[z]
        out[z] = np.where(code == 0, mine, np.where(mine == 0, code, np.minimum(mine, code)))
    return out


def strip_markers(tmx: tm.TiledMap, markers) -> int:
    """Quita del mapa (en memoria) los tiles que solo llevan material. Devuelve cuántos."""
    if not markers:
        return 0
    marker_ids = np.array(sorted(markers), np.uint32)
    removed = 0

    def clean(tiles):
        nonlocal removed
        arr = np.frombuffer(tiles, np.uint32).copy()
        hit = np.isin(arr & GID_MASK, marker_ids)
        if hit.any():
            removed += int(hit.sum())
            arr[hit] = 0
            return tm.array.array("I", arr.tolist())
        return tiles
    for layer in _tile_layers(tmx):
        if layer.is_infinite:
            for ch in layer.data.chunks:
                ch.tiles = clean(ch.tiles)
        else:
            layer.data.tiles = clean(layer.data.tiles)
    return removed


def apply_materials(obstacles: np.ndarray, materials_by_cell: np.ndarray) -> np.ndarray:
    """Guarda el material en las celdas con suelo (en las demás no significa nada)."""
    has_floor = (C.floor_code(obstacles, 1) != C.FLOOR_NONE) | (C.floor_code(obstacles, 2) != C.FLOOR_NONE)
    code = np.where(has_floor, materials_by_cell, 0).astype(np.uint32)
    return (obstacles & ~np.uint32(C.MATERIAL_MASK << C.MATERIAL_SHIFT)) | (code << np.uint32(C.MATERIAL_SHIFT))


def _positions(mask: np.ndarray, world: World) -> str:
    zs, ys, xs = np.nonzero(mask)
    listed = [f"(x={x + world.origin_x}, y={y + world.origin_y}, planta={world.get_level_value(z)})"
              for z, y, x in zip(zs[:MAX_LISTED], ys[:MAX_LISTED], xs[:MAX_LISTED])]
    more = f" y {len(zs) - MAX_LISTED} más" if len(zs) > MAX_LISTED else ""
    return ", ".join(listed) + more


def cell_values(world: World, specs: Dict[int, CellSpec], report: Report):
    """(valores (H, D, W) uint32 sin paredes S/E, pared sur (H, D, W), pared este (H, D, W))."""
    slots = world.tiles.reshape(-1, world.tiles.shape[-1])
    combos, inverse = np.unique(slots, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    values = np.zeros(len(combos), np.uint32)
    south = np.zeros(len(combos), bool)
    east = np.zeros(len(combos), bool)
    shape = world.tiles.shape[:-1]

    for i, combo in enumerate(combos):
        spec = CellSpec()
        try:
            for raw in combo:
                gid = int(raw) & GID_MASK
                part = specs.get(gid)
                if part is None:
                    continue
                if int(raw) & FLIP_H_BIT:
                    part = part.mirrored()
                spec = spec.merge(part)
            values[i] = spec.encode()
        except CellConflict as e:
            where = _positions((inverse == i).reshape(shape), world)
            report.errors.append(f"{e}: {where}")
            continue
        south[i], east[i] = "S" in spec.walls, "E" in spec.walls
        for problem in spec.problems():
            report.warnings.append(f"{problem}: {_positions((inverse == i).reshape(shape), world)}")

    if (slots & FLIP_OTHER_BITS).any():
        report.warnings.append("Tiles con volteo vertical o diagonal: se ignora (solo el horizontal "
                               "tiene sentido en vista oblicua)")
    return values[inverse].reshape(shape), south[inverse].reshape(shape), east[inverse].reshape(shape)


# --- 3. Paredes sur y este -----------------------------------------------------------

def transfer_walls(values: np.ndarray, south: np.ndarray, east: np.ndarray) -> np.ndarray:
    """Pared sur de (y, x) -> pared norte de (y+1, x); este -> oeste de (y, x+1)."""
    values = values.copy()
    values[:, 1:, :] |= np.where(south[:, :-1, :], C.WALL_N, 0).astype(np.uint32)
    values[:, :, 1:] |= np.where(east[:, :, :-1], C.WALL_W, 0).astype(np.uint32)
    return values          # las del borde del mapa caen fuera: fuera ya es macizo


# Puntos dentro de cada mitad (diagonal, mitad): cerca de sus tres esquinas y en el centro
_HALF_SAMPLES = {
    (0, 1): [(0.12, 0.12), (0.76, 0.12), (0.12, 0.76), (0.3, 0.3)],    # "/" mitad NO
    (0, 2): [(0.88, 0.88), (0.88, 0.24), (0.24, 0.88), (0.7, 0.7)],    # "/" mitad SE
    (1, 1): [(0.88, 0.12), (0.24, 0.12), (0.88, 0.76), (0.7, 0.3)],    # "\\" mitad NE
    (1, 2): [(0.12, 0.88), (0.12, 0.24), (0.76, 0.88), (0.3, 0.7)],    # "\\" mitad SO
}


def check_overflow(world: World, obstacles: np.ndarray, report: Report):
    """Una mitad cuyo suelo sobresale de su celda necesita libre esa zona de la celda de arriba
    (lo que haya en la otra mitad no estorba)."""
    blocked = np.zeros(obstacles.shape, bool)
    split = C.split(obstacles).astype(int)
    for half in (1, 2):
        over = np.nan_to_num(C.FLOOR_MAX[C.floor_code(obstacles, half)]) > 1.0
        over[-1] = False                                   # la planta más alta no tiene celda encima
        if not over.any():
            continue
        z, y, x = np.nonzero(over)
        above = obstacles[z + 1, y, x]
        hit = np.zeros(len(z), bool)
        for s_val in (0, 1):
            sel = split[z, y, x] == s_val
            for u, v in _HALF_SAMPLES[(s_val, half)]:
                occupied = np.isfinite(C.surface_at(above[sel], u, v)) | C.fill_contains(C.fill(above[sel]), u, v)
                hit[sel] |= occupied
        blocked[z[hit], y[hit], x[hit]] = True
    if blocked.any():
        report.warnings.append("Suelo que sobresale de su celda con esa zona de la celda de arriba ocupada: "
                               + _positions(blocked, world))


# --- 4. Atlas ------------------------------------------------------------------------

def _tileset_extra_maps(tmx: tm.TiledMap, tmx_path: Path):
    """GID -> {"normal": RGBA8, "position": RGBA16, "albedo": RGBA8} de los tilesets que los tengan."""
    out: Dict[int, Dict[str, object]] = {}
    for ts in tmx.tilesets:
        if not ts.image or ts.columns <= 0:
            continue
        base = tmx_path.parent / Path(ts.source).parent if ts.source else tmx_path.parent
        maps = {}
        for kind, prop in (("normal", "normal_map"), ("position", "position_map"), ("albedo", "albedo_map")):
            if prop in ts.properties:
                path = base / str(ts.properties[prop].value)
                maps[kind] = png16.read_rgba16(path) if kind == "position" else \
                    np.asarray(Image.open(path).convert("RGBA"))
        for tile_id in range(ts.tilecount):
            col, row = tile_id % ts.columns, tile_id // ts.columns
            x = ts.margin + col * (ts.tilewidth + ts.spacing)
            y = ts.margin + row * (ts.tileheight + ts.spacing)
            out[ts.firstgid + tile_id] = {k: m[y:y + ts.tileheight, x:x + ts.tilewidth]
                                          for k, m in maps.items()}
    return out


def build_atlas(tmx: tm.TiledMap, tmx_path: Path, used_gids: np.ndarray, report: Report):
    """(tabla GID -> región, filas de regiones, páginas {(página, tipo): png})."""
    images = load_tileset_images(tmx, tmx_path)
    extras = _tileset_extra_maps(tmx, tmx_path)
    gids = [int(g) for g in used_gids if int(g) in images
            and np.asarray(images[int(g)].convert("RGBA"))[..., 3].any()]     # sin nada visible: fuera
    missing = sorted(int(g) for g in used_gids if int(g) not in images)
    if missing:
        report.warnings.append(f"GIDs sin imagen (no se dibujan): {missing[:MAX_LISTED]}")

    b = ATLAS_BORDER

    def aligned(n):
        return -(-n // b) * b
    sizes = [(aligned(images[g].width + 2 * b), aligned(images[g].height + 2 * b)) for g in gids]
    placements, page_sizes = pack_shelves(sizes, ATLAS_MAX) if gids else ([], [])
    page_sizes = [(aligned(w), aligned(h)) for w, h in page_sizes]
    if any(pl.x % b or pl.y % b for pl in placements):
        raise AssertionError("Regiones del atlas sin alinear: los niveles reducidos mezclarían tiles")
    kinds = ["color"] + [k for k in ("normal", "position", "albedo")
                         if gids and all(k in extras.get(g, {}) for g in gids)]

    pages = {}
    for kind in kinds:
        dtype = np.uint16 if kind == "position" else np.uint8
        canvases = [np.zeros((h, w, 4), dtype) for w, h in page_sizes]
        for g, pl in zip(gids, placements):
            src = np.asarray(images[g].convert("RGBA")) if kind == "color" else extras[g][kind]
            h0, w0 = src.shape[:2]
            extra_h, extra_w = aligned(h0 + 2 * b) - (h0 + 2 * b), aligned(w0 + 2 * b) - (w0 + 2 * b)
            tile = np.pad(src, ((b, b + extra_h), (b, b + extra_w), (0, 0)), mode="edge")
            canvases[pl.page][pl.y:pl.y + tile.shape[0], pl.x:pl.x + tile.shape[1]] = tile
        for page, canvas in enumerate(canvases):
            if kind == "position":
                pages[page, kind] = png16.encode_rgba16(canvas)
            else:
                buf = io.BytesIO()
                Image.fromarray(canvas).save(buf, format="PNG")
                pages[page, kind] = buf.getvalue()

    table = np.zeros(max([0] + [int(g) for g in used_gids]) + 1, np.uint32)
    rows = []
    for rid, (g, pl) in enumerate(zip(gids, placements), start=1):
        table[g] = rid
        rows.append((rid, pl.page, pl.x + b, pl.y + b, images[g].width, images[g].height, b))
    return table, rows, pages


def visual_values(world: World, region_of_gid: np.ndarray) -> np.ndarray:
    tiles = world.tiles
    gid = tiles & GID_MASK
    region = region_of_gid[np.minimum(gid, len(region_of_gid) - 1)] * (gid < len(region_of_gid))
    roles = np.stack([world.role_of(k) for k in range(tiles.shape[-1])], axis=-1)
    opacity_of = np.ones(256)
    opacity_of[:len(world.layers)] = [l.opacity for l in world.layers]
    return V.pack(region, roles, (tiles & FLIP_H_BIT) != 0, opacity_of[world.source])


# --- Luces ----------------------------------------------------------------------------

LIGHT_DEFAULTS = {"radius": 6.0, "intensity": 1.5, "height": 2.2, "color": "#ffd9a0"}


def _parse_color(value) -> tuple:
    """'#RRGGBB' o '#AARRGGBB' (sRGB) -> (r, g, b) lineales."""
    text = str(value).lstrip("#")
    if len(text) == 8:
        text = text[2:]
    r, g, b = (int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(c ** 2.2 for c in (r, g, b))


def collect_lights(tmx: tm.TiledMap, world: World, cell_w: float, cell_h: float, report: Report):
    """Objetos de clase "light" de las capas de objetos; su planta es la del grupo (propiedad Z)."""
    rows = []

    def visit(layers, level):
        for layer in layers:
            if isinstance(layer, tm.LayerGroup):
                z = layer.properties.get("Z")
                visit(layer.layers, int(z.value) if z is not None else level)
            elif isinstance(layer, tm.ObjectGroup):
                for obj in layer.objects:
                    if obj.type != "light":
                        continue
                    props = {k: p.value for k, p in obj.properties.items()}
                    try:
                        r, g, b = _parse_color(props.get("color", LIGHT_DEFAULTS["color"]))
                        radius = float(props.get("radius", LIGHT_DEFAULTS["radius"]))
                        intensity = float(props.get("intensity", LIGHT_DEFAULTS["intensity"]))
                        height = float(props.get("height", LIGHT_DEFAULTS["height"]))
                    except (TypeError, ValueError) as e:
                        report.warnings.append(f"Luz '{obj.name or obj.id}' con propiedades no válidas: {e}")
                        continue
                    x = obj.x / tmx.tilewidth * cell_w
                    z = obj.y / tmx.tileheight * cell_w
                    rows.append((x, level * cell_h + height, z, r, g, b, radius, intensity))

    visit(tmx.layers, 0)
    return rows


# --- 5. Columnas y chunks ------------------------------------------------------------

def build_chunks(world: World, obstacles: np.ndarray, visual: np.ndarray, size: int):
    content = (obstacles != 0) | visual.any(axis=-1)              # (H, D, W)
    has = content.any(axis=0)
    first = np.argmax(content, axis=0)
    last = content.shape[0] - 1 - np.argmax(content[::-1], axis=0)
    base_level = world.get_level_value(0)

    x0, y0 = world.origin_x, world.origin_y
    for cy in range(math.floor(y0 / size), math.floor((y0 + world.D - 1) / size) + 1):
        for cx in range(math.floor(x0 / size), math.floor((x0 + world.W - 1) / size) + 1):
            gy = cy * size + np.arange(size)[:, None] - y0          # índices en el mundo integrado
            gx = cx * size + np.arange(size)[None, :] - x0
            inside = (gy >= 0) & (gy < world.D) & (gx >= 0) & (gx < world.W)
            iy, ix = np.where(inside, gy, 0), np.where(inside, gx, 0)
            col_has = (has[iy, ix] & inside).ravel()
            if not col_has.any():
                continue                                             # chunk vacío: no se guarda
            lo, hi = first[iy, ix].ravel(), last[iy, ix].ravel()
            col_len = np.where(col_has, hi - lo + 1, 0).astype(np.uint8)
            col_start = np.where(col_has, lo + base_level, base_level).astype(np.int16)

            col_of_cell = np.repeat(np.arange(size * size), col_len)
            offset = np.concatenate([[0], np.cumsum(col_len, dtype=np.int64)[:-1]])
            z = lo[col_of_cell] + (np.arange(len(col_of_cell)) - offset[col_of_cell])
            cy_idx, cx_idx = iy.ravel()[col_of_cell], ix.ravel()[col_of_cell]
            yield ChunkData(cx, cy, size, col_start, col_len,
                            obstacles[z, cy_idx, cx_idx], visual[z, cy_idx, cx_idx])


# --- Todo junto ----------------------------------------------------------------------

def compile_tmx(tmx_path, out_path, chunk_size: int = 32, compression: str = "zlib",
                force: bool = False, materials: Materials = None) -> Report:
    t0 = time.perf_counter()
    tmx_path, out_path = Path(tmx_path), Path(out_path)
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError("El tamaño de chunk debe ser potencia de dos")
    report = Report()

    tmx = tm.TiledMap.load(tmx_path)
    materials = materials if materials is not None else load_materials()
    codes, markers = tile_materials(tmx, materials, report)
    original = copy.deepcopy(tmx) if markers else tmx
    report.stats["marcadores_de_material"] = strip_markers(tmx, markers)
    world = World(tmx)
    if world.dropped_tiles:
        report.warnings.append(f"{world.dropped_tiles} tiles descartados por celdas con más de 4")
    if world.H > 255:
        report.errors.append(f"{world.H} plantas: una columna admite como mucho 255")

    specs = tile_specs(tmx, report)
    values, south, east = cell_values(world, specs, report)
    obstacles = transfer_walls(values, south, east)
    check_overflow(world, obstacles, report)
    obstacles = apply_materials(obstacles, cell_materials(original, world, codes))

    used = np.unique(world.tiles & GID_MASK)
    region_of_gid, regions, pages = build_atlas(tmx, tmx_path, used[used != 0], report)
    visual = visual_values(world, region_of_gid)

    if report.errors and not force:
        report.stats["tiempo_s"] = round(time.perf_counter() - t0, 2)
        return report

    chunks = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with SQLiteWorldStore(out_path, "w") as store:
        store.set_meta({"format_version": FORMAT_VERSION, "compression": compression})
        for chunk in build_chunks(world, obstacles, visual, chunk_size):
            store.write_chunk(chunk)
            chunks += 1
        store.write_atlas(regions, pages)
        cell_w = float(next((p.value for ts in tmx.tilesets for k, p in ts.properties.items()
                             if k == "cell_width_m"), 2.0))
        cell_h = float(next((p.value for ts in tmx.tilesets for k, p in ts.properties.items()
                             if k == "cell_height_m"), 3.0))
        lights = collect_lights(tmx, world, cell_w, cell_h, report)
        store.write_lights(lights)
        report.stats["luces"] = len(lights)
        props = {k: p.value for ts in tmx.tilesets for k, p in ts.properties.items()}
        report.stats.update({
            "celdas_con_contenido": int(((obstacles != 0) | visual.any(-1)).sum()),
            "combinaciones_de_tiles": int(len(np.unique(world.tiles.reshape(-1, 4), axis=0))),
            "chunks": chunks, "regiones_de_atlas": len(regions),
            "paginas_de_atlas": len({p for p, _ in pages}),
            "mapas_del_atlas": sorted({k for _, k in pages}),
        })
        store.set_meta({
            "chunk_size": chunk_size,
            "bounds": {"x0": world.origin_x, "y0": world.origin_y, "width": world.W, "depth": world.D,
                       "level_min": world.get_level_value(0),
                       "level_max": world.get_level_value(world.H - 1)},
            "cell_width_m": float(props.get("cell_width_m", 2.0)),
            "cell_height_m": float(props.get("cell_height_m", 3.0)),
            "tile_width_px": tmx.tilewidth, "row_height_px": tmx.tileheight,
            "level_height_px": world.level_height,
            "position_range_m": props.get("position_range_m"),
            "solid_value": C.CELL_SOLID,
            "source": str(tmx_path), "compiled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "compiler_version": COMPILER_VERSION,
            "materials": materials.to_json(),
        })
        report.stats["tiempo_s"] = round(time.perf_counter() - t0, 2)
        store.set_meta({"report": report.as_dict()})
    report.stats["tamaño_kb"] = round(out_path.stat().st_size / 1024, 1)
    return report
