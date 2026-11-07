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
from engine.world.store import FORMAT_VERSION, SQLiteWorldStore

COMPILER_VERSION = "0.1"
GID_MASK, FLIP_H_BIT, FLIP_OTHER_BITS = 0x0FFFFFFF, 0x80000000, 0x70000000
SPEC_KEYS = {"walls", "fill", "floor", "floor_part", "ceiling", "ceiling_part", "flags"}
ATLAS_MAX = 4096
ATLAS_BORDER = 1
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


def check_overflow(world: World, obstacles: np.ndarray, report: Report):
    """Una rampa que sobresale de su celda necesita la celda de arriba libre de geometría."""
    top = np.fmax(C.FLOOR_MAX[C.floor_code(obstacles, 1)], C.FLOOR_MAX[C.floor_code(obstacles, 2)])
    overflow = np.nan_to_num(top) > 1.0
    blocked = np.zeros_like(overflow)
    blocked[:-1] = overflow[:-1] & ((obstacles[1:] & C.GEOMETRY_MASK) != 0)
    if blocked.any():
        report.warnings.append("Rampa que sobresale de su celda con la celda de arriba ocupada: "
                               + _positions(blocked, world))


# --- 4. Atlas ------------------------------------------------------------------------

def _tileset_extra_maps(tmx: tm.TiledMap, tmx_path: Path):
    """GID -> {"normal": PIL, "position": array uint16} para los tilesets que tengan esos mapas."""
    out: Dict[int, Dict[str, object]] = {}
    for ts in tmx.tilesets:
        if not ts.image or ts.columns <= 0:
            continue
        base = tmx_path.parent / Path(ts.source).parent if ts.source else tmx_path.parent
        maps = {}
        for kind, prop in (("normal", "normal_map"), ("position", "position_map")):
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
    gids = [int(g) for g in used_gids if int(g) in images]
    missing = sorted(set(int(g) for g in used_gids) - set(gids))
    if missing:
        report.warnings.append(f"GIDs sin imagen (no se dibujan): {missing[:MAX_LISTED]}")

    b = ATLAS_BORDER
    sizes = [(images[g].width + 2 * b, images[g].height + 2 * b) for g in gids]
    placements, page_sizes = pack_shelves(sizes, ATLAS_MAX) if gids else ([], [])
    kinds = ["color"] + [k for k in ("normal", "position") if gids and all(k in extras.get(g, {}) for g in gids)]

    pages = {}
    for kind in kinds:
        dtype = np.uint16 if kind == "position" else np.uint8
        canvases = [np.zeros((h, w, 4), dtype) for w, h in page_sizes]
        for g, pl in zip(gids, placements):
            src = np.asarray(images[g].convert("RGBA")) if kind == "color" else extras[g][kind]
            tile = np.pad(src, ((b, b), (b, b), (0, 0)), mode="edge")
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
    for rid, (g, pl, (w, h)) in enumerate(zip(gids, placements, sizes), start=1):
        table[g] = rid
        rows.append((rid, pl.page, pl.x + b, pl.y + b, w - 2 * b, h - 2 * b, b))
    return table, rows, pages


def visual_values(world: World, region_of_gid: np.ndarray) -> np.ndarray:
    tiles = world.tiles
    gid = tiles & GID_MASK
    region = region_of_gid[np.minimum(gid, len(region_of_gid) - 1)] * (gid < len(region_of_gid))
    roles = np.stack([world.role_of(k) for k in range(tiles.shape[-1])], axis=-1)
    opacity_of = np.ones(256)
    opacity_of[:len(world.layers)] = [l.opacity for l in world.layers]
    return V.pack(region, roles, (tiles & FLIP_H_BIT) != 0, opacity_of[world.source])


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
                force: bool = False) -> Report:
    t0 = time.perf_counter()
    tmx_path, out_path = Path(tmx_path), Path(out_path)
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError("El tamaño de chunk debe ser potencia de dos")
    report = Report()

    tmx = tm.TiledMap.load(tmx_path)
    world = World(tmx)
    if world.dropped_tiles:
        report.warnings.append(f"{world.dropped_tiles} tiles descartados por celdas con más de 4")
    if world.H > 255:
        report.errors.append(f"{world.H} plantas: una columna admite como mucho 255")

    specs = tile_specs(tmx, report)
    values, south, east = cell_values(world, specs, report)
    obstacles = transfer_walls(values, south, east)
    check_overflow(world, obstacles, report)

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
        })
        report.stats["tiempo_s"] = round(time.perf_counter() - t0, 2)
        store.set_meta({"report": report.as_dict()})
    report.stats["tamaño_kb"] = round(out_path.stat().st_size / 1024, 1)
    return report
