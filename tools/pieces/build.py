"""Genera las piezas (.obj), el tileset (color, normales, posición + .tsx) y la galería (.tmx).

    python -m tools.pieces --out assets/pieces --gallery maps/gallery/gallery.tmx
"""

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

import tmx_manager as tm

from engine import png16
from .catalog import Piece, build_catalog
from .mesh import mesh_for
from .objio import MTL_NAME, read_obj, write_mtl, write_obj
from .render import DEFAULT_PROJECTION, Projection, render

COLUMNS = 16
TILESET_NAME = "pieces"


def build_tileset(out: Path, proj: Projection = DEFAULT_PROJECTION, pieces: List[Piece] = None) -> Path:
    """Escribe obj/, los tres atlas y pieces.tsx en ``out``. Devuelve la ruta del .tsx."""
    pieces = pieces or build_catalog()
    obj_dir = out / "obj"
    obj_dir.mkdir(parents=True, exist_ok=True)
    write_mtl(obj_dir / MTL_NAME)

    tw, th = proj.image_size
    rows = math.ceil(len(pieces) / COLUMNS)
    color = np.zeros((rows * th, COLUMNS * tw, 4), np.uint8)
    normal = np.zeros_like(color)
    position = np.zeros((rows * th, COLUMNS * tw, 4), np.uint16)

    for i, piece in enumerate(pieces):
        obj_path = obj_dir / f"{piece.name}.obj"
        write_obj(obj_path, mesh_for(piece.spec, proj.cell), piece.name)
        c, n, p = render(read_obj(obj_path), proj)          # el tile sale del .obj
        r, col = divmod(i, COLUMNS)
        sl = (slice(r * th, (r + 1) * th), slice(col * tw, (col + 1) * tw))
        color[sl], normal[sl], position[sl] = c, n, p

    Image.fromarray(color).save(out / f"{TILESET_NAME}.png")
    Image.fromarray(normal).save(out / f"{TILESET_NAME}.n.png")
    png16.write_rgba16(out / f"{TILESET_NAME}.p.png", position)

    tileset = tm.Tileset(firstgid=1, name=TILESET_NAME, tilewidth=tw, tileheight=th,
                         tilecount=len(pieces), columns=COLUMNS,
                         image=tm.Image(source=f"{TILESET_NAME}.png", width=COLUMNS * tw, height=rows * th))
    (xr, yr, zr) = proj.position_range
    meta = {
        "cell_width_m": proj.cell.width, "cell_height_m": proj.cell.height,
        "row_height_px": proj.row_h, "level_height_px": proj.level_h,
        "normal_map": f"{TILESET_NAME}.n.png", "position_map": f"{TILESET_NAME}.p.png",
        "position_range_m": f"x {xr[0]:g} {xr[1]:g}; y {yr[0]:g} {yr[1]:g}; z {zr[0]:g} {zr[1]:g}",
    }
    for key, value in meta.items():
        kind = "float" if isinstance(value, float) else "int" if isinstance(value, int) else "string"
        tileset.properties[key] = tm.Property(key, kind, value)
    for i, piece in enumerate(pieces):
        props = {"piece": piece.name, **piece.spec.to_properties()}
        tile = tm.Tile(id=i, type=piece.layer,
                       properties={k: tm.Property(k, "string", v) for k, v in props.items()})
        tileset.tiles[i] = tile

    root = tileset.to_xml(include_firstgid=False)
    root.attrib = {"version": "1.10", "tiledversion": "1.11.0", **root.attrib}
    tm.TiledMap._indent(root)
    tsx = out / f"{TILESET_NAME}.tsx"
    ET.ElementTree(root).write(tsx, encoding="utf-8", xml_declaration=True)
    return tsx


def _text_object(oid: int, x: float, y: float, width: float, text: str) -> tm.MapObject:
    obj = tm.MapObject(id=oid, name=text, x=x, y=y, width=width, height=16)
    label = ET.Element("text", {"pixelsize": "12", "wrap": "1", "halign": "center"})
    label.text = text
    obj.extra_children.append(label)
    return obj


def build_gallery(tmx_path: Path, tsx_path: Path, proj: Projection = DEFAULT_PROJECTION,
                  pieces: List[Piece] = None) -> tm.TiledMap:
    """Galería: cada pieza en su celda, separadas, con su nombre debajo y suelo alrededor."""
    pieces = pieces or build_catalog()
    by_category: Dict[str, List[int]] = {}
    for i, piece in enumerate(pieces):
        by_category.setdefault(piece.category, []).append(i)

    per_row = 15
    places = []                                        # (índice de pieza, x, y)
    y = 1
    for indices in by_category.values():
        for k, idx in enumerate(indices):
            if k and k % per_row == 0:
                y += 3
            places.append((idx, 1 + 2 * (k % per_row), y))
        y += 3
    width, height = 2 * per_row + 1, y

    gid = {p.name: i + 1 for i, p in enumerate(pieces)}
    m = tm.create_empty_map(width, height, proj.tile_w, proj.row_h)
    m.tiledversion = "1.11.0"
    m.properties["level_height"] = tm.Property("level_height", "int", proj.level_h)
    m.tilesets.append(tm.Tileset(firstgid=1, name=TILESET_NAME, tilewidth=proj.image_size[0],
                                 tileheight=proj.image_size[1],
                                 source=os.path.relpath(tsx_path, tmx_path.parent)))

    floor = tm.create_layer("suelo", width, height); floor.klass = "floor"; floor.id = 2
    things = tm.create_layer("piezas", width, height); things.klass = "object"; things.id = 3
    roofs = tm.create_layer("techos", width, height); roofs.klass = "overhead"; roofs.id = 4
    labels = tm.ObjectGroup(name="etiquetas", id=5)
    target = {"floor": None, "object": things, "overhead": roofs}

    occupied = {(x, yy): pieces[idx] for idx, x, yy in places}
    for x in range(width):
        for yy in range(height):
            piece = occupied.get((x, yy))
            if piece is None or not piece.spec.floors and not piece.spec.fill:
                floor.set_tile_gid(x, yy, gid["floor_flat_000"])        # suelo de referencia
            elif piece.spec.fill in ("half_nw", "half_se", "half_ne", "half_sw"):
                other = {"half_nw": "se", "half_se": "nw", "half_ne": "sw", "half_sw": "ne"}
                floor.set_tile_gid(x, yy, gid[f"floor_{other[piece.spec.fill]}_flat_000"])

    for n, (idx, x, yy) in enumerate(places):
        piece = pieces[idx]
        layer = target[piece.layer] or floor
        layer.set_tile_gid(x, yy, gid[piece.name])
        labels.objects.append(_text_object(n + 1, (x - 0.5) * proj.tile_w, (yy + 1) * proj.row_h + 2,
                                           2 * proj.tile_w, piece.name))

    group = tm.LayerGroup(name="Planta 0", id=1, properties={"Z": tm.Property("Z", "int", 0)})
    group.layers = [floor, things, roofs]
    m.layers = [group, labels]
    m.extra_attrib.update({"nextlayerid": "6", "nextobjectid": str(len(places) + 1)})
    tmx_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(tmx_path, encoding="csv")
    return m
