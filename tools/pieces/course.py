"""Mapa de recorrido: terreno con varias plantas que combina las piezas de prueba.

Zonas (coordenadas de celda, planta 0 salvo que se diga):
    plataforma   x 3-8, y 3-7: un bloque de una planta con suelo encima (planta 1),
                 escalera recta de dos celdas por el sur (x 5, y 8-9) y acantilado al oeste
    habitación   x 12-17, y 3-7: paredes finas con una puerta al sur
    muro diag.   medias celdas macizas en diagonal, de (12, 14) a (17, 9)
    túnel        x 3-8, y 13: techo bajo a ½ planta (pasa un perro, no una persona)
    terraza      x 19-24, y 6-9: un bloque de una planta con suelo encima (planta 1), con una
                 rampa recta de cuatro celdas de ancho por el sur (x 20-23, y 10-11)
    torre        x 22-23, y 2-3: dos plantas de bloques con suelo arriba (planta 2)

Las rampas diagonales no están aquí: suben a la vez hacia el este y hacia el norte, así que
solo enlazan con suelo plano a lo largo de una diagonal (en pasillos diagonales). Un campo
de rampas diagonales en una zona de cuadrícula queda rodeado de escalones demasiado altos.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import tmx_manager as tm

from .catalog import Piece, build_catalog
from .render import DEFAULT_PROJECTION, Projection

ROLE_LAYER = {"floor": "suelo", "object": "piezas", "overhead": "techos"}


class CourseBuilder:
    def __init__(self, width: int, height: int, pieces: List[Piece]):
        self.width, self.height = width, height
        self.pieces = {p.name: p for p in pieces}
        self.gid = {p.name: i + 1 for i, p in enumerate(pieces)}
        self.cells: Dict[Tuple[int, str], Dict[Tuple[int, int], int]] = {}
        self.lights: Dict[int, list] = {}
        self.events: Dict[int, list] = {}
        self.markers: Dict[int, list] = {}

    def light(self, level: int, x: float, y: float, color: str = "#ffd9a0", radius: float = 6.0,
              intensity: float = 1.5, height: float = 2.2):
        """Luz puntual en la celda (x, y) de esa planta (coordenadas de celda, admiten decimales)."""
        self.lights.setdefault(level, []).append((x, y, color, radius, intensity, height))

    def put(self, level: int, x: int, y: int, name: str, flip: bool = False, replace: bool = False):
        """Pone una pieza; si ya hay otra en su capa, va en otra capa encima (o, con
        ``replace``, la sustituye)."""
        layer = ROLE_LAYER[self.pieces[name].layer]
        if name.startswith("fill_"):
            layer = "piezas"
        cells = self.cells.setdefault((level, layer), {})
        if (x, y) in cells and not replace:    # una pieza más en la misma celda: otra capa
            n = 2
            while (x, y) in self.cells.setdefault((level, f"{layer}{n}"), {}):
                n += 1
            cells = self.cells[(level, f"{layer}{n}")]
        cells[(x, y)] = self.gid[name] | (0x80000000 if flip else 0)

    def fill_rect(self, level, x0, y0, x1, y1, name):
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.put(level, x, y, name)

    def event(self, level: int, x0: int, y0: int, x1: int, y1: int, name: str, **props):
        """Suceso: rectángulo de celdas [x0, x1) × [y0, y1); propiedades when, who, times, if, do…"""
        self.events.setdefault(level, []).append((x0, y0, x1, y1, name, props))

    def marker(self, level: int, x: float, y: float, name: str):
        """Marcador con nombre (destino de teleport), en coordenadas de celda."""
        self.markers.setdefault(level, []).append((x, y, name))

    def to_map(self, tsx_path: Path, tmx_path: Path, proj: Projection) -> tm.TiledMap:
        m = tm.create_empty_map(self.width, self.height, proj.tile_w, proj.row_h)
        m.tiledversion = "1.11.0"
        m.properties["level_height"] = tm.Property("level_height", "int", proj.level_h)
        m.tilesets.append(tm.Tileset(firstgid=1, name="pieces", tilewidth=proj.image_size[0],
                                     tileheight=proj.image_size[1],
                                     source=os.path.relpath(tsx_path, tmx_path.parent)))
        next_id = 1
        for level in sorted({lv for lv, _ in self.cells}):
            group = tm.LayerGroup(name=f"Planta {level}", id=next_id,
                                  properties={"Z": tm.Property("Z", "int", level)},
                                  offsety=-level * proj.level_h)       # vista previa correcta en Tiled
            next_id += 1
            for (lv, name), cells in sorted(self.cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                if lv != level:
                    continue
                layer = tm.create_layer(name, self.width, self.height)
                layer.id, next_id = next_id, next_id + 1
                base = name.rstrip("0123456789")
                layer.klass = {"suelo": "floor", "piezas": "object", "techos": "overhead"}[base]
                for (x, y), g in cells.items():
                    layer.set_tile_gid(x, y, g)
                group.layers.append(layer)
            if level in self.lights:
                group.layers.append(self._light_layer(level, next_id, proj))
                next_id += 1
            if level in self.events or level in self.markers:
                group.layers.append(self._event_layer(level, next_id, proj))
                next_id += 1
            m.layers.append(group)
        m.extra_attrib.update({"nextlayerid": str(next_id), "nextobjectid": str(self._next_object)})
        return m

    _next_object = 1

    def _event_layer(self, level, layer_id, proj):
        import xml.etree.ElementTree as ET
        og = tm.ObjectGroup(name="sucesos", id=layer_id)
        for x0, y0, x1, y1, name, props in self.events.get(level, []):
            og.objects.append(tm.MapObject(
                id=self._next_object, name=name, type="event", x=x0 * proj.tile_w, y=y0 * proj.row_h,
                width=(x1 - x0) * proj.tile_w, height=(y1 - y0) * proj.row_h,
                properties={k: tm.Property(k, "int" if isinstance(v, int) else "string", v)
                            for k, v in props.items()}))
            self._next_object += 1
        for x, y, name in self.markers.get(level, []):
            obj = tm.MapObject(id=self._next_object, name=name, type="marker", x=x * proj.tile_w, y=y * proj.row_h)
            obj.extra_children.append(ET.Element("point"))
            og.objects.append(obj)
            self._next_object += 1
        return og

    def _light_layer(self, level, layer_id, proj):
        import xml.etree.ElementTree as ET
        og = tm.ObjectGroup(name="luces", id=layer_id)
        for x, y, color, radius, intensity, height in self.lights[level]:
            obj = tm.MapObject(id=self._next_object, name="luz", type="light",
                               x=x * proj.tile_w, y=y * proj.row_h, properties={
                                   "color": tm.Property("color", "color", color),
                                   "radius": tm.Property("radius", "float", radius),
                                   "intensity": tm.Property("intensity", "float", intensity),
                                   "height": tm.Property("height", "float", height)})
            obj.extra_children.append(ET.Element("point"))
            og.objects.append(obj)
            self._next_object += 1
        return og


def build_course(tmx_path: Path, tsx_path: Path, proj: Projection = DEFAULT_PROJECTION,
                 pieces: List[Piece] = None) -> tm.TiledMap:
    b = CourseBuilder(26, 17, pieces or build_catalog())
    W, H = b.width, b.height

    # Suelo general, salvo donde hay bloques o rampas en la planta 0
    busy = set()
    busy |= {(x, y) for x in range(3, 9) for y in range(3, 8)}            # plataforma
    busy |= {(5, 8), (5, 9)}                                               # escalera
    busy |= {(22, 2), (23, 2), (22, 3), (23, 3)}                           # torre
    terrace = {(x, y) for x in range(19, 25) for y in range(6, 10)}
    ramp = {(x, y) for x in range(20, 24) for y in (10, 11)}
    busy |= terrace | ramp
    diagonal = [(12 + i, 14 - i) for i in range(6)]
    busy |= set(diagonal)
    for y in range(H):
        for x in range(W):
            if (x, y) not in busy:
                b.put(0, x, y, "floor_flat_000")

    # Plataforma: bloques en la planta 0 y suelo en la 1; escalera de dos celdas al sur
    b.fill_rect(0, 3, 3, 8, 7, "fill_solid")
    b.fill_rect(1, 3, 3, 8, 7, "floor_flat_000")
    b.put(0, 5, 9, "floor_ramp_N_000")
    b.put(0, 5, 8, "floor_ramp_N_050")
    for y in (8, 9):                                                       # barandillas de la escalera
        b.put(0, 5, y, "wall_W")
        b.put(0, 5, y, "wall_E")

    # Habitación con puerta al sur (x = 14)
    for x in range(12, 18):
        b.put(0, x, 3, "wall_N")
        if x != 14:
            b.put(0, x, 7, "wall_S")
    for y in range(3, 8):
        b.put(0, 12, y, "wall_W")
        b.put(0, 17, y, "wall_E")

    # Muro diagonal: medias celdas macizas con suelo en la otra mitad
    for x, y in diagonal:
        b.put(0, x, y, "fill_half_nw")
        b.put(0, x, y, "floor_se_flat_000")

    # Túnel: techo bajo a ½ planta, con paredes a los lados
    for x in range(3, 9):
        b.put(0, x, 13, "ceiling_050")
        b.put(0, x, 12, "wall_S")
        b.put(0, x, 14, "wall_N")

    # Terraza con rampa ancha: dos celdas de rampa recta (una planta en 4 m) por el sur
    for x, y in terrace:
        b.put(0, x, y, "fill_solid")
        b.put(1, x, y, "floor_flat_000")
    for x in range(20, 24):
        b.put(0, x, 11, "floor_ramp_N_000")
        b.put(0, x, 10, "floor_ramp_N_050")

    # Torre: dos plantas de bloques con suelo arriba
    for x in (22, 23):
        for y in (2, 3):
            b.put(0, x, y, "fill_solid")
            b.put(1, x, y, "fill_solid")
            b.put(2, x, y, "floor_flat_000")

    # Luces: farolas en el suelo, una en la habitación, otra en lo alto de la plataforma y la torre
    for x, y in ((10.5, 10.5), (2.5, 10.0), (18.5, 13.5), (11.0, 15.5)):
        b.light(0, x, y, "#ffcf8a", radius=7.0, intensity=1.6)
    b.light(0, 15.0, 5.0, "#ffe2b8", radius=6.0, intensity=1.4)
    b.light(1, 6.0, 5.0, "#9ec8ff", radius=7.0, intensity=1.5)
    b.light(1, 21.5, 7.5, "#ffb070", radius=6.0, intensity=1.4)
    b.light(2, 23.0, 3.0, "#ff7050", radius=6.0, intensity=1.6)

    m = b.to_map(tsx_path, tmx_path, proj)
    tmx_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(tmx_path, encoding="csv")
    return m
