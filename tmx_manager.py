#!/usr/bin/env python3
"""Lectura, modificación y escritura de archivos TMX (Tiled Map Format).

Soporta TMX 1.11 y anteriores, incluidos mapas infinitos (chunks).

Ida y vuelta sin pérdidas: todo atributo o elemento que esta librería no modela
(capas de imagen, animaciones, formas de colisión de tiles, polígonos, textos,
``editorsettings``, ``nextobjectid``...) se conserva tal cual y se vuelve a
escribir al guardar. Cada clase lo guarda en ``extra_attrib`` / ``extra_children``.

Sin dependencias fuera de la biblioteca estándar (``zstandard`` solo para zstd).
"""

import array
import base64
import copy
import logging
import sys
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

# Bits altos de un GID: volteo horizontal, vertical, diagonal y rotación hexagonal
FLIP_MASK = 0xF0000000
GID_MASK = 0x0FFFFFFF


# --- Utilidades comunes ---

def _read_extras(obj, elem: ET.Element, known_attrs: Iterable[str], known_tags: Iterable[str]):
    """Guarda en ``obj`` los atributos e hijos de ``elem`` que no se modelan."""
    known_attrs, known_tags = set(known_attrs), set(known_tags)
    obj.extra_attrib = {k: v for k, v in elem.attrib.items() if k not in known_attrs}
    obj.extra_children = [copy.deepcopy(c) for c in elem if c.tag not in known_tags]


def _write_extras(obj, elem: ET.Element, children_first: bool = False):
    for k, v in obj.extra_attrib.items():
        elem.set(k, v)
    extras = [copy.deepcopy(c) for c in obj.extra_children]
    if children_first:
        for i, child in enumerate(extras):
            elem.insert(i, child)
    else:
        elem.extend(extras)


def _fmt(value: float) -> str:
    """Número como lo escribe Tiled: '16' en vez de '16.0'."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _set_if(elem: ET.Element, name: str, value, default):
    if value != default and value is not None:
        elem.set(name, _fmt(value) if isinstance(value, float) else str(value))


def _read_properties(elem: ET.Element) -> Dict[str, "Property"]:
    props_elem = elem.find("properties")
    if props_elem is None:
        return {}
    return {p.name: p for p in map(Property.from_xml, props_elem.findall("property"))}


def _write_properties(elem: ET.Element, properties: Dict[str, "Property"]):
    if properties:
        props_elem = ET.SubElement(elem, "properties")
        for prop in properties.values():
            props_elem.append(prop.to_xml())


def _tiles_from_bytes(raw: bytes) -> array.array:
    tiles = array.array("I")
    if tiles.itemsize != 4:                      # plataformas exóticas
        tiles = array.array("L")
    tiles.frombytes(raw)
    if sys.byteorder == "big":                   # TMX siempre es little-endian
        tiles.byteswap()
    return tiles


def _tiles_to_bytes(tiles: array.array) -> bytes:
    if sys.byteorder == "big":
        tiles = array.array(tiles.typecode, tiles)
        tiles.byteswap()
    return tiles.tobytes()


def _compress(raw: bytes, compression: Optional[str]) -> bytes:
    if not compression:
        return raw
    if compression == "zlib":
        return zlib.compress(raw)
    if compression == "gzip":
        import gzip
        return gzip.compress(raw)
    if compression == "zstd":
        return _zstd().ZstdCompressor().compress(raw)
    raise ValueError(f"Unsupported compression: {compression}")


def _decompress(raw: bytes, compression: Optional[str]) -> bytes:
    if not compression:
        return raw
    if compression == "zlib":
        return zlib.decompress(raw)
    if compression == "gzip":
        import gzip
        return gzip.decompress(raw)
    if compression == "zstd":
        return _zstd().ZstdDecompressor().decompress(raw)
    raise ValueError(f"Unsupported compression: {compression}")


def _zstd():
    try:
        import zstandard
    except ImportError:
        raise ImportError("zstandard library required for zstd compression. "
                          "Install with: pip install zstandard") from None
    return zstandard


# --- Propiedades ---

@dataclass
class Property:
    """Propiedad custom. ``value`` ya convertido: int, float, bool, str o, para el
    tipo ``class``, un dict de Property."""
    name: str
    type: str = "string"
    value: Any = None
    propertytype: Optional[str] = None      # nombre del tipo custom (enum o class)

    @classmethod
    def from_xml(cls, elem: ET.Element) -> "Property":
        prop_type = elem.get("type", "string")
        if prop_type == "class":
            value = _read_properties(elem)
        else:
            # Las cadenas multilínea van como texto del elemento, no como atributo
            raw = elem.get("value")
            if raw is None:
                raw = elem.text or ""
            value = raw
            if prop_type in ("int", "object"):
                value = int(raw or 0)
            elif prop_type == "float":
                value = float(raw or 0)
            elif prop_type == "bool":
                value = raw.lower() == "true"
            # color (#AARRGGBB) y file se quedan como string
        return cls(name=elem.get("name"), type=prop_type, value=value,
                   propertytype=elem.get("propertytype"))

    def to_xml(self) -> ET.Element:
        elem = ET.Element("property")
        elem.set("name", self.name)
        if self.type != "string":
            elem.set("type", self.type)
        if self.propertytype:
            elem.set("propertytype", self.propertytype)
        if self.type == "class":
            _write_properties(elem, self.value or {})
        elif self.type == "bool":
            elem.set("value", "true" if self.value else "false")
        elif isinstance(self.value, str) and "\n" in self.value:
            elem.text = self.value
        elif isinstance(self.value, float):
            elem.set("value", _fmt(self.value))
        else:
            elem.set("value", "" if self.value is None else str(self.value))
        return elem


# --- Imágenes y tilesets ---

@dataclass
class Image:
    source: str
    width: Optional[int] = None
    height: Optional[int] = None
    trans: Optional[str] = None
    extra_attrib: Dict[str, str] = field(default_factory=dict)
    extra_children: List[ET.Element] = field(default_factory=list)

    @classmethod
    def from_xml(cls, elem: ET.Element) -> "Image":
        img = cls(source=elem.get("source", ""),
                  width=int(elem.get("width")) if elem.get("width") else None,
                  height=int(elem.get("height")) if elem.get("height") else None,
                  trans=elem.get("trans"))
        _read_extras(img, elem, ("source", "width", "height", "trans"), ())
        return img

    def to_xml(self) -> ET.Element:
        elem = ET.Element("image")
        elem.set("source", self.source)
        _set_if(elem, "trans", self.trans, None)
        _set_if(elem, "width", self.width, None)
        _set_if(elem, "height", self.height, None)
        _write_extras(self, elem)
        return elem


@dataclass
class Tile:
    """Tile con metadatos dentro de un tileset. ``type`` es su clase (atributo
    ``class`` en Tiled 1.9+, ``type`` en versiones anteriores). Las animaciones y
    las formas de colisión se conservan en ``extra_children``."""
    id: int
    type: str = ""
    properties: Dict[str, Property] = field(default_factory=dict)
    image: Optional[Image] = None
    type_attr: str = "class"
    extra_attrib: Dict[str, str] = field(default_factory=dict)
    extra_children: List[ET.Element] = field(default_factory=list)

    @classmethod
    def from_xml(cls, elem: ET.Element) -> "Tile":
        type_attr = "class" if "class" in elem.attrib else "type"
        tile = cls(id=int(elem.get("id", 0)), type=elem.get(type_attr, ""),
                   properties=_read_properties(elem), type_attr=type_attr)
        img_elem = elem.find("image")
        if img_elem is not None:
            tile.image = Image.from_xml(img_elem)
        _read_extras(tile, elem, ("id", "class", "type"), ("properties", "image"))
        return tile

    def to_xml(self) -> ET.Element:
        elem = ET.Element("tile")
        elem.set("id", str(self.id))
        if self.type:
            elem.set(self.type_attr, self.type)
        _write_properties(elem, self.properties)
        if self.image:
            elem.append(self.image.to_xml())
        _write_extras(self, elem)
        return elem


@dataclass
class Tileset:
    firstgid: int
    name: str
    tilewidth: int
    tileheight: int
    tilecount: int = 0
    columns: int = 0
    spacing: int = 0
    margin: int = 0
    image: Optional[Image] = None
    tiles: Dict[int, Tile] = field(default_factory=dict)
    properties: Dict[str, Property] = field(default_factory=dict)
    source: Optional[str] = None             # ruta del TSX si es externo
    extra_attrib: Dict[str, str] = field(default_factory=dict)
    extra_children: List[ET.Element] = field(default_factory=list)

    _ATTRS = ("firstgid", "source", "name", "tilewidth", "tileheight", "tilecount",
              "columns", "spacing", "margin")

    @classmethod
    def from_xml(cls, elem: ET.Element, firstgid: int) -> "Tileset":
        ts = cls(firstgid=firstgid, name=elem.get("name", ""),
                 tilewidth=int(elem.get("tilewidth", 0)),
                 tileheight=int(elem.get("tileheight", 0)),
                 tilecount=int(elem.get("tilecount", 0)),
                 columns=int(elem.get("columns", 0)),
                 spacing=int(elem.get("spacing", 0)),
                 margin=int(elem.get("margin", 0)),
                 source=elem.get("source"),
                 properties=_read_properties(elem))
        img_elem = elem.find("image")
        if img_elem is not None:
            ts.image = Image.from_xml(img_elem)
        for tile_elem in elem.findall("tile"):
            tile = Tile.from_xml(tile_elem)
            ts.tiles[tile.id] = tile
        # tileoffset, grid, wangsets, transformations... se conservan
        _read_extras(ts, elem, cls._ATTRS, ("properties", "image", "tile"))
        return ts

    def to_xml(self, include_firstgid: bool = True) -> ET.Element:
        elem = ET.Element("tileset")
        if include_firstgid:
            elem.set("firstgid", str(self.firstgid))
        if self.source:                         # externo: el TSX no se toca
            elem.set("source", self.source)
            return elem
        elem.set("name", self.name)
        elem.set("tilewidth", str(self.tilewidth))
        elem.set("tileheight", str(self.tileheight))
        _set_if(elem, "spacing", self.spacing, 0)
        _set_if(elem, "margin", self.margin, 0)
        elem.set("tilecount", str(self.tilecount))
        elem.set("columns", str(self.columns))
        for k, v in self.extra_attrib.items():
            elem.set(k, v)
        # Orden de Tiled: hijos desconocidos tipo tileoffset/grid antes que la imagen
        head = [c for c in self.extra_children if c.tag in ("tileoffset", "grid")]
        tail = [c for c in self.extra_children if c not in head]
        elem.extend([copy.deepcopy(c) for c in head])
        _write_properties(elem, self.properties)
        if self.image:
            elem.append(self.image.to_xml())
        elem.extend([copy.deepcopy(c) for c in tail])
        for tile in self.tiles.values():
            elem.append(tile.to_xml())
        return elem

    def contains(self, gid: int) -> bool:
        local = (gid & GID_MASK) - self.firstgid
        return 0 <= local < max(self.tilecount, 1)


# --- Datos de capas ---

@dataclass
class Chunk:
    """Trozo de un mapa infinito. x, y pueden ser negativos."""
    x: int
    y: int
    width: int
    height: int
    tiles: array.array = field(default_factory=lambda: array.array("I"))


def _decode_tiles(elem: ET.Element, encoding: Optional[str],
                  compression: Optional[str]) -> array.array:
    if encoding == "csv":
        text = (elem.text or "").replace("\n", "")
        return array.array("I", [int(v) for v in text.split(",") if v.strip()])
    if encoding == "base64":
        raw = base64.b64decode((elem.text or "").strip())
        return _tiles_from_bytes(_decompress(raw, compression))
    return array.array("I", [int(t.get("gid", 0)) for t in elem.findall("tile")])


def _encode_tiles(elem: ET.Element, tiles: array.array, width: int,
                  encoding: str, compression: Optional[str]):
    if encoding == "csv":
        rows = [",".join(map(str, tiles[y * width:(y + 1) * width]))
                for y in range(len(tiles) // width if width else 0)]
        elem.text = "\n" + ",\n".join(rows) + "\n"
    elif encoding == "base64":
        raw = _compress(_tiles_to_bytes(tiles), compression)
        elem.text = "\n" + base64.b64encode(raw).decode("ascii") + "\n"
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


@dataclass
class LayerData:
    """GIDs de una capa. En mapas finitos, ``tiles`` (fila a fila); en infinitos, ``chunks``."""
    encoding: Optional[str] = None          # csv, base64 o None (XML antiguo)
    compression: Optional[str] = None       # zlib, gzip, zstd o None
    tiles: array.array = field(default_factory=lambda: array.array("I"))
    chunks: List[Chunk] = field(default_factory=list)

    def decode_data(self, data_elem: ET.Element, width: int, height: int):
        self.encoding = data_elem.get("encoding")
        self.compression = data_elem.get("compression")
        chunk_elems = data_elem.findall("chunk")
        if chunk_elems:
            self.chunks = []
            for c in chunk_elems:
                self.chunks.append(Chunk(int(c.get("x")), int(c.get("y")),
                                         int(c.get("width")), int(c.get("height")),
                                         _decode_tiles(c, self.encoding, self.compression)))
            self.tiles = array.array("I")
        else:
            self.tiles = _decode_tiles(data_elem, self.encoding, self.compression)

    def to_xml(self, width: int, height: int) -> ET.Element:
        elem = ET.Element("data")
        if not self.encoding:
            self.encoding = "csv"               # el formato XML antiguo no se escribe
        elem.set("encoding", self.encoding)
        if self.compression:
            elem.set("compression", self.compression)
        if self.chunks:
            for ch in self.chunks:
                c = ET.SubElement(elem, "chunk", x=str(ch.x), y=str(ch.y),
                                  width=str(ch.width), height=str(ch.height))
                _encode_tiles(c, ch.tiles, ch.width, self.encoding, self.compression)
        else:
            _encode_tiles(elem, self.tiles, width, self.encoding, self.compression)
        return elem


# --- Capas ---

_COMMON_LAYER_ATTRS = ("id", "name", "class", "visible", "opacity", "offsetx", "offsety",
                       "parallaxx", "parallaxy", "tintcolor")


def _read_common(layer, elem: ET.Element):
    layer.name = elem.get("name", "")
    layer.id = int(elem.get("id", 0))
    layer.klass = elem.get("class", "")
    layer.visible = elem.get("visible", "1") == "1"
    layer.opacity = float(elem.get("opacity", 1.0))
    layer.offsetx = float(elem.get("offsetx", 0))
    layer.offsety = float(elem.get("offsety", 0))
    layer.parallaxx = float(elem.get("parallaxx", 1.0))
    layer.parallaxy = float(elem.get("parallaxy", 1.0))
    layer.tintcolor = elem.get("tintcolor")
    layer.properties = _read_properties(elem)


def _write_common(layer, elem: ET.Element):
    elem.set("id", str(layer.id))
    elem.set("name", layer.name)
    _set_if(elem, "class", layer.klass, "")
    _set_if(elem, "visible", None if layer.visible else 0, None)
    _set_if(elem, "opacity", layer.opacity, 1.0)
    _set_if(elem, "tintcolor", layer.tintcolor, None)
    _set_if(elem, "offsetx", layer.offsetx, 0)
    _set_if(elem, "offsety", layer.offsety, 0)
    _set_if(elem, "parallaxx", layer.parallaxx, 1.0)
    _set_if(elem, "parallaxy", layer.parallaxy, 1.0)


@dataclass
class TileLayer:
    name: str
    width: int
    height: int
    id: int = 0
    klass: str = ""                          # clase de la capa en Tiled (atributo "class")
    visible: bool = True
    opacity: float = 1.0
    offsetx: float = 0
    offsety: float = 0
    parallaxx: float = 1.0
    parallaxy: float = 1.0
    tintcolor: Optional[str] = None
    properties: Dict[str, Property] = field(default_factory=dict)
    data: LayerData = field(default_factory=LayerData)
    extra_attrib: Dict[str, str] = field(default_factory=dict)
    extra_children: List[ET.Element] = field(default_factory=list)

    @classmethod
    def from_xml(cls, elem: ET.Element) -> "TileLayer":
        layer = cls(name="", width=int(elem.get("width", 0)), height=int(elem.get("height", 0)))
        _read_common(layer, elem)
        data_elem = elem.find("data")
        if data_elem is not None:
            layer.data.decode_data(data_elem, layer.width, layer.height)
        _read_extras(layer, elem, _COMMON_LAYER_ATTRS + ("width", "height"),
                     ("properties", "data"))
        return layer

    def to_xml(self) -> ET.Element:
        elem = ET.Element("layer")
        _write_common(self, elem)
        elem.set("width", str(self.width))
        elem.set("height", str(self.height))
        _write_extras(self, elem)
        _write_properties(elem, self.properties)
        elem.append(self.data.to_xml(self.width, self.height))
        return elem

    # --- Acceso a tiles (finitos e infinitos) ---

    @property
    def is_infinite(self) -> bool:
        return bool(self.data.chunks)

    def _chunk_at(self, x: int, y: int) -> Optional[Chunk]:
        for ch in self.data.chunks:
            if ch.x <= x < ch.x + ch.width and ch.y <= y < ch.y + ch.height:
                return ch
        return None

    def get_tile_gid(self, x: int, y: int) -> int:
        """GID en (x, y), con bits de flip incluidos. Fuera de la capa: 0."""
        if self.is_infinite:
            ch = self._chunk_at(x, y)
            return ch.tiles[(y - ch.y) * ch.width + (x - ch.x)] if ch else 0
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data.tiles[y * self.width + x]
        return 0

    def set_tile_gid(self, x: int, y: int, gid: int, chunk_size: int = 16):
        if self.is_infinite:
            ch = self._chunk_at(x, y)
            if ch is None:                       # crea el chunk alineado que falta
                cx, cy = x // chunk_size * chunk_size, y // chunk_size * chunk_size
                ch = Chunk(cx, cy, chunk_size, chunk_size,
                           array.array("I", [0] * chunk_size * chunk_size))
                self.data.chunks.append(ch)
            ch.tiles[(y - ch.y) * ch.width + (x - ch.x)] = gid
        elif 0 <= x < self.width and 0 <= y < self.height:
            self.data.tiles[y * self.width + x] = gid

    def bounds(self) -> Tuple[int, int, int, int]:
        """(x0, y0, x1, y1) en tiles que cubren los datos. En infinitos, la caja de los chunks."""
        if not self.is_infinite:
            return 0, 0, self.width, self.height
        return (min(c.x for c in self.data.chunks), min(c.y for c in self.data.chunks),
                max(c.x + c.width for c in self.data.chunks),
                max(c.y + c.height for c in self.data.chunks))

    def dense_tiles(self, x0: int, y0: int, width: int, height: int) -> array.array:
        """GIDs de la región dada, fila a fila (0 donde no hay datos)."""
        out = array.array("I", [0] * (width * height))
        if not self.is_infinite:
            for y in range(max(0, y0), min(self.height, y0 + height)):
                sx0, sx1 = max(0, x0), min(self.width, x0 + width)
                if sx0 < sx1:
                    out[(y - y0) * width + sx0 - x0:(y - y0) * width + sx1 - x0] = \
                        self.data.tiles[y * self.width + sx0:y * self.width + sx1]
            return out
        for ch in self.data.chunks:
            for row in range(ch.height):
                y = ch.y + row
                if not y0 <= y < y0 + height:
                    continue
                sx0, sx1 = max(ch.x, x0), min(ch.x + ch.width, x0 + width)
                if sx0 < sx1:
                    out[(y - y0) * width + sx0 - x0:(y - y0) * width + sx1 - x0] = \
                        ch.tiles[row * ch.width + sx0 - ch.x:row * ch.width + sx1 - ch.x]
        return out


@dataclass
class MapObject:
    """Objeto de una capa de objetos. Polígonos, puntos, elipses, textos y plantillas
    se conservan en ``extra_children`` / ``extra_attrib``."""
    id: int
    name: str = ""
    type: str = ""                           # clase (atributo "class" o "type")
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    rotation: float = 0
    gid: Optional[int] = None
    visible: bool = True
    properties: Dict[str, Property] = field(default_factory=dict)
    type_attr: str = "class"
    extra_attrib: Dict[str, str] = field(default_factory=dict)
    extra_children: List[ET.Element] = field(default_factory=list)

    @classmethod
    def from_xml(cls, elem: ET.Element) -> "MapObject":
        type_attr = "class" if "class" in elem.attrib else "type"
        obj = cls(id=int(elem.get("id", 0)), name=elem.get("name", ""),
                  type=elem.get(type_attr, ""), type_attr=type_attr,
                  x=float(elem.get("x", 0)), y=float(elem.get("y", 0)),
                  width=float(elem.get("width", 0)), height=float(elem.get("height", 0)),
                  rotation=float(elem.get("rotation", 0)),
                  visible=elem.get("visible", "1") == "1",
                  properties=_read_properties(elem))
        if elem.get("gid"):
            obj.gid = int(elem.get("gid"))
        _read_extras(obj, elem, ("id", "name", "class", "type", "x", "y", "width", "height",
                                 "rotation", "gid", "visible"), ("properties",))
        return obj

    @property
    def shape(self) -> str:
        """rectangle, ellipse, point, polygon, polyline, text o tile."""
        if self.gid is not None:
            return "tile"
        for child in self.extra_children:
            if child.tag in ("ellipse", "point", "polygon", "polyline", "text"):
                return child.tag
        return "rectangle"

    def to_xml(self) -> ET.Element:
        elem = ET.Element("object")
        elem.set("id", str(self.id))
        _set_if(elem, "name", self.name, "")
        if self.type:
            elem.set(self.type_attr, self.type)
        _set_if(elem, "gid", self.gid, None)
        elem.set("x", _fmt(self.x))
        elem.set("y", _fmt(self.y))
        _set_if(elem, "width", float(self.width), 0.0)
        _set_if(elem, "height", float(self.height), 0.0)
        _set_if(elem, "rotation", float(self.rotation), 0.0)
        _set_if(elem, "visible", None if self.visible else 0, None)
        for k, v in self.extra_attrib.items():
            elem.set(k, v)
        _write_properties(elem, self.properties)
        elem.extend([copy.deepcopy(c) for c in self.extra_children])
        return elem


@dataclass
class ObjectGroup:
    name: str
    id: int = 0
    klass: str = ""
    visible: bool = True
    opacity: float = 1.0
    offsetx: float = 0
    offsety: float = 0
    parallaxx: float = 1.0
    parallaxy: float = 1.0
    tintcolor: Optional[str] = None
    properties: Dict[str, Property] = field(default_factory=dict)
    objects: List[MapObject] = field(default_factory=list)
    extra_attrib: Dict[str, str] = field(default_factory=dict)
    extra_children: List[ET.Element] = field(default_factory=list)

    @classmethod
    def from_xml(cls, elem: ET.Element) -> "ObjectGroup":
        group = cls(name="")
        _read_common(group, elem)
        group.objects = [MapObject.from_xml(o) for o in elem.findall("object")]
        _read_extras(group, elem, _COMMON_LAYER_ATTRS, ("properties", "object"))
        return group

    def to_xml(self) -> ET.Element:
        elem = ET.Element("objectgroup")
        _write_common(self, elem)
        _write_extras(self, elem)
        _write_properties(elem, self.properties)
        elem.extend([obj.to_xml() for obj in self.objects])
        return elem


@dataclass
class RawLayer:
    """Capa de un tipo no modelado (p. ej. ``imagelayer``): se conserva el XML tal cual."""
    element: ET.Element

    @property
    def name(self) -> str:
        return self.element.get("name", "")

    @property
    def tag(self) -> str:
        return self.element.tag

    @property
    def visible(self) -> bool:
        return self.element.get("visible", "1") == "1"

    def to_xml(self) -> ET.Element:
        return copy.deepcopy(self.element)


AnyLayer = Union[TileLayer, ObjectGroup, "LayerGroup", RawLayer]


def _parse_layer(elem: ET.Element) -> Optional[AnyLayer]:
    if elem.tag == "layer":
        return TileLayer.from_xml(elem)
    if elem.tag == "objectgroup":
        return ObjectGroup.from_xml(elem)
    if elem.tag == "group":
        return LayerGroup.from_xml(elem)
    if elem.tag == "imagelayer":
        return RawLayer(copy.deepcopy(elem))
    return None


@dataclass
class LayerGroup:
    name: str
    id: int = 0
    klass: str = ""
    visible: bool = True
    opacity: float = 1.0
    offsetx: float = 0
    offsety: float = 0
    parallaxx: float = 1.0
    parallaxy: float = 1.0
    tintcolor: Optional[str] = None
    properties: Dict[str, Property] = field(default_factory=dict)
    layers: List[AnyLayer] = field(default_factory=list)
    extra_attrib: Dict[str, str] = field(default_factory=dict)
    extra_children: List[ET.Element] = field(default_factory=list)

    _LAYER_TAGS = ("layer", "objectgroup", "group", "imagelayer")

    @classmethod
    def from_xml(cls, elem: ET.Element) -> "LayerGroup":
        group = cls(name="")
        _read_common(group, elem)
        group.layers = [l for l in map(_parse_layer, elem) if l is not None]
        _read_extras(group, elem, _COMMON_LAYER_ATTRS, ("properties",) + cls._LAYER_TAGS)
        return group

    def to_xml(self) -> ET.Element:
        elem = ET.Element("group")
        _write_common(self, elem)
        _write_extras(self, elem)
        _write_properties(elem, self.properties)
        elem.extend([layer.to_xml() for layer in self.layers])
        return elem


# --- Mapa ---

@dataclass
class TiledMap:
    version: str = "1.10"
    tiledversion: str = ""
    orientation: str = "orthogonal"
    renderorder: str = "right-down"
    width: int = 0
    height: int = 0
    tilewidth: int = 0
    tileheight: int = 0
    infinite: bool = False
    properties: Dict[str, Property] = field(default_factory=dict)
    tilesets: List[Tileset] = field(default_factory=list)
    layers: List[AnyLayer] = field(default_factory=list)
    extra_attrib: Dict[str, str] = field(default_factory=dict)      # nextobjectid, class...
    extra_children: List[ET.Element] = field(default_factory=list)  # editorsettings...

    _ATTRS = ("version", "tiledversion", "orientation", "renderorder", "width", "height",
              "tilewidth", "tileheight", "infinite")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TiledMap":
        filepath = Path(filepath)
        root = ET.parse(filepath).getroot()
        m = cls(version=root.get("version", "1.0"),
                tiledversion=root.get("tiledversion", ""),
                orientation=root.get("orientation", "orthogonal"),
                renderorder=root.get("renderorder", "right-down"),
                width=int(root.get("width", 0)), height=int(root.get("height", 0)),
                tilewidth=int(root.get("tilewidth", 0)),
                tileheight=int(root.get("tileheight", 0)),
                infinite=root.get("infinite", "0") == "1",
                properties=_read_properties(root))

        for ts_elem in root.findall("tileset"):
            m.tilesets.append(cls._load_tileset(ts_elem, filepath.parent, m))
        m.layers = [l for l in map(_parse_layer, root) if l is not None]
        _read_extras(m, root, cls._ATTRS,
                     ("properties", "tileset") + LayerGroup._LAYER_TAGS)
        return m

    @staticmethod
    def _load_tileset(ts_elem: ET.Element, base: Path, m: "TiledMap") -> Tileset:
        firstgid = int(ts_elem.get("firstgid"))
        source = ts_elem.get("source")
        if not source:
            return Tileset.from_xml(ts_elem, firstgid)
        tsx_path = base / source
        try:
            tileset = Tileset.from_xml(ET.parse(tsx_path).getroot(), firstgid)
        except FileNotFoundError:
            log.warning("Tileset externo no encontrado: %s", tsx_path)
            tileset = Tileset(firstgid=firstgid, name=Path(source).stem,
                              tilewidth=m.tilewidth, tileheight=m.tileheight)
        tileset.source = source
        return tileset

    def save(self, filepath: Union[str, Path], encoding: Optional[str] = None,
             compression: Optional[str] = None):
        """Guarda el mapa. Sin ``encoding``, cada capa conserva su codificación y
        compresión originales; con él, se aplica a todas las capas de tiles."""
        root = ET.Element("map")
        root.set("version", self.version)
        _set_if(root, "tiledversion", self.tiledversion, "")
        for name in ("orientation", "renderorder", "width", "height", "tilewidth", "tileheight"):
            root.set(name, str(getattr(self, name)))
        root.set("infinite", "1" if self.infinite else "0")
        _write_extras(self, root, children_first=True)
        _write_properties(root, self.properties)
        root.extend([ts.to_xml() for ts in self.tilesets])

        if encoding is not None:
            for layer in self.iter_layers():
                if isinstance(layer, TileLayer):
                    layer.data.encoding, layer.data.compression = encoding, compression
        root.extend([layer.to_xml() for layer in self.layers])

        self._indent(root)
        ET.ElementTree(root).write(Path(filepath), encoding="utf-8", xml_declaration=True)

    @staticmethod
    def _indent(elem: ET.Element, level: int = 0):
        """Sangrado legible sin tocar el texto de los datos."""
        indent = "\n" + " " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + " "
            for child in elem:
                TiledMap._indent(child, level + 1)
                if not child.tail or not child.tail.strip():
                    child.tail = indent + " "
            elem[-1].tail = indent
        elif level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent

    # --- Consultas ---

    def get_tileset_for_gid(self, gid: int) -> Optional[Tileset]:
        """Tileset al que pertenece un GID (los bits de flip se ignoran)."""
        gid &= GID_MASK
        if gid == 0:
            return None
        for tileset in sorted(self.tilesets, key=lambda t: t.firstgid, reverse=True):
            if gid >= tileset.firstgid:
                return tileset
        return None

    def iter_layers(self, layers: Optional[List[AnyLayer]] = None):
        """Todas las capas, entrando en los grupos (los grupos también se devuelven)."""
        for layer in self.layers if layers is None else layers:
            yield layer
            if isinstance(layer, LayerGroup):
                yield from self.iter_layers(layer.layers)

    def get_layer_by_name(self, name: str) -> Optional[AnyLayer]:
        return next((l for l in self.iter_layers() if l.name == name), None)

    def get_all_layers_flat(self) -> List[Union[TileLayer, ObjectGroup, RawLayer]]:
        """Todas las capas que no son grupos, en orden de dibujo."""
        return [l for l in self.iter_layers() if not isinstance(l, LayerGroup)]


# --- Utilidades ---

def create_empty_map(width: int, height: int, tilewidth: int, tileheight: int,
                     orientation: str = "orthogonal") -> TiledMap:
    return TiledMap(width=width, height=height, tilewidth=tilewidth,
                    tileheight=tileheight, orientation=orientation)


def create_layer(name: str, width: int, height: int) -> TileLayer:
    """Capa de tiles vacía (todo GID 0)."""
    layer = TileLayer(name=name, width=width, height=height)
    layer.data.tiles = array.array("I", [0] * (width * height))
    return layer
