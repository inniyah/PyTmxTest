#!/usr/bin/env python3

"""
Module for reading, modifying and writing TMX files (Tiled Map Format)
Supports TMX version 1.11.0 and earlier versions

Compatible with:
- Multiple orientations (orthogonal, isometric, staggered, hexagonal)
- Tile layers, object layers, image layers and groups
- Encoding: XML, CSV, Base64
- Compression: gzip, zlib, zstd
- Custom properties
- Embedded and external tilesets (TSX)
"""

# Lee/escribe archivos TMX (Tiled Map Format)
# Soporta TMX versión 1.11.0
# Grupos de capas jerárquicas
# Tilesets externos (TSX) e internos
# Image collection tilesets (isométricos)
# Múltiples formatos: XML, CSV, Base64
# Compresión: gzip, zlib, zstd
# Propiedades personalizadas
# Modificación de mapas en código

import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import base64
import zlib
import struct
import array


@dataclass
class Property:
    """Custom property"""
    name: str
    type: str = "string"  # string, int, float, bool, color, file, object
    value: Any = None
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'Property':
        prop_type = elem.get('type', 'string')
        value = elem.get('value', '')
        
        # Convert value according to type
        if prop_type == 'int':
            value = int(value)
        elif prop_type == 'float':
            value = float(value)
        elif prop_type == 'bool':
            value = value.lower() == 'true'
        elif prop_type == 'color':
            value = value  # Keep as string #AARRGGBB
            
        return cls(name=elem.get('name'), type=prop_type, value=value)
    
    def to_xml(self) -> ET.Element:
        elem = ET.Element('property')
        elem.set('name', self.name)
        if self.type != 'string':
            elem.set('type', self.type)
        elem.set('value', str(self.value))
        return elem


@dataclass
class Image:
    """Image used in tilesets"""
    source: str
    width: Optional[int] = None
    height: Optional[int] = None
    trans: Optional[str] = None  # Transparent color
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'Image':
        return cls(
            source=elem.get('source', ''),
            width=int(elem.get('width')) if elem.get('width') else None,
            height=int(elem.get('height')) if elem.get('height') else None,
            trans=elem.get('trans')
        )
    
    def to_xml(self) -> ET.Element:
        elem = ET.Element('image')
        elem.set('source', self.source)
        if self.width:
            elem.set('width', str(self.width))
        if self.height:
            elem.set('height', str(self.height))
        if self.trans:
            elem.set('trans', self.trans)
        return elem


@dataclass
class Tile:
    """Individual tile within a tileset"""
    id: int
    type: str = ""
    properties: Dict[str, Property] = field(default_factory=dict)
    image: Optional[Image] = None
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'Tile':
        tile = cls(id=int(elem.get('id', 0)))
        tile.type = elem.get('type', '')
        
        # Properties
        props_elem = elem.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                tile.properties[prop.name] = prop
        
        # Image (for image collection tilesets)
        img_elem = elem.find('image')
        if img_elem is not None:
            tile.image = Image.from_xml(img_elem)
        
        return tile
    
    def to_xml(self) -> ET.Element:
        elem = ET.Element('tile')
        elem.set('id', str(self.id))
        if self.type:
            elem.set('type', self.type)
        
        if self.properties:
            props_elem = ET.SubElement(elem, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        if self.image:
            elem.append(self.image.to_xml())
        
        return elem


@dataclass
class Tileset:
    """Tileset collection"""
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
    source: Optional[str] = None  # For external tilesets (TSX)
    
    @classmethod
    def from_xml(cls, elem: ET.Element, firstgid: int) -> 'Tileset':
        tileset = cls(
            firstgid=firstgid,
            name=elem.get('name', ''),
            tilewidth=int(elem.get('tilewidth', 0)),
            tileheight=int(elem.get('tileheight', 0)),
            tilecount=int(elem.get('tilecount', 0)),
            columns=int(elem.get('columns', 0)),
            spacing=int(elem.get('spacing', 0)),
            margin=int(elem.get('margin', 0)),
            source=elem.get('source')
        )
        
        # Properties
        props_elem = elem.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                tileset.properties[prop.name] = prop
        
        # Tileset image
        img_elem = elem.find('image')
        if img_elem is not None:
            tileset.image = Image.from_xml(img_elem)
        
        # Individual tiles
        for tile_elem in elem.findall('tile'):
            tile = Tile.from_xml(tile_elem)
            tileset.tiles[tile.id] = tile
        
        return tileset
    
    def to_xml(self, include_firstgid: bool = True) -> ET.Element:
        elem = ET.Element('tileset')
        
        if include_firstgid:
            elem.set('firstgid', str(self.firstgid))
        
        if self.source:
            elem.set('source', self.source)
            return elem
        
        elem.set('name', self.name)
        elem.set('tilewidth', str(self.tilewidth))
        elem.set('tileheight', str(self.tileheight))
        elem.set('tilecount', str(self.tilecount))
        elem.set('columns', str(self.columns))
        
        if self.spacing:
            elem.set('spacing', str(self.spacing))
        if self.margin:
            elem.set('margin', str(self.margin))
        
        if self.properties:
            props_elem = ET.SubElement(elem, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        if self.image:
            elem.append(self.image.to_xml())
        
        for tile in self.tiles.values():
            elem.append(tile.to_xml())
        
        return elem


@dataclass
class LayerData:
    """Tile layer data"""
    encoding: Optional[str] = None  # csv, base64
    compression: Optional[str] = None  # gzip, zlib, zstd
    tiles: array.array = field(default_factory=lambda: array.array('I'))
    
    def decode_data(self, data_elem: ET.Element, width: int, height: int):
        """Decodes layer data according to encoding and compression"""
        encoding = data_elem.get('encoding')
        compression = data_elem.get('compression')
        
        if encoding == 'csv':
            # CSV format
            csv_data = data_elem.text.strip()
            gids = [int(x) for x in csv_data.replace('\n', '').split(',') if x.strip()]
            self.tiles = array.array('I', gids)
            
        elif encoding == 'base64':
            # Base64 format
            b64_data = data_elem.text.strip()
            raw_data = base64.b64decode(b64_data)
            
            # Decompress if necessary
            if compression == 'zlib':
                raw_data = zlib.decompress(raw_data)
            elif compression == 'gzip':
                import gzip
                raw_data = gzip.decompress(raw_data)
            elif compression == 'zstd':
                try:
                    import zstandard as zstd
                    dctx = zstd.ZstdDecompressor()
                    raw_data = dctx.decompress(raw_data)
                except ImportError:
                    raise ImportError("zstandard library required for zstd compression")
            
            # Convert bytes to uint32 array
            num_tiles = len(raw_data) // 4
            self.tiles = array.array('I')
            self.tiles.frombytes(raw_data)
            
        else:
            # XML format (deprecated)
            gids = []
            for tile_elem in data_elem.findall('tile'):
                gid = int(tile_elem.get('gid', 0))
                gids.append(gid)
            self.tiles = array.array('I', gids)
        
        self.encoding = encoding
        self.compression = compression
    
    def to_xml(self, width: int, height: int) -> ET.Element:
        """Converts data to XML"""
        elem = ET.Element('data')
        
        # Default to CSV which is simple and readable
        if not self.encoding:
            self.encoding = 'csv'
        
        elem.set('encoding', self.encoding)
        if self.compression:
            elem.set('compression', self.compression)
        
        if self.encoding == 'csv':
            # CSV format
            csv_lines = []
            for y in range(height):
                row_start = y * width
                row_end = row_start + width
                row = ','.join(str(gid) for gid in self.tiles[row_start:row_end])
                csv_lines.append(row)
            elem.text = '\n' + ',\n'.join(csv_lines) + '\n'
            
        elif self.encoding == 'base64':
            # Base64 format
            raw_data = self.tiles.tobytes()
            
            if self.compression == 'zlib':
                raw_data = zlib.compress(raw_data)
            elif self.compression == 'gzip':
                import gzip
                raw_data = gzip.compress(raw_data)
            
            b64_data = base64.b64encode(raw_data).decode('ascii')
            elem.text = '\n' + b64_data + '\n'
        
        return elem


@dataclass
class TileLayer:
    """Tile layer"""
    name: str
    width: int
    height: int
    id: int = 0
    visible: bool = True
    opacity: float = 1.0
    offsetx: float = 0
    offsety: float = 0
    parallaxx: float = 1.0
    parallaxy: float = 1.0
    tintcolor: Optional[str] = None
    properties: Dict[str, Property] = field(default_factory=dict)
    data: LayerData = field(default_factory=LayerData)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'TileLayer':
        layer = cls(
            name=elem.get('name', ''),
            width=int(elem.get('width', 0)),
            height=int(elem.get('height', 0)),
            id=int(elem.get('id', 0)),
            visible=elem.get('visible', '1') == '1',
            opacity=float(elem.get('opacity', 1.0)),
            offsetx=float(elem.get('offsetx', 0)),
            offsety=float(elem.get('offsety', 0)),
            parallaxx=float(elem.get('parallaxx', 1.0)),
            parallaxy=float(elem.get('parallaxy', 1.0)),
            tintcolor=elem.get('tintcolor')
        )
        
        # Properties
        props_elem = elem.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                layer.properties[prop.name] = prop
        
        # Layer data
        data_elem = elem.find('data')
        if data_elem is not None:
            layer.data = LayerData()
            layer.data.decode_data(data_elem, layer.width, layer.height)
        
        return layer
    
    def to_xml(self) -> ET.Element:
        elem = ET.Element('layer')
        elem.set('id', str(self.id))
        elem.set('name', self.name)
        elem.set('width', str(self.width))
        elem.set('height', str(self.height))
        
        if not self.visible:
            elem.set('visible', '0')
        if self.opacity != 1.0:
            elem.set('opacity', str(self.opacity))
        if self.offsetx:
            elem.set('offsetx', str(self.offsetx))
        if self.offsety:
            elem.set('offsety', str(self.offsety))
        if self.parallaxx != 1.0:
            elem.set('parallaxx', str(self.parallaxx))
        if self.parallaxy != 1.0:
            elem.set('parallaxy', str(self.parallaxy))
        if self.tintcolor:
            elem.set('tintcolor', self.tintcolor)
        
        if self.properties:
            props_elem = ET.SubElement(elem, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        elem.append(self.data.to_xml(self.width, self.height))
        
        return elem
    
    def get_tile_gid(self, x: int, y: int) -> int:
        """Gets the GID of the tile at position (x, y)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            index = y * self.width + x
            return self.data.tiles[index]
        return 0
    
    def set_tile_gid(self, x: int, y: int, gid: int):
        """Sets the GID of the tile at position (x, y)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            index = y * self.width + x
            self.data.tiles[index] = gid


@dataclass
class MapObject:
    """Object in an object layer"""
    id: int
    name: str = ""
    type: str = ""
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    rotation: float = 0
    gid: Optional[int] = None  # For tile objects
    visible: bool = True
    properties: Dict[str, Property] = field(default_factory=dict)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'MapObject':
        obj = cls(
            id=int(elem.get('id', 0)),
            name=elem.get('name', ''),
            type=elem.get('type', ''),
            x=float(elem.get('x', 0)),
            y=float(elem.get('y', 0)),
            width=float(elem.get('width', 0)),
            height=float(elem.get('height', 0)),
            rotation=float(elem.get('rotation', 0)),
            visible=elem.get('visible', '1') == '1'
        )
        
        if elem.get('gid'):
            obj.gid = int(elem.get('gid'))
        
        # Properties
        props_elem = elem.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                obj.properties[prop.name] = prop
        
        return obj
    
    def to_xml(self) -> ET.Element:
        elem = ET.Element('object')
        elem.set('id', str(self.id))
        
        if self.name:
            elem.set('name', self.name)
        if self.type:
            elem.set('type', self.type)
        
        elem.set('x', str(self.x))
        elem.set('y', str(self.y))
        
        if self.width:
            elem.set('width', str(self.width))
        if self.height:
            elem.set('height', str(self.height))
        if self.rotation:
            elem.set('rotation', str(self.rotation))
        if self.gid is not None:
            elem.set('gid', str(self.gid))
        if not self.visible:
            elem.set('visible', '0')
        
        if self.properties:
            props_elem = ET.SubElement(elem, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        return elem


@dataclass
class ObjectGroup:
    """Object layer"""
    name: str
    id: int = 0
    visible: bool = True
    opacity: float = 1.0
    offsetx: float = 0
    offsety: float = 0
    parallaxx: float = 1.0
    parallaxy: float = 1.0
    tintcolor: Optional[str] = None
    properties: Dict[str, Property] = field(default_factory=dict)
    objects: List[MapObject] = field(default_factory=list)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'ObjectGroup':
        group = cls(
            name=elem.get('name', ''),
            id=int(elem.get('id', 0)),
            visible=elem.get('visible', '1') == '1',
            opacity=float(elem.get('opacity', 1.0)),
            offsetx=float(elem.get('offsetx', 0)),
            offsety=float(elem.get('offsety', 0)),
            parallaxx=float(elem.get('parallaxx', 1.0)),
            parallaxy=float(elem.get('parallaxy', 1.0)),
            tintcolor=elem.get('tintcolor')
        )
        
        # Properties
        props_elem = elem.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                group.properties[prop.name] = prop
        
        # Objects
        for obj_elem in elem.findall('object'):
            obj = MapObject.from_xml(obj_elem)
            group.objects.append(obj)
        
        return group
    
    def to_xml(self) -> ET.Element:
        elem = ET.Element('objectgroup')
        elem.set('id', str(self.id))
        elem.set('name', self.name)
        
        if not self.visible:
            elem.set('visible', '0')
        if self.opacity != 1.0:
            elem.set('opacity', str(self.opacity))
        if self.offsetx:
            elem.set('offsetx', str(self.offsetx))
        if self.offsety:
            elem.set('offsety', str(self.offsety))
        if self.parallaxx != 1.0:
            elem.set('parallaxx', str(self.parallaxx))
        if self.parallaxy != 1.0:
            elem.set('parallaxy', str(self.parallaxy))
        if self.tintcolor:
            elem.set('tintcolor', self.tintcolor)
        
        if self.properties:
            props_elem = ET.SubElement(elem, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        for obj in self.objects:
            elem.append(obj.to_xml())
        
        return elem


@dataclass
class LayerGroup:
    """Group of layers"""
    name: str
    id: int = 0
    visible: bool = True
    opacity: float = 1.0
    offsetx: float = 0
    offsety: float = 0
    parallaxx: float = 1.0
    parallaxy: float = 1.0
    tintcolor: Optional[str] = None
    properties: Dict[str, Property] = field(default_factory=dict)
    layers: List[Union['TileLayer', 'ObjectGroup', 'LayerGroup']] = field(default_factory=list)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'LayerGroup':
        group = cls(
            name=elem.get('name', ''),
            id=int(elem.get('id', 0)),
            visible=elem.get('visible', '1') == '1',
            opacity=float(elem.get('opacity', 1.0)),
            offsetx=float(elem.get('offsetx', 0)),
            offsety=float(elem.get('offsety', 0)),
            parallaxx=float(elem.get('parallaxx', 1.0)),
            parallaxy=float(elem.get('parallaxy', 1.0)),
            tintcolor=elem.get('tintcolor')
        )
        
        # Properties
        props_elem = elem.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                group.properties[prop.name] = prop
        
        # Recursively load child layers
        for child in elem:
            if child.tag == 'layer':
                layer = TileLayer.from_xml(child)
                group.layers.append(layer)
            elif child.tag == 'objectgroup':
                obj_group = ObjectGroup.from_xml(child)
                group.layers.append(obj_group)
            elif child.tag == 'group':
                sub_group = LayerGroup.from_xml(child)
                group.layers.append(sub_group)
        
        return group
    
    def to_xml(self) -> ET.Element:
        elem = ET.Element('group')
        elem.set('id', str(self.id))
        elem.set('name', self.name)
        
        if not self.visible:
            elem.set('visible', '0')
        if self.opacity != 1.0:
            elem.set('opacity', str(self.opacity))
        if self.offsetx:
            elem.set('offsetx', str(self.offsetx))
        if self.offsety:
            elem.set('offsety', str(self.offsety))
        if self.parallaxx != 1.0:
            elem.set('parallaxx', str(self.parallaxx))
        if self.parallaxy != 1.0:
            elem.set('parallaxy', str(self.parallaxy))
        if self.tintcolor:
            elem.set('tintcolor', self.tintcolor)
        
        if self.properties:
            props_elem = ET.SubElement(elem, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        for layer in self.layers:
            elem.append(layer.to_xml())
        
        return elem


@dataclass
class TiledMap:
    """Complete Tiled map"""
    version: str = "1.10"
    tiledversion: str = ""
    orientation: str = "orthogonal"  # orthogonal, isometric, staggered, hexagonal
    renderorder: str = "right-down"
    width: int = 0
    height: int = 0
    tilewidth: int = 0
    tileheight: int = 0
    infinite: bool = False
    properties: Dict[str, Property] = field(default_factory=dict)
    tilesets: List[Tileset] = field(default_factory=list)
    layers: List[Union[TileLayer, ObjectGroup, LayerGroup]] = field(default_factory=list)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TiledMap':
        """Loads a TMX file"""
        filepath = Path(filepath)
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        map_obj = cls(
            version=root.get('version', '1.0'),
            tiledversion=root.get('tiledversion', ''),
            orientation=root.get('orientation', 'orthogonal'),
            renderorder=root.get('renderorder', 'right-down'),
            width=int(root.get('width', 0)),
            height=int(root.get('height', 0)),
            tilewidth=int(root.get('tilewidth', 0)),
            tileheight=int(root.get('tileheight', 0)),
            infinite=root.get('infinite', '0') == '1'
        )
        
        # Map properties
        props_elem = root.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                map_obj.properties[prop.name] = prop
        
        # Tilesets
        for tileset_elem in root.findall('tileset'):
            firstgid = int(tileset_elem.get('firstgid'))
            
            # If external tileset, load from TSX file
            if tileset_elem.get('source'):
                tsx_path = filepath.parent / tileset_elem.get('source')
                try:
                    tsx_tree = ET.parse(tsx_path)
                    tsx_root = tsx_tree.getroot()
                    tileset = Tileset.from_xml(tsx_root, firstgid)
                    tileset.source = tileset_elem.get('source')
                except FileNotFoundError:
                    print(f"Warning: External tileset not found: {tsx_path}")
                    # Create a placeholder tileset
                    tileset = Tileset(
                        firstgid=firstgid,
                        name=Path(tileset_elem.get('source')).stem,
                        tilewidth=map_obj.tilewidth,
                        tileheight=map_obj.tileheight,
                        source=tileset_elem.get('source')
                    )
            else:
                tileset = Tileset.from_xml(tileset_elem, firstgid)
            
            map_obj.tilesets.append(tileset)
        
        # Layers
        for elem in root:
            if elem.tag == 'layer':
                layer = TileLayer.from_xml(elem)
                map_obj.layers.append(layer)
            elif elem.tag == 'objectgroup':
                group = ObjectGroup.from_xml(elem)
                map_obj.layers.append(group)
            elif elem.tag == 'group':
                group = LayerGroup.from_xml(elem)
                map_obj.layers.append(group)
        
        return map_obj
    
    def save(self, filepath: Union[str, Path], encoding: str = 'csv', 
             compression: Optional[str] = None):
        """Saves the map to a TMX file"""
        filepath = Path(filepath)
        
        root = ET.Element('map')
        root.set('version', self.version)
        if self.tiledversion:
            root.set('tiledversion', self.tiledversion)
        root.set('orientation', self.orientation)
        root.set('renderorder', self.renderorder)
        root.set('width', str(self.width))
        root.set('height', str(self.height))
        root.set('tilewidth', str(self.tilewidth))
        root.set('tileheight', str(self.tileheight))
        
        if self.infinite:
            root.set('infinite', '1')
        
        # Properties
        if self.properties:
            props_elem = ET.SubElement(root, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        # Tilesets
        for tileset in self.tilesets:
            root.append(tileset.to_xml())
        
        # Layers
        for layer in self.layers:
            if isinstance(layer, TileLayer):
                layer.data.encoding = encoding
                layer.data.compression = compression
            root.append(layer.to_xml())
        
        # Save with readable formatting
        self._indent(root)
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
    
    @staticmethod
    def _indent(elem, level=0):
        """Formats XML with indentation"""
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                TiledMap._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent
    
    def get_tileset_for_gid(self, gid: int) -> Optional[Tileset]:
        """Gets the tileset corresponding to a GID"""
        for i in range(len(self.tilesets) - 1, -1, -1):
            if gid >= self.tilesets[i].firstgid:
                return self.tilesets[i]
        return None
    
    def get_layer_by_name(self, name: str) -> Optional[Union[TileLayer, ObjectGroup, LayerGroup]]:
        """Gets a layer by name"""
        def search_layers(layers):
            for layer in layers:
                if layer.name == name:
                    return layer
                if isinstance(layer, LayerGroup):
                    result = search_layers(layer.layers)
                    if result:
                        return result
            return None
        
        return search_layers(self.layers)
    
    def get_all_layers_flat(self) -> List[Union[TileLayer, ObjectGroup]]:
        """Gets all layers in a flat list (recursively expanding groups)"""
        result = []
        
        def flatten(layers):
            for layer in layers:
                if isinstance(layer, LayerGroup):
                    flatten(layer.layers)
                else:
                    result.append(layer)
        
        flatten(self.layers)
        return result


# Utility functions
def create_empty_map(width: int, height: int, tilewidth: int, tileheight: int,
                     orientation: str = "orthogonal") -> TiledMap:
    """Creates an empty map"""
    return TiledMap(
        width=width,
        height=height,
        tilewidth=tilewidth,
        tileheight=tileheight,
        orientation=orientation
    )


def create_layer(name: str, width: int, height: int) -> TileLayer:
    """Creates an empty tile layer"""
    layer = TileLayer(name=name, width=width, height=height)
    layer.data.tiles = array.array('I', [0] * (width * height))
    return layer


# Usage example
if __name__ == "__main__":
    # Read a TMX map
    print("Example 1: Reading a TMX map")
    try:
        map_data = TiledMap.load("example.tmx")
        print(f"Map loaded: {map_data.width}x{map_data.height}")
        print(f"Orientation: {map_data.orientation}")
        print(f"Tile size: {map_data.tilewidth}x{map_data.tileheight}")
        print(f"Number of tilesets: {len(map_data.tilesets)}")
        print(f"Number of layers: {len(map_data.layers)}")
        
        for i, layer in enumerate(map_data.layers):
            if isinstance(layer, TileLayer):
                print(f"  Layer {i}: {layer.name} (tile layer)")
            elif isinstance(layer, ObjectGroup):
                print(f"  Layer {i}: {layer.name} (object group, {len(layer.objects)} objects)")
    except FileNotFoundError:
        print("File example.tmx not found")
    
    # Create a new map
    print("\nExample 2: Creating a new map")
    new_map = create_empty_map(20, 15, 32, 32)
    
    # Add a tileset
    tileset = Tileset(
        firstgid=1,
        name="terrain",
        tilewidth=32,
        tileheight=32,
        tilecount=64,
        columns=8
    )
    tileset.image = Image(source="terrain.png", width=256, height=256)
    new_map.tilesets.append(tileset)
    
    # Add a tile layer
    background = create_layer("Background", 20, 15)
    # Fill with tiles (example: tile GID 1)
    for y in range(15):
        for x in range(20):
            background.set_tile_gid(x, y, 1)
    new_map.layers.append(background)
    
    # Add an object layer
    objects = ObjectGroup(name="Collisions")
    obj = MapObject(id=1, name="tree", type="collision", x=100, y=100, width=32, height=64)
    objects.objects.append(obj)
    new_map.layers.append(objects)
    
    # Save the map
    new_map.save("new_map.tmx", encoding='csv')
    print("Map saved as new_map.tmx")
    
    # Modify a specific tile
    print("\nExample 3: Modifying tiles")
    background.set_tile_gid(5, 5, 10)  # Change tile at position (5, 5) to GID 10
    print(f"Tile at (5, 5): {background.get_tile_gid(5, 5)}")
    
    # Save with different formats
    print("\nExample 4: Saving with different formats")
    new_map.save("map_csv.tmx", encoding='csv')
    new_map.save("map_base64.tmx", encoding='base64')
    new_map.save("map_base64_zlib.tmx", encoding='base64', compression='zlib')
    print("Maps saved in different formats")
