#!/usr/bin/env python3

"""
Module for reading, modifying and writing TMX files (Tiled Map Format)
Supports TMX version 1.11.0 and earlier versions

=============================================================================
WHAT IS TMX?
=============================================================================

TMX (Tiled Map XML) is the native format of the Tiled Map Editor, the most
popular tile map editor for 2D games. TMX files describe:

- Map dimensions and tile sizes
- Tilesets (collections of tile graphics)
- Layers (tile layers, object layers, groups)
- Custom properties (metadata on any element)

This module provides complete read/write support for TMX files, allowing:
- Loading maps created in Tiled into your game
- Modifying maps programmatically
- Creating maps entirely in code
- Saving maps back to TMX format

=============================================================================
TMX FILE STRUCTURE
=============================================================================

A TMX file is XML with this basic structure:

    <map version="1.10" orientation="orthogonal" width="100" height="100"
         tilewidth="32" tileheight="32">
         
        <tileset firstgid="1" name="terrain" tilewidth="32" tileheight="32">
            <image source="terrain.png" width="256" height="256"/>
        </tileset>
        
        <layer name="Ground" width="100" height="100">
            <data encoding="csv">
                1,2,3,4,5,...
            </data>
        </layer>
        
        <objectgroup name="Collisions">
            <object id="1" x="100" y="200" width="32" height="32"/>
        </objectgroup>
    </map>

=============================================================================
GLOBAL TILE IDs (GIDs)
=============================================================================

Tiles are referenced by Global IDs (GIDs) across all tilesets:

    Tileset A (firstgid=1):   tiles 1-100
    Tileset B (firstgid=101): tiles 101-200
    
    GID 0 = empty tile (no graphic)
    GID 50 = tile 50 from tileset A
    GID 150 = tile 50 from tileset B (150 - 101 = 49, so tile index 49)

Local tile ID within tileset = GID - tileset.firstgid

=============================================================================
SUPPORTED FEATURES
=============================================================================

Map Orientations:
- Orthogonal (standard square tiles)
- Isometric (diamond-shaped tiles)
- Staggered (offset rows/columns)
- Hexagonal (hexagon tiles)

Layer Types:
- TileLayer: Grid of tile references
- ObjectGroup: Vector objects (rectangles, points, etc.)
- LayerGroup: Folder containing other layers

Data Encodings:
- XML (deprecated, verbose)
- CSV (human-readable, good for debugging)
- Base64 (compact, binary)

Compressions (with Base64):
- None (uncompressed)
- zlib (standard compression)
- gzip (gzip format)
- zstd (modern, high-ratio compression)

=============================================================================
"""

import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import base64
import zlib
import struct
import array


# =============================================================================
# PROPERTY CLASS
# =============================================================================

@dataclass
class Property:
    """
    Custom property attached to any TMX element.
    
    Tiled allows adding custom properties to maps, layers, tiles, objects, etc.
    Properties are key-value pairs with typed values.
    
    ==========================================================================
    SUPPORTED TYPES
    ==========================================================================
    
    - string: Text value (default)
    - int: Integer number
    - float: Decimal number
    - bool: True/False
    - color: Color in #AARRGGBB format
    - file: File path reference
    - object: Reference to another object by ID
    
    ==========================================================================
    USE CASES
    ==========================================================================
    
    Properties enable data-driven game design:
    
    On tiles:
        solid=true         → Mark tiles as collision
        damage=10          → Damage dealt by hazard tiles
        animation_speed=2  → Custom animation timing
    
    On objects:
        spawn_type="enemy" → What to spawn here
        dialogue_id=42     → NPC dialogue reference
        trigger_event="boss_fight" → Event to fire
    
    On layers:
        Z=2                → Height level for 3D maps
        parallax_factor=0.5 → Parallax scrolling speed
    
    ==========================================================================
    """
    name: str                    # Property name (key)
    type: str = "string"         # Value type
    value: Any = None            # The actual value
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'Property':
        """
        Parse property from XML element.
        
        XML format:
            <property name="solid" type="bool" value="true"/>
            <property name="health" type="int" value="100"/>
            <property name="description" value="A wooden door"/>  (type defaults to string)
        """
        prop_type = elem.get('type', 'string')  # Default to string if not specified
        value = elem.get('value', '')
        
        # -----------------------------------------------------------------
        # TYPE CONVERSION
        # -----------------------------------------------------------------
        # Convert string value to appropriate Python type
        # This makes properties usable directly in game code
        
        if prop_type == 'int':
            value = int(value)
        elif prop_type == 'float':
            value = float(value)
        elif prop_type == 'bool':
            # XML stores as "true"/"false" strings
            value = value.lower() == 'true'
        elif prop_type == 'color':
            # Keep as string - parsing #AARRGGBB is caller's responsibility
            value = value
            
        return cls(name=elem.get('name'), type=prop_type, value=value)
    
    def to_xml(self) -> ET.Element:
        """Convert property back to XML element."""
        elem = ET.Element('property')
        elem.set('name', self.name)
        
        # Only include type attribute if not string (string is default)
        if self.type != 'string':
            elem.set('type', self.type)
        
        # Convert value to string for XML
        elem.set('value', str(self.value))
        return elem


# =============================================================================
# IMAGE CLASS
# =============================================================================

@dataclass
class Image:
    """
    Image reference used in tilesets.
    
    Represents an image file that contains tile graphics.
    Used by both regular tilesets (one image, many tiles) and
    image collection tilesets (one image per tile).
    
    ==========================================================================
    ATTRIBUTES
    ==========================================================================
    
    source: Path to image file (relative to TMX/TSX file)
    width:  Image width in pixels (optional, for validation)
    height: Image height in pixels (optional)
    trans:  Transparent color in hex (e.g., "ff00ff" for magenta)
            Pixels of this color become transparent
    
    ==========================================================================
    """
    source: str                          # Path to image file
    width: Optional[int] = None          # Image width (pixels)
    height: Optional[int] = None         # Image height (pixels)
    trans: Optional[str] = None          # Transparent color (#RRGGBB)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'Image':
        """Parse image from XML element."""
        return cls(
            source=elem.get('source', ''),
            # Width/height are optional - use None if not present
            width=int(elem.get('width')) if elem.get('width') else None,
            height=int(elem.get('height')) if elem.get('height') else None,
            trans=elem.get('trans')
        )
    
    def to_xml(self) -> ET.Element:
        """Convert image back to XML element."""
        elem = ET.Element('image')
        elem.set('source', self.source)
        
        # Only include optional attributes if they have values
        if self.width:
            elem.set('width', str(self.width))
        if self.height:
            elem.set('height', str(self.height))
        if self.trans:
            elem.set('trans', self.trans)
        return elem


# =============================================================================
# TILE CLASS
# =============================================================================

@dataclass
class Tile:
    """
    Individual tile within a tileset.
    
    Represents metadata for a specific tile. Not all tiles need Tile objects -
    only tiles with custom properties, animations, or (for image collections)
    individual images.
    
    ==========================================================================
    TILE IDs
    ==========================================================================
    
    The 'id' is LOCAL to the tileset (0-based index).
    To get the Global ID (GID): gid = tileset.firstgid + tile.id
    
    Example:
        Tileset with firstgid=101
        Tile with id=5
        GID = 101 + 5 = 106
    
    ==========================================================================
    IMAGE COLLECTION TILESETS
    ==========================================================================
    
    Some tilesets use individual images per tile instead of a spritesheet.
    These are called "image collection" tilesets, common for:
    - Isometric buildings (different sizes)
    - Character portraits
    - Items with varied dimensions
    
    For these, each Tile has its own Image reference.
    
    ==========================================================================
    """
    id: int                                          # Local tile ID (within tileset)
    type: str = ""                                   # Tile type/class
    properties: Dict[str, Property] = field(default_factory=dict)  # Custom properties
    image: Optional[Image] = None                    # Image (for collection tilesets)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'Tile':
        """Parse tile from XML element."""
        tile = cls(id=int(elem.get('id', 0)))
        tile.type = elem.get('type', '')
        
        # -----------------------------------------------------------------
        # PARSE PROPERTIES
        # -----------------------------------------------------------------
        props_elem = elem.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                # Store by name for O(1) lookup
                tile.properties[prop.name] = prop
        
        # -----------------------------------------------------------------
        # PARSE IMAGE (for image collection tilesets)
        # -----------------------------------------------------------------
        img_elem = elem.find('image')
        if img_elem is not None:
            tile.image = Image.from_xml(img_elem)
        
        return tile
    
    def to_xml(self) -> ET.Element:
        """Convert tile back to XML element."""
        elem = ET.Element('tile')
        elem.set('id', str(self.id))
        
        if self.type:
            elem.set('type', self.type)
        
        # Add properties container if we have any
        if self.properties:
            props_elem = ET.SubElement(elem, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        # Add image for collection tilesets
        if self.image:
            elem.append(self.image.to_xml())
        
        return elem


# =============================================================================
# TILESET CLASS
# =============================================================================

@dataclass
class Tileset:
    """
    Tileset collection - a set of tile graphics.
    
    ==========================================================================
    TILESET TYPES
    ==========================================================================
    
    1. SPRITESHEET TILESET (most common):
       One large image divided into a grid of tiles.
       
       +---+---+---+---+
       | 0 | 1 | 2 | 3 |
       +---+---+---+---+
       | 4 | 5 | 6 | 7 |
       +---+---+---+---+
       
       Attributes used: image, tilewidth, tileheight, columns, spacing, margin
       
    2. IMAGE COLLECTION TILESET:
       Each tile is a separate image file.
       
       tiles/
       ├── tree.png     (tile 0)
       ├── house.png    (tile 1)
       └── rock.png     (tile 2)
       
       Each tile has its own Image reference.
       Used for: isometric objects, varied-size tiles
    
    ==========================================================================
    EMBEDDED vs EXTERNAL TILESETS
    ==========================================================================
    
    EMBEDDED: Tileset data is inside the TMX file
        <tileset firstgid="1" name="terrain" tilewidth="32" ...>
            <image source="terrain.png"/>
        </tileset>
    
    EXTERNAL (TSX): Tileset data is in separate .tsx file
        <tileset firstgid="1" source="terrain.tsx"/>
        
        The TSX file contains the actual tileset definition.
        Advantages:
        - Share tileset across multiple maps
        - Edit tileset without opening each map
        - Smaller TMX files
    
    ==========================================================================
    SPACING AND MARGIN
    ==========================================================================
    
    For spritesheets with gaps between tiles:
    
    margin = pixels around the EDGE of the entire image
    spacing = pixels BETWEEN tiles
    
    +--+===+===+===+--+
    |  | 0 | 1 | 2 |  |  <- margin
    +--+===+===+===+--+
    |  | 3 | 4 | 5 |  |
    +--+===+===+===+--+
         ^
         spacing between tiles
    
    ==========================================================================
    """
    firstgid: int                                    # First Global ID
    name: str                                        # Tileset name
    tilewidth: int                                   # Tile width in pixels
    tileheight: int                                  # Tile height in pixels
    tilecount: int = 0                               # Total number of tiles
    columns: int = 0                                 # Tiles per row (for spritesheet)
    spacing: int = 0                                 # Pixels between tiles
    margin: int = 0                                  # Pixels around edge
    image: Optional[Image] = None                    # Spritesheet image
    tiles: Dict[int, Tile] = field(default_factory=dict)  # Tile metadata
    properties: Dict[str, Property] = field(default_factory=dict)
    source: Optional[str] = None                     # TSX file path (if external)
    
    @classmethod
    def from_xml(cls, elem: ET.Element, firstgid: int) -> 'Tileset':
        """
        Parse tileset from XML element.
        
        Parameters:
        -----------
        elem : ET.Element
            The <tileset> XML element
        firstgid : int
            First Global ID (from parent TMX, not the TSX itself)
        """
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
        
        # Parse properties
        props_elem = elem.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                tileset.properties[prop.name] = prop
        
        # Parse tileset image (for spritesheet tilesets)
        img_elem = elem.find('image')
        if img_elem is not None:
            tileset.image = Image.from_xml(img_elem)
        
        # Parse individual tile definitions
        # Only tiles with metadata (properties, animations, images) are listed
        for tile_elem in elem.findall('tile'):
            tile = Tile.from_xml(tile_elem)
            tileset.tiles[tile.id] = tile
        
        return tileset
    
    def to_xml(self, include_firstgid: bool = True) -> ET.Element:
        """
        Convert tileset back to XML element.
        
        Parameters:
        -----------
        include_firstgid : bool
            Whether to include firstgid attribute.
            True for TMX embedding, False for TSX export.
        """
        elem = ET.Element('tileset')
        
        if include_firstgid:
            elem.set('firstgid', str(self.firstgid))
        
        # If this is an external tileset reference, only output source
        if self.source:
            elem.set('source', self.source)
            return elem  # External tilesets only have firstgid and source
        
        # Full tileset definition
        elem.set('name', self.name)
        elem.set('tilewidth', str(self.tilewidth))
        elem.set('tileheight', str(self.tileheight))
        elem.set('tilecount', str(self.tilecount))
        elem.set('columns', str(self.columns))
        
        # Only include spacing/margin if non-zero
        if self.spacing:
            elem.set('spacing', str(self.spacing))
        if self.margin:
            elem.set('margin', str(self.margin))
        
        # Properties
        if self.properties:
            props_elem = ET.SubElement(elem, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        # Tileset image
        if self.image:
            elem.append(self.image.to_xml())
        
        # Individual tile definitions
        for tile in self.tiles.values():
            elem.append(tile.to_xml())
        
        return elem


# =============================================================================
# LAYER DATA CLASS
# =============================================================================

@dataclass
class LayerData:
    """
    Tile layer data storage and encoding.
    
    Handles the actual tile data within a layer - the grid of GIDs that
    defines which tile appears at each position.
    
    ==========================================================================
    DATA ENCODINGS
    ==========================================================================
    
    TMX supports multiple encodings for tile data:
    
    1. XML (deprecated):
       <data>
           <tile gid="1"/><tile gid="2"/><tile gid="3"/>...
       </data>
       Verbose, slow to parse, not recommended.
    
    2. CSV:
       <data encoding="csv">
           1,2,3,4,5,
           6,7,8,9,10
       </data>
       Human-readable, good for debugging, moderate file size.
    
    3. Base64:
       <data encoding="base64">
           AQAAAAIAAAADAAAABAAAAAUAAAAGAAAABwAAAA==
       </data>
       Compact binary, optionally compressed.
    
    ==========================================================================
    COMPRESSION (with Base64 only)
    ==========================================================================
    
    - zlib: Standard compression, good ratio, fast
    - gzip: Similar to zlib with header, widely compatible
    - zstd: Modern, best ratio, requires extra library
    
    Compression is almost always worth it:
    - 100x100 map CSV: ~50KB
    - 100x100 map Base64+zlib: ~5KB
    
    ==========================================================================
    INTERNAL STORAGE
    ==========================================================================
    
    Internally, tiles are stored as array.array('I') - unsigned 32-bit ints.
    This is:
    - Memory efficient (4 bytes per tile)
    - Fast to access (C-level array)
    - Easy to convert to/from binary formats
    
    Index calculation: tiles[y * width + x]
    
    ==========================================================================
    """
    encoding: Optional[str] = None      # csv, base64, or None (XML)
    compression: Optional[str] = None   # gzip, zlib, zstd, or None
    tiles: array.array = field(default_factory=lambda: array.array('I'))
    
    def decode_data(self, data_elem: ET.Element, width: int, height: int):
        """
        Decode tile data from XML element.
        
        Parameters:
        -----------
        data_elem : ET.Element
            The <data> XML element containing tile data
        width, height : int
            Layer dimensions (needed for validation)
        """
        encoding = data_elem.get('encoding')
        compression = data_elem.get('compression')
        
        if encoding == 'csv':
            # -----------------------------------------------------------------
            # CSV FORMAT
            # -----------------------------------------------------------------
            # Data looks like: "1,2,3,4,\n5,6,7,8,\n..."
            # Each number is a GID (Global ID)
            csv_data = data_elem.text.strip()
            
            # Remove newlines, split by comma, convert to integers
            # Filter empty strings (trailing commas create empty elements)
            gids = [int(x) for x in csv_data.replace('\n', '').split(',') 
                    if x.strip()]
            self.tiles = array.array('I', gids)
            
        elif encoding == 'base64':
            # -----------------------------------------------------------------
            # BASE64 FORMAT
            # -----------------------------------------------------------------
            # Binary data encoded as Base64 string
            b64_data = data_elem.text.strip()
            raw_data = base64.b64decode(b64_data)
            
            # -----------------------------------------------------------------
            # DECOMPRESSION (if compressed)
            # -----------------------------------------------------------------
            if compression == 'zlib':
                raw_data = zlib.decompress(raw_data)
                
            elif compression == 'gzip':
                import gzip
                raw_data = gzip.decompress(raw_data)
                
            elif compression == 'zstd':
                # zstd requires external library (not in stdlib)
                try:
                    import zstandard as zstd
                    dctx = zstd.ZstdDecompressor()
                    raw_data = dctx.decompress(raw_data)
                except ImportError:
                    raise ImportError(
                        "zstandard library required for zstd compression. "
                        "Install with: pip install zstandard"
                    )
            
            # -----------------------------------------------------------------
            # CONVERT BYTES TO UINT32 ARRAY
            # -----------------------------------------------------------------
            # Each tile is 4 bytes (little-endian uint32)
            # Total tiles = bytes / 4
            num_tiles = len(raw_data) // 4
            self.tiles = array.array('I')
            self.tiles.frombytes(raw_data)
            
        else:
            # -----------------------------------------------------------------
            # XML FORMAT (deprecated)
            # -----------------------------------------------------------------
            # Each tile is an element: <tile gid="5"/>
            gids = []
            for tile_elem in data_elem.findall('tile'):
                gid = int(tile_elem.get('gid', 0))
                gids.append(gid)
            self.tiles = array.array('I', gids)
        
        # Store encoding info for round-trip saving
        self.encoding = encoding
        self.compression = compression
    
    def to_xml(self, width: int, height: int) -> ET.Element:
        """
        Convert tile data back to XML element.
        
        Parameters:
        -----------
        width, height : int
            Layer dimensions (for CSV formatting)
        """
        elem = ET.Element('data')
        
        # Default to CSV if no encoding set (readable, debuggable)
        if not self.encoding:
            self.encoding = 'csv'
        
        elem.set('encoding', self.encoding)
        if self.compression:
            elem.set('compression', self.compression)
        
        if self.encoding == 'csv':
            # -----------------------------------------------------------------
            # CSV OUTPUT
            # -----------------------------------------------------------------
            # Format as rows for readability
            csv_lines = []
            for y in range(height):
                row_start = y * width
                row_end = row_start + width
                # Join GIDs with commas
                row = ','.join(str(gid) for gid in self.tiles[row_start:row_end])
                csv_lines.append(row)
            
            # Combine rows with newlines, wrap in newlines for formatting
            elem.text = '\n' + ',\n'.join(csv_lines) + '\n'
            
        elif self.encoding == 'base64':
            # -----------------------------------------------------------------
            # BASE64 OUTPUT
            # -----------------------------------------------------------------
            # Convert array to bytes
            raw_data = self.tiles.tobytes()
            
            # Compress if requested
            if self.compression == 'zlib':
                raw_data = zlib.compress(raw_data)
            elif self.compression == 'gzip':
                import gzip
                raw_data = gzip.compress(raw_data)
            # Note: zstd writing not implemented (rarely needed)
            
            # Encode to Base64
            b64_data = base64.b64encode(raw_data).decode('ascii')
            elem.text = '\n' + b64_data + '\n'
        
        return elem


# =============================================================================
# TILE LAYER CLASS
# =============================================================================

@dataclass
class TileLayer:
    """
    Tile layer - a grid of tile references.
    
    The main content layer type in TMX. Contains a 2D grid where each cell
    references a tile by its Global ID (GID).
    
    ==========================================================================
    LAYER PROPERTIES
    ==========================================================================
    
    Rendering properties:
    - visible: Whether layer is rendered
    - opacity: Transparency (0.0 = invisible, 1.0 = opaque)
    - tintcolor: Color tint applied to all tiles
    
    Positioning:
    - offsetx, offsety: Pixel offset from map origin
    - parallaxx, parallaxy: Parallax scrolling factors
      (1.0 = normal, 0.5 = half speed, 0 = static background)
    
    ==========================================================================
    TILE ACCESS
    ==========================================================================
    
    Access tiles using get_tile_gid(x, y) and set_tile_gid(x, y, gid):
    
        gid = layer.get_tile_gid(5, 10)  # Get tile at column 5, row 10
        layer.set_tile_gid(5, 10, 42)    # Set tile to GID 42
    
    GID 0 = empty (no tile)
    GID > 0 = reference to tileset tile
    
    ==========================================================================
    """
    name: str                                        # Layer name
    width: int                                       # Width in tiles
    height: int                                      # Height in tiles
    id: int = 0                                      # Unique layer ID
    visible: bool = True                             # Is layer rendered?
    opacity: float = 1.0                             # Transparency
    offsetx: float = 0                               # X pixel offset
    offsety: float = 0                               # Y pixel offset
    parallaxx: float = 1.0                           # Parallax X factor
    parallaxy: float = 1.0                           # Parallax Y factor
    tintcolor: Optional[str] = None                  # Color tint (#AARRGGBB)
    properties: Dict[str, Property] = field(default_factory=dict)
    data: LayerData = field(default_factory=LayerData)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'TileLayer':
        """Parse tile layer from XML element."""
        layer = cls(
            name=elem.get('name', ''),
            width=int(elem.get('width', 0)),
            height=int(elem.get('height', 0)),
            id=int(elem.get('id', 0)),
            # '1' is default for visible (absent means visible)
            visible=elem.get('visible', '1') == '1',
            opacity=float(elem.get('opacity', 1.0)),
            offsetx=float(elem.get('offsetx', 0)),
            offsety=float(elem.get('offsety', 0)),
            parallaxx=float(elem.get('parallaxx', 1.0)),
            parallaxy=float(elem.get('parallaxy', 1.0)),
            tintcolor=elem.get('tintcolor')
        )
        
        # Parse properties
        props_elem = elem.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                layer.properties[prop.name] = prop
        
        # Parse tile data
        data_elem = elem.find('data')
        if data_elem is not None:
            layer.data = LayerData()
            layer.data.decode_data(data_elem, layer.width, layer.height)
        
        return layer
    
    def to_xml(self) -> ET.Element:
        """Convert tile layer back to XML element."""
        elem = ET.Element('layer')
        elem.set('id', str(self.id))
        elem.set('name', self.name)
        elem.set('width', str(self.width))
        elem.set('height', str(self.height))
        
        # Only include non-default values (reduces file size)
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
        
        # Properties
        if self.properties:
            props_elem = ET.SubElement(elem, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        # Tile data
        elem.append(self.data.to_xml(self.width, self.height))
        
        return elem
    
    def get_tile_gid(self, x: int, y: int) -> int:
        """
        Get the GID of the tile at position (x, y).
        
        Parameters:
        -----------
        x : int
            Column (0 to width-1)
        y : int
            Row (0 to height-1)
            
        Returns:
        --------
        int : Global tile ID (0 = empty, >0 = tile reference)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            # Convert 2D coords to 1D index: row-major order
            index = y * self.width + x
            return self.data.tiles[index]
        return 0  # Out of bounds = empty
    
    def set_tile_gid(self, x: int, y: int, gid: int):
        """
        Set the GID of the tile at position (x, y).
        
        Parameters:
        -----------
        x, y : int
            Tile coordinates
        gid : int
            Global tile ID to set (0 = clear tile)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            index = y * self.width + x
            self.data.tiles[index] = gid


# =============================================================================
# MAP OBJECT CLASS
# =============================================================================

@dataclass
class MapObject:
    """
    Object in an object layer.
    
    Objects are vector shapes placed on the map, used for:
    - Collision shapes (rectangles, polygons)
    - Spawn points (position only)
    - Trigger areas
    - Entity placement (tile objects)
    
    ==========================================================================
    OBJECT TYPES
    ==========================================================================
    
    Basic rectangle:
        x, y, width, height define the bounds
        
    Point:
        Just x, y (width and height are 0)
        
    Tile object:
        Has a gid - displays a tile graphic at this position
        Used for placing decorations, items, etc.
        
    Polygon/Polyline:
        Has list of points (not implemented in this basic version)
    
    ==========================================================================
    """
    id: int                                          # Unique object ID
    name: str = ""                                   # Object name
    type: str = ""                                   # Object type/class
    x: float = 0                                     # X position
    y: float = 0                                     # Y position
    width: float = 0                                 # Width (0 for points)
    height: float = 0                                # Height (0 for points)
    rotation: float = 0                              # Rotation in degrees
    gid: Optional[int] = None                        # Tile GID (for tile objects)
    visible: bool = True                             # Is object visible?
    properties: Dict[str, Property] = field(default_factory=dict)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'MapObject':
        """Parse object from XML element."""
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
        
        # GID only present for tile objects
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
        """Convert object back to XML element."""
        elem = ET.Element('object')
        elem.set('id', str(self.id))
        
        # Only include non-empty optional attributes
        if self.name:
            elem.set('name', self.name)
        if self.type:
            elem.set('type', self.type)
        
        # Position is always included
        elem.set('x', str(self.x))
        elem.set('y', str(self.y))
        
        # Size only if non-zero
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
        
        # Properties
        if self.properties:
            props_elem = ET.SubElement(elem, 'properties')
            for prop in self.properties.values():
                props_elem.append(prop.to_xml())
        
        return elem


# =============================================================================
# OBJECT GROUP CLASS
# =============================================================================

@dataclass
class ObjectGroup:
    """
    Object layer - contains vector objects.
    
    Used for non-tile data:
    - Collision shapes
    - Spawn points
    - Triggers and zones
    - Entity placements
    
    Objects are stored in a list (order may matter for some games).
    """
    name: str                                        # Layer name
    id: int = 0                                      # Unique layer ID
    visible: bool = True                             # Is layer visible?
    opacity: float = 1.0                             # Transparency
    offsetx: float = 0                               # X pixel offset
    offsety: float = 0                               # Y pixel offset
    parallaxx: float = 1.0                           # Parallax X factor
    parallaxy: float = 1.0                           # Parallax Y factor
    tintcolor: Optional[str] = None                  # Color tint
    properties: Dict[str, Property] = field(default_factory=dict)
    objects: List[MapObject] = field(default_factory=list)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'ObjectGroup':
        """Parse object group from XML element."""
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
        """Convert object group back to XML element."""
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


# =============================================================================
# LAYER GROUP CLASS
# =============================================================================

@dataclass
class LayerGroup:
    """
    Group of layers - a folder containing other layers.
    
    Layer groups help organize complex maps:
    
    Layers:
    ├── Background (group)
    │   ├── Sky
    │   └── Mountains
    ├── Gameplay (group)
    │   ├── Ground
    │   ├── Objects
    │   └── Collisions
    └── Foreground
    
    Groups can be nested (groups within groups).
    """
    name: str                                        # Group name
    id: int = 0                                      # Unique ID
    visible: bool = True                             # Is group visible?
    opacity: float = 1.0                             # Transparency (affects all children)
    offsetx: float = 0                               # X offset (affects all children)
    offsety: float = 0                               # Y offset
    parallaxx: float = 1.0                           # Parallax X
    parallaxy: float = 1.0                           # Parallax Y
    tintcolor: Optional[str] = None                  # Color tint
    properties: Dict[str, Property] = field(default_factory=dict)
    # Recursive type: can contain TileLayer, ObjectGroup, or more LayerGroups
    layers: List[Union['TileLayer', 'ObjectGroup', 'LayerGroup']] = field(default_factory=list)
    
    @classmethod
    def from_xml(cls, elem: ET.Element) -> 'LayerGroup':
        """Parse layer group from XML element."""
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
        
        # -----------------------------------------------------------------
        # RECURSIVELY LOAD CHILD LAYERS
        # -----------------------------------------------------------------
        # Groups can contain any layer type, including nested groups
        for child in elem:
            if child.tag == 'layer':
                layer = TileLayer.from_xml(child)
                group.layers.append(layer)
            elif child.tag == 'objectgroup':
                obj_group = ObjectGroup.from_xml(child)
                group.layers.append(obj_group)
            elif child.tag == 'group':
                # Recursive: group within group
                sub_group = LayerGroup.from_xml(child)
                group.layers.append(sub_group)
        
        return group
    
    def to_xml(self) -> ET.Element:
        """Convert layer group back to XML element."""
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
        
        # Recursively convert child layers
        for layer in self.layers:
            elem.append(layer.to_xml())
        
        return elem


# =============================================================================
# TILED MAP CLASS (Main Entry Point)
# =============================================================================

@dataclass
class TiledMap:
    """
    Complete Tiled map - the root object for TMX files.
    
    This is the main class you'll interact with. It contains:
    - Map metadata (size, orientation, tile size)
    - Tilesets (collections of tile graphics)
    - Layers (tile layers, object layers, groups)
    - Custom properties
    
    ==========================================================================
    MAP ORIENTATIONS
    ==========================================================================
    
    ORTHOGONAL (most common):
        Standard square grid, tiles aligned in rows and columns.
        +---+---+---+
        | 0 | 1 | 2 |
        +---+---+---+
        | 3 | 4 | 5 |
        +---+---+---+
    
    ISOMETRIC:
        Diamond-shaped tiles for pseudo-3D effect.
           /\
          /0 \
         /\  /\
        /1 \/2 \
        \  /\  /
         \/3 \/
    
    STAGGERED:
        Offset rows/columns (isometric without rotation).
    
    HEXAGONAL:
        Hexagon tiles for strategy games.
    
    ==========================================================================
    RENDER ORDER
    ==========================================================================
    
    Determines which corner rendering starts from:
    - right-down: Left-to-right, top-to-bottom (most common)
    - right-up: Left-to-right, bottom-to-top
    - left-down: Right-to-left, top-to-bottom
    - left-up: Right-to-left, bottom-to-top
    
    ==========================================================================
    USAGE
    ==========================================================================
    
    Loading:
        map_data = TiledMap.load("level1.tmx")
        print(f"Map size: {map_data.width}x{map_data.height}")
    
    Accessing layers:
        ground = map_data.get_layer_by_name("Ground")
        tile_gid = ground.get_tile_gid(5, 10)
    
    Modifying:
        ground.set_tile_gid(5, 10, 42)
        map_data.save("modified.tmx")
    
    ==========================================================================
    """
    version: str = "1.10"                            # TMX format version
    tiledversion: str = ""                           # Tiled editor version
    orientation: str = "orthogonal"                  # Map orientation
    renderorder: str = "right-down"                  # Render order
    width: int = 0                                   # Map width in tiles
    height: int = 0                                  # Map height in tiles
    tilewidth: int = 0                               # Tile width in pixels
    tileheight: int = 0                              # Tile height in pixels
    infinite: bool = False                           # Is map infinite?
    properties: Dict[str, Property] = field(default_factory=dict)
    tilesets: List[Tileset] = field(default_factory=list)
    layers: List[Union[TileLayer, ObjectGroup, LayerGroup]] = field(default_factory=list)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TiledMap':
        """
        Load a TMX file from disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the .tmx file
            
        Returns:
        --------
        TiledMap : Parsed map object
        
        Raises:
        -------
        FileNotFoundError : If TMX file doesn't exist
        xml.etree.ElementTree.ParseError : If XML is malformed
        """
        filepath = Path(filepath)
        
        # Parse XML
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # -----------------------------------------------------------------
        # PARSE MAP ATTRIBUTES
        # -----------------------------------------------------------------
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
        
        # -----------------------------------------------------------------
        # PARSE MAP PROPERTIES
        # -----------------------------------------------------------------
        props_elem = root.find('properties')
        if props_elem is not None:
            for prop_elem in props_elem.findall('property'):
                prop = Property.from_xml(prop_elem)
                map_obj.properties[prop.name] = prop
        
        # -----------------------------------------------------------------
        # PARSE TILESETS
        # -----------------------------------------------------------------
        for tileset_elem in root.findall('tileset'):
            firstgid = int(tileset_elem.get('firstgid'))
            
            # Check if external tileset (TSX file)
            if tileset_elem.get('source'):
                # ---------------------------------------------------------
                # EXTERNAL TILESET (TSX)
                # ---------------------------------------------------------
                # The TMX only contains a reference; actual data is in TSX
                tsx_path = filepath.parent / tileset_elem.get('source')
                
                try:
                    # Parse the external TSX file
                    tsx_tree = ET.parse(tsx_path)
                    tsx_root = tsx_tree.getroot()
                    
                    # Load tileset from TSX, but use firstgid from TMX
                    tileset = Tileset.from_xml(tsx_root, firstgid)
                    tileset.source = tileset_elem.get('source')
                    
                except FileNotFoundError:
                    # TSX file missing - create placeholder
                    print(f"Warning: External tileset not found: {tsx_path}")
                    tileset = Tileset(
                        firstgid=firstgid,
                        name=Path(tileset_elem.get('source')).stem,
                        tilewidth=map_obj.tilewidth,
                        tileheight=map_obj.tileheight,
                        source=tileset_elem.get('source')
                    )
            else:
                # ---------------------------------------------------------
                # EMBEDDED TILESET
                # ---------------------------------------------------------
                tileset = Tileset.from_xml(tileset_elem, firstgid)
            
            map_obj.tilesets.append(tileset)
        
        # -----------------------------------------------------------------
        # PARSE LAYERS
        # -----------------------------------------------------------------
        # Iterate through direct children of <map>
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
            # Note: tileset and properties are handled above
        
        return map_obj
    
    def save(self, filepath: Union[str, Path], encoding: str = 'csv', 
             compression: Optional[str] = None):
        """
        Save the map to a TMX file.
        
        Parameters:
        -----------
        filepath : str or Path
            Output file path
        encoding : str
            Tile data encoding: 'csv' or 'base64'
            Default 'csv' for human-readable output
        compression : str, optional
            Compression for base64: 'zlib', 'gzip', or None
        """
        filepath = Path(filepath)
        
        # Build XML tree
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
        
        # Layers (set encoding/compression for tile layers)
        for layer in self.layers:
            if isinstance(layer, TileLayer):
                layer.data.encoding = encoding
                layer.data.compression = compression
            root.append(layer.to_xml())
        
        # Format XML with indentation for readability
        self._indent(root)
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
    
    @staticmethod
    def _indent(elem, level=0):
        """
        Add indentation to XML for readable output.
        
        ElementTree doesn't include pretty-printing by default.
        This recursive method adds newlines and spaces.
        """
        indent = "\n" + "  " * level
        
        if len(elem):  # Has children
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            
            for child in elem:
                TiledMap._indent(child, level + 1)
            
            # Last child's tail
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent
    
    def get_tileset_for_gid(self, gid: int) -> Optional[Tileset]:
        """
        Find which tileset contains a given GID.
        
        Parameters:
        -----------
        gid : int
            Global tile ID
            
        Returns:
        --------
        Tileset or None : The tileset containing this GID
        
        =======================================================================
        ALGORITHM
        =======================================================================
        
        Tilesets are sorted by firstgid (ascending).
        A GID belongs to the tileset with the largest firstgid <= gid.
        
        Example:
            Tileset A: firstgid=1
            Tileset B: firstgid=101
            Tileset C: firstgid=201
            
            GID 50:  50 >= 1, 50 < 101  → Tileset A
            GID 150: 150 >= 101, 150 < 201 → Tileset B
            GID 250: 250 >= 201 → Tileset C
        
        We iterate backwards (largest firstgid first) for efficiency.
        """
        # Iterate from last tileset (highest firstgid) to first
        for i in range(len(self.tilesets) - 1, -1, -1):
            if gid >= self.tilesets[i].firstgid:
                return self.tilesets[i]
        return None
    
    def get_layer_by_name(self, name: str) -> Optional[Union[TileLayer, ObjectGroup, LayerGroup]]:
        """
        Find a layer by name (searches recursively through groups).
        
        Parameters:
        -----------
        name : str
            Layer name to find
            
        Returns:
        --------
        Layer or None : The found layer, or None if not found
        """
        def search_layers(layers):
            for layer in layers:
                if layer.name == name:
                    return layer
                # Recursively search inside groups
                if isinstance(layer, LayerGroup):
                    result = search_layers(layer.layers)
                    if result:
                        return result
            return None
        
        return search_layers(self.layers)
    
    def get_all_layers_flat(self) -> List[Union[TileLayer, ObjectGroup]]:
        """
        Get all layers in a flat list (expanding groups recursively).
        
        Useful when you need to iterate through all layers regardless
        of group hierarchy.
        
        Returns:
        --------
        List : All TileLayer and ObjectGroup objects (groups themselves excluded)
        """
        result = []
        
        def flatten(layers):
            for layer in layers:
                if isinstance(layer, LayerGroup):
                    # Recurse into group, don't add group itself
                    flatten(layer.layers)
                else:
                    result.append(layer)
        
        flatten(self.layers)
        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_empty_map(width: int, height: int, tilewidth: int, tileheight: int,
                     orientation: str = "orthogonal") -> TiledMap:
    """
    Create an empty map with specified dimensions.
    
    Convenience function for creating maps programmatically.
    
    Parameters:
    -----------
    width, height : int
        Map size in tiles
    tilewidth, tileheight : int
        Tile size in pixels
    orientation : str
        Map orientation (default "orthogonal")
        
    Returns:
    --------
    TiledMap : Empty map ready for adding tilesets and layers
    """
    return TiledMap(
        width=width,
        height=height,
        tilewidth=tilewidth,
        tileheight=tileheight,
        orientation=orientation
    )


def create_layer(name: str, width: int, height: int) -> TileLayer:
    """
    Create an empty tile layer filled with GID 0 (empty tiles).
    
    Parameters:
    -----------
    name : str
        Layer name
    width, height : int
        Layer size in tiles
        
    Returns:
    --------
    TileLayer : Empty layer ready for setting tiles
    """
    layer = TileLayer(name=name, width=width, height=height)
    # Initialize with empty tiles (GID 0)
    layer.data.tiles = array.array('I', [0] * (width * height))
    return layer


# =============================================================================
# EXAMPLE USAGE (when run directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TMX Manager - Example Usage")
    print("=" * 60)
    
    # Example 1: Create a new map
    print("\n--- Creating New Map ---")
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
    ground = create_layer("Ground", 20, 15)
    for y in range(15):
        for x in range(20):
            ground.set_tile_gid(x, y, 1)  # Fill with tile GID 1
    new_map.layers.append(ground)
    
    # Add an object layer
    objects = ObjectGroup(name="Collisions")
    obj = MapObject(id=1, name="tree", type="collision", 
                   x=100, y=100, width=32, height=64)
    objects.objects.append(obj)
    new_map.layers.append(objects)
    
    print(f"Created map: {new_map.width}x{new_map.height} tiles")
    print(f"Tilesets: {len(new_map.tilesets)}")
    print(f"Layers: {len(new_map.layers)}")
    
    # Save the map
    new_map.save("example_output.tmx", encoding='csv')
    print("Saved to: example_output.tmx")
