# TMX Manager - Complete User Guide

A Python library for reading, modifying, and writing TMX files (Tiled Map Format).

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Loading Maps](#loading-maps)
6. [Accessing Map Data](#accessing-map-data)
7. [Modifying Maps](#modifying-maps)
8. [Creating Maps from Scratch](#creating-maps-from-scratch)
9. [Working with Tilesets](#working-with-tilesets)
10. [Working with Objects](#working-with-objects)
11. [Custom Properties](#custom-properties)
12. [Saving Maps](#saving-maps)
13. [Common Patterns](#common-patterns)
14. [API Reference](#api-reference)

---

## Overview

TMX Manager provides complete read/write support for TMX files created with the [Tiled Map Editor](https://www.mapeditor.org/). It allows you to:

- **Load** existing TMX maps into your Python game
- **Modify** tiles, objects, and properties programmatically
- **Create** new maps entirely in code
- **Save** maps back to TMX format

### Supported Features

| Feature | Support |
|---------|---------|
| Map Orientations | Orthogonal, Isometric, Staggered, Hexagonal |
| Layer Types | Tile Layers, Object Layers, Layer Groups |
| Data Encoding | XML, CSV, Base64 |
| Compression | None, zlib, gzip, zstd |
| Tilesets | Embedded, External (TSX), Image Collections |
| Properties | All types (string, int, float, bool, color, file, object) |

---

## Installation

Simply copy `tmx_manager.py` into your project:

```
my_game/
├── tmx_manager.py
├── maps/
│   └── level1.tmx
└── main.py
```

**Dependencies:**
- Python 3.7+
- Standard library only (no external dependencies)
- Optional: `zstandard` for zstd compression

---

## Quick Start

```python
from tmx_manager import TiledMap

# Load a map
game_map = TiledMap.load("maps/level1.tmx")

# Print basic info
print(f"Map size: {game_map.width}x{game_map.height} tiles")
print(f"Tile size: {game_map.tilewidth}x{game_map.tileheight} pixels")

# Access a layer
ground = game_map.get_layer_by_name("Ground")

# Get a tile
tile_gid = ground.get_tile_gid(5, 10)
print(f"Tile at (5,10): GID {tile_gid}")

# Modify a tile
ground.set_tile_gid(5, 10, 42)

# Save changes
game_map.save("maps/level1_modified.tmx")
```

---

## Core Concepts

### Global Tile IDs (GIDs)

Every tile in a TMX map is referenced by a **Global ID (GID)**. GIDs are unique across all tilesets in a map.

```
Tileset "terrain" (firstgid=1):    tiles 1-100
Tileset "objects" (firstgid=101):  tiles 101-200

GID 0  = Empty tile (no graphic)
GID 1  = First tile of "terrain"
GID 50 = 50th tile of "terrain"
GID 101 = First tile of "objects"
GID 150 = 50th tile of "objects"
```

**Converting GID to local tile ID:**
```python
local_id = gid - tileset.firstgid
```

### Layer Types

| Type | Description | Use Case |
|------|-------------|----------|
| `TileLayer` | Grid of tile GIDs | Terrain, walls, decorations |
| `ObjectGroup` | Vector objects | Collisions, spawn points, triggers |
| `LayerGroup` | Folder of layers | Organization |

### Map Structure

```
TiledMap
├── properties      (custom metadata)
├── tilesets[]      (tile graphics)
└── layers[]        (content)
    ├── TileLayer "Ground"
    ├── TileLayer "Walls"
    ├── ObjectGroup "Collisions"
    └── LayerGroup "Decorations"
        ├── TileLayer "Trees"
        └── TileLayer "Flowers"
```

---

## Loading Maps

### Basic Loading

```python
from tmx_manager import TiledMap

# Load from file path
game_map = TiledMap.load("maps/level1.tmx")

# Load using Path object
from pathlib import Path
game_map = TiledMap.load(Path("maps") / "level1.tmx")
```

### Handling External Tilesets

External tilesets (`.tsx` files) are loaded automatically:

```python
# If level1.tmx references "tilesets/terrain.tsx",
# the library looks for it relative to the TMX file location
game_map = TiledMap.load("maps/level1.tmx")
# Automatically loads: maps/tilesets/terrain.tsx
```

### Error Handling

```python
from tmx_manager import TiledMap

try:
    game_map = TiledMap.load("maps/level1.tmx")
except FileNotFoundError:
    print("Map file not found!")
except Exception as e:
    print(f"Error loading map: {e}")
```

---

## Accessing Map Data

### Map Properties

```python
# Basic dimensions
print(f"Size: {game_map.width}x{game_map.height} tiles")
print(f"Tile size: {game_map.tilewidth}x{game_map.tileheight} px")
print(f"Pixel size: {game_map.width * game_map.tilewidth}x{game_map.height * game_map.tileheight}")

# Map type
print(f"Orientation: {game_map.orientation}")  # orthogonal, isometric, etc.
print(f"Render order: {game_map.renderorder}")  # right-down, etc.
```

### Accessing Layers

```python
# By name (searches recursively through groups)
ground = game_map.get_layer_by_name("Ground")

# All layers (flat list, expands groups)
all_layers = game_map.get_all_layers_flat()
for layer in all_layers:
    print(f"Layer: {layer.name}")

# Direct access (top-level only)
first_layer = game_map.layers[0]

# Filter by type
from tmx_manager import TileLayer, ObjectGroup

tile_layers = [l for l in game_map.get_all_layers_flat() 
               if isinstance(l, TileLayer)]
object_layers = [l for l in game_map.get_all_layers_flat() 
                 if isinstance(l, ObjectGroup)]
```

### Reading Tiles

```python
layer = game_map.get_layer_by_name("Ground")

# Single tile
gid = layer.get_tile_gid(x=5, y=10)

# Check if empty
if gid == 0:
    print("Empty tile")

# Iterate all tiles
for y in range(layer.height):
    for x in range(layer.width):
        gid = layer.get_tile_gid(x, y)
        if gid > 0:
            print(f"Tile at ({x},{y}): GID {gid}")
```

### Finding Tileset for GID

```python
gid = layer.get_tile_gid(5, 10)
tileset = game_map.get_tileset_for_gid(gid)

if tileset:
    local_id = gid - tileset.firstgid
    print(f"Tileset: {tileset.name}, Local ID: {local_id}")
```

---

## Modifying Maps

### Changing Tiles

```python
layer = game_map.get_layer_by_name("Ground")

# Set single tile
layer.set_tile_gid(5, 10, 42)

# Clear a tile
layer.set_tile_gid(5, 10, 0)

# Fill a region
for y in range(5, 15):
    for x in range(10, 20):
        layer.set_tile_gid(x, y, 1)

# Replace all tiles of one type with another
for y in range(layer.height):
    for x in range(layer.width):
        if layer.get_tile_gid(x, y) == 5:
            layer.set_tile_gid(x, y, 10)
```

### Modifying Layer Properties

```python
layer = game_map.get_layer_by_name("Clouds")

# Visibility
layer.visible = False

# Transparency
layer.opacity = 0.5  # 50% transparent

# Parallax scrolling
layer.parallaxx = 0.5  # Scroll at half speed
layer.parallaxy = 0.5
```

---

## Creating Maps from Scratch

### Basic Map Creation

```python
from tmx_manager import (
    TiledMap, Tileset, Image, TileLayer, 
    create_empty_map, create_layer
)

# Create empty map
game_map = create_empty_map(
    width=50,       # 50 tiles wide
    height=40,      # 40 tiles tall
    tilewidth=32,   # 32px per tile
    tileheight=32
)

# Add a tileset
tileset = Tileset(
    firstgid=1,
    name="terrain",
    tilewidth=32,
    tileheight=32,
    tilecount=256,
    columns=16
)
tileset.image = Image(source="terrain.png", width=512, height=512)
game_map.tilesets.append(tileset)

# Add a layer
ground = create_layer("Ground", 50, 40)
game_map.layers.append(ground)

# Fill with grass (GID 1)
for y in range(40):
    for x in range(50):
        ground.set_tile_gid(x, y, 1)

# Save
game_map.save("new_map.tmx")
```

### Procedural Map Generation

```python
import random
from tmx_manager import create_empty_map, create_layer, Tileset, Image

def generate_dungeon(width, height):
    # Create map
    game_map = create_empty_map(width, height, 32, 32)
    
    # Add tileset (assuming: 1=floor, 2=wall)
    tileset = Tileset(
        firstgid=1, name="dungeon",
        tilewidth=32, tileheight=32,
        tilecount=64, columns=8
    )
    tileset.image = Image(source="dungeon.png", width=256, height=256)
    game_map.tilesets.append(tileset)
    
    # Create layers
    floor = create_layer("Floor", width, height)
    walls = create_layer("Walls", width, height)
    game_map.layers.extend([floor, walls])
    
    # Generate: walls on edges, floor inside
    FLOOR, WALL = 1, 2
    
    for y in range(height):
        for x in range(width):
            # Border walls
            if x == 0 or x == width-1 or y == 0 or y == height-1:
                walls.set_tile_gid(x, y, WALL)
            else:
                floor.set_tile_gid(x, y, FLOOR)
                # Random internal walls
                if random.random() < 0.1:
                    walls.set_tile_gid(x, y, WALL)
    
    return game_map

# Generate and save
dungeon = generate_dungeon(30, 20)
dungeon.save("dungeon.tmx")
```

---

## Working with Tilesets

### Accessing Tileset Data

```python
for tileset in game_map.tilesets:
    print(f"Tileset: {tileset.name}")
    print(f"  First GID: {tileset.firstgid}")
    print(f"  Tile count: {tileset.tilecount}")
    print(f"  Tile size: {tileset.tilewidth}x{tileset.tileheight}")
    print(f"  Columns: {tileset.columns}")
    
    if tileset.image:
        print(f"  Image: {tileset.image.source}")
```

### Tile Metadata

```python
tileset = game_map.tilesets[0]

# Access tiles with custom properties
for local_id, tile in tileset.tiles.items():
    gid = tileset.firstgid + local_id
    print(f"Tile GID {gid}:")
    print(f"  Type: {tile.type}")
    
    for prop_name, prop in tile.properties.items():
        print(f"  {prop_name} = {prop.value}")
```

### Image Collection Tilesets

```python
# For tilesets where each tile is a separate image
for local_id, tile in tileset.tiles.items():
    if tile.image:
        print(f"Tile {local_id}: {tile.image.source}")
        print(f"  Size: {tile.image.width}x{tile.image.height}")
```

---

## Working with Objects

### Reading Objects

```python
from tmx_manager import ObjectGroup

# Find object layer
collisions = game_map.get_layer_by_name("Collisions")

if isinstance(collisions, ObjectGroup):
    for obj in collisions.objects:
        print(f"Object: {obj.name} (type: {obj.type})")
        print(f"  Position: ({obj.x}, {obj.y})")
        print(f"  Size: {obj.width}x{obj.height}")
        
        # Check for tile object
        if obj.gid:
            print(f"  Tile GID: {obj.gid}")
```

### Creating Objects

```python
from tmx_manager import ObjectGroup, MapObject

# Create object layer
spawn_points = ObjectGroup(name="SpawnPoints", id=1)

# Add spawn point objects
player_spawn = MapObject(
    id=1,
    name="PlayerSpawn",
    type="spawn",
    x=100,
    y=200
)
spawn_points.objects.append(player_spawn)

# Add enemy spawn with properties
enemy_spawn = MapObject(
    id=2,
    name="EnemySpawn",
    type="spawn",
    x=500,
    y=300
)
enemy_spawn.properties["enemy_type"] = Property(
    name="enemy_type", 
    type="string", 
    value="goblin"
)
enemy_spawn.properties["count"] = Property(
    name="count", 
    type="int", 
    value=5
)
spawn_points.objects.append(enemy_spawn)

game_map.layers.append(spawn_points)
```

### Finding Objects by Type

```python
def find_objects_by_type(game_map, obj_type):
    """Find all objects of a specific type."""
    results = []
    
    for layer in game_map.get_all_layers_flat():
        if isinstance(layer, ObjectGroup):
            for obj in layer.objects:
                if obj.type == obj_type:
                    results.append(obj)
    
    return results

# Usage
spawn_points = find_objects_by_type(game_map, "spawn")
collision_rects = find_objects_by_type(game_map, "collision")
triggers = find_objects_by_type(game_map, "trigger")
```

---

## Custom Properties

### Reading Properties

```python
# Map properties
if "author" in game_map.properties:
    author = game_map.properties["author"].value
    print(f"Map author: {author}")

# Layer properties
layer = game_map.get_layer_by_name("Ground")
if "Z" in layer.properties:
    z_level = layer.properties["Z"].value
    print(f"Z level: {z_level}")

# Tile properties (for collision, etc.)
tileset = game_map.tilesets[0]
for local_id, tile in tileset.tiles.items():
    if "solid" in tile.properties:
        is_solid = tile.properties["solid"].value
        gid = tileset.firstgid + local_id
        print(f"Tile GID {gid} solid: {is_solid}")
```

### Setting Properties

```python
from tmx_manager import Property

# Add property to map
game_map.properties["version"] = Property(
    name="version",
    type="string",
    value="1.0.0"
)

# Add property to layer
layer.properties["Z"] = Property(
    name="Z",
    type="int",
    value=2
)

# Add property to object
obj.properties["health"] = Property(
    name="health",
    type="int",
    value=100
)
```

### Building a Solid Tile Lookup

```python
def build_solid_lookup(game_map):
    """Create GID -> is_solid mapping from tile properties."""
    solid_tiles = set()
    
    for tileset in game_map.tilesets:
        for local_id, tile in tileset.tiles.items():
            if "solid" in tile.properties:
                if tile.properties["solid"].value:
                    gid = tileset.firstgid + local_id
                    solid_tiles.add(gid)
    
    return solid_tiles

# Usage
solid = build_solid_lookup(game_map)
layer = game_map.get_layer_by_name("Walls")

for y in range(layer.height):
    for x in range(layer.width):
        gid = layer.get_tile_gid(x, y)
        if gid in solid:
            print(f"Solid tile at ({x}, {y})")
```

---

## Saving Maps

### Basic Save

```python
# Save with CSV encoding (human-readable)
game_map.save("output.tmx")

# Explicit encoding
game_map.save("output.tmx", encoding="csv")
```

### Compressed Save

```python
# Base64 + zlib (smaller file size)
game_map.save("output.tmx", encoding="base64", compression="zlib")

# Base64 + gzip
game_map.save("output.tmx", encoding="base64", compression="gzip")
```

### File Size Comparison

| Encoding | 100x100 Map |
|----------|-------------|
| CSV | ~50 KB |
| Base64 | ~15 KB |
| Base64 + zlib | ~5 KB |

---

## Common Patterns

### Game Loading Pattern

```python
class GameLevel:
    def __init__(self, tmx_path):
        self.map_data = TiledMap.load(tmx_path)
        self.width = self.map_data.width
        self.height = self.map_data.height
        
        # Cache layers
        self.ground = self.map_data.get_layer_by_name("Ground")
        self.walls = self.map_data.get_layer_by_name("Walls")
        self.objects = self.map_data.get_layer_by_name("Objects")
        
        # Build collision map
        self.solid_tiles = self._build_solid_lookup()
        
        # Load spawn points
        self.spawn_points = self._load_spawn_points()
    
    def _build_solid_lookup(self):
        solid = set()
        for tileset in self.map_data.tilesets:
            for local_id, tile in tileset.tiles.items():
                if tile.properties.get("solid", Property("", "bool", False)).value:
                    solid.add(tileset.firstgid + local_id)
        return solid
    
    def _load_spawn_points(self):
        spawns = {}
        if isinstance(self.objects, ObjectGroup):
            for obj in self.objects.objects:
                if obj.type == "spawn":
                    spawns[obj.name] = (obj.x, obj.y)
        return spawns
    
    def is_solid(self, tile_x, tile_y):
        if self.walls:
            gid = self.walls.get_tile_gid(tile_x, tile_y)
            return gid in self.solid_tiles
        return False
```

### Map Editor Tool Pattern

```python
class MapEditor:
    def __init__(self):
        self.current_map = None
        self.current_layer = None
        self.current_tile = 1
    
    def new_map(self, width, height):
        self.current_map = create_empty_map(width, height, 32, 32)
        layer = create_layer("Layer 1", width, height)
        self.current_map.layers.append(layer)
        self.current_layer = layer
    
    def load_map(self, path):
        self.current_map = TiledMap.load(path)
        if self.current_map.layers:
            self.current_layer = self.current_map.layers[0]
    
    def save_map(self, path):
        if self.current_map:
            self.current_map.save(path)
    
    def paint_tile(self, x, y):
        if self.current_layer and isinstance(self.current_layer, TileLayer):
            self.current_layer.set_tile_gid(x, y, self.current_tile)
    
    def erase_tile(self, x, y):
        if self.current_layer and isinstance(self.current_layer, TileLayer):
            self.current_layer.set_tile_gid(x, y, 0)
    
    def fill_region(self, x1, y1, x2, y2, gid):
        if self.current_layer and isinstance(self.current_layer, TileLayer):
            for y in range(min(y1, y2), max(y1, y2) + 1):
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    self.current_layer.set_tile_gid(x, y, gid)
```

---

## API Reference

### Main Classes

| Class | Description |
|-------|-------------|
| `TiledMap` | Root map object |
| `Tileset` | Collection of tile graphics |
| `Tile` | Individual tile metadata |
| `TileLayer` | Grid of tile GIDs |
| `ObjectGroup` | Layer of vector objects |
| `LayerGroup` | Folder containing layers |
| `MapObject` | Single object (rect, point, etc.) |
| `Property` | Custom key-value property |
| `Image` | Image file reference |

### Utility Functions

| Function | Description |
|----------|-------------|
| `create_empty_map(w, h, tw, th)` | Create new empty map |
| `create_layer(name, w, h)` | Create empty tile layer |

### TiledMap Methods

| Method | Description |
|--------|-------------|
| `TiledMap.load(path)` | Load TMX file |
| `save(path, encoding, compression)` | Save to TMX file |
| `get_layer_by_name(name)` | Find layer by name |
| `get_tileset_for_gid(gid)` | Find tileset containing GID |
| `get_all_layers_flat()` | Get all layers (expand groups) |

### TileLayer Methods

| Method | Description |
|--------|-------------|
| `get_tile_gid(x, y)` | Get tile GID at position |
| `set_tile_gid(x, y, gid)` | Set tile GID at position |

---

## Tips and Best Practices

1. **Use meaningful layer names** in Tiled - they're your API to access content
2. **Use custom properties** for game logic (solid, damage, spawn_type, etc.)
3. **Organize with layer groups** for complex maps
4. **Use external tilesets (.tsx)** to share across maps
5. **Choose CSV encoding** for debugging, **Base64+zlib** for production
6. **Cache layer references** instead of calling `get_layer_by_name()` repeatedly
7. **Build lookup tables** for tile properties at load time
