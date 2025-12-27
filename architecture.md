# Game Engine Architecture Guide

A comprehensive technical document explaining the structure, data flow, and performance optimizations of the TMX-based 2D game engine.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Diagram](#system-diagram)
3. [Data Flow Pipeline](#data-flow-pipeline)
4. [The Map System](#the-map-system)
5. [The Collision System](#the-collision-system)
6. [The Rendering Pipeline](#the-rendering-pipeline)
7. [The Entity System](#the-entity-system)
8. [The Sprite System](#the-sprite-system)
9. [The Camera System](#the-camera-system)
10. [Performance Optimizations](#performance-optimizations)
11. [Memory Management](#memory-management)
12. [Frame Update Cycle](#frame-update-cycle)

---

## Architecture Overview

This engine is designed for **tile-based 2D games with height levels** (pseudo-3D), optimized for performance through batched rendering, spatial data structures, and careful memory management.

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Separation of Concerns** | Each module handles one responsibility |
| **Data-Oriented Design** | NumPy arrays for bulk data operations |
| **Batched Rendering** | Minimize draw calls via sprite batching |
| **Lazy Initialization** | Defer expensive operations until needed |
| **Resource Sharing** | Cache and reuse textures/sprites |

### Module Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│                        GAME APPLICATION                         │
├─────────────────────────────────────────────────────────────────┤
│  TMX Manager    │  Map3D Structure  │  Entity Manager           │
│  (File I/O)     │  (World Data)     │  (Game Objects)           │
├─────────────────────────────────────────────────────────────────┤
│  Collision Map  │  Tileset Renderer │  Animated Sprites         │
│  (Physics)      │  (Tile Textures)  │  (Character Graphics)     │
├─────────────────────────────────────────────────────────────────┤
│  Camera         │  Sprite Batch     │  OpenGL Renderer          │
│  (View)         │  (Geometry)       │  (GPU Interface)          │
├─────────────────────────────────────────────────────────────────┤
│                      OpenGL / GPU Hardware                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Diagram

### Component Relationships

```
                    ┌──────────────┐
                    │   TMX File   │
                    │  (level.tmx) │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  TMX Manager │ ◄── Parses XML, handles encoding
                    │  (TiledMap)  │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Tilesets   │ │   Layers    │ │  Objects    │
    │  (graphics) │ │ (tile data) │ │  (spawns)   │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Tileset    │ │    Map3D    │ │   Entity    │
    │  Renderer   │ │  Structure  │ │   Manager   │
    │  (textures) │ │  (4D array) │ │ (characters)│
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           │               ▼               │
           │        ┌─────────────┐        │
           │        │  Collision  │        │
           │        │     Map     │◄───────┤
           │        │  (3D array) │        │
           │        └─────────────┘        │
           │                               │
           ▼                               ▼
    ┌─────────────────────────────────────────┐
    │            OpenGL Renderer              │
    │  ┌─────────────┐   ┌─────────────────┐  │
    │  │   Sprite    │   │    Texture      │  │
    │  │    Batch    │   │     Cache       │  │
    │  └─────────────┘   └─────────────────┘  │
    └─────────────────────────────────────────┘
                        │
                        ▼
                   ┌─────────┐
                   │   GPU   │
                   │ (Screen)│
                   └─────────┘
```

---

## Data Flow Pipeline

### Loading Phase (Startup)

```
1. TMX FILE PARSING
   ─────────────────
   TMX File → XML Parser → TiledMap Object
   
   • Decode tile data (CSV/Base64/zlib)
   • Load external tilesets (.tsx)
   • Parse custom properties

2. 3D MAP CONSTRUCTION  
   ────────────────────
   TiledMap → Map3DStructure
   
   • Extract Z levels from layer properties
   • Build 4D NumPy array [Z, Y, X, Layer]
   • Calculate level offsets for negative Z

3. COLLISION MAP BUILDING
   ──────────────────────
   Map3DStructure + Tile Properties → CollisionMap
   
   • Scan tile properties for "solid" flag
   • Build 3D collision array [Z, Y, X]
   • Store as uint16 flags (extensible)

4. TEXTURE LOADING
   ────────────────
   Tilesets → TilesetRenderer → GPU Textures
   
   • Load tileset images (PIL)
   • Cut individual tiles from spritesheets
   • Add 1px border (bleeding fix)
   • Upload to GPU memory

5. ENTITY SPAWNING
   ────────────────
   Object Layers → EntityManager → Characters
   
   • Parse spawn points from objects
   • Create player/NPC instances
   • Load and cache character sprites
```

### Runtime Phase (Game Loop)

```
┌─────────────────────────────────────────────────────────────┐
│                      FRAME UPDATE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INPUT                                                   │
│     └─► Player.move(dx, dy, dz)                             │
│                                                             │
│  2. UPDATE (EntityManager.update)                           │
│     ├─► NPC AI behaviors (wander, patrol, follow)           │
│     ├─► Apply velocities to positions                       │
│     ├─► Collision detection (CollisionMap)                  │
│     └─► Animation frame advancement                         │
│                                                             │
│  3. RENDER                                                  │
│     ├─► Calculate visible tile range (Camera)               │
│     ├─► Batch tiles by texture                              │
│     ├─► Collect entity render data                          │
│     ├─► Sort by depth                                       │
│     └─► Draw batched geometry (SpriteBatch)                 │
│                                                             │
│  4. PRESENT                                                 │
│     └─► Swap buffers (display frame)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## The Map System

### Why 4D Array?

The map uses a **4D NumPy array** with shape `[H, D, W, N]`:

| Dimension | Meaning | Purpose |
|-----------|---------|---------|
| H | Height levels (Z) | Floors, bridges, underground |
| D | Depth (Y tiles) | Rows in top-down view |
| W | Width (X tiles) | Columns in top-down view |
| N | Layer index | Multiple layers per level |

```python
# Access tile at position (x=5, y=10, z=0, layer=2)
gid = map_3d.mapa[0, 10, 5, 2]
```

### Why Not 3D?

A 3D array `[Z, Y, X]` would only allow **one tile per position**. But maps need:

- Ground layer + decoration layer at same Z
- Transparency/overlay effects
- Separate collision layer from visual layers

The 4th dimension (N) preserves layer separation.

### Level Offset System

TMX layers can have negative Z values (underground):

```
TMX Levels:  -2, -1, 0, 1, 2
Array Index:  0,  1, 2, 3, 4
Offset = 2 (add to convert level → index)
```

```python
# Convert level to array index
z_index = level + self.level_offset

# Convert array index back to level
level = z_index - self.level_offset
```

### Memory Layout

```
Map3DStructure
├── mapa: np.ndarray[uint16]     # 4D tile GIDs
│   └── Shape: (H, D, W, N)
│   └── Memory: H × D × W × N × 2 bytes
│
├── layer_names: List[str]        # Layer identification
├── layer_levels: List[int]       # Original Z values
│
└── collision: CollisionMap       # Parallel collision data
```

**Example Memory Usage:**
- 100×100 map, 3 height levels, 5 layers
- 3 × 100 × 100 × 5 × 2 bytes = 300 KB

---

## The Collision System

### Parallel Array Design

The collision map is a **3D NumPy array** parallel to the visual map:

```
Visual Map:    mapa[z, y, x, layer] = GID (what to draw)
Collision Map: data[z, y, x] = flags (can we walk here?)
```

**Why separate?**
- Collision doesn't need layer dimension (it's binary per position)
- Different access patterns (visual needs all layers, collision needs one answer)
- Can update independently

### Flag-Based Collision

Each cell stores `uint16` flags (currently using 1 bit):

```
Bit 0: Solid (blocks movement)
Bit 1: (reserved) Water
Bit 2: (reserved) Ladder
Bit 3: (reserved) Damage
...
```

```python
# Check if solid
is_blocked = collision_map.data[z, y, x] != 0

# Future: Check specific flags
is_water = collision_map.data[z, y, x] & 0x02
```

### Collision Detection Algorithm

```python
def can_move_to(px, py, z, char_width, char_depth, char_height):
    """
    Check if character can occupy a position.
    
    1. Convert pixel position to bounding box
    2. Get corner tiles of bounding box
    3. Get Z levels character occupies
    4. Check all corners at all Z levels
    5. Return False if ANY is solid
    """
    
    # Character bounding box in pixels
    left = px - (char_width * tile_width) / 2
    right = px + (char_width * tile_width) / 2
    top = py - (char_depth * tile_height) / 2
    bottom = py + (char_depth * tile_height) / 2
    
    # Check four corners
    corners = [(left, top), (right, top), 
               (left, bottom), (right, bottom)]
    
    # Z levels (character might span 2 levels)
    z_levels = [int(z), int(z + char_height)]
    
    for (cx, cy) in corners:
        tile_x = int(cx // tile_width)
        tile_y = int(cy // tile_height)
        for tz in z_levels:
            if collision_map.is_solid(tile_x, tile_y, tz):
                return False
    
    return True
```

### Separate Axis Collision

Movement is checked **separately for X and Y**:

```python
# Try X movement
if can_move_to(new_x, current_y, z):
    x = new_x

# Try Y movement (independently)
if can_move_to(current_x, new_y, z):
    y = new_y
```

**Why?** Enables **wall sliding**:
- Moving diagonally into a vertical wall
- X blocked, but Y allowed
- Character slides along the wall

Without separate checks, hitting any wall stops ALL movement.

---

## The Rendering Pipeline

### The Batching Problem

**Without batching:**
```
For each tile (10,000 tiles):
    Bind texture         ← GPU state change
    Upload 4 vertices    ← Small data transfer
    Draw 1 quad          ← Draw call overhead
    
Result: 10,000 draw calls = ~5 FPS
```

**With batching:**
```
Group tiles by texture (10 textures)
For each texture group:
    Bind texture once    ← 1 state change
    Upload all vertices  ← Bulk transfer
    Draw all quads       ← 1 draw call
    
Result: 10 draw calls = 60+ FPS
```

### SpriteBatch Architecture

```
SpriteBatch
├── vertices: np.array[float32]   # CPU-side vertex buffer
│   └── Pre-allocated for max_sprites × 4 vertices × 9 floats
│
├── indices: np.array[uint32]     # Index buffer (static)
│   └── Pre-generated: [0,1,2, 0,2,3, 4,5,6, 4,6,7, ...]
│
├── VAO, VBO, EBO                 # GPU buffer handles
│
└── current_texture               # Active texture for batch
```

### Vertex Format

Each vertex contains 9 floats (36 bytes):

```
[x, y, u, v, r, g, b, a, depth]
 ╰──╯  ╰──╯  ╰────────╯  ╰────╯
 pos   tex     color     z-order
```

**Stride calculation:**
```
9 floats × 4 bytes = 36 bytes per vertex
4 vertices × 36 bytes = 144 bytes per sprite
```

### Indexed Drawing

Each sprite (quad) uses 4 vertices but 6 indices:

```
Vertices:        Indices (2 triangles):
0───1            Triangle 1: 0, 1, 2
│ ╲ │            Triangle 2: 0, 2, 3
3───2            

Without indexing: 6 vertices × 9 floats = 54 floats/sprite
With indexing:    4 vertices × 9 floats + 6 indices = 42 values/sprite

Savings: ~22% less data per sprite
```

### Render Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    FRAME RENDERING                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. BEGIN FRAME                                             │
│     └─► glClear(COLOR | DEPTH)                              │
│                                                             │
│  2. TILE BATCHING                                           │
│     ┌─────────────────────────────────────────────────┐     │
│     │  For each visible tile:                         │     │
│     │    texture = tileset.get_texture(gid)           │     │
│     │    tile_batches[texture].append(tile_data)      │     │
│     └─────────────────────────────────────────────────┘     │
│                                                             │
│  3. BATCH RENDERING                                         │
│     ┌─────────────────────────────────────────────────┐     │
│     │  For each (texture, tiles) in batches:          │     │
│     │    batch.begin(texture)   # Bind texture        │     │
│     │    For each tile:                               │     │
│     │      batch.add_sprite(x, y, w, h, depth)        │     │
│     │    batch.flush()          # Draw all at once    │     │
│     └─────────────────────────────────────────────────┘     │
│                                                             │
│  4. ENTITY RENDERING                                        │
│     └─► Similar batching for character sprites              │
│                                                             │
│  5. UI RENDERING                                            │
│     └─► Text overlay (depth test disabled)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Depth Buffer Usage

Instead of sorting tiles back-to-front (CPU expensive), we use the **GPU depth buffer**:

```python
# Each tile gets a depth value based on position
depth = base_offset + (y / tile_height) + z_level

# GPU automatically discards pixels "behind" existing pixels
glEnable(GL_DEPTH_TEST)
glDepthFunc(GL_LESS)  # Keep closer pixels
```

**Benefits:**
- No CPU sorting needed
- Correct overlap for free
- Works with any render order

---

## The Entity System

### Unified Character Design

Both players and NPCs use the same `Character` class:

```
Character
├── Position: x, y, z (floats)
├── Velocity: velocity_x, velocity_y, velocity_z
├── Collision: width, depth, height (in tile units)
├── Animation: AnimatedSprite reference
├── AI: behavior, target, patrol_points (NPC only)
└── State: is_npc, direction, walking
```

**Why unified?**
- Same movement physics
- Same collision detection
- Same animation system
- Simpler interaction code

### Entity Manager Pattern

```
EntityManager
├── characters: List[Character]   # All entities
├── player: Character             # Quick reference
├── _sprite_cache: Dict           # Shared sprite data
│
├── Factory Methods:
│   ├── create_character()
│   ├── create_npc()
│   ├── create_npc_wanderer()
│   ├── create_npc_patrol()
│   └── create_npc_follower()
│
└── Runtime Methods:
    ├── update(dt)                # Update all entities
    ├── collect_render_data()     # Prepare for rendering
    └── get_characters_at(x, y)   # Spatial query
```

### NPC Behavior System

Simple state machine with behavior types:

```python
class NPCBehavior(Enum):
    IDLE = "idle"      # Stand still
    WANDER = "wander"  # Random movement in area
    PATROL = "patrol"  # Walk between waypoints
    FOLLOW = "follow"  # Chase target character
```

Each behavior method sets velocity, main `update()` applies it:

```python
def _update_npc_behavior(self, dt):
    if self.behavior == NPCBehavior.WANDER:
        self._behavior_wander(dt)  # Sets velocity_x, velocity_y
    # ... etc

def update(self, dt):
    if self.is_npc:
        self._update_npc_behavior(dt)
    
    # Apply velocity (same for player and NPC)
    new_x = self.x + self.velocity_x * dt
    # ... collision checks and position update
```

---

## The Sprite System

### Spritesheet Layout

Standard 4×4 character spritesheet:

```
     Col 0    Col 1    Col 2    Col 3
    ┌────────┬────────┬────────┬────────┐
Row │  Down  │  Down  │  Down  │  Down  │
 0  │  Idle  │ Walk 1 │  Idle  │ Walk 2 │
    ├────────┼────────┼────────┼────────┤
Row │  Left  │  Left  │  Left  │  Left  │
 1  │  Idle  │ Walk 1 │  Idle  │ Walk 2 │
    ├────────┼────────┼────────┼────────┤
Row │ Right  │ Right  │ Right  │ Right  │
 2  │  Idle  │ Walk 1 │  Idle  │ Walk 2 │
    ├────────┼────────┼────────┼────────┤
Row │   Up   │   Up   │   Up   │   Up   │
 3  │  Idle  │ Walk 1 │  Idle  │ Walk 2 │
    └────────┴────────┴────────┴────────┘
```

### Frame Pre-Cutting

Frames are cut at load time, not runtime:

```python
# At load time (once):
for direction in [DOWN, LEFT, RIGHT, UP]:
    for frame in range(4):
        image = spritesheet.crop(frame_rect)
        self.frames[direction].append(image)

# At runtime (every frame):
current_frame = self.frames[direction][frame_index]  # O(1) lookup
```

**Why pre-cut?**
- `crop()` allocates new image = slow
- Dictionary lookup = fast
- No garbage collection pressure during gameplay

### Sprite Sharing (Memory Optimization)

Multiple characters can share the same sprite frames:

```python
# Load template once
npc_template = AnimatedSprite("villager.png")

# Create 100 NPCs sharing frame data
for i in range(100):
    npc_sprite = AnimatedSprite.from_cached(npc_template)
    # npc_sprite.frames points to SAME images as template
    # Only animation STATE is independent
```

**Memory comparison:**
```
Without sharing: 100 NPCs × 16 frames × 64×64×4 bytes = 26 MB
With sharing:    1 template + 100 tiny state objects = 0.3 MB
```

### Time-Based Animation

Animation uses delta time, not frame counting:

```python
def update(self, dt):
    self.animation_timer += dt * self.animation_speed
    
    if self.animation_timer >= 1.0:
        frames_to_advance = int(self.animation_timer)
        self.animation_timer -= frames_to_advance
        self.current_frame += frames_to_advance
```

**Why time-based?**
- Same animation speed at 30 FPS or 144 FPS
- Smooth timing regardless of frame drops
- Predictable behavior

---

## The Camera System

### Camera Properties

```
Camera
├── x, y: float          # World position (top-left of view)
├── zoom: float          # Scale factor (1.0 = normal)
├── width, height: int   # Viewport size (pixels)
```

### Coordinate Transformation

```
World → Screen:
  screen_x = (world_x - camera_x) × zoom
  screen_y = (world_y - camera_y) × zoom

Screen → World:
  world_x = camera_x + screen_x / zoom
  world_y = camera_y + screen_y / zoom
```

### Zoom-Compensated Panning

Mouse drag distance must be adjusted for zoom:

```python
def move(self, dx, dy):
    # dx, dy are screen pixels
    self.x += dx / self.zoom
    self.y += dy / self.zoom
```

**Why divide by zoom?**
- At zoom 2.0: Drag 100px → should move 50 world units
- At zoom 0.5: Drag 100px → should move 200 world units
- Keeps "feel" consistent at any zoom level

### Center-Point Zooming

Zoom maintains the center point:

```python
def set_zoom(self, new_zoom):
    # 1. Find current center in world coords
    center_x = self.x + self.width / (2 * self.zoom)
    center_y = self.y + self.height / (2 * self.zoom)
    
    # 2. Apply new zoom
    self.zoom = clamp(new_zoom, MIN_ZOOM, MAX_ZOOM)
    
    # 3. Reposition so center stays fixed
    self.x = center_x - self.width / (2 * self.zoom)
    self.y = center_y - self.height / (2 * self.zoom)
```

### Visible Bounds (Culling)

```python
def get_visible_bounds(self):
    left = self.x
    top = self.y
    right = self.x + self.width / self.zoom
    bottom = self.y + self.height / self.zoom
    return left, top, right, bottom
```

**Used for frustum culling:**
- Only render tiles within visible bounds
- 1000×1000 map but only ~50×40 tiles visible
- Massive performance improvement

---

## Performance Optimizations

### Summary Table

| Optimization | Technique | Impact |
|--------------|-----------|--------|
| Draw Call Reduction | Sprite Batching | 100× fewer draw calls |
| Memory Efficiency | Sprite Sharing | 99% less sprite memory |
| Render Culling | Visible Bounds | Only render visible tiles |
| Collision Speed | NumPy Arrays | O(1) tile lookup |
| Texture Management | Pre-loading + Caching | No runtime loading |
| Animation Efficiency | Pre-cut Frames | No per-frame allocation |
| Depth Sorting | GPU Depth Buffer | No CPU sorting |
| Tile Bleeding | 1px Border Extrusion | Correct rendering |

### Detailed Breakdown

#### 1. Sprite Batching

```
Before: 10,000 tiles × 1 draw call = 10,000 draw calls
After:  10,000 tiles ÷ ~1000 per batch × 10 textures = ~100 draw calls

Improvement: 100× reduction
```

#### 2. Texture Caching

```python
# TilesetRenderer caches all tile textures
tile_texture_cache: Dict[int, Texture]  # GID → Texture

# EntityManager caches sprite templates
_sprite_cache: Dict[str, AnimatedSprite]  # path → template
```

#### 3. Pre-allocated Buffers

```python
# SpriteBatch pre-allocates for max sprites
vertices = np.zeros(max_sprites × 4 × 9, dtype=np.float32)
indices = pre_generate_indices(max_sprites)  # Never changes

# glBufferSubData updates existing buffer (no reallocation)
```

#### 4. Lazy Initialization

```python
# Character textures created on first render
def get_texture(self, renderer):
    if not self._textures_initialized:
        self._init_textures(renderer)  # Deferred
    return self._texture_cache[key]
```

#### 5. Squared Distance Comparison

```python
# Avoid expensive sqrt() for distance checks
def get_characters_at(x, y, radius):
    radius_squared = radius * radius
    for char in characters:
        dx = char.x - x
        dy = char.y - y
        if dx*dx + dy*dy <= radius_squared:  # No sqrt!
            yield char
```

---

## Memory Management

### Memory Layout Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU MEMORY                               │
├─────────────────────────────────────────────────────────────┤
│  Tile Textures          │  ~5-20 MB (depends on tilesets)   │
│  Character Textures     │  ~1-5 MB (shared sprites)         │
│  Vertex Buffers (VBO)   │  ~3 MB (pre-allocated)            │
│  Index Buffers (EBO)    │  ~0.5 MB (static)                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    CPU MEMORY                               │
├─────────────────────────────────────────────────────────────┤
│  Map3D.mapa (NumPy)     │  H×D×W×N×2 bytes                  │
│  CollisionMap (NumPy)   │  H×D×W×2 bytes                    │
│  Vertex Array (NumPy)   │  max_sprites×144 bytes            │
│  Entity State           │  ~1 KB per character              │
│  Sprite Frame Cache     │  Shared references (minimal)      │
└─────────────────────────────────────────────────────────────┘
```

### Resource Lifecycle

```
LOAD PHASE:
  1. Parse TMX → Python objects (temporary)
  2. Build NumPy arrays → CPU memory (persistent)
  3. Upload textures → GPU memory (persistent)
  4. Discard parsed TMX → garbage collected

RUNTIME:
  - NumPy arrays: Read-only (map data)
  - GPU buffers: Updated each frame (vertex data)
  - Textures: Static after upload
  
CLEANUP:
  - Texture.__del__() calls glDeleteTextures()
  - NumPy arrays freed by Python GC
```

---

## Frame Update Cycle

### Complete Frame Timeline

```
┌──────────────────────────────────────────────────────────────┐
│  FRAME N                                           ~16.67ms  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐                                         │
│  │  1. INPUT       │  Poll keyboard/mouse                    │
│  │     ~0.1ms      │  Set player velocity                    │
│  └────────┬────────┘                                         │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │  2. UPDATE      │  EntityManager.update(dt)               │
│  │     ~1-2ms      │  ├─ NPC AI decisions                    │
│  │                 │  ├─ Apply velocities                    │
│  │                 │  ├─ Collision detection                 │
│  │                 │  └─ Animation updates                   │
│  └────────┬────────┘                                         │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │  3. RENDER      │  Prepare render data                    │
│  │     ~2-5ms      │  ├─ Calculate visible bounds            │
│  │                 │  ├─ Batch tiles by texture              │
│  │                 │  ├─ Collect entity data                 │
│  │                 │  └─ Sort by depth                       │
│  └────────┬────────┘                                         │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │  4. DRAW        │  GPU commands                           │
│  │     ~1-3ms      │  ├─ glClear()                           │
│  │                 │  ├─ SpriteBatch.flush() × N             │
│  │                 │  └─ Text overlay                        │
│  └────────┬────────┘                                         │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │  5. PRESENT     │  Swap buffers                           │
│  │     ~0-8ms      │  (may wait for vsync)                   │
│  └─────────────────┘                                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Performance Targets

| Component | Target | Notes |
|-----------|--------|-------|
| Input | < 0.5ms | Minimal processing |
| Update | < 3ms | 100 entities |
| Render Prep | < 3ms | Batching, sorting |
| Draw Calls | < 5ms | GPU-bound |
| **Total** | **< 16.67ms** | **60 FPS** |

---

## Appendix: Key Data Structures

### NumPy Array Choices

| Array | dtype | Why |
|-------|-------|-----|
| mapa (tiles) | uint16 | GIDs 0-65535, 2 bytes each |
| collision | uint16 | 16 flag bits, 2 bytes each |
| vertices | float32 | GPU requires 32-bit floats |
| indices | uint32 | >65535 vertices possible |

### Dictionary Caches

| Cache | Key | Value | Purpose |
|-------|-----|-------|---------|
| tile_texture_cache | GID (int) | Texture | Tile rendering |
| _sprite_cache | "path_w_h" | AnimatedSprite | Sprite sharing |
| _texture_cache | (Direction, frame) | Texture | Character frames |
| frames | Direction | List[Image] | Animation frames |

---

## Conclusion

This architecture achieves high performance through:

1. **Data-Oriented Design**: NumPy arrays for bulk data
2. **Batched Rendering**: Minimize GPU state changes
3. **Resource Sharing**: Avoid duplicate memory
4. **Lazy Initialization**: Defer expensive operations
5. **Spatial Optimization**: Only process visible content

The result is a tile engine capable of rendering large maps (1000×1000+) with hundreds of animated entities at 60+ FPS.
