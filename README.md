# 🎮 TMX Game Engine

A high-performance 2D tile-based game engine with height levels (pseudo-3D), built with Python and OpenGL.

## ✨ Features

### 🗺️ Map System
- **TMX Support** - Full read/write support for Tiled Map Editor files
- **Height Levels** - Pseudo-3D with multiple Z layers (floors, bridges, underground)
- **4D Map Structure** - Efficient NumPy arrays `[Z, Y, X, Layer]`
- **All Encodings** - CSV, Base64, zlib, gzip, zstd compression

### 🎨 Rendering
- **Batched Rendering** - 100x fewer draw calls via sprite batching
- **Depth Buffer** - Automatic depth sorting without CPU overhead
- **Tile Bleeding Fix** - 1px border extrusion for perfect tile rendering
- **Camera System** - Smooth pan, zoom, and coordinate conversion

### 🏃 Entities & Animation
- **Unified Character System** - Players and NPCs share the same codebase
- **NPC Behaviors** - Idle, Wander, Patrol, Follow AI patterns
- **Sprite Sharing** - 99% memory reduction for duplicate sprites
- **Time-Based Animation** - Consistent speed regardless of framerate

### 💥 Collision
- **Tile-Based Collision** - O(1) lookup with NumPy arrays
- **3D Collision Boxes** - Width, depth, and height in tile units
- **Wall Sliding** - Separate axis collision for smooth movement
- **Flag System** - Extensible for water, ladders, damage zones

### 🎮 Input
- **Keyboard & Mouse** - Full support with GLFW
- **Gamepad Support** - Xbox-style layout with SDL_GameControllerDB
- **Deadzone Handling** - Proper analog stick processing

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- OpenGL 3.3+ compatible graphics

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
PyOpenGL>=3.1.6
glfw>=2.5.0
numpy>=1.21.0
Pillow>=9.0.0
```

### Optional Dependencies

```bash
# For zstd compression support in TMX files
pip install zstandard

# For gamepad support (download SDL_GameControllerDB)
curl -o gamecontrollerdb.txt https://raw.githubusercontent.com/gabomdq/SDL_GameControllerDB/master/gamecontrollerdb.txt
```

---

## 🚀 Quick Start

### Basic Example

```python
from tmx_manager import TiledMap
from engine import GameEngine

# Load a map created in Tiled
game_map = TiledMap.load("maps/level1.tmx")

# Create game engine
engine = GameEngine(width=1280, height=720)
engine.load_map(game_map)

# Create player
player = engine.entities.create_character(
    "sprites/hero.png",
    x=400, y=300, z=0,
    is_player=True
)

# Run game loop
engine.run()
```

### Creating NPCs

```python
# Wandering villager
villager = engine.entities.create_npc_wanderer(
    "sprites/villager.png",
    x=500, y=400,
    radius=200,  # Wander radius in pixels
    speed=60
)

# Patrolling guard
guard = engine.entities.create_npc_patrol(
    "sprites/guard.png",
    points=[(100, 100), (300, 100), (300, 300), (100, 300)],
    speed=70
)

# Companion that follows player
pet = engine.entities.create_npc_follower(
    "sprites/dog.png",
    target=player,
    speed=90
)
```

---

## 📁 Project Structure

```
tmx-game-engine/
├── 📄 tmx_manager.py        # TMX file parser (read/write)
├── 📁 engine/
│   ├── 📁 map/
│   │   ├── structure.py     # Map3DStructure (4D NumPy array)
│   │   └── collision.py     # CollisionMap (3D collision grid)
│   ├── 📁 renderer/
│   │   ├── opengl_renderer.py  # Main OpenGL renderer
│   │   ├── sprite_batch.py     # Batched sprite rendering
│   │   ├── texture.py          # Texture management
│   │   └── tileset_renderer.py # Tileset loading
│   ├── 📁 entities/
│   │   ├── manager.py       # EntityManager (factory + updates)
│   │   ├── character.py     # Character class (player/NPC)
│   │   └── sprite.py        # AnimatedSprite system
│   ├── 📁 input/
│   │   └── gamepad.py       # Gamepad/joystick support
│   └── camera.py            # 2D camera with zoom
├── 📁 assets/
│   ├── 📁 maps/             # TMX map files
│   ├── 📁 tilesets/         # Tileset images + TSX files
│   └── 📁 sprites/          # Character spritesheets
├── 📁 docs/
│   ├── architecture.md      # Technical architecture guide
│   ├── tmx-guide.md         # TMX Manager user guide
│   └── api-reference.md     # API documentation
├── 📄 requirements.txt
├── 📄 main.py               # Example game
└── 📄 README.md
```

---

## 🗺️ Working with Tiled Maps

### Setting Up Layers in Tiled

1. **Create tile layers** for terrain, walls, decorations
2. **Add Z property** to layers for height levels:
   - Select layer → Add Property → Name: `Z`, Type: `int`
   - Ground = 0, Bridge = 1, Underground = -1

3. **Mark solid tiles** in tileset:
   - Open tileset editor
   - Select tile → Add Property → Name: `solid`, Type: `bool`, Value: `true`

4. **Add spawn points** with object layer:
   - Create object layer "Spawns"
   - Add point objects with `type` = "player_spawn", "npc_spawn", etc.

### Loading Maps in Code

```python
from tmx_manager import TiledMap
from engine.map import Map3DStructure

# Load TMX
tmx_map = TiledMap.load("maps/dungeon.tmx")

# Create 3D structure (extracts Z levels, builds collision)
map_3d = Map3DStructure(tmx_map)

print(f"Dimensions: {map_3d.W}x{map_3d.D}x{map_3d.H}")
print(f"Layers: {map_3d.N}")
```

---

## 🎨 Sprite Requirements

### Character Spritesheets

Standard 4×4 layout (RPG Maker compatible):

```
     Col 0    Col 1    Col 2    Col 3
    ┌────────┬────────┬────────┬────────┐
    │  Down  │  Down  │  Down  │  Down  │  Row 0
    │  Idle  │ Walk 1 │  Idle  │ Walk 2 │
    ├────────┼────────┼────────┼────────┤
    │  Left  │  Left  │  Left  │  Left  │  Row 1
    ├────────┼────────┼────────┼────────┤
    │ Right  │ Right  │ Right  │ Right  │  Row 2
    ├────────┼────────┼────────┼────────┤
    │   Up   │   Up   │   Up   │   Up   │  Row 3
    └────────┴────────┴────────┴────────┘
```

### Tileset Requirements

- **Format:** PNG with transparency
- **Tile size:** Consistent (16×16, 32×32, etc.)
- **Spacing/Margin:** Supported if configured in Tiled

---

## ⚡ Performance

### Benchmarks

| Scenario | Without Optimization | With Optimization |
|----------|---------------------|-------------------|
| 10,000 tiles | ~5 FPS | 60+ FPS |
| 100 NPCs | ~15 FPS | 60+ FPS |
| Sprite memory (100 chars) | 26 MB | 0.3 MB |

### Key Optimizations

| Technique | Impact |
|-----------|--------|
| Sprite Batching | 100× fewer draw calls |
| Texture Caching | Zero runtime texture loading |
| Sprite Sharing | 99% memory reduction |
| NumPy Collision | O(1) tile lookup |
| Depth Buffer | No CPU sorting needed |
| Frustum Culling | Only render visible tiles |

---

## 🎮 Controls

### Keyboard

| Key | Action |
|-----|--------|
| WASD / Arrows | Move |
| Q / E | Change height level |
| Mouse Drag | Pan camera |
| Scroll Wheel | Zoom |
| R | Reset camera |
| ESC | Quit |

### Gamepad

| Input | Action |
|-------|--------|
| Left Stick / D-Pad | Move |
| LB / RB | Change height level |
| A | Interact |
| Start | Pause |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture Guide](architecture.md) | Technical deep-dive into engine design |
| [TMX Manager Guide](tmx_manager.md) | Complete guide to TMX file handling |

---

## 🛠️ API Overview

### TMX Manager

```python
from tmx_manager import TiledMap, create_empty_map, create_layer

# Load
map_data = TiledMap.load("level.tmx")

# Access
layer = map_data.get_layer_by_name("Ground")
gid = layer.get_tile_gid(x, y)

# Modify
layer.set_tile_gid(x, y, new_gid)

# Save
map_data.save("modified.tmx", encoding="csv")
```

### Entity Manager

```python
from engine.entities import EntityManager

entities = EntityManager(tile_width=32, tile_height=32, collision_map=collision)

# Create characters
player = entities.create_character("hero.png", x=100, y=100, is_player=True)
npc = entities.create_npc_wanderer("villager.png", x=200, y=200, radius=150)

# Game loop
entities.update(dt)
render_data = entities.collect_render_data(renderer, level_offset)
```

### Camera

```python
from engine import Camera

camera = Camera(width=1280, height=720)
camera.move(dx, dy)           # Pan
camera.zoom_by(1.1)           # Zoom in
camera.screen_to_world(mx, my) # Mouse picking
```

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/yourusername/tmx-game-engine.git
cd tmx-game-engine
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python main.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Tiled Map Editor](https://www.mapeditor.org/) - Excellent map editor
- [SDL_GameControllerDB](https://github.com/gabomdq/SDL_GameControllerDB) - Gamepad mappings
- [OpenGameArt](https://opengameart.org/) - Free game assets
- [GLFW](https://www.glfw.org/) - Window/input handling

---

## 📬 Contact

- **Issues:** [GitHub Issues](https://github.com/inniyah/PyTmxTest/issues)
- **Discussions:** [GitHub Discussions](https://github.com/inniyah/PyTmxTest/discussions)

---

<p align="center">
  Made with ❤️ and 🐍
</p>
