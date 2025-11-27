# ============================================
# tmx_explorer/__init__.py
# ============================================
"""
TMX Map Explorer - OpenGL-based TMX map viewer

Optimized features:
- 2D culling (X/Y only, all Z heights visible)
- Layer-by-layer rendering: Y (far→near), X (left→right), Z (bottom→top), N (0→max)
- Tile bleeding prevention with NEAREST filtering and edge extrusion
"""

from .app import TMXExplorer
from .camera import Camera
from .map.structure import Map3DStructure
from .map.tileset_renderer import TilesetRenderer
from .renderer.opengl_renderer import OpenGLRenderer
from .renderer.texture import Texture
from .renderer.sprite_batch import SpriteBatch

__version__ = "1.0.0"
__all__ = [
    "TMXExplorer",
    "Camera", 
    "Map3DStructure",
    "TilesetRenderer",
    "OpenGLRenderer",
    "Texture",
    "SpriteBatch",
]
