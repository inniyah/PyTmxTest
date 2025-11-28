"""
TMX Map Explorer - GLFW/OpenGL version

Requisitos:
    pip install glfw PyOpenGL PyOpenGL_accelerate pillow numpy
"""

from .app import TMXExplorer
from .camera import Camera
from .map.structure import Map3DStructure
from .map.tileset_renderer import TilesetRenderer
from .renderer.opengl_renderer import OpenGLRenderer
from .renderer.texture import Texture
from .renderer.sprite_batch import SpriteBatch
from .entities import (
    AnimatedSprite, Direction, AnimationState,
    Character, EntityManager
)

__version__ = "2.0.0"
__all__ = [
    "TMXExplorer",
    "Camera", 
    "Map3DStructure",
    "TilesetRenderer",
    "OpenGLRenderer",
    "Texture",
    "SpriteBatch",
    "AnimatedSprite",
    "Direction",
    "AnimationState",
    "Character",
    "EntityManager",
]
