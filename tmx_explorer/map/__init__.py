"""Mundo 3D, celdas de colisión y carga de tilesets."""

from .collision import CollisionMap
from .world import LayerRole, World

Map3DStructure = World

__all__ = ["World", "Map3DStructure", "LayerRole", "CollisionMap", "TilesetRenderer"]


def __getattr__(name):
    # TilesetRenderer necesita OpenGL al usarse; se importa bajo demanda
    if name == "TilesetRenderer":
        from .tileset_renderer import TilesetRenderer
        return TilesetRenderer
    raise AttributeError(name)
