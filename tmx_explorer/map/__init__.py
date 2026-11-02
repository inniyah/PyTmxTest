"""Estructura del mapa, colisión y carga de tilesets."""

from .collision import CollisionMap
from .structure import Map3DStructure

__all__ = ["Map3DStructure", "CollisionMap", "TilesetRenderer"]


def __getattr__(name):
    # TilesetRenderer necesita OpenGL al usarse; se importa bajo demanda
    if name == "TilesetRenderer":
        from .tileset_renderer import TilesetRenderer
        return TilesetRenderer
    raise AttributeError(name)
