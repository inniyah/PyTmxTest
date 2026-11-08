"""TMX Explorer: visor OpenGL de mapas Tiled con niveles de altura y personajes."""

__version__ = "3.5.2"

# Los módulos que dependen de OpenGL/GLFW se importan bajo demanda para que
# ``tmx_explorer.map`` o ``tmx_explorer.entities`` se puedan usar (y testear) sin GPU.
_LAZY = {
    "TMXExplorer": ".app",
    "Camera": ".camera",
    "Map3DStructure": ".map.structure",
    "TilesetRenderer": ".map.tileset_renderer",
    "OpenGLRenderer": ".renderer.opengl_renderer",
    "Texture": ".renderer.texture",
    "SpriteBatch": ".renderer.sprite_batch",
    "AnimatedSprite": ".entities.sprite",
    "Direction": ".entities.sprite",
    "AnimationState": ".entities.sprite",
    "Character": ".entities.character",
    "EntityManager": ".entities.manager",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
