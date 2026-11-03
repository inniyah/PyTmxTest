"""Compatibilidad: ``Map3DStructure`` es ahora ``World`` (ver world.py).

La estructura antigua ``mapa[z, y, x, capa]`` se sustituyó por el mundo integrado
``tiles[z, y, x, hueco]``; ``get_tile`` recibe ahora un hueco en lugar de una capa.
"""

from .world import EMPTY_SOURCE, SLOTS, LayerInfo, LayerRole, World

Map3DStructure = World

__all__ = ["World", "Map3DStructure", "LayerRole", "LayerInfo", "SLOTS", "EMPTY_SOURCE"]
