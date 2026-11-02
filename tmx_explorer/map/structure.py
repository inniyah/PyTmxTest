"""Mapa TMX convertido a una estructura 3D con niveles de altura.

Cada capa de tiles se asigna a un nivel con la propiedad custom ``Z`` (o ``z``,
``level``) en Tiled. Ejes: X = ancho (W), Y = filas/fondo (D), Z = altura (H).

Datos:
    mapa[z, y, x, n]   GID (sin bits de flip) de la capa n en el nivel z
    flips[z, y, x, n]  bits de flip de Tiled (>> 28): H=8, V=4, D=2, hex=1
    collision          CollisionMap construido a partir de la propiedad ``solid``
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from tmx_manager import LayerGroup, TiledMap, TileLayer

from ..config import DEFAULT_COLLISION_SIZE, GID_MASK
from .collision import FLAG_SOLID, CollisionMap

log = logging.getLogger(__name__)

LEVEL_PROPERTY_NAMES = ("Z", "z", "level")


class Map3DStructure:
    """Capas TMX organizadas por nivel de altura, más su mapa de colisión."""

    def __init__(self, tmx_map: TiledMap):
        self.map_width = tmx_map.width
        self.map_height = tmx_map.height
        self.tile_width = tmx_map.tilewidth
        self.tile_height = tmx_map.tileheight
        self.W = self.map_width
        self.D = self.map_height

        # (capa, nivel, nombre completo "grupo/capa")
        self.layer_info: List[Tuple[TileLayer, int, str]] = self._extract_tile_layers(tmx_map)

        levels = sorted({level for _, level, _ in self.layer_info}) or [0]
        self.min_level, self.max_level = levels[0], levels[-1]
        self.H = self.max_level - self.min_level + 1
        self.level_offset = -self.min_level     # índice z = nivel + level_offset
        self.N = len(self.layer_info)

        self.layer_names: List[str] = [name for _, _, name in self.layer_info]
        self.layer_levels: List[int] = [level for _, level, _ in self.layer_info]

        self.mapa = np.zeros((self.H, self.D, self.W, self.N), dtype=np.uint32)
        self.flips = np.zeros((self.H, self.D, self.W, self.N), dtype=np.uint8)
        self._load_layers()

        self.collision = CollisionMap(self.W, self.H, self.D)
        self._build_collision_map(tmx_map.tilesets)

        log.info("Map3D: W=%d D=%d H=%d N=%d", self.W, self.D, self.H, self.N)

    # --- Construcción ---

    def _extract_tile_layers(self, tmx_map: TiledMap) -> List[Tuple[TileLayer, int, str]]:
        """Aplana grupos y devuelve las capas de tiles con su nivel."""
        result = []

        def visit(layer, path: str):
            if isinstance(layer, TileLayer):
                result.append((layer, self._get_layer_level(layer), path))
            elif isinstance(layer, LayerGroup):
                for sub in layer.layers:
                    visit(sub, f"{path}/{sub.name}")

        for layer in tmx_map.layers:
            visit(layer, layer.name)
        return result

    @staticmethod
    def _get_layer_level(layer: TileLayer) -> int:
        for prop_name in LEVEL_PROPERTY_NAMES:
            if prop_name in layer.properties:
                value = layer.properties[prop_name].value
                return int(value, 0) if isinstance(value, str) else int(value)
        return 0

    def _load_layers(self):
        for n, (layer, level, name) in enumerate(self.layer_info):
            raw = np.frombuffer(layer.data.tiles, dtype=np.uint32)
            if raw.size != layer.width * layer.height:
                log.warning("Capa '%s': %d tiles, se esperaban %d; se ignora",
                            name, raw.size, layer.width * layer.height)
                continue
            grid = raw.reshape(layer.height, layer.width)[:self.D, :self.W]
            z = level + self.level_offset
            h, w = grid.shape
            self.mapa[z, :h, :w, n] = grid & GID_MASK
            self.flips[z, :h, :w, n] = (grid >> 28).astype(np.uint8)

    def _build_collision_map(self, tilesets):
        """Marca como sólida cada celda donde alguna capa de su nivel tenga un tile 'solid'."""
        solid_gids = []
        for tileset in tilesets:
            for tile_id, tile in tileset.tiles.items():
                prop = tile.properties.get("solid")
                if prop is not None and _as_bool(prop.value):
                    solid_gids.append(tileset.firstgid + tile_id)

        if not solid_gids:
            return

        max_gid = max(int(self.mapa.max()), max(solid_gids))
        is_solid_gid = np.zeros(max_gid + 1, dtype=bool)
        is_solid_gid[solid_gids] = True

        layer_levels = np.array(self.layer_levels)
        for z in range(self.H):
            layers_here = np.flatnonzero(layer_levels == self.get_level_value(z))
            if layers_here.size == 0:
                continue
            solid = is_solid_gid[self.mapa[z][:, :, layers_here]].any(axis=-1)
            self.collision.data[z][solid] = FLAG_SOLID

        stats = self.collision.get_stats()
        log.info("Colisión: %d GIDs sólidos, %d celdas sólidas (%.1f%%)",
                 len(solid_gids), stats["solid_tiles"], stats["solid_percent"])

    # --- Consultas ---

    def get_level_value(self, z: int) -> int:
        """Índice z del array -> valor de nivel en Tiled (puede ser negativo)."""
        return z - self.level_offset

    def _in_bounds(self, x, y, z, layer) -> bool:
        return 0 <= x < self.W and 0 <= y < self.D and 0 <= z < self.H and 0 <= layer < self.N

    def get_tile(self, x: int, y: int, z: int, layer: int) -> int:
        return int(self.mapa[z, y, x, layer]) if self._in_bounds(x, y, z, layer) else 0

    def set_tile(self, x: int, y: int, z: int, layer: int, gid: int):
        if self._in_bounds(x, y, z, layer):
            self.mapa[z, y, x, layer] = gid & GID_MASK
            self.flips[z, y, x, layer] = (gid >> 28) & 0xF

    def is_walkable(self, px: float, py: float, z: float) -> bool:
        """¿Es transitable el punto (px, py) en píxeles al nivel z?"""
        tx, ty = self.collision.pixel_to_tile(px, py, self.tile_width, self.tile_height)
        return self.collision.is_walkable(tx, ty, int(z))

    def is_walkable_with_size(self, px: float, py: float, z: float,
                              char_width: float = DEFAULT_COLLISION_SIZE[0],
                              char_depth: float = DEFAULT_COLLISION_SIZE[1],
                              char_height: float = DEFAULT_COLLISION_SIZE[2]) -> bool:
        """¿Cabe una caja de ese tamaño (en tiles) centrada en (px, py)?"""
        return self.collision.can_move_to_with_size(
            px, py, z, char_width, char_depth, char_height,
            self.tile_width, self.tile_height)


def _as_bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"
