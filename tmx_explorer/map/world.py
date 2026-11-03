"""Mundo 3D de celdas construido a partir de las capas de Tiled.

Las capas de Tiled son solo la forma de editar. Al cargar, todas las capas de un
nivel se integran en las celdas de ese nivel:

    tiles[z, y, x, k]   uint32  GID de Tiled (con sus bits de flip) del hueco k
    source[z, y, x, k]  uint8   índice de la capa de origen (EMPTY_SOURCE si vacío)
    roles[z, y, x]      uint8   rol de cada hueco, 2 bits por hueco (ver role_of)
    cells[z, y, x]      uint16  forma de colisión + flags (formato de cells.py)

Cada celda tiene SLOTS huecos apilados de abajo arriba, en el orden de las capas
de Tiled. Los tiles vacíos no ocupan hueco. Si una celda recibe más de SLOTS
tiles, se conservan los de más abajo y se avisa.

Nivel de una capa: propiedad ``Z`` (o ``z``, ``level``) de la capa o del grupo más
cercano que la tenga. Rol: clase de la capa en Tiled (``floor``, ``object``,
``overhead``, ``roof``) o propiedad ``role``, buscando también en sus grupos;
por defecto ``object``. Un tile puede imponer su rol con su clase en el tileset.
Las capas ocultas en Tiled no se cargan; la opacidad se combina con la de sus grupos.
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import numpy as np

from tmx_manager import LayerGroup, TiledMap, TileLayer

from ..config import DEFAULT_COLLISION_SIZE, GID_MASK, LEVEL_HEIGHT_OFFSET
from . import cells as cellfmt
from .collision import CollisionMap

log = logging.getLogger(__name__)

SLOTS = 4
EMPTY_SOURCE = 255
LEVEL_PROPERTY_NAMES = ("Z", "z", "level")
NO_ROLE = 255


class LayerRole(IntEnum):
    FLOOR = 0       # plano, bajo los personajes de su nivel
    OBJECT = 1      # ordenado por fila junto con los personajes
    OVERHEAD = 2    # siempre por encima de los personajes de su nivel
    ROOF = 3        # como OVERHEAD; se ocultará cuando el jugador esté debajo

    @classmethod
    def parse(cls, name) -> Optional["LayerRole"]:
        try:
            return cls[str(name).strip().upper()] if name else None
        except KeyError:
            return None


@dataclass
class LayerInfo:
    index: int          # valor en world.source
    name: str           # "grupo/capa"
    level: int          # valor de nivel en Tiled (puede ser negativo)
    z: int              # índice de nivel en el mundo
    role: LayerRole
    opacity: float


def _prop_value(layer, names):
    for name in names:
        if name in layer.properties:
            return layer.properties[name].value
    return None


class World:
    """Mundo integrado. Ver el docstring del módulo."""

    def __init__(self, tmx_map: TiledMap):
        self.tile_width = tmx_map.tilewidth
        self.tile_height = tmx_map.tileheight
        level_height = tmx_map.properties.get("level_height")
        self.level_height = int(level_height.value) if level_height else LEVEL_HEIGHT_OFFSET

        entries = self._collect_layers(tmx_map)       # (TileLayer, nivel, rol, opacidad, nombre)
        if len(entries) >= EMPTY_SOURCE:
            raise ValueError(f"Demasiadas capas de tiles ({len(entries)}); máximo {EMPTY_SOURCE - 1}")

        levels = sorted({e[1] for e in entries}) or [0]
        self.min_level, self.max_level = levels[0], levels[-1]
        self.H = self.max_level - self.min_level + 1
        self.level_offset = -self.min_level

        # Extensión en tiles (en mapas infinitos, la caja de todos los chunks)
        if entries and any(e[0].is_infinite for e in entries):
            boxes = [e[0].bounds() for e in entries]
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        else:
            x0, y0, x1, y1 = 0, 0, tmx_map.width, tmx_map.height
        self.origin_x, self.origin_y = x0, y0           # tile de Tiled que es (0, 0) aquí
        self.W, self.D = x1 - x0, y1 - y0

        self.layers: List[LayerInfo] = [
            LayerInfo(i, name, level, level + self.level_offset, role, opacity)
            for i, (_, level, role, opacity, name) in enumerate(entries)]

        shape4 = (self.H, self.D, self.W, SLOTS)
        self.tiles = np.zeros(shape4, dtype=np.uint32)
        self.source = np.full(shape4, EMPTY_SOURCE, dtype=np.uint8)
        self.roles = np.zeros((self.H, self.D, self.W), dtype=np.uint8)
        self.cells = np.zeros((self.H, self.D, self.W), dtype=np.uint16)
        self.dropped_tiles = 0
        self.slots_used = np.zeros(self.H, dtype=np.int32)   # huecos ocupados por nivel

        tile_roles = self._tile_role_table(tmx_map.tilesets)
        for z in range(self.H):
            self._integrate_level(z, [e[0] for e in entries], tile_roles)

        self._build_cells(tmx_map.tilesets)
        self.collision = CollisionMap(self.W, self.H, self.D, data=self.cells)

        log.info("Mundo: W=%d D=%d H=%d, %d capas de tiles, origen (%d, %d)",
                 self.W, self.D, self.H, len(self.layers), self.origin_x, self.origin_y)

    # --- Construcción ---

    def _collect_layers(self, tmx_map: TiledMap):
        """Capas de tiles visibles con nivel, rol y opacidad heredados de sus grupos."""
        out = []

        def visit(layer, path, level, role, opacity, visible):
            visible = visible and layer.visible
            opacity = opacity * layer.opacity
            own_level = _prop_value(layer, LEVEL_PROPERTY_NAMES)
            if own_level is not None:
                level = int(own_level, 0) if isinstance(own_level, str) else int(own_level)
            # Ojo: FLOOR vale 0, así que nada de "a or b" con roles
            for candidate in (getattr(layer, "klass", ""), _prop_value(layer, ("role",))):
                parsed = LayerRole.parse(candidate)
                if parsed is not None:
                    role = parsed
                    break
            name = f"{path}/{layer.name}" if path else layer.name

            if isinstance(layer, LayerGroup):
                for child in layer.layers:
                    visit(child, name, level, role, opacity, visible)
            elif isinstance(layer, TileLayer):
                if not visible:
                    log.info("Capa '%s' oculta en Tiled: no se carga", name)
                    return
                out.append((layer, level, LayerRole.OBJECT if role is None else role,
                            opacity, name))

        for layer in tmx_map.layers:
            visit(layer, "", 0, None, 1.0, True)
        return out

    @staticmethod
    def _tile_role_table(tilesets) -> np.ndarray:
        """GID -> rol impuesto por la clase del tile en el tileset (NO_ROLE si no impone)."""
        max_gid = max((ts.firstgid + max(ts.tilecount, max(ts.tiles, default=0) + 1)
                       for ts in tilesets), default=1)
        table = np.full(max_gid + 1, NO_ROLE, dtype=np.uint8)
        for ts in tilesets:
            for tile_id, tile in ts.tiles.items():
                role = LayerRole.parse(tile.type)
                if role is not None:
                    table[ts.firstgid + tile_id] = role
        return table

    def _integrate_level(self, z: int, tile_layers, tile_roles: np.ndarray):
        indices = [l.index for l in self.layers if l.z == z]
        if not indices:
            return
        # (L, D, W): las capas del nivel apiladas en el orden de Tiled (abajo -> arriba)
        gids = np.stack([np.frombuffer(tile_layers[i].dense_tiles(
            self.origin_x, self.origin_y, self.W, self.D), dtype=np.uint32).reshape(self.D, self.W)
            for i in indices])
        clean = gids & GID_MASK
        roles = np.broadcast_to(np.array([self.layers[i].role for i in indices],
                                         dtype=np.uint8)[:, None, None], gids.shape)
        forced = tile_roles[np.minimum(clean, len(tile_roles) - 1)]
        roles = np.where((forced != NO_ROLE) & (clean < len(tile_roles)), forced, roles)
        sources = np.broadcast_to(np.array(indices, dtype=np.uint8)[:, None, None], gids.shape)

        filled = clean != 0
        overflow = filled.sum(axis=0) > SLOTS
        if overflow.any():
            ys, xs = np.nonzero(overflow)
            extra = int((filled.sum(axis=0) - SLOTS)[overflow].sum())
            self.dropped_tiles += extra
            names = ", ".join(self.layers[i].name for i in indices)
            log.warning("Nivel %d: %d celdas con más de %d tiles (capas: %s); se descartan %d "
                        "tiles de las capas más altas. Primera en (x=%d, y=%d)",
                        self.get_level_value(z), len(ys), SLOTS, names, extra,
                        xs[0] + self.origin_x, ys[0] + self.origin_y)

        # Compacta hacia abajo los tiles no vacíos conservando su orden
        order = np.argsort(~filled, axis=0, kind="stable")[:SLOTS]
        k = order.shape[0]
        self.slots_used[z] = min(SLOTS, int(filled.sum(axis=0).max(initial=0)))
        take = lambda a: np.take_along_axis(a, order, axis=0)
        keep = take(filled)
        kept_roles = take(roles)
        self.tiles[z, :, :, :k] = np.moveaxis(np.where(keep, take(gids), 0), 0, -1)
        self.source[z, :, :, :k] = np.moveaxis(np.where(keep, take(sources), EMPTY_SOURCE), 0, -1)
        packed = np.zeros((self.D, self.W), dtype=np.uint8)
        for slot in range(k):
            packed |= (np.where(keep[slot], kept_roles[slot], 0) << (2 * slot)).astype(np.uint8)
        self.roles[z] = packed

    def _build_cells(self, tilesets):
        """Geometría del hueco más alto que defina una; flags combinados de todos."""
        max_gid = int((self.tiles & GID_MASK).max(initial=0))
        table = cellfmt.build_gid_table(tilesets, max_gid)
        geometry = np.zeros(self.cells.shape, dtype=np.uint16)
        flags = np.zeros(self.cells.shape, dtype=np.uint16)
        for slot in range(SLOTS):
            value = table[self.tiles[..., slot] & GID_MASK]
            geo = value & cellfmt.GEOMETRY_MASK
            geometry = np.where(geo != 0, geo, geometry)
            flags |= value & cellfmt.FLAGS_MASK
        self.cells[...] = geometry | flags

        unimplemented = np.isin(cellfmt.shape_of(geometry),
                                list(set(cellfmt.SHAPES.values()) - cellfmt.IMPLEMENTED_SHAPES))
        if unimplemented.any():
            log.warning("%d celdas usan formas aún no implementadas; se tratan como bloque",
                        int(unimplemented.sum()))
        stats = self.collision_stats()
        log.info("Celdas sólidas: %d (%.1f%%)", stats["solid"], stats["solid_percent"])

    # --- Consultas ---

    def role_of(self, slot: int, roles: Optional[np.ndarray] = None) -> np.ndarray:
        """Rol del hueco ``slot`` (array del mismo tamaño que ``roles``, por defecto todo el mundo)."""
        roles = self.roles if roles is None else roles
        return (roles >> (2 * slot)) & 3

    def get_level_value(self, z: int) -> int:
        """Índice z del mundo -> valor de nivel en Tiled (puede ser negativo)."""
        return z - self.level_offset

    def _in_bounds(self, x, y, z) -> bool:
        return 0 <= x < self.W and 0 <= y < self.D and 0 <= z < self.H

    def get_tile(self, x: int, y: int, z: int, slot: int) -> int:
        """GID sin bits de flip del hueco ``slot`` de la celda, o 0."""
        if self._in_bounds(x, y, z) and 0 <= slot < SLOTS:
            return int(self.tiles[z, y, x, slot]) & GID_MASK
        return 0

    def get_cell(self, x: int, y: int, z: int) -> int:
        return int(self.cells[z, y, x]) if self._in_bounds(x, y, z) else cellfmt.BLOCK

    def is_walkable(self, px: float, py: float, z: float) -> bool:
        tx, ty = self.collision.pixel_to_tile(px, py, self.tile_width, self.tile_height)
        return self.collision.is_walkable(tx, ty, int(z))

    def is_walkable_with_size(self, px: float, py: float, z: float,
                              char_width: float = DEFAULT_COLLISION_SIZE[0],
                              char_depth: float = DEFAULT_COLLISION_SIZE[1],
                              char_height: float = DEFAULT_COLLISION_SIZE[2]) -> bool:
        return self.collision.can_move_to_with_size(
            px, py, z, char_width, char_depth, char_height, self.tile_width, self.tile_height)

    def collision_stats(self) -> dict:
        total = self.cells.size
        solid = int(np.count_nonzero(cellfmt.is_solid(self.cells)))
        return {"total": total, "solid": solid,
                "solid_percent": solid / total * 100 if total else 0.0}

    @property
    def pixel_size(self):
        return self.W * self.tile_width, self.D * self.tile_height

    @property
    def memory_bytes(self) -> int:
        return sum(a.nbytes for a in (self.tiles, self.source, self.roles, self.cells))
