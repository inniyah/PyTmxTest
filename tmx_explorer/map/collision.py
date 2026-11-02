"""Mapa de colisión por tiles: un array numpy [z, y, x] de flags uint16.

Unidades: 1 tile = 1 m de ancho x 1 m de fondo; 1 nivel Z = 2 m de alto.
Un humano mide ~0.5 x 0.5 tiles de base y ~0.85 niveles de alto.
Ver docs/architecture.md para la explicación completa.
"""

from typing import List, Tuple

import numpy as np

# Flags por celda. Solo FLAG_SOLID se usa hoy; el resto de bits queda libre
# (agua, escaleras, daño...).
FLAG_SOLID = 1 << 0


class CollisionMap:
    """Rejilla 3D de flags de colisión, indexada como data[z, y, x]."""

    def __init__(self, width: int, height: int, depth: int):
        self.W = width    # tiles en X
        self.H = height   # niveles en Z
        self.D = depth    # tiles en Y
        self.data = np.zeros((self.H, self.D, self.W), dtype=np.uint16)

    # --- Acceso a celdas ---

    def _in_bounds(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.D and 0 <= z < self.H

    def set_flags(self, x: int, y: int, z: int, flags: int):
        if self._in_bounds(x, y, z):
            self.data[z, y, x] = flags

    def get_flags(self, x: int, y: int, z: int) -> int:
        """Flags de la celda. Fuera del mapa se considera sólido."""
        if self._in_bounds(x, y, z):
            return int(self.data[z, y, x])
        return FLAG_SOLID

    def is_solid(self, x: int, y: int, z: int) -> bool:
        return bool(self.get_flags(x, y, z) & FLAG_SOLID)

    def is_walkable(self, x: int, y: int, z: int) -> bool:
        return not self.is_solid(x, y, z)

    # --- Consultas en píxeles ---

    @staticmethod
    def pixel_to_tile(px: float, py: float, tile_width: int, tile_height: int) -> Tuple[int, int]:
        return int(px // tile_width), int(py // tile_height)

    def get_z_levels_to_check(self, z: float, char_height: float = 0.85) -> List[int]:
        """Niveles que ocupa un cuerpo desde z hasta z + char_height."""
        z_floor = int(z)
        z_top = int(z + char_height)
        levels = []
        if 0 <= z_floor < self.H:
            levels.append(z_floor)
        if z_top != z_floor and 0 <= z_top < self.H:
            levels.append(z_top)
        return levels

    def can_move_to_with_size(self, px: float, py: float, z: float,
                              char_width: float, char_depth: float, char_height: float,
                              tile_width: int, tile_height: int) -> bool:
        """True si la caja del personaje (centrada en px, py; tamaños en tiles) cabe.

        Comprueba las 4 esquinas de la base en cada nivel que ocupa el cuerpo.
        """
        half_w = char_width * tile_width / 2
        half_d = char_depth * tile_height / 2
        corners = ((px - half_w, py - half_d), (px + half_w, py - half_d),
                   (px - half_w, py + half_d), (px + half_w, py + half_d))
        z_levels = self.get_z_levels_to_check(z, char_height)

        for cx, cy in corners:
            tx, ty = self.pixel_to_tile(cx, cy, tile_width, tile_height)
            for tz in z_levels:
                if self.is_solid(tx, ty, tz):
                    return False
        return True

    def can_change_height(self, px: float, py: float, current_z: float, new_z: float,
                          char_width: float, char_depth: float, char_height: float,
                          tile_width: int, tile_height: int) -> bool:
        if not 0 <= new_z < self.H:
            return False
        return self.can_move_to_with_size(px, py, new_z, char_width, char_depth,
                                          char_height, tile_width, tile_height)

    def get_stats(self) -> dict:
        total = self.data.size
        solid = int(np.count_nonzero(self.data & FLAG_SOLID))
        return {
            "total_tiles": total,
            "solid_tiles": solid,
            "empty_tiles": total - solid,
            "solid_percent": solid / total * 100 if total else 0.0,
        }
