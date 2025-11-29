"""
Collision system using a parallel numpy array
"""

import numpy as np
from typing import Tuple, List


class CollisionMap:
    """
    Mapa de colisiones paralelo al mapa de tiles.
    
    Dimensiones: [H, D, W] donde:
    - H: Niveles de altura (Z)
    - D: Profundidad (Y en tiles)
    - W: Ancho (X en tiles)
    
    Cada celda es un uint16 con flags de colisión.
    Por ahora: 0 = vacío/caminable, != 0 = sólido
    
    Unidades de tile: 1 tile = 1m x 1m x 2m (ancho x largo x alto)
    """
    
    def __init__(self, width: int, height: int, depth: int):
        self.W = width
        self.H = height
        self.D = depth
        
        self.data = np.zeros((self.H, self.D, self.W), dtype=np.uint16)
        
        print(f"CollisionMap creado: {self.W}x{self.D}x{self.H} (W x D x H)")
    
    def set_flags(self, x: int, y: int, z: int, flags: int):
        if self._in_bounds(x, y, z):
            self.data[z, y, x] = flags
    
    def get_flags(self, x: int, y: int, z: int) -> int:
        if self._in_bounds(x, y, z):
            return int(self.data[z, y, x])
        return 1  # Fuera de límites = sólido
    
    def is_solid(self, x: int, y: int, z: int) -> bool:
        return self.get_flags(x, y, z) != 0
    
    def is_walkable(self, x: int, y: int, z: int) -> bool:
        return self.get_flags(x, y, z) == 0
    
    def _in_bounds(self, x: int, y: int, z: int) -> bool:
        return (0 <= x < self.W and 0 <= y < self.D and 0 <= z < self.H)
    
    def pixel_to_tile(self, px: float, py: float, 
                      tile_width: int, tile_height: int) -> Tuple[int, int]:
        tx = int(px // tile_width)
        ty = int(py // tile_height)
        return tx, ty
    
    def get_z_levels_to_check(self, z: float, char_height: float = 0.85) -> List[int]:
        """
        Obtiene los niveles de Z que ocupa el personaje.
        
        Args:
            z: Posición Z actual (nivel de los pies)
            char_height: Altura del personaje en niveles (0.85 = 1.7m si tile=2m)
        """
        z_floor = int(z)
        z_top = z + char_height
        z_ceil = int(z_top)
        
        levels = []
        if 0 <= z_floor < self.H:
            levels.append(z_floor)
        if z_ceil != z_floor and 0 <= z_ceil < self.H:
            levels.append(z_ceil)
        
        return levels
    
    def can_move_to_with_size(self, px: float, py: float, z: float,
                               char_width: float, char_depth: float, char_height: float,
                               tile_width: int, tile_height: int) -> bool:
        """
        Comprueba colisión considerando el tamaño del personaje en unidades de tile.
        
        Args:
            px, py: Posición en píxeles (centro-abajo del personaje)
            z: Nivel Z actual
            char_width: Ancho del personaje en tiles (ej: 0.5 = 50cm)
            char_depth: Profundidad del personaje en tiles (ej: 0.5 = 50cm)  
            char_height: Altura del personaje en niveles (ej: 0.85 = 1.7m)
            tile_width, tile_height: Tamaño de tiles en píxeles
        """
        # Convertir tamaño del personaje de tiles a píxeles
        half_width_px = (char_width * tile_width) / 2
        half_depth_px = (char_depth * tile_height) / 2
        
        # Bounding box del personaje (centrado en px, py)
        left = px - half_width_px
        right = px + half_width_px
        top = py - half_depth_px
        bottom = py + half_depth_px
        
        # Esquinas a comprobar
        corners = [
            (left, top),
            (right, top),
            (left, bottom),
            (right, bottom),
        ]
        
        # Niveles Z a comprobar
        z_levels = self.get_z_levels_to_check(z, char_height)
        
        for cx, cy in corners:
            tx, ty = self.pixel_to_tile(cx, cy, tile_width, tile_height)
            for tz in z_levels:
                if self.is_solid(tx, ty, tz):
                    return False
        
        return True
    
    def can_change_height(self, px: float, py: float, 
                          current_z: float, new_z: float,
                          char_width: float, char_depth: float, char_height: float,
                          tile_width: int, tile_height: int) -> bool:
        """
        Comprueba si el personaje puede cambiar de altura.
        """
        # No permitir bajar por debajo de 0
        if new_z < 0:
            return False
        
        # No permitir subir por encima del máximo
        if new_z >= self.H:
            return False
        
        # Comprobar que hay espacio en la nueva altura
        return self.can_move_to_with_size(
            px, py, new_z,
            char_width, char_depth, char_height,
            tile_width, tile_height
        )
    
    def get_stats(self) -> dict:
        total = self.W * self.D * self.H
        solid = np.count_nonzero(self.data)
        empty = total - solid
        
        return {
            'total_tiles': total,
            'solid_tiles': solid,
            'empty_tiles': empty,
            'solid_percent': (solid / total * 100) if total > 0 else 0
        }
