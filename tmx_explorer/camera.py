"""Cámara 2D: posición (esquina superior izquierda, en píxeles de mundo) y zoom.

    pantalla = (mundo - cámara) * zoom
"""

from typing import Tuple


class Camera:
    MIN_ZOOM = 0.1
    MAX_ZOOM = 5.0

    def __init__(self, width: int, height: int):
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0
        self.width = width      # tamaño de la ventana en píxeles
        self.height = height

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height

    def move(self, dx: float, dy: float):
        """Desplaza en píxeles de *pantalla* (misma sensación con cualquier zoom)."""
        self.x += dx / self.zoom
        self.y += dy / self.zoom

    def set_zoom(self, zoom: float):
        """Cambia el zoom manteniendo fijo el centro de la pantalla."""
        center_x, center_y = self.center
        self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
        self.center_on(center_x, center_y)

    def zoom_by(self, factor: float):
        self.set_zoom(self.zoom * factor)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / (2 * self.zoom),
                self.y + self.height / (2 * self.zoom))

    def center_on(self, world_x: float, world_y: float):
        self.x = world_x - self.width / (2 * self.zoom)
        self.y = world_y - self.height / (2 * self.zoom)

    def reset(self, map_width: int, map_height: int, tile_width: int, tile_height: int):
        """Vuelve al origen con un zoom que encaje el mapa (sin pasar de 1:1)."""
        self.x = self.y = 0.0
        self.zoom = min(self.width / (map_width * tile_width),
                        self.height / (map_height * tile_height), 1.0)

    def screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        return self.x + screen_x / self.zoom, self.y + screen_y / self.zoom

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        return (world_x - self.x) * self.zoom, (world_y - self.y) * self.zoom

    def get_visible_bounds(self) -> Tuple[float, float, float, float]:
        """(izquierda, arriba, derecha, abajo) visibles, en píxeles de mundo."""
        return (self.x, self.y,
                self.x + self.width / self.zoom, self.y + self.height / self.zoom)
