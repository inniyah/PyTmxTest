"""
Camera system for 2D map navigation
"""


class Camera:
    """2D camera with pan and zoom capabilities"""
    
    MIN_ZOOM = 0.1
    MAX_ZOOM = 5.0
    
    def __init__(self, width: int, height: int):
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0
        self.width = width
        self.height = height

    def move(self, dx: float, dy: float):
        """Move camera by delta, adjusted for zoom"""
        self.x += dx / self.zoom
        self.y += dy / self.zoom

    def set_zoom(self, zoom: float):
        """Set zoom level, maintaining center point"""
        center_x = self.x + self.width / (2 * self.zoom)
        center_y = self.y + self.height / (2 * self.zoom)
        
        self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
        
        self.x = center_x - self.width / (2 * self.zoom)
        self.y = center_y - self.height / (2 * self.zoom)

    def zoom_by(self, factor: float):
        """Multiply current zoom by factor"""
        self.set_zoom(self.zoom * factor)

    def reset(self, map_width: int, map_height: int, 
              tile_width: int, tile_height: int):
        """Reset camera to fit map in view"""
        self.x = 0
        self.y = 0
        zoom_x = self.width / (map_width * tile_width)
        zoom_y = self.height / (map_height * tile_height)
        self.zoom = min(zoom_x, zoom_y, 1.0)

    def screen_to_world(self, screen_x: float, screen_y: float) -> tuple:
        """Convert screen coordinates to world coordinates"""
        world_x = self.x + screen_x / self.zoom
        world_y = self.y + screen_y / self.zoom
        return world_x, world_y

    def world_to_screen(self, world_x: float, world_y: float) -> tuple:
        """Convert world coordinates to screen coordinates"""
        screen_x = (world_x - self.x) * self.zoom
        screen_y = (world_y - self.y) * self.zoom
        return screen_x, screen_y

    def get_visible_bounds(self) -> tuple:
        """Get visible world bounds (left, top, right, bottom)"""
        left = self.x
        top = self.y
        right = self.x + self.width / self.zoom
        bottom = self.y + self.height / self.zoom
        return left, top, right, bottom
