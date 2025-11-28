"""
Character entity with position, movement and animation
"""

from typing import Dict, Tuple, TYPE_CHECKING
from .sprite import AnimatedSprite, Direction

if TYPE_CHECKING:
    from ..renderer.texture import Texture
    from ..renderer.opengl_renderer import OpenGLRenderer


class Character:
    """Character entity in the game world (PIL version)"""
    
    def __init__(self, sprite: AnimatedSprite,
                 x: float = 0.0, y: float = 0.0, z: int = 0,
                 speed: float = 100.0):
        self.sprite = sprite
        self.x = x
        self.y = y
        self.z = z
        self.speed = speed
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        
        self._texture_cache: Dict[Tuple[Direction, int], 'Texture'] = {}
        self._textures_initialized = False

    def _init_textures(self, renderer: 'OpenGLRenderer'):
        """Pre-create all frame textures"""
        from ..renderer.texture import Texture
        
        for direction in Direction:
            for frame_idx in range(4):
                frame_image = self.sprite.get_frame(direction, frame_idx)
                self._texture_cache[(direction, frame_idx)] = Texture.from_pil(frame_image)
        
        self._textures_initialized = True

    def update(self, dt: float):
        is_moving = (self.velocity_x != 0 or self.velocity_y != 0)
        
        if is_moving:
            self.x += self.velocity_x * dt
            self.y += self.velocity_y * dt
            self._update_direction()
        
        self.sprite.set_walking(is_moving)
        self.sprite.update(dt)

    def _update_direction(self):
        if abs(self.velocity_y) > abs(self.velocity_x):
            if self.velocity_y > 0:
                self.sprite.set_direction(Direction.DOWN)
            else:
                self.sprite.set_direction(Direction.UP)
        else:
            if self.velocity_x > 0:
                self.sprite.set_direction(Direction.RIGHT)
            else:
                self.sprite.set_direction(Direction.LEFT)

    def move(self, dx: float, dy: float):
        if dx == 0 and dy == 0:
            self.velocity_x = 0
            self.velocity_y = 0
            return
        
        if dx != 0 and dy != 0:
            factor = 0.7071
            dx *= factor
            dy *= factor
        
        self.velocity_x = dx * self.speed
        self.velocity_y = dy * self.speed

    def get_render_position(self) -> tuple:
        render_x = self.x - self.sprite.width / 2
        render_y = self.y - self.sprite.height
        return render_x, render_y

    def get_depth(self) -> float:
        base_offset = -1000.0
        return base_offset + (self.y / 32.0) + self.z + 0.5

    def get_texture(self, renderer: 'OpenGLRenderer') -> 'Texture':
        if not self._textures_initialized:
            self._init_textures(renderer)
        
        key = (self.sprite.direction, self.sprite.current_frame)
        return self._texture_cache[key]

    @property
    def width(self) -> int:
        return self.sprite.width
    
    @property
    def height(self) -> int:
        return self.sprite.height
