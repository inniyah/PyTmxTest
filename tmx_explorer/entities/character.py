"""
Character entity with position, movement and animation
"""

import pygame
from typing import Optional, Dict, Tuple, TYPE_CHECKING
from .sprite import AnimatedSprite, Direction, AnimationState

if TYPE_CHECKING:
    from ..renderer.texture import Texture
    from ..renderer.opengl_renderer import OpenGLRenderer


class Character:
    """
    A character entity in the game world.
    
    Coordinates are in world space (tile pixels).
    The anchor point is at the bottom-center of the sprite.
    """
    
    def __init__(
        self,
        sprite: AnimatedSprite,
        x: float = 0.0,
        y: float = 0.0,
        z: int = 0,
        speed: float = 100.0
    ):
        self.sprite = sprite
        
        # World position (bottom-center anchor)
        self.x = x
        self.y = y
        self.z = z  # Height level
        
        # Movement
        self.speed = speed  # Pixels per second
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        
        # Pre-cached textures for all frames (created on first use)
        self._texture_cache: Dict[Tuple[Direction, int], 'Texture'] = {}
        self._textures_initialized = False

    def _init_textures(self, renderer: 'OpenGLRenderer'):
        """Pre-create all frame textures"""
        from ..renderer.texture import Texture
        
        border = 1
        for direction in Direction:
            for frame_idx in range(4):  # 0=idle, 1-3=walk
                frame_surface = self.sprite.get_frame(direction, frame_idx)
                
                # Create bordered surface to prevent bleeding
                bordered = pygame.Surface(
                    (self.sprite.width + border * 2, self.sprite.height + border * 2),
                    pygame.SRCALPHA
                )
                bordered.blit(frame_surface, (border, border))
                
                self._texture_cache[(direction, frame_idx)] = Texture(bordered)
        
        self._textures_initialized = True

    def update(self, dt: float):
        """Update character state"""
        # Check if moving
        is_moving = (self.velocity_x != 0 or self.velocity_y != 0)
        
        # Update position based on velocity
        if is_moving:
            self.x += self.velocity_x * dt
            self.y += self.velocity_y * dt
            self._update_direction()
        
        # Update animation state
        self.sprite.set_walking(is_moving)
        
        # Update animation
        self.sprite.update(dt)

    def _update_direction(self):
        """Set sprite direction based on velocity"""
        # Prioritize vertical movement for isometric feel
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
        """Set movement velocity (normalized). dx=dy=0 stops movement."""
        if dx == 0 and dy == 0:
            self.velocity_x = 0
            self.velocity_y = 0
            return
            
        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            factor = 0.7071  # 1/sqrt(2)
            dx *= factor
            dy *= factor
        
        self.velocity_x = dx * self.speed
        self.velocity_y = dy * self.speed

    def set_position(self, x: float, y: float, z: int = 0):
        """Set world position"""
        self.x = x
        self.y = y
        self.z = z

    def get_render_position(self) -> tuple:
        """
        Get top-left position for rendering.
        Anchor is bottom-center, so offset by half width and full height.
        """
        render_x = self.x - self.sprite.width / 2
        render_y = self.y - self.sprite.height
        return render_x, render_y

    def get_depth(self) -> float:
        """
        Calculate depth for z-ordering.
        Uses same formula as tiles: base_offset + y + z
        Characters are rendered slightly in front of tiles at same position.
        """
        base_offset = -1000.0
        return base_offset + (self.y / 32.0) + self.z + 0.5  # +0.5 to be in front of tiles

    def get_texture(self, renderer: 'OpenGLRenderer') -> 'Texture':
        """Get cached texture for current frame"""
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

    @property
    def direction(self) -> Direction:
        return self.sprite.direction
    
    @direction.setter
    def direction(self, value: Direction):
        self.sprite.set_direction(value)
