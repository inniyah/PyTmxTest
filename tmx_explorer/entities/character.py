"""
Character entity - Player and NPC support
"""

import random
import math
from typing import Dict, Tuple, TYPE_CHECKING, Optional
from enum import Enum
from .sprite import AnimatedSprite, Direction

if TYPE_CHECKING:
    from ..renderer.texture import Texture
    from ..renderer.opengl_renderer import OpenGLRenderer


class NPCBehavior(Enum):
    """NPC behavior types"""
    IDLE = "idle"
    WANDER = "wander"
    PATROL = "patrol"
    FOLLOW = "follow"


class Character:
    """Character entity - Player or NPC"""
    
    def __init__(self, sprite: AnimatedSprite,
                 x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 speed: float = 100.0,
                 tile_height: int = 32,
                 is_npc: bool = False,
                 behavior: NPCBehavior = NPCBehavior.IDLE):
        self.sprite = sprite
        self.x = x
        self.y = y
        self.z = z
        self.speed = speed
        self.tile_height = tile_height
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        
        # NPC settings
        self.is_npc = is_npc
        self.behavior = behavior
        self.target: Optional['Character'] = None
        
        # Patrol settings
        self.patrol_points: list[Tuple[float, float]] = []
        self.patrol_index = 0
        
        # Wander settings
        self.wander_timer = 0.0
        self.wander_interval = 2.0
        self.wander_radius = 200.0
        self.home_x = x
        self.home_y = y
        
        # Idle pause settings
        self.idle_timer = 0.0
        self.is_idle_pausing = False
        
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
        """Update character state"""
        if self.is_npc:
            self._update_npc_behavior(dt)
        
        is_moving = (self.velocity_x != 0 or self.velocity_y != 0)
        
        if is_moving:
            self.x += self.velocity_x * dt
            self.y += self.velocity_y * dt
            self._update_direction()
        
        if self.velocity_z != 0:
            self.z += self.velocity_z * dt
        
        self.sprite.set_walking(is_moving)
        self.sprite.update(dt)

    def _update_npc_behavior(self, dt: float):
        """Update NPC AI behavior"""
        if self.behavior == NPCBehavior.IDLE:
            self._behavior_idle()
        elif self.behavior == NPCBehavior.WANDER:
            self._behavior_wander(dt)
        elif self.behavior == NPCBehavior.PATROL:
            self._behavior_patrol(dt)
        elif self.behavior == NPCBehavior.FOLLOW:
            self._behavior_follow(dt)

    def _behavior_idle(self):
        """Just stand still"""
        self.velocity_x = 0
        self.velocity_y = 0

    def _behavior_wander(self, dt: float):
        """Wander randomly around home position"""
        if self.is_idle_pausing:
            self.idle_timer -= dt
            if self.idle_timer <= 0:
                self.is_idle_pausing = False
            else:
                self.velocity_x = 0
                self.velocity_y = 0
                return
        
        self.wander_timer -= dt
        
        if self.wander_timer <= 0:
            if random.random() < 0.3:
                self.is_idle_pausing = True
                self.idle_timer = random.uniform(1.0, 3.0)
                self.velocity_x = 0
                self.velocity_y = 0
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = self.speed * random.uniform(0.3, 0.7)
                self.velocity_x = math.cos(angle) * speed
                self.velocity_y = math.sin(angle) * speed
            
            self.wander_timer = random.uniform(1.0, self.wander_interval)
        
        # Stay within wander radius
        dist_from_home = math.sqrt((self.x - self.home_x)**2 + (self.y - self.home_y)**2)
        if dist_from_home > self.wander_radius:
            dx = self.home_x - self.x
            dy = self.home_y - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                self.velocity_x = (dx / dist) * self.speed * 0.5
                self.velocity_y = (dy / dist) * self.speed * 0.5

    def _behavior_patrol(self, dt: float):
        """Patrol between defined points"""
        if not self.patrol_points:
            self._behavior_idle()
            return
        
        target_x, target_y = self.patrol_points[self.patrol_index]
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < 10:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
        else:
            speed = self.speed * 0.5
            self.velocity_x = (dx / dist) * speed
            self.velocity_y = (dy / dist) * speed

    def _behavior_follow(self, dt: float):
        """Follow target character"""
        if not self.target:
            self._behavior_idle()
            return
        
        dx = self.target.x - self.x
        dy = self.target.y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        min_distance = 80
        
        if dist > min_distance:
            speed = self.speed * 0.6
            self.velocity_x = (dx / dist) * speed
            self.velocity_y = (dy / dist) * speed
        else:
            self.velocity_x = 0
            self.velocity_y = 0

    def _update_direction(self):
        if abs(self.velocity_y) > abs(self.velocity_x):
            if self.velocity_y > 0:
                self.sprite.set_direction(Direction.DOWN)
            else:
                self.sprite.set_direction(Direction.UP)
        elif self.velocity_x != 0:
            if self.velocity_x > 0:
                self.sprite.set_direction(Direction.RIGHT)
            else:
                self.sprite.set_direction(Direction.LEFT)

    def move(self, dx: float, dy: float, dz: float = 0.0):
        """Set movement velocity (for player control)"""
        if dx == 0 and dy == 0:
            self.velocity_x = 0
            self.velocity_y = 0
        else:
            if dx != 0 and dy != 0:
                factor = 0.7071
                dx *= factor
                dy *= factor
            
            self.velocity_x = dx * self.speed
            self.velocity_y = dy * self.speed
        
        self.velocity_z = dz * self.speed * 0.01

    def set_patrol_points(self, points: list[Tuple[float, float]]):
        """Set patrol path for NPC"""
        self.patrol_points = points
        self.patrol_index = 0

    def get_render_position(self) -> tuple:
        render_x = self.x - self.sprite.width / 2
        render_y = self.y - self.sprite.height
        return render_x, render_y

    def get_depth(self) -> float:
        base_offset = -1000.0
        tile_y = self.y / self.tile_height
        return base_offset + tile_y + self.z + 0.5

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
