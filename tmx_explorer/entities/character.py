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
    IDLE = "idle"
    WANDER = "wander"
    PATROL = "patrol"
    FOLLOW = "follow"


class Character:
    """
    Character entity - Player or NPC
    
    Tamaño por defecto (en unidades de tile, donde 1 tile = 1m x 1m x 2m):
    - collision_width: 0.5 (50cm de ancho)
    - collision_depth: 0.5 (50cm de largo/profundidad)
    - collision_height: 0.85 (1.7m de alto, permite pasar por huecos de 2m)
    """
    
    # Valores por defecto para colisiones (en unidades de tile)
    DEFAULT_COLLISION_WIDTH = 0.5   # 50cm
    DEFAULT_COLLISION_DEPTH = 0.5   # 50cm  
    DEFAULT_COLLISION_HEIGHT = 0.85 # 1.7m (si 1 nivel = 2m)
    
    def __init__(self, sprite: AnimatedSprite,
                 x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 speed: float = 100.0,
                 tile_height: int = 32,
                 is_npc: bool = False,
                 behavior: NPCBehavior = NPCBehavior.IDLE,
                 collision_width: float = None,
                 collision_depth: float = None,
                 collision_height: float = None):
        self.sprite = sprite
        self.x = x
        self.y = y
        self.z = z
        self.speed = speed
        self.tile_height = tile_height
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        
        # Tamaño de colisión en unidades de tile (personalizable por personaje)
        self.collision_width = collision_width or self.DEFAULT_COLLISION_WIDTH
        self.collision_depth = collision_depth or self.DEFAULT_COLLISION_DEPTH
        self.collision_height = collision_height or self.DEFAULT_COLLISION_HEIGHT
        
        # Collision reference (set by EntityManager)
        self.collision_map = None
        self.tile_width = 32
        
        # Límites de altura
        self.min_z = 0.0
        self.max_z = 10.0
        
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

    def set_collision_size(self, width: float, depth: float, height: float):
        """Configura el tamaño de colisión del personaje (en unidades de tile)"""
        self.collision_width = width
        self.collision_depth = depth
        self.collision_height = height

    def _init_textures(self, renderer: 'OpenGLRenderer'):
        from ..renderer.texture import Texture
        
        for direction in Direction:
            for frame_idx in range(4):
                frame_image = self.sprite.get_frame(direction, frame_idx)
                self._texture_cache[(direction, frame_idx)] = Texture.from_pil(frame_image)
        
        self._textures_initialized = True

    def update(self, dt: float):
        if self.is_npc:
            self._update_npc_behavior(dt)
        
        is_moving = (self.velocity_x != 0 or self.velocity_y != 0)
        
        if is_moving:
            new_x = self.x + self.velocity_x * dt
            new_y = self.y + self.velocity_y * dt
            
            if self.collision_map is not None:
                if self._can_move_to(new_x, self.y, self.z):
                    self.x = new_x
                if self._can_move_to(self.x, new_y, self.z):
                    self.y = new_y
            else:
                self.x = new_x
                self.y = new_y
            
            self._update_direction()
        
        # Movimiento en Z con validación
        if self.velocity_z != 0:
            new_z = self.z + self.velocity_z * dt
            new_z = max(self.min_z, min(self.max_z, new_z))
            
            if self.collision_map is not None:
                if self._can_change_height(new_z):
                    self.z = new_z
            else:
                self.z = new_z
        
        self.sprite.set_walking(is_moving)
        self.sprite.update(dt)
    
    def _can_move_to(self, px: float, py: float, z: float) -> bool:
        if self.collision_map is None:
            return True
        
        return self.collision_map.can_move_to_with_size(
            px, py, z,
            self.collision_width,
            self.collision_depth,
            self.collision_height,
            self.tile_width, self.tile_height
        )
    
    def _can_change_height(self, new_z: float) -> bool:
        if self.collision_map is None:
            return True
        
        return self.collision_map.can_change_height(
            self.x, self.y,
            self.z, new_z,
            self.collision_width,
            self.collision_depth,
            self.collision_height,
            self.tile_width, self.tile_height
        )

    def _update_npc_behavior(self, dt: float):
        if self.behavior == NPCBehavior.IDLE:
            self._behavior_idle()
        elif self.behavior == NPCBehavior.WANDER:
            self._behavior_wander(dt)
        elif self.behavior == NPCBehavior.PATROL:
            self._behavior_patrol(dt)
        elif self.behavior == NPCBehavior.FOLLOW:
            self._behavior_follow(dt)

    def _behavior_idle(self):
        self.velocity_x = 0
        self.velocity_y = 0

    def _behavior_wander(self, dt: float):
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
        
        dist_from_home = math.sqrt((self.x - self.home_x)**2 + (self.y - self.home_y)**2)
        if dist_from_home > self.wander_radius:
            dx = self.home_x - self.x
            dy = self.home_y - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                self.velocity_x = (dx / dist) * self.speed * 0.5
                self.velocity_y = (dy / dist) * self.speed * 0.5

    def _behavior_patrol(self, dt: float):
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
