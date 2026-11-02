"""Personaje (jugador o NPC): posición, movimiento con colisión, IA simple y animación.

Coordenadas: (x, y) en píxeles de mundo, en los *pies* del personaje; z en niveles
(índice del array, 0..H-1). La caja de colisión se mide en tiles.
"""

import math
import random
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

from .. import config
from .sprite import AnimatedSprite, Direction

if TYPE_CHECKING:
    from ..map.collision import CollisionMap
    from ..renderer.texture import Texture

# Parámetros de comportamiento
WANDER_PAUSE_CHANCE = 0.3
WANDER_PAUSE_RANGE = (1.0, 3.0)
WANDER_SPEED_RANGE = (0.3, 0.7)
RETURN_HOME_SPEED = 0.5
PATROL_SPEED = 0.5
PATROL_ARRIVE_DISTANCE = 10.0
FOLLOW_SPEED = 0.6
FOLLOW_MIN_DISTANCE = 80.0


class NPCBehavior(Enum):
    IDLE = "idle"
    WANDER = "wander"
    PATROL = "patrol"
    FOLLOW = "follow"


class Character:
    DEFAULT_COLLISION_WIDTH, DEFAULT_COLLISION_DEPTH, DEFAULT_COLLISION_HEIGHT = \
        config.DEFAULT_COLLISION_SIZE

    def __init__(self, sprite: AnimatedSprite,
                 x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 speed: float = 100.0, tile_height: int = 32,
                 is_npc: bool = False, behavior: NPCBehavior = NPCBehavior.IDLE,
                 collision_width: float = None, collision_depth: float = None,
                 collision_height: float = None):
        self.sprite = sprite
        self.x, self.y, self.z = x, y, z
        self.speed = speed
        self.velocity_x = self.velocity_y = self.velocity_z = 0.0

        # Colisión (el EntityManager rellena collision_map, tile_width y límites de z)
        self.collision_width = collision_width or self.DEFAULT_COLLISION_WIDTH
        self.collision_depth = collision_depth or self.DEFAULT_COLLISION_DEPTH
        self.collision_height = collision_height or self.DEFAULT_COLLISION_HEIGHT
        self.collision_map: Optional["CollisionMap"] = None
        self.tile_width = 32
        self.tile_height = tile_height
        self.min_z = 0.0
        self.max_z = 10.0

        # IA
        self.is_npc = is_npc
        self.behavior = behavior
        self.target: Optional["Character"] = None
        self.patrol_points: List[Tuple[float, float]] = []
        self.patrol_index = 0
        self.wander_timer = 0.0
        self.wander_interval = 2.0
        self.wander_radius = 200.0
        self.home_x, self.home_y = x, y
        self.idle_timer = 0.0
        self.is_idle_pausing = False

    def set_collision_size(self, width: float, depth: float, height: float):
        self.collision_width, self.collision_depth, self.collision_height = width, depth, height

    # --- Bucle principal ---

    def update(self, dt: float):
        if self.is_npc:
            self._update_npc_behavior(dt)

        is_moving = self.velocity_x != 0 or self.velocity_y != 0
        if is_moving:
            # Ejes por separado: así se puede "deslizar" a lo largo de una pared
            new_x = self.x + self.velocity_x * dt
            if self._can_move_to(new_x, self.y, self.z):
                self.x = new_x
            new_y = self.y + self.velocity_y * dt
            if self._can_move_to(self.x, new_y, self.z):
                self.y = new_y
            self._update_direction()

        if self.velocity_z != 0:
            new_z = min(self.max_z, max(self.min_z, self.z + self.velocity_z * dt))
            if self._can_change_height(new_z):
                self.z = new_z

        self.sprite.set_walking(is_moving)
        self.sprite.update(dt)

    def move(self, dx: float, dy: float, dz: float = 0.0):
        """Fija la dirección de movimiento; dx, dy, dz en -1..1."""
        if dx and dy:
            dx *= config.DIAGONAL_FACTOR
            dy *= config.DIAGONAL_FACTOR
        self.velocity_x = dx * self.speed
        self.velocity_y = dy * self.speed
        self.velocity_z = dz * self.speed * config.HEIGHT_SPEED_FACTOR

    def stop(self):
        self.move(0, 0, 0)

    # --- Colisión ---

    def _can_move_to(self, px: float, py: float, z: float) -> bool:
        if self.collision_map is None:
            return True
        return self.collision_map.can_move_to_with_size(
            px, py, z, self.collision_width, self.collision_depth, self.collision_height,
            self.tile_width, self.tile_height)

    def _can_change_height(self, new_z: float) -> bool:
        if self.collision_map is None:
            return True
        return self.collision_map.can_change_height(
            self.x, self.y, self.z, new_z,
            self.collision_width, self.collision_depth, self.collision_height,
            self.tile_width, self.tile_height)

    # --- IA de NPC ---

    def _update_npc_behavior(self, dt: float):
        if self.behavior == NPCBehavior.WANDER:
            self._behavior_wander(dt)
        elif self.behavior == NPCBehavior.PATROL:
            self._behavior_patrol()
        elif self.behavior == NPCBehavior.FOLLOW:
            self._behavior_follow()
        else:
            self._behavior_idle()

    def _behavior_idle(self):
        self.velocity_x = self.velocity_y = 0.0

    def _head_towards(self, tx: float, ty: float, speed: float) -> float:
        """Apunta la velocidad hacia (tx, ty). Devuelve la distancia hasta allí."""
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.velocity_x = dx / dist * speed
            self.velocity_y = dy / dist * speed
        return dist

    def _behavior_wander(self, dt: float):
        if self.is_idle_pausing:
            self.idle_timer -= dt
            if self.idle_timer > 0:
                self._behavior_idle()
                return
            self.is_idle_pausing = False

        self.wander_timer -= dt
        if self.wander_timer <= 0:
            if random.random() < WANDER_PAUSE_CHANCE:
                self.is_idle_pausing = True
                self.idle_timer = random.uniform(*WANDER_PAUSE_RANGE)
                self._behavior_idle()
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = self.speed * random.uniform(*WANDER_SPEED_RANGE)
                self.velocity_x = math.cos(angle) * speed
                self.velocity_y = math.sin(angle) * speed
            self.wander_timer = random.uniform(1.0, self.wander_interval)

        if math.hypot(self.x - self.home_x, self.y - self.home_y) > self.wander_radius:
            self._head_towards(self.home_x, self.home_y, self.speed * RETURN_HOME_SPEED)

    def _behavior_patrol(self):
        if not self.patrol_points:
            self._behavior_idle()
            return
        tx, ty = self.patrol_points[self.patrol_index]
        if math.hypot(tx - self.x, ty - self.y) < PATROL_ARRIVE_DISTANCE:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
        else:
            self._head_towards(tx, ty, self.speed * PATROL_SPEED)

    def _behavior_follow(self):
        if not self.target:
            self._behavior_idle()
            return
        dist = math.hypot(self.target.x - self.x, self.target.y - self.y)
        if dist > FOLLOW_MIN_DISTANCE:
            self._head_towards(self.target.x, self.target.y, self.speed * FOLLOW_SPEED)
        else:
            self._behavior_idle()

    def set_patrol_points(self, points: List[Tuple[float, float]]):
        self.patrol_points = list(points)
        self.patrol_index = 0

    # --- Render ---

    def _update_direction(self):
        vx, vy = self.velocity_x, self.velocity_y
        if abs(vy) > abs(vx):
            self.sprite.set_direction(Direction.DOWN if vy > 0 else Direction.UP)
        elif vx != 0:
            self.sprite.set_direction(Direction.RIGHT if vx > 0 else Direction.LEFT)

    def get_render_position(self) -> Tuple[float, float]:
        """Esquina superior izquierda del sprite (los pies quedan en x, y)."""
        return self.x - self.sprite.width / 2, self.y - self.sprite.height

    def get_depth(self) -> float:
        return (config.DEPTH_BASE + self.y / self.tile_height + self.z
                + config.DEPTH_CHARACTER_BIAS)

    def get_texture(self, renderer=None) -> "Texture":
        """Textura del frame actual (``renderer`` se acepta por compatibilidad)."""
        return self.sprite.get_texture()

    @property
    def width(self) -> int:
        return self.sprite.width

    @property
    def height(self) -> int:
        return self.sprite.height
