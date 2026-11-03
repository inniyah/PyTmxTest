"""Crea, actualiza y prepara para dibujar todos los personajes de la escena."""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .character import Character, NPCBehavior
from .sprite import AnimatedSprite

CollisionSize = Optional[Tuple[float, float, float]]   # (ancho, fondo, alto) en tiles


class EntityManager:
    def __init__(self, tile_width: int = 32, tile_height: int = 32,
                 collision_map=None, max_z: int = 10):
        self.characters: List[Character] = []
        self.player: Optional[Character] = None
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.collision_map = collision_map
        self.max_z = max_z
        self._sprite_cache: Dict[Tuple, AnimatedSprite] = {}

    # --- Sprites ---

    def load_sprite(self, path: str, frame_width: int = None, frame_height: int = None,
                    animation_speed: float = 8.0) -> AnimatedSprite:
        """Sprite con estado propio que comparte imagen y texturas con los demás del mismo archivo."""
        key = (path, frame_width, frame_height)
        cached = self._sprite_cache.get(key)
        if cached is not None:
            return AnimatedSprite.from_cached(cached, animation_speed)
        sprite = AnimatedSprite(path, frame_width, frame_height, animation_speed)
        self._sprite_cache[key] = sprite
        return sprite

    # --- Creación ---

    def _add(self, character: Character) -> Character:
        character.collision_map = self.collision_map
        character.tile_width = self.tile_width
        character.min_z = 0.0
        character.max_z = float(self.max_z - 1)
        self.characters.append(character)
        return character

    def create_character(self, spritesheet_path: str,
                         x: float = 0.0, y: float = 0.0, z: float = 0.0,
                         speed: float = 100.0,
                         frame_width: int = None, frame_height: int = None,
                         is_player: bool = False,
                         collision_width: float = None, collision_depth: float = None,
                         collision_height: float = None) -> Character:
        sprite = self.load_sprite(spritesheet_path, frame_width, frame_height)
        character = self._add(Character(
            sprite, x, y, z, speed, tile_height=self.tile_height, is_npc=False,
            collision_width=collision_width, collision_depth=collision_depth,
            collision_height=collision_height))
        if is_player:
            self.player = character
        return character

    def create_npc(self, spritesheet_path: str,
                   x: float = 0.0, y: float = 0.0, z: float = 0.0,
                   speed: float = 80.0, behavior: NPCBehavior = NPCBehavior.WANDER,
                   frame_width: int = None, frame_height: int = None,
                   collision_width: float = None, collision_depth: float = None,
                   collision_height: float = None) -> Character:
        sprite = self.load_sprite(spritesheet_path, frame_width, frame_height)
        return self._add(Character(
            sprite, x, y, z, speed, tile_height=self.tile_height, is_npc=True,
            behavior=behavior, collision_width=collision_width,
            collision_depth=collision_depth, collision_height=collision_height))

    def _create_npc_sized(self, path, x, y, z, speed, behavior, collision_size: CollisionSize):
        cw, cd, ch = collision_size or (None, None, None)
        return self.create_npc(path, x, y, z, speed, behavior,
                               collision_width=cw, collision_depth=cd, collision_height=ch)

    def create_npc_wanderer(self, spritesheet_path: str, x: float, y: float, z: float = 0.0,
                            radius: float = 200.0, speed: float = 60.0,
                            collision_size: CollisionSize = None) -> Character:
        npc = self._create_npc_sized(spritesheet_path, x, y, z, speed,
                                     NPCBehavior.WANDER, collision_size)
        npc.wander_radius = radius
        npc.home_x, npc.home_y = x, y
        return npc

    def create_npc_patrol(self, spritesheet_path: str,
                          points: Sequence[Tuple[float, float]] = None,
                          patrol_points: Sequence[Tuple[float, float]] = None,
                          z: float = 0.0, speed: float = 70.0,
                          collision_size: CollisionSize = None) -> Character:
        """NPC que recorre ``points`` en bucle, empezando en el primero.

        ``patrol_points`` es un alias antiguo de ``points``.
        """
        points = points or patrol_points
        if not points:
            raise ValueError("Patrol points cannot be empty")
        x, y = points[0]
        npc = self._create_npc_sized(spritesheet_path, x, y, z, speed,
                                     NPCBehavior.PATROL, collision_size)
        npc.set_patrol_points(points)
        return npc

    def create_npc_follower(self, spritesheet_path: str, target: Character,
                            x: float = 0.0, y: float = 0.0, z: float = 0.0,
                            speed: float = 90.0,
                            collision_size: CollisionSize = None) -> Character:
        npc = self._create_npc_sized(spritesheet_path, x, y, z, speed,
                                     NPCBehavior.FOLLOW, collision_size)
        npc.target = target
        return npc

    # --- Por frame ---

    def update(self, dt: float):
        for character in self.characters:
            character.update(dt)

    def collect_render_data(self, level_height_offset: float = 0,
                            max_z: Optional[int] = None) -> Dict[object, np.ndarray]:
        """``{textura: vértices}`` de los personajes hasta el nivel ``max_z`` (None = todos),
        en coordenadas de mundo.

        Los personajes que comparten spritesheet comparten textura, así que salen en
        el mismo lote. No hace falta ordenar: el depth buffer decide qué queda delante.
        """
        from ..renderer.sprite_batch import quad_vertices

        groups = defaultdict(list)
        for c in self.characters:
            if max_z is not None and int(c.z) > max_z:
                continue
            texture, region = c.get_region()
            rx, ry = c.get_render_position()
            groups[texture].append((rx, ry - c.z * level_height_offset,
                                    region.width, region.height, c.get_depth(), *region.uv))
        return {texture: quad_vertices(*np.asarray(rows, dtype=np.float32).T)
                for texture, rows in groups.items()}

    # --- Consultas ---

    def get_characters_at(self, x: float, y: float, radius: float = 32.0) -> List[Character]:
        r2 = radius * radius
        return [c for c in self.characters if (c.x - x) ** 2 + (c.y - y) ** 2 <= r2]

    @property
    def npc_count(self) -> int:
        return sum(1 for c in self.characters if c.is_npc)

    @property
    def character_count(self) -> int:
        return len(self.characters)
