"""
Entity Manager - Manages all characters and NPCs
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
from .character import Character, NPCBehavior
from .sprite import AnimatedSprite

if TYPE_CHECKING:
    from ..renderer.opengl_renderer import OpenGLRenderer


class EntityManager:
    """Manages all characters/entities"""
    
    def __init__(self, tile_width: int = 32, tile_height: int = 32, 
                 collision_map = None, max_z: int = 10):
        self.characters: List[Character] = []
        self.player: Optional[Character] = None
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.collision_map = collision_map
        self.max_z = max_z
        self._sprite_cache = {}

    def load_sprite(self, path: str, frame_width: int = None,
                    frame_height: int = None, animation_speed: float = 8.0) -> AnimatedSprite:
        cache_key = f"{path}_{frame_width}_{frame_height}"
        
        if cache_key in self._sprite_cache:
            cached = self._sprite_cache[cache_key]
            return AnimatedSprite.from_cached(cached, animation_speed)
        else:
            sprite = AnimatedSprite(path, frame_width, frame_height, animation_speed)
            self._sprite_cache[cache_key] = sprite
            return sprite

    def _setup_character(self, character: Character):
        """Configura las referencias de colisión y límites para un personaje"""
        character.collision_map = self.collision_map
        character.tile_width = self.tile_width
        character.min_z = 0.0
        character.max_z = float(self.max_z - 1)

    def create_character(self, spritesheet_path: str,
                         x: float = 0.0, y: float = 0.0, z: float = 0.0,
                         speed: float = 100.0, frame_width: int = None,
                         frame_height: int = None, is_player: bool = False,
                         collision_width: float = None,
                         collision_depth: float = None,
                         collision_height: float = None) -> Character:
        """
        Create player character.
        
        Args:
            collision_width: Ancho de colisión en tiles (default 0.5 = 50cm)
            collision_depth: Profundidad de colisión en tiles (default 0.5 = 50cm)
            collision_height: Altura de colisión en niveles (default 0.85 = 1.7m)
        """
        sprite = self.load_sprite(spritesheet_path, frame_width, frame_height)
        character = Character(
            sprite, x, y, z, speed,
            tile_height=self.tile_height,
            is_npc=False,
            collision_width=collision_width,
            collision_depth=collision_depth,
            collision_height=collision_height
        )
        
        self._setup_character(character)
        
        self.characters.append(character)
        if is_player:
            self.player = character
        
        return character

    def create_npc(self, spritesheet_path: str,
                   x: float = 0.0, y: float = 0.0, z: float = 0.0,
                   speed: float = 80.0,
                   behavior: NPCBehavior = NPCBehavior.WANDER,
                   frame_width: int = None,
                   frame_height: int = None,
                   collision_width: float = None,
                   collision_depth: float = None,
                   collision_height: float = None) -> Character:
        """Create NPC character with optional custom collision size"""
        sprite = self.load_sprite(spritesheet_path, frame_width, frame_height)
        npc = Character(
            sprite, x, y, z, speed,
            tile_height=self.tile_height,
            is_npc=True,
            behavior=behavior,
            collision_width=collision_width,
            collision_depth=collision_depth,
            collision_height=collision_height
        )
        
        self._setup_character(npc)
        
        self.characters.append(npc)
        return npc

    def create_npc_wanderer(self, spritesheet_path: str,
                            x: float, y: float, z: float = 0.0,
                            radius: float = 200.0, speed: float = 60.0,
                            collision_size: Tuple[float, float, float] = None) -> Character:
        """
        Create NPC that wanders around a point.
        
        Args:
            collision_size: Tuple (width, depth, height) en unidades de tile
        """
        cw, cd, ch = collision_size if collision_size else (None, None, None)
        npc = self.create_npc(spritesheet_path, x, y, z, speed, NPCBehavior.WANDER,
                              collision_width=cw, collision_depth=cd, collision_height=ch)
        npc.wander_radius = radius
        npc.home_x = x
        npc.home_y = y
        return npc

    def create_npc_patrol(self, spritesheet_path: str,
                          points: List[Tuple[float, float]] = None,
                          patrol_points: List[Tuple[float, float]] = None,
                          z: float = 0.0, speed: float = 70.0,
                          collision_size: Tuple[float, float, float] = None) -> Character:
        """Create NPC that patrols between points"""
        # Aceptar tanto 'points' como 'patrol_points' para compatibilidad
        actual_points = points or patrol_points
        if not actual_points:
            raise ValueError("Patrol points cannot be empty")
        
        cw, cd, ch = collision_size if collision_size else (None, None, None)
        npc = self.create_npc(
            spritesheet_path, actual_points[0][0], actual_points[0][1], z,
            speed, NPCBehavior.PATROL,
            collision_width=cw, collision_depth=cd, collision_height=ch
        )
        npc.set_patrol_points(actual_points)
        return npc

    def create_npc_follower(self, spritesheet_path: str,
                            target: Character,
                            x: float = 0.0, y: float = 0.0, z: float = 0.0,
                            speed: float = 90.0,
                            collision_size: Tuple[float, float, float] = None) -> Character:
        """Create NPC that follows a target"""
        cw, cd, ch = collision_size if collision_size else (None, None, None)
        npc = self.create_npc(spritesheet_path, x, y, z, speed, NPCBehavior.FOLLOW,
                              collision_width=cw, collision_depth=cd, collision_height=ch)
        npc.target = target
        return npc

    def update(self, dt: float):
        for character in self.characters:
            character.update(dt)

    def collect_render_data(self, renderer: 'OpenGLRenderer', 
                            level_height_offset: int) -> List[Tuple]:
        render_data = []
        
        for character in self.characters:
            texture = character.get_texture(renderer)
            rx, ry = character.get_render_position()
            
            level_y_offset = character.z * level_height_offset
            ry -= level_y_offset
            
            depth = character.get_depth()
            
            render_data.append((
                texture,
                rx, ry,
                character.width, character.height,
                depth
            ))
        
        render_data.sort(key=lambda x: x[5])
        
        return render_data

    def get_characters_at(self, x: float, y: float, radius: float = 32.0) -> List[Character]:
        result = []
        for char in self.characters:
            dx = char.x - x
            dy = char.y - y
            if dx*dx + dy*dy <= radius*radius:
                result.append(char)
        return result

    @property
    def npc_count(self) -> int:
        """Número de NPCs (personajes que no son el jugador)"""
        return sum(1 for c in self.characters if c.is_npc)
    
    @property
    def character_count(self) -> int:
        """Número total de personajes"""
        return len(self.characters)
