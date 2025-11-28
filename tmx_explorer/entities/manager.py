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
    
    def __init__(self, tile_height: int = 32):
        self.characters: List[Character] = []
        self.player: Optional[Character] = None
        self.tile_height = tile_height
        self._sprite_cache = {}

    def load_sprite(self, path: str, frame_width: int = None,
                    frame_height: int = None, animation_speed: float = 8.0) -> AnimatedSprite:
        """Load sprite - cachea los frames pero cada personaje tiene su propio estado"""
        cache_key = f"{path}_{frame_width}_{frame_height}"
        
        if cache_key in self._sprite_cache:
            # Crear nueva instancia que comparte los frames cacheados
            cached = self._sprite_cache[cache_key]
            return AnimatedSprite.from_cached(cached, animation_speed)
        else:
            # Primera carga - crear y cachear
            sprite = AnimatedSprite(path, frame_width, frame_height, animation_speed)
            self._sprite_cache[cache_key] = sprite
            return sprite

    def create_character(self, spritesheet_path: str,
                         x: float = 0.0, y: float = 0.0, z: float = 0.0,
                         speed: float = 100.0, frame_width: int = None,
                         frame_height: int = None, is_player: bool = False) -> Character:
        """Create player character"""
        sprite = self.load_sprite(spritesheet_path, frame_width, frame_height)
        character = Character(
            sprite, x, y, z, speed,
            tile_height=self.tile_height,
            is_npc=False
        )
        
        self.characters.append(character)
        if is_player:
            self.player = character
        
        return character

    def create_npc(self, spritesheet_path: str,
                   x: float = 0.0, y: float = 0.0, z: float = 0.0,
                   speed: float = 80.0,
                   behavior: NPCBehavior = NPCBehavior.WANDER,
                   frame_width: int = None,
                   frame_height: int = None) -> Character:
        """Create NPC character"""
        sprite = self.load_sprite(spritesheet_path, frame_width, frame_height)
        npc = Character(
            sprite, x, y, z, speed,
            tile_height=self.tile_height,
            is_npc=True,
            behavior=behavior
        )
        
        self.characters.append(npc)
        return npc

    def create_npc_wanderer(self, spritesheet_path: str,
                            x: float, y: float, z: float = 0.0,
                            radius: float = 200.0, speed: float = 60.0) -> Character:
        """Create NPC that wanders around a point"""
        npc = self.create_npc(spritesheet_path, x, y, z, speed, NPCBehavior.WANDER)
        npc.wander_radius = radius
        npc.home_x = x
        npc.home_y = y
        return npc

    def create_npc_patrol(self, spritesheet_path: str,
                          patrol_points: List[Tuple[float, float]],
                          z: float = 0.0, speed: float = 70.0) -> Character:
        """Create NPC that patrols between points"""
        if not patrol_points:
            return None
        x, y = patrol_points[0]
        npc = self.create_npc(spritesheet_path, x, y, z, speed, NPCBehavior.PATROL)
        npc.set_patrol_points(patrol_points)
        return npc

    def create_npc_follower(self, spritesheet_path: str,
                            target: Character,
                            x: float = 0.0, y: float = 0.0, z: float = 0.0,
                            speed: float = 90.0) -> Character:
        """Create NPC that follows another character"""
        npc = self.create_npc(spritesheet_path, x, y, z, speed, NPCBehavior.FOLLOW)
        npc.target = target
        return npc

    def update(self, dt: float):
        for character in self.characters:
            character.update(dt)

    def collect_render_data(self, renderer: 'OpenGLRenderer',
                           level_height_offset: int = 0) -> list:
        render_data = []
        
        for character in self.characters:
            texture = character.get_texture(renderer)
            rx, ry = character.get_render_position()
            ry -= character.z * level_height_offset
            depth = character.get_depth()
            
            render_data.append((
                texture, rx, ry,
                character.width, character.height,
                depth
            ))
        
        return render_data
    
    @property
    def npc_count(self) -> int:
        return sum(1 for c in self.characters if c.is_npc)
