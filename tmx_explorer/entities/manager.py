"""
Character/Entity manager for handling multiple characters
"""

from typing import List, Optional, TYPE_CHECKING
from .character import Character
from .sprite import AnimatedSprite

if TYPE_CHECKING:
    from ..renderer.opengl_renderer import OpenGLRenderer


class EntityManager:
    """Manages all characters/entities"""
    
    def __init__(self):
        self.characters: List[Character] = []
        self.player: Optional[Character] = None
        self._sprite_cache = {}

    def load_sprite(self, path: str, frame_width: int = None,
                    frame_height: int = None, animation_speed: float = 8.0) -> AnimatedSprite:
        cache_key = f"{path}_{frame_width}_{frame_height}"
        if cache_key not in self._sprite_cache:
            self._sprite_cache[cache_key] = AnimatedSprite(
                path, frame_width, frame_height, animation_speed
            )
        return self._sprite_cache[cache_key]

    def create_character(self, spritesheet_path: str,
                         x: float = 0.0, y: float = 0.0, z: int = 0,
                         speed: float = 100.0, frame_width: int = None,
                         frame_height: int = None, is_player: bool = False) -> Character:
        sprite = self.load_sprite(spritesheet_path, frame_width, frame_height)
        character = Character(sprite, x, y, z, speed)
        
        self.characters.append(character)
        if is_player:
            self.player = character
        
        return character

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
