"""
Character/Entity manager for handling multiple characters
"""

from typing import List, Dict, Optional, TYPE_CHECKING
from .character import Character
from .sprite import AnimatedSprite, Direction

if TYPE_CHECKING:
    from ..renderer.opengl_renderer import OpenGLRenderer


class EntityManager:
    """
    Manages all characters/entities in the scene.
    Handles updates and rendering data collection.
    """
    
    def __init__(self):
        self.characters: List[Character] = []
        self.player: Optional[Character] = None
        self._sprite_cache: Dict[str, AnimatedSprite] = {}

    def load_sprite(
        self,
        path: str,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        animation_speed: float = 8.0
    ) -> AnimatedSprite:
        """Load or get cached sprite"""
        cache_key = f"{path}_{frame_width}_{frame_height}"
        
        if cache_key not in self._sprite_cache:
            self._sprite_cache[cache_key] = AnimatedSprite(
                path, frame_width, frame_height, animation_speed
            )
        
        return self._sprite_cache[cache_key]

    def create_character(
        self,
        spritesheet_path: str,
        x: float = 0.0,
        y: float = 0.0,
        z: int = 0,
        speed: float = 100.0,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        is_player: bool = False
    ) -> Character:
        """Create a new character"""
        sprite = self.load_sprite(spritesheet_path, frame_width, frame_height)
        character = Character(sprite, x, y, z, speed)
        
        self.characters.append(character)
        
        if is_player:
            self.player = character
        
        return character

    def update(self, dt: float):
        """Update all characters"""
        for character in self.characters:
            character.update(dt)

    def collect_render_data(
        self,
        renderer: 'OpenGLRenderer',
        level_height_offset: int = 0
    ) -> List[tuple]:
        """
        Collect render data for all visible characters.
        
        Returns list of (texture, x, y, w, h, depth) tuples.
        """
        render_data = []
        
        for character in self.characters:
            texture = character.get_texture(renderer)
            rx, ry = character.get_render_position()
            
            # Apply Z level offset (same as tiles)
            ry -= character.z * level_height_offset
            
            depth = character.get_depth()
            
            render_data.append((
                texture,
                rx, ry,
                character.width, character.height,
                depth
            ))
        
        return render_data

    def remove_character(self, character: Character):
        """Remove a character from the manager"""
        if character in self.characters:
            self.characters.remove(character)
        if character is self.player:
            self.player = None

    def clear(self):
        """Remove all characters"""
        self.characters.clear()
        self.player = None
