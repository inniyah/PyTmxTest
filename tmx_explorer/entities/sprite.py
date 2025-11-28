"""
Animated sprite system for character spritesheets

Spritesheet format (4x4 grid):
- Columns: 0=idle, 1-3=walk animation
- Rows: 0=down, 1=left, 2=up, 3=right
"""

from PIL import Image
from enum import IntEnum
from typing import List, Dict
from pathlib import Path


class Direction(IntEnum):
    """Character facing direction (matches spritesheet row order)"""
    DOWN = 0
    LEFT = 1
    RIGHT = 2
    UP = 3


class AnimationState(IntEnum):
    """Animation states"""
    IDLE = 0
    WALKING = 1


class AnimatedSprite:
    """Animated sprite from a 4x4 spritesheet (PIL version)"""
    
    COLS = 4
    ROWS = 4
    
    def __init__(self, spritesheet_path: str,
                 frame_width: int = None, frame_height: int = None,
                 animation_speed: float = 8.0):
        
        self.spritesheet = Image.open(spritesheet_path).convert('RGBA')
        
        sheet_w = self.spritesheet.width
        sheet_h = self.spritesheet.height
        
        self.frame_width = frame_width or (sheet_w // self.COLS)
        self.frame_height = frame_height or (sheet_h // self.ROWS)
        
        self.animation_speed = animation_speed
        self.animation_timer = 0.0
        self.current_frame = 0
        
        self.direction = Direction.DOWN
        self.state = AnimationState.IDLE
        
        # Pre-cut all frames
        self.frames: Dict[Direction, List[Image.Image]] = {}
        self._cut_frames()
        
        print(f"Loaded spritesheet: {Path(spritesheet_path).name} "
              f"({self.frame_width}x{self.frame_height} per frame)")

    def _cut_frames(self):
        """Pre-cut all frames from spritesheet"""
        for direction in Direction:
            self.frames[direction] = []
            row = direction.value
            
            for col in range(self.COLS):
                x = col * self.frame_width
                y = row * self.frame_height
                
                frame = self.spritesheet.crop((
                    x, y, x + self.frame_width, y + self.frame_height
                ))
                self.frames[direction].append(frame)

    def update(self, dt: float):
        """Update animation state"""
        if self.state == AnimationState.WALKING:
            self.animation_timer += dt * self.animation_speed
            frames_to_advance = int(self.animation_timer)
            if frames_to_advance > 0:
                self.animation_timer -= frames_to_advance
                self.current_frame += frames_to_advance
                while self.current_frame > 3:
                    self.current_frame -= 3
        else:
            self.current_frame = 0
            self.animation_timer = 0.0

    def set_direction(self, direction: Direction):
        self.direction = direction

    def set_walking(self, walking: bool):
        if walking:
            if self.state != AnimationState.WALKING:
                self.state = AnimationState.WALKING
                self.current_frame = 1
                self.animation_timer = 0.0
        else:
            if self.state != AnimationState.IDLE:
                self.state = AnimationState.IDLE
                self.current_frame = 0
                self.animation_timer = 0.0

    def get_current_frame(self) -> Image.Image:
        return self.frames[self.direction][self.current_frame]

    def get_frame(self, direction: Direction, frame_index: int) -> Image.Image:
        return self.frames[direction][frame_index]

    @property
    def width(self) -> int:
        return self.frame_width
    
    @property
    def height(self) -> int:
        return self.frame_height
