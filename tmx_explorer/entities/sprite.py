"""
Animated sprite system for character spritesheets

Spritesheet format (4x4 grid):
- Columns: 0=idle, 1-3=walk animation
- Rows: 0=down, 1=left, 2=up, 3=right
"""

import pygame
from enum import IntEnum
from typing import Optional, List, Dict, Tuple
from pathlib import Path


class Direction(IntEnum):
    """Character facing direction (matches spritesheet row order)"""
    DOWN = 0   # Hacia cámara (fila 0)
    LEFT = 1   # Hacia izquierda (fila 1)
    RIGHT = 2  # Hacia derecha (fila 2) - era UP
    UP = 3     # Hacia lejos/arriba en pantalla (fila 3) - era RIGHT


class AnimationState(IntEnum):
    """Animation states"""
    IDLE = 0
    WALKING = 1


class AnimatedSprite:
    """
    Animated sprite from a 4x4 spritesheet.
    
    Args:
        spritesheet_path: Path to the spritesheet image
        frame_width: Width of each frame (default: auto-detect as sheet_width/4)
        frame_height: Height of each frame (default: auto-detect as sheet_height/4)
        animation_speed: Frames per second for animation
    """
    
    COLS = 4  # Idle + 3 walk frames
    ROWS = 4  # 4 directions
    
    def __init__(
        self,
        spritesheet_path: str,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        animation_speed: float = 8.0
    ):
        self.spritesheet = pygame.image.load(spritesheet_path).convert_alpha()
        
        # Auto-detect frame size if not provided
        sheet_w = self.spritesheet.get_width()
        sheet_h = self.spritesheet.get_height()
        
        self.frame_width = frame_width or (sheet_w // self.COLS)
        self.frame_height = frame_height or (sheet_h // self.ROWS)
        
        # Animation settings
        self.animation_speed = animation_speed  # FPS
        self.animation_timer = 0.0
        self.current_frame = 0
        
        # State
        self.direction = Direction.DOWN
        self.state = AnimationState.IDLE
        
        # Pre-cut all frames
        self.frames: Dict[Direction, List[pygame.Surface]] = {}
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
                
                frame = pygame.Surface(
                    (self.frame_width, self.frame_height),
                    pygame.SRCALPHA
                )
                frame.blit(
                    self.spritesheet, (0, 0),
                    pygame.Rect(x, y, self.frame_width, self.frame_height)
                )
                self.frames[direction].append(frame)

    def update(self, dt: float):
        """
        Update animation state.
        
        Args:
            dt: Delta time in seconds
        """
        if self.state == AnimationState.WALKING:
            self.animation_timer += dt * self.animation_speed
            
            # Cycle through walk frames (1, 2, 3) continuously
            frames_to_advance = int(self.animation_timer)
            if frames_to_advance > 0:
                self.animation_timer -= frames_to_advance
                self.current_frame += frames_to_advance
                # Wrap around: 1 -> 2 -> 3 -> 1 -> 2 -> 3...
                while self.current_frame > 3:
                    self.current_frame -= 3  # 4->1, 5->2, 6->3, 7->1...
        else:
            # Idle: always frame 0
            self.current_frame = 0
            self.animation_timer = 0.0

    def set_direction(self, direction: Direction):
        """Set facing direction"""
        self.direction = direction

    def set_walking(self, walking: bool):
        """Set walking state"""
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

    def get_current_frame(self) -> pygame.Surface:
        """Get the current animation frame"""
        return self.frames[self.direction][self.current_frame]

    def get_frame(self, direction: Direction, frame_index: int) -> pygame.Surface:
        """Get a specific frame"""
        return self.frames[direction][frame_index]

    @property
    def width(self) -> int:
        return self.frame_width
    
    @property
    def height(self) -> int:
        return self.frame_height
