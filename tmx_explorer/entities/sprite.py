"""Sprites animados a partir de un spritesheet 4x4 (estilo RPG Maker).

Filas = dirección (abajo, izquierda, derecha, arriba); columnas = frames.
La columna 0 es el frame en reposo; al caminar se cicla 1 -> 2 -> 3.

Las instancias creadas con ``from_cached`` comparten los frames PIL *y* las
texturas de GPU con la original; cada una solo tiene su propio estado de
animación (dirección, frame, temporizador).
"""

import logging
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

from PIL import Image

from ..config import ANIMATION_FPS, SPRITE_COLS, SPRITE_ROWS

if TYPE_CHECKING:
    from ..renderer.texture import Texture

log = logging.getLogger(__name__)


class Direction(IntEnum):
    DOWN = 0
    LEFT = 1
    RIGHT = 2
    UP = 3


class AnimationState(IntEnum):
    IDLE = 0
    WALKING = 1


IDLE_FRAME = 0
FIRST_WALK_FRAME = 1
LAST_WALK_FRAME = SPRITE_COLS - 1


class AnimatedSprite:
    COLS = SPRITE_COLS
    ROWS = SPRITE_ROWS

    def __init__(self, spritesheet_path: str, frame_width: int = None,
                 frame_height: int = None, animation_speed: float = ANIMATION_FPS):
        with Image.open(spritesheet_path) as img:
            self.spritesheet = img.convert("RGBA")
        self.frame_width = frame_width or self.spritesheet.width // self.COLS
        self.frame_height = frame_height or self.spritesheet.height // self.ROWS

        # Datos compartidos entre instancias (ver from_cached)
        self.frames: Dict[Direction, List[Image.Image]] = self._cut_frames()
        self.textures: Dict[Tuple[Direction, int], "Texture"] = {}

        self._reset_animation(animation_speed)
        log.info("Spritesheet %s: %dx%d por frame", Path(spritesheet_path).name,
                 self.frame_width, self.frame_height)

    @classmethod
    def from_cached(cls, cached: "AnimatedSprite",
                    animation_speed: float = ANIMATION_FPS) -> "AnimatedSprite":
        """Nueva instancia que comparte imagen, frames y texturas con ``cached``."""
        instance = object.__new__(cls)
        instance.spritesheet = cached.spritesheet
        instance.frame_width = cached.frame_width
        instance.frame_height = cached.frame_height
        instance.frames = cached.frames
        instance.textures = cached.textures      # mismo dict: una sola copia en GPU
        instance._reset_animation(animation_speed)
        return instance

    def _reset_animation(self, animation_speed: float):
        self.animation_speed = animation_speed
        self.animation_timer = 0.0
        self.current_frame = IDLE_FRAME
        self.direction = Direction.DOWN
        self.state = AnimationState.IDLE

    def _cut_frames(self) -> Dict[Direction, List[Image.Image]]:
        fw, fh = self.frame_width, self.frame_height
        return {
            direction: [self.spritesheet.crop((col * fw, direction * fh,
                                               (col + 1) * fw, (direction + 1) * fh))
                        for col in range(self.COLS)]
            for direction in Direction
        }

    # --- Animación ---

    def update(self, dt: float):
        if self.state != AnimationState.WALKING:
            self.current_frame = IDLE_FRAME
            self.animation_timer = 0.0
            return

        self.animation_timer += dt * self.animation_speed
        steps = int(self.animation_timer)
        if steps > 0:
            self.animation_timer -= steps
            cycle = LAST_WALK_FRAME - FIRST_WALK_FRAME + 1
            self.current_frame = (FIRST_WALK_FRAME
                                  + (self.current_frame - FIRST_WALK_FRAME + steps) % cycle)

    def set_direction(self, direction: Direction):
        self.direction = direction

    def set_walking(self, walking: bool):
        new_state = AnimationState.WALKING if walking else AnimationState.IDLE
        if new_state != self.state:
            self.state = new_state
            self.current_frame = FIRST_WALK_FRAME if walking else IDLE_FRAME
            self.animation_timer = 0.0

    # --- Acceso a frames ---

    def get_current_frame(self) -> Image.Image:
        return self.frames[self.direction][self.current_frame]

    def get_frame(self, direction: Direction, frame_index: int) -> Image.Image:
        return self.frames[direction][frame_index]

    def get_texture(self) -> "Texture":
        """Textura GPU del frame actual. Se crean la primera vez y se comparten."""
        if not self.textures:
            from ..renderer.texture import Texture
            for direction, frames in self.frames.items():
                for index, frame in enumerate(frames):
                    self.textures[direction, index] = Texture.from_pil(frame)
        return self.textures[self.direction, self.current_frame]

    @property
    def width(self) -> int:
        return self.frame_width

    @property
    def height(self) -> int:
        return self.frame_height
