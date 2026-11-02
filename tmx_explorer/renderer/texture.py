"""Texturas OpenGL a partir de imágenes PIL."""

import numpy as np
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE, GL_NEAREST, GL_RGBA, GL_TEXTURE0, GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_UNSIGNED_BYTE, glActiveTexture, glBindTexture,
    glDeleteTextures, glGenTextures, glTexImage2D, glTexParameteri,
)
from PIL import Image

from ..config import TEXTURE_BORDER


class Texture:
    """Textura 2D RGBA con filtrado NEAREST (pixel art).

    Si se crea con borde, la imagen se extruye ``border`` píxeles por cada lado
    (repitiendo el píxel del borde) para evitar líneas entre tiles al hacer zoom.
    ``width``/``height`` incluyen ese borde; ``content_width``/``content_height`` no.
    """

    def __init__(self, width: int, height: int, data: bytes, border: int = 0):
        self.width = width
        self.height = height
        self.border = border
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data)
        glBindTexture(GL_TEXTURE_2D, 0)

    @classmethod
    def from_pil(cls, image: Image.Image, add_border: bool = True) -> "Texture":
        pixels = np.asarray(image.convert("RGBA"))
        border = TEXTURE_BORDER if add_border else 0
        if border:
            pixels = np.pad(pixels, ((border, border), (border, border), (0, 0)), mode="edge")
        pixels = np.ascontiguousarray(pixels[::-1])   # OpenGL empieza por abajo
        h, w = pixels.shape[:2]
        return cls(w, h, pixels.tobytes(), border)

    @classmethod
    def from_file(cls, filepath: str, add_border: bool = True) -> "Texture":
        with Image.open(filepath) as image:
            return cls.from_pil(image, add_border)

    @property
    def content_width(self) -> int:
        return self.width - 2 * self.border

    @property
    def content_height(self) -> int:
        return self.height - 2 * self.border

    def uv_rect(self) -> tuple:
        """(u_min, v_min, u_max, v_max) del contenido, sin el borde extruido."""
        u0 = self.border / self.width
        v0 = self.border / self.height
        return u0, v0, 1.0 - u0, 1.0 - v0

    def bind(self, slot: int = 0):
        glActiveTexture(GL_TEXTURE0 + slot)
        glBindTexture(GL_TEXTURE_2D, self.id)

    def delete(self):
        """Libera la textura en la GPU. Llamar mientras el contexto GL siga vivo."""
        if self.id:
            glDeleteTextures([self.id])
            self.id = 0
