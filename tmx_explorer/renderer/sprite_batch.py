"""Acumula quads de una misma textura y los dibuja en una sola draw call.

Uso: ``begin(texture)`` -> ``add_sprite(...)`` x N -> ``flush()``.

Cada sprite son 4 vértices + 6 índices; cada vértice son 9 floats:
``[x, y, u, v, r, g, b, a, depth]``.
"""

import ctypes
from typing import Optional, Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_ELEMENT_ARRAY_BUFFER, GL_FALSE, GL_FLOAT,
    GL_STATIC_DRAW, GL_TRIANGLES, GL_UNSIGNED_INT, glBindBuffer, glBindVertexArray,
    glBufferData, glBufferSubData, glDrawElements, glEnableVertexAttribArray,
    glGenBuffers, glGenVertexArrays, glVertexAttribPointer,
)

from .texture import Texture

FLOAT_SIZE = 4


class SpriteBatch:
    FLOATS_PER_VERTEX = 9
    VERTICES_PER_SPRITE = 4
    INDICES_PER_SPRITE = 6
    FLOATS_PER_SPRITE = FLOATS_PER_VERTEX * VERTICES_PER_SPRITE

    def __init__(self, max_sprites: int = 20000):
        self.max_sprites = max_sprites
        self.sprite_count = 0
        self.vertices = np.zeros(max_sprites * self.FLOATS_PER_SPRITE, dtype=np.float32)
        self.indices = self._create_indices(max_sprites)
        self.current_texture: Optional[Texture] = None
        self._setup_buffers()

    @staticmethod
    def _create_indices(max_sprites: int) -> np.ndarray:
        """Dos triángulos por quad: (0, 1, 2) y (0, 2, 3)."""
        base = np.arange(max_sprites, dtype=np.uint32)[:, None] * 4
        return (base + np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)).ravel()

    def _setup_buffers(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, None, GL_DYNAMIC_DRAW)

        stride = self.FLOATS_PER_VERTEX * FLOAT_SIZE
        # (location, componentes, offset en floats) -> coincide con VERTEX_SHADER
        for location, size, offset in ((0, 2, 0), (1, 2, 2), (2, 4, 4), (3, 1, 8)):
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, stride,
                                  ctypes.c_void_p(offset * FLOAT_SIZE))

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

    def begin(self, texture: Texture):
        self.sprite_count = 0
        self.current_texture = texture

    @property
    def is_full(self) -> bool:
        return self.sprite_count >= self.max_sprites

    def add_sprite(self, x: float, y: float, width: float, height: float,
                   depth: float = 0.0,
                   color: Tuple[float, float, float, float] = (1, 1, 1, 1)) -> bool:
        """Añade un quad en coordenadas de pantalla. Devuelve False si el batch está lleno.

        Las UV salen de la textura actual (excluyen su borde extruido), así que no
        dependen del tamaño en pantalla ni del zoom.
        """
        if self.is_full:
            return False

        u0, v0, u1, v1 = self.current_texture.uv_rect() if self.current_texture else (0, 0, 1, 1)
        r, g, b, a = color
        x1, y1 = x + width, y + height

        i = self.sprite_count * self.FLOATS_PER_SPRITE
        # La textura está volteada en vertical, así que la y de arriba usa v1
        self.vertices[i:i + self.FLOATS_PER_SPRITE] = (
            x,  y,  u0, v1, r, g, b, a, depth,
            x1, y,  u1, v1, r, g, b, a, depth,
            x1, y1, u1, v0, r, g, b, a, depth,
            x,  y1, u0, v0, r, g, b, a, depth,
        )
        self.sprite_count += 1
        return True

    def flush(self):
        if self.sprite_count == 0:
            return
        if self.current_texture:
            self.current_texture.bind(0)

        active = self.vertices[:self.sprite_count * self.FLOATS_PER_SPRITE]
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, active.nbytes, active)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.sprite_count * self.INDICES_PER_SPRITE,
                       GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        self.sprite_count = 0
