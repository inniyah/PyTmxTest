"""Construcción vectorizada de quads y su dibujo por lotes.

Cada quad son 4 vértices (arriba-izq, arriba-der, abajo-der, abajo-izq) de 9 floats:
``[x, y, u, v, r, g, b, a, depth]``. ``quad_vertices`` genera N quads de una vez con
numpy; ``SpriteBatch.draw`` los sube y dibuja en una sola draw call por textura
(o varias si superan ``max_sprites``).
"""

import ctypes
from typing import Optional

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_ELEMENT_ARRAY_BUFFER, GL_FALSE, GL_FLOAT,
    GL_STATIC_DRAW, GL_TRIANGLES, GL_UNSIGNED_INT, glBindBuffer, glBindVertexArray,
    glBufferData, glBufferSubData, glDrawElements, glEnableVertexAttribArray,
    glGenBuffers, glGenVertexArrays, glVertexAttribPointer,
)

from ..config import GID_FLIP_D, GID_FLIP_H, GID_FLIP_V

FLOAT_SIZE = 4
FLOATS_PER_VERTEX = 9
VERTICES_PER_QUAD = 4
INDICES_PER_QUAD = 6

# Bits de flip tal como los guarda Map3DStructure.flips (GID >> 28)
FLIP_H = GID_FLIP_H >> 28
FLIP_V = GID_FLIP_V >> 28
FLIP_D = GID_FLIP_D >> 28

# Coordenadas normalizadas (s, t) de cada esquina, con t creciendo hacia abajo
_CORNER_S = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
_CORNER_T = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)


def quad_vertices(x, y, w, h, depth, u0, v0, u1, v1,
                  flips: Optional[np.ndarray] = None, color=(1.0, 1.0, 1.0, 1.0)) -> np.ndarray:
    """Vértices de N quads como array float32 (N*4, 9).

    Los argumentos posicionales son escalares o arrays de longitud N.
    ``u0, v0, u1, v1`` delimitan el contenido en la textura, con v1 = borde superior
    (convención de ``Texture.uv_rect`` y ``AtlasRegion``).
    ``flips`` usa los bits de Tiled (H=8, V=4, D=2). Según la especificación, primero
    se aplica la diagonal y después los volteos horizontal y vertical. Con diagonal,
    el llamador debe pasar w y h ya intercambiados.
    """
    x, y, w, h, depth, u0, v0, u1, v1 = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(a, dtype=np.float32))
          for a in (x, y, w, h, depth, u0, v0, u1, v1)))
    n = x.size
    out = np.empty((n, VERTICES_PER_QUAD, FLOATS_PER_VERTEX), dtype=np.float32)

    out[:, :, 0] = x[:, None] + _CORNER_S * w[:, None]
    out[:, :, 1] = y[:, None] + _CORNER_T * h[:, None]

    s = np.broadcast_to(_CORNER_S, (n, 4))
    t = np.broadcast_to(_CORNER_T, (n, 4))
    if flips is not None and np.any(flips):
        # Para cada esquina mostrada, qué punto de la imagen original le toca:
        # se deshacen las transformaciones en orden inverso (V, luego H, luego diagonal)
        f = np.asarray(flips)[:, None]
        t = np.where(f & FLIP_V, 1.0 - t, t)
        s = np.where(f & FLIP_H, 1.0 - s, s)
        s, t = np.where(f & FLIP_D, t, s), np.where(f & FLIP_D, s, t)

    out[:, :, 2] = u0[:, None] + s * (u1 - u0)[:, None]
    out[:, :, 3] = v1[:, None] - t * (v1 - v0)[:, None]      # t = 0 (arriba) -> v1
    out[:, :, 4:8] = np.asarray(color, dtype=np.float32)
    out[:, :, 8] = depth[:, None]
    return out.reshape(-1, FLOATS_PER_VERTEX)


class SpriteBatch:
    """VAO/VBO/EBO reutilizables para dibujar arrays de quads."""

    def __init__(self, max_sprites: int = 20000):
        self.max_sprites = max_sprites
        self.draw_calls = 0                 # el renderer lo pone a 0 cada frame
        self._setup_buffers()

    def _setup_buffers(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.max_sprites * VERTICES_PER_QUAD * FLOATS_PER_VERTEX * FLOAT_SIZE,
                     None, GL_DYNAMIC_DRAW)

        stride = FLOATS_PER_VERTEX * FLOAT_SIZE
        # (location, componentes, offset en floats) -> coincide con VERTEX_SHADER
        for location, size, offset in ((0, 2, 0), (1, 2, 2), (2, 4, 4), (3, 1, 8)):
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, stride,
                                  ctypes.c_void_p(offset * FLOAT_SIZE))

        # Dos triángulos por quad: (0, 1, 2) y (0, 2, 3)
        base = np.arange(self.max_sprites, dtype=np.uint32)[:, None] * VERTICES_PER_QUAD
        indices = (base + np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)).ravel()
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

    def draw(self, texture, vertices: np.ndarray):
        """Dibuja los quads de ``vertices`` (salida de quad_vertices) con ``texture``."""
        quads = len(vertices) // VERTICES_PER_QUAD
        if quads == 0:
            return
        texture.bind(0)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        for start in range(0, quads, self.max_sprites):
            count = min(self.max_sprites, quads - start)
            chunk = np.ascontiguousarray(
                vertices[start * VERTICES_PER_QUAD:(start + count) * VERTICES_PER_QUAD])
            glBufferSubData(GL_ARRAY_BUFFER, 0, chunk.nbytes, chunk)
            glDrawElements(GL_TRIANGLES, count * INDICES_PER_QUAD, GL_UNSIGNED_INT, None)
            self.draw_calls += 1
        glBindVertexArray(0)
