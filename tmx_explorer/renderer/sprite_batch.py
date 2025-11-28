"""
Batched sprite rendering for efficient tile drawing (GLFW version)
"""

import ctypes
import numpy as np
from OpenGL.GL import *
from typing import Optional, Tuple
from .texture import Texture


class SpriteBatch:
    """Efficient batched sprite renderer"""
    
    FLOATS_PER_VERTEX = 9  # x, y, u, v, r, g, b, a, depth
    VERTICES_PER_SPRITE = 4
    INDICES_PER_SPRITE = 6
    
    def __init__(self, max_sprites: int = 20000):
        self.max_sprites = max_sprites
        self.sprite_count = 0
        self.vertices = np.zeros(
            max_sprites * self.VERTICES_PER_SPRITE * self.FLOATS_PER_VERTEX,
            dtype=np.float32
        )
        self.indices = self._create_indices(max_sprites)
        self.current_texture: Optional[Texture] = None
        
        self._setup_buffers()

    def _create_indices(self, max_sprites: int) -> np.ndarray:
        """Pre-generate index buffer for all possible sprites"""
        indices = []
        for i in range(max_sprites):
            offset = i * 4
            indices.extend([
                offset + 0, offset + 1, offset + 2,
                offset + 0, offset + 2, offset + 3
            ])
        return np.array(indices, dtype=np.uint32)

    def _setup_buffers(self):
        """Initialize OpenGL buffers"""
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, None, GL_DYNAMIC_DRAW)

        stride = self.FLOATS_PER_VERTEX * 4  # 4 bytes per float
        
        # Position (vec2)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # TexCoord (vec2)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        # Color (vec4)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        # Depth (float)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

    def begin(self, texture: Texture):
        """Start a new batch with the given texture"""
        self.sprite_count = 0
        self.current_texture = texture

    def add_sprite(
        self,
        x: float, y: float,
        width: float, height: float,
        depth: float = 0.0,
        color: Tuple[float, float, float, float] = (1, 1, 1, 1),
        border: int = 1
    ) -> bool:
        """Add a sprite to the current batch. Returns False if batch is full."""
        if self.sprite_count >= self.max_sprites:
            return False

        # Account for border in texture (for tile bleeding fix)
        total_width = width + border * 2
        total_height = height + border * 2

        # UV coordinates mapping to center of bordered texture
        u_min = border / total_width
        v_min = border / total_height
        u_max = (border + width) / total_width
        v_max = (border + height) / total_height

        idx = self.sprite_count * self.VERTICES_PER_SPRITE * self.FLOATS_PER_VERTEX
        r, g, b, a = color

        # Top-left, Top-right, Bottom-right, Bottom-left
        # Note: v is flipped because texture was flipped during load
        self.vertices[idx:idx+9] = [x, y, u_min, v_max, r, g, b, a, depth]
        self.vertices[idx+9:idx+18] = [x + width, y, u_max, v_max, r, g, b, a, depth]
        self.vertices[idx+18:idx+27] = [x + width, y + height, u_max, v_min, r, g, b, a, depth]
        self.vertices[idx+27:idx+36] = [x, y + height, u_min, v_min, r, g, b, a, depth]

        self.sprite_count += 1
        return True

    def flush(self):
        """Render all batched sprites"""
        if self.sprite_count == 0:
            return
            
        if self.current_texture:
            self.current_texture.bind(0)
            
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        data_size = self.sprite_count * self.VERTICES_PER_SPRITE * self.FLOATS_PER_VERTEX * 4
        glBufferSubData(GL_ARRAY_BUFFER, 0, data_size,
                       self.vertices[:self.sprite_count * self.VERTICES_PER_SPRITE * self.FLOATS_PER_VERTEX])
        
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.sprite_count * self.INDICES_PER_SPRITE, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        self.sprite_count = 0
