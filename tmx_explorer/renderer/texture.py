"""
OpenGL Texture management with tile bleeding fix
"""

import pygame
from OpenGL.GL import *


class Texture:
    """OpenGL texture wrapper with proper filtering for pixel art"""
    
    def __init__(self, surface: pygame.Surface):
        self.width = surface.get_width()
        self.height = surface.get_height()
        texture_data = pygame.image.tostring(surface, "RGBA", False)
        
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id)

        # Use CLAMP_TO_EDGE to prevent texture wrapping
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # NEAREST_MIPMAP_NEAREST for zoom out without bleeding
        # NEAREST for magnification (zoom in)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height,
            0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data
        )

        # Generate mipmaps for better quality at different zoom levels
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def bind(self, slot: int = 0):
        """Bind texture to specified slot"""
        glActiveTexture(GL_TEXTURE0 + slot)
        glBindTexture(GL_TEXTURE_2D, self.id)

    def __del__(self):
        if hasattr(self, 'id'):
            try:
                glDeleteTextures([self.id])
            except:
                pass
