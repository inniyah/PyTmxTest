"""
OpenGL Texture management (GLFW version - uses PIL)
"""

from OpenGL.GL import *
from PIL import Image
import numpy as np


class Texture:
    """OpenGL texture wrapper with proper filtering for pixel art"""
    
    def __init__(self, width: int, height: int, data: bytes):
        """Create texture from raw RGBA data"""
        self.width = width
        self.height = height
        
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id)

        # Use CLAMP_TO_EDGE to prevent texture wrapping
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # NEAREST for pixel art
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, width, height,
            0, GL_RGBA, GL_UNSIGNED_BYTE, data
        )

        glBindTexture(GL_TEXTURE_2D, 0)

    @classmethod
    def from_pil(cls, image: Image.Image, add_border: bool = True) -> 'Texture':
        """Create texture from PIL Image with optional 1px border"""
        # Ensure RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        if add_border:
            # Add 1px border by creating larger image
            border = 1
            new_width = image.width + border * 2
            new_height = image.height + border * 2
            
            bordered = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
            bordered.paste(image, (border, border))
            
            # Extrude edges
            # Top edge
            for x in range(image.width):
                pixel = image.getpixel((x, 0))
                bordered.putpixel((x + border, 0), pixel)
            # Bottom edge
            for x in range(image.width):
                pixel = image.getpixel((x, image.height - 1))
                bordered.putpixel((x + border, new_height - 1), pixel)
            # Left edge
            for y in range(image.height):
                pixel = image.getpixel((0, y))
                bordered.putpixel((0, y + border), pixel)
            # Right edge
            for y in range(image.height):
                pixel = image.getpixel((image.width - 1, y))
                bordered.putpixel((new_width - 1, y + border), pixel)
            # Corners
            bordered.putpixel((0, 0), image.getpixel((0, 0)))
            bordered.putpixel((new_width - 1, 0), image.getpixel((image.width - 1, 0)))
            bordered.putpixel((0, new_height - 1), image.getpixel((0, image.height - 1)))
            bordered.putpixel((new_width - 1, new_height - 1), image.getpixel((image.width - 1, image.height - 1)))
            
            image = bordered
        
        # Flip for OpenGL (top-down to bottom-up)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Get raw data
        data = image.tobytes()
        
        return cls(image.width, image.height, data)

    @classmethod
    def from_file(cls, filepath: str, add_border: bool = True) -> 'Texture':
        """Load texture from file"""
        image = Image.open(filepath)
        return cls.from_pil(image, add_border)

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
