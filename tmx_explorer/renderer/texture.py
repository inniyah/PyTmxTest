"""
OpenGL Texture management (GLFW version - uses PIL)

=============================================================================
WHAT IS A TEXTURE?
=============================================================================

A texture is an image stored in GPU memory that can be "painted" onto
geometry (triangles/quads). In 2D games, textures are used for:
- Tile graphics (terrain, walls, objects)
- Character sprites
- UI elements
- Backgrounds

This class handles:
1. Loading images from files or PIL Image objects
2. Uploading pixel data to GPU memory
3. Configuring how OpenGL samples the texture
4. Binding textures for rendering

=============================================================================
COORDINATE SYSTEMS (Important!)
=============================================================================

There's a mismatch between image formats and OpenGL:

Most image formats (PNG, JPEG, PIL):
    (0,0) -----> X
      |
      |
      v
      Y
    Origin at TOP-LEFT, Y increases DOWNWARD

OpenGL textures:
      Y
      ^
      |
      |
    (0,0) -----> X
    Origin at BOTTOM-LEFT, Y increases UPWARD

This means we need to FLIP images vertically when loading them into
OpenGL textures, otherwise everything appears upside-down!

=============================================================================
TILE BLEEDING PROBLEM
=============================================================================

When tiles are rendered at non-integer positions or with zoom, the GPU
may sample pixels from adjacent tiles due to floating-point precision
issues. This causes ugly lines/gaps between tiles.

SOLUTION: Add a 1-pixel border around each tile that duplicates the
edge pixels (called "edge extrusion"). When the GPU accidentally samples
outside the tile bounds, it gets the same color as the edge.

Before (16x16 tile):          After (18x18 with border):
+----------------+            +------------------+
|GGGGGGGGGGGGGGGG|            |GGGGGGGGGGGGGGGGGG|
|G              G|            |G+----------------+G|
|G    GRASS     G|   --->     |G|GGGGGGGGGGGGGGGG|G|
|G              G|            |G|G              G|G|
|GGGGGGGGGGGGGGGG|            |G|G    GRASS     G|G|
+----------------+            |G|G              G|G|
                              |G|GGGGGGGGGGGGGGGG|G|
                              |G+----------------+G|
                              |GGGGGGGGGGGGGGGGGG|
                              +------------------+

The border pixels are copies of the adjacent edge pixels.

=============================================================================
"""

from OpenGL.GL import *
from PIL import Image
import numpy as np


class Texture:
    """
    OpenGL texture wrapper with proper filtering for pixel art.
    
    This class encapsulates OpenGL texture creation and management,
    providing a clean interface for loading and using textures.
    
    ==========================================================================
    USAGE EXAMPLES
    ==========================================================================
    
    ```python
    # Load from file
    texture = Texture.from_file("tiles.png")
    
    # Load from PIL Image
    pil_image = Image.open("sprite.png")
    texture = Texture.from_pil(pil_image)
    
    # Use in rendering
    texture.bind(0)  # Bind to texture unit 0
    # ... draw geometry ...
    ```
    
    ==========================================================================
    TEXTURE FILTERING MODES
    ==========================================================================
    
    OpenGL needs to know how to sample textures when:
    - The texture is SMALLER than screen pixels (magnification)
    - The texture is LARGER than screen pixels (minification)
    
    For pixel art, we use GL_NEAREST (nearest-neighbor):
    - Picks the closest texel (texture pixel) with no interpolation
    - Preserves sharp pixel edges
    - Gives that classic "pixel art" look
    
    Alternative GL_LINEAR would blur pixels together - bad for pixel art,
    but good for photorealistic textures.
    
    ==========================================================================
    """
    
    def __init__(self, width: int, height: int, data: bytes):
        """
        Create texture from raw RGBA data.
        
        This is the low-level constructor. Usually you'll use from_pil()
        or from_file() instead, which handle image loading and conversion.
        
        Parameters:
        -----------
        width : int
            Texture width in pixels
        height : int
            Texture height in pixels
        data : bytes
            Raw RGBA pixel data (4 bytes per pixel)
            Length must be width × height × 4 bytes
            Pixel order: left-to-right, BOTTOM-to-top (OpenGL convention)
            
        =======================================================================
        OPENGL TEXTURE SETUP EXPLAINED
        =======================================================================
        
        Creating a texture in OpenGL involves:
        1. Generate a texture ID (like a handle/reference)
        2. Bind it (make it the "current" texture)
        3. Set texture parameters (filtering, wrapping)
        4. Upload pixel data
        5. Unbind (good practice)
        
        =======================================================================
        """
        self.width = width
        self.height = height
        
        # ---------------------------------------------------------------------
        # GENERATE TEXTURE ID
        # ---------------------------------------------------------------------
        # OpenGL textures are identified by integer IDs, not pointers.
        # glGenTextures creates a new ID that we'll use to reference this texture.
        self.id = glGenTextures(1)
        
        # ---------------------------------------------------------------------
        # BIND TEXTURE
        # ---------------------------------------------------------------------
        # "Binding" makes this texture the current 2D texture.
        # All subsequent texture operations affect the bound texture.
        # GL_TEXTURE_2D = standard 2D texture (vs 1D, 3D, cube maps, etc.)
        glBindTexture(GL_TEXTURE_2D, self.id)

        # ---------------------------------------------------------------------
        # TEXTURE WRAPPING MODE
        # ---------------------------------------------------------------------
        # What happens when UV coordinates go outside [0, 1] range?
        #
        # Options:
        # - GL_REPEAT: Tile the texture (u=1.5 samples same as u=0.5)
        # - GL_MIRRORED_REPEAT: Tile with alternating mirror
        # - GL_CLAMP_TO_EDGE: Clamp to edge pixels (no repeat)
        # - GL_CLAMP_TO_BORDER: Use a border color
        #
        # For tiles, GL_CLAMP_TO_EDGE prevents edge pixels from wrapping
        # around and causing visual artifacts.
        #
        # S = horizontal axis (U), T = vertical axis (V)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # ---------------------------------------------------------------------
        # TEXTURE FILTERING MODE
        # ---------------------------------------------------------------------
        # How to sample when texture size doesn't match screen pixels?
        #
        # MIN_FILTER: When texture is LARGER than screen area (zoomed out)
        # MAG_FILTER: When texture is SMALLER than screen area (zoomed in)
        #
        # GL_NEAREST: Pick nearest texel (sharp pixels, good for pixel art)
        # GL_LINEAR: Blend nearby texels (smooth, good for photos)
        #
        # For pixel art games, NEAREST preserves the crisp pixel look.
        # LINEAR would make everything blurry.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # ---------------------------------------------------------------------
        # UPLOAD PIXEL DATA TO GPU
        # ---------------------------------------------------------------------
        # glTexImage2D transfers pixel data from CPU memory to GPU memory.
        #
        # Parameters:
        # - GL_TEXTURE_2D: Target texture type
        # - 0: Mipmap level (0 = base level, we don't use mipmaps)
        # - GL_RGBA: Internal format (how GPU stores it)
        # - width, height: Dimensions
        # - 0: Border (must be 0, legacy parameter)
        # - GL_RGBA: Input format (how our data is organized)
        # - GL_UNSIGNED_BYTE: Input data type (8 bits per channel)
        # - data: The actual pixel bytes
        glTexImage2D(
            GL_TEXTURE_2D,      # Target
            0,                  # Mipmap level (0 = base)
            GL_RGBA,            # Internal format (GPU storage)
            width, height,      # Dimensions
            0,                  # Border (must be 0)
            GL_RGBA,            # Input format
            GL_UNSIGNED_BYTE,   # Input data type
            data                # Pixel data
        )

        # ---------------------------------------------------------------------
        # UNBIND TEXTURE
        # ---------------------------------------------------------------------
        # Unbind to prevent accidental modification.
        # Binding texture 0 means "no texture".
        glBindTexture(GL_TEXTURE_2D, 0)

    # =========================================================================
    # FACTORY METHODS (Alternative Constructors)
    # =========================================================================

    @classmethod
    def from_pil(cls, image: Image.Image, add_border: bool = True) -> 'Texture':
        """
        Create texture from PIL Image with optional 1px border.
        
        This is the primary way to create textures. It handles:
        1. Converting to RGBA format if needed
        2. Adding edge-extruded border (for tile bleeding fix)
        3. Flipping for OpenGL's coordinate system
        4. Extracting raw bytes
        
        Parameters:
        -----------
        image : PIL.Image.Image
            Source image (any PIL-supported format/mode)
        add_border : bool
            If True, adds 1-pixel extruded border to prevent tile bleeding.
            Set False for UI elements or textures that don't need it.
            
        Returns:
        --------
        Texture : New texture object uploaded to GPU
        
        =======================================================================
        BORDER EXTRUSION ALGORITHM
        =======================================================================
        
        The border is NOT transparent or a solid color - it's an "extrusion"
        of the edge pixels. This is critical for the tile bleeding fix.
        
        Original 4x4 image:        With extruded 1px border (6x6):
        
        [A][B][C][D]               [A][A][B][C][D][D]
        [E][F][G][H]               [A][A][B][C][D][D]
        [I][J][K][L]     --->      [E][E][F][G][H][H]
        [M][N][O][P]               [I][I][J][K][L][L]
                                   [M][M][N][O][P][P]
                                   [M][M][N][O][P][P]
        
        - Top border row copies the top row of the original
        - Bottom border row copies the bottom row
        - Left border column copies the left column
        - Right border column copies the right column
        - Corners copy the corner pixels
        
        This way, if the GPU accidentally samples outside the tile bounds,
        it gets the same color as the nearest edge pixel.
        
        =======================================================================
        """
        # ---------------------------------------------------------------------
        # ENSURE RGBA FORMAT
        # ---------------------------------------------------------------------
        # OpenGL expects RGBA (4 channels). Convert if image is:
        # - RGB (3 channels) - common for JPEG
        # - L (grayscale)
        # - P (palette/indexed)
        # - LA (grayscale + alpha)
        # etc.
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        if add_border:
            # -----------------------------------------------------------------
            # CREATE BORDERED IMAGE
            # -----------------------------------------------------------------
            border = 1  # 1 pixel on each side
            new_width = image.width + border * 2
            new_height = image.height + border * 2
            
            # Create new larger image with transparent background
            # (0, 0, 0, 0) = fully transparent black
            bordered = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
            
            # Paste original image in the center (offset by border size)
            bordered.paste(image, (border, border))
            
            # -----------------------------------------------------------------
            # EXTRUDE EDGES
            # -----------------------------------------------------------------
            # Copy edge pixels to the border area.
            # This prevents tile bleeding when GPU samples outside bounds.
            
            # TOP EDGE: Copy top row of original to top border row
            # For each column, copy the pixel from y=0 to y=-1 (border)
            for x in range(image.width):
                pixel = image.getpixel((x, 0))
                bordered.putpixel((x + border, 0), pixel)
            
            # BOTTOM EDGE: Copy bottom row of original to bottom border row
            for x in range(image.width):
                pixel = image.getpixel((x, image.height - 1))
                bordered.putpixel((x + border, new_height - 1), pixel)
            
            # LEFT EDGE: Copy left column of original to left border column
            for y in range(image.height):
                pixel = image.getpixel((0, y))
                bordered.putpixel((0, y + border), pixel)
            
            # RIGHT EDGE: Copy right column of original to right border column
            for y in range(image.height):
                pixel = image.getpixel((image.width - 1, y))
                bordered.putpixel((new_width - 1, y + border), pixel)
            
            # -----------------------------------------------------------------
            # EXTRUDE CORNERS
            # -----------------------------------------------------------------
            # The four corners need to be filled too.
            # Each corner gets the color of the nearest original corner pixel.
            
            # Top-left corner
            bordered.putpixel((0, 0), image.getpixel((0, 0)))
            
            # Top-right corner
            bordered.putpixel((new_width - 1, 0), 
                             image.getpixel((image.width - 1, 0)))
            
            # Bottom-left corner
            bordered.putpixel((0, new_height - 1), 
                             image.getpixel((0, image.height - 1)))
            
            # Bottom-right corner
            bordered.putpixel((new_width - 1, new_height - 1), 
                             image.getpixel((image.width - 1, image.height - 1)))
            
            # Use bordered image from now on
            image = bordered
        
        # ---------------------------------------------------------------------
        # FLIP IMAGE FOR OPENGL
        # ---------------------------------------------------------------------
        # PIL images have origin at top-left (y=0 at top)
        # OpenGL textures have origin at bottom-left (y=0 at bottom)
        # 
        # We flip here so that when OpenGL renders, the image appears
        # right-side up. The SpriteBatch compensates by also flipping
        # the V texture coordinates.
        #
        # Image.FLIP_TOP_BOTTOM mirrors the image vertically.
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # ---------------------------------------------------------------------
        # EXTRACT RAW BYTES
        # ---------------------------------------------------------------------
        # tobytes() returns raw pixel data as bytes object.
        # Format: RGBARGBARGBA... (4 bytes per pixel, row by row)
        # Total size: width × height × 4 bytes
        data = image.tobytes()
        
        # Create and return new Texture instance
        return cls(image.width, image.height, data)

    @classmethod
    def from_file(cls, filepath: str, add_border: bool = True) -> 'Texture':
        """
        Load texture from file.
        
        Convenience method that loads an image file and creates a texture.
        Supports any format PIL can read: PNG, JPEG, BMP, GIF, etc.
        
        Parameters:
        -----------
        filepath : str
            Path to image file
        add_border : bool
            If True, adds edge-extruded border for tile bleeding fix
            
        Returns:
        --------
        Texture : New texture object
        
        Example:
        --------
        ```python
        # Load a tileset
        tileset = Texture.from_file("assets/tiles.png")
        
        # Load UI element without border (not a tile)
        button = Texture.from_file("assets/button.png", add_border=False)
        ```
        """
        image = Image.open(filepath)
        return cls.from_pil(image, add_border)

    # =========================================================================
    # TEXTURE OPERATIONS
    # =========================================================================

    def bind(self, slot: int = 0):
        """
        Bind texture to specified texture unit slot.
        
        =======================================================================
        TEXTURE UNITS EXPLAINED
        =======================================================================
        
        OpenGL can have multiple textures active simultaneously, each in
        a different "texture unit" (numbered 0, 1, 2, ...).
        
        This is useful for:
        - Multi-texturing (combining textures in shader)
        - Normal maps + diffuse maps
        - Shadow maps
        
        For our 2D tile renderer, we only need one texture at a time,
        so we always use slot 0.
        
        The shader samples from a specific unit via a uniform:
            uniform sampler2D texture0;  // Samples from unit 0
        
        Parameters:
        -----------
        slot : int
            Texture unit to bind to (0-15 typically, GPU dependent)
            Default 0 is fine for single-texture rendering
            
        =======================================================================
        BINDING SEQUENCE
        =======================================================================
        
        1. glActiveTexture(GL_TEXTURE0 + slot) - Select which unit to configure
        2. glBindTexture(GL_TEXTURE_2D, id) - Bind our texture to that unit
        
        After this, any rendering that samples from unit `slot` will get
        this texture's pixels.
        """
        # Select texture unit (GL_TEXTURE0, GL_TEXTURE1, etc.)
        glActiveTexture(GL_TEXTURE0 + slot)
        
        # Bind this texture to the selected unit
        glBindTexture(GL_TEXTURE_2D, self.id)

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def __del__(self):
        """
        Clean up OpenGL texture when object is garbage collected.
        
        =======================================================================
        RESOURCE MANAGEMENT IN OPENGL
        =======================================================================
        
        OpenGL resources (textures, buffers, shaders, etc.) are stored in
        GPU memory and referenced by integer IDs. They're NOT automatically
        freed when Python objects are garbage collected.
        
        We must explicitly call glDeleteTextures() to:
        1. Free GPU memory
        2. Release the texture ID for reuse
        
        Without this, loading/unloading textures would leak GPU memory!
        
        =======================================================================
        WHY THE TRY/EXCEPT?
        =======================================================================
        
        The destructor might be called:
        - After OpenGL context is destroyed (program shutdown)
        - During Python interpreter shutdown (modules unloaded)
        - When the OpenGL context is invalid for other reasons
        
        In these cases, glDeleteTextures() would raise an error.
        We catch and ignore it because:
        1. If context is gone, GPU resources are already freed
        2. Crashing in __del__ causes confusing error messages
        3. It's cleanup code - failing silently is acceptable
        
        =======================================================================
        HASATTR CHECK
        =======================================================================
        
        If __init__ fails before self.id is assigned (e.g., OpenGL error),
        we'd get an AttributeError trying to access self.id.
        The hasattr check prevents this edge case crash.
        """
        if hasattr(self, 'id'):
            try:
                # Delete texture from GPU memory
                # Takes a list of IDs (can delete multiple at once)
                glDeleteTextures([self.id])
            except:
                # Ignore errors during cleanup (context may be gone)
                pass
