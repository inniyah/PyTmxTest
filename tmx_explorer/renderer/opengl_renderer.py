"""
OpenGL-based renderer for TMX maps (GLFW version - uses PIL for images)

=============================================================================
ARCHITECTURE OVERVIEW
=============================================================================

This renderer implements a 2D tile-based rendering system using modern OpenGL.
It's designed for rendering TMX (Tiled Map Editor) maps efficiently.

Key Design Decisions:
1. BATCHED RENDERING: Instead of drawing tiles one-by-one (which would be 
   extremely slow due to draw call overhead), we group tiles by texture 
   and render them in batches. This is THE most important optimization.

2. TEXTURE ATLASING: Tiles sharing the same tileset texture are rendered 
   together, minimizing texture binding operations (which are expensive).

3. SPRITE BATCHING: Uses a SpriteBatch class to accumulate geometry and 
   send it to the GPU in large chunks rather than individual draw calls.

4. DEPTH BUFFER: Uses OpenGL's depth buffer for proper layer ordering,
   avoiding the need for manual back-to-front sorting.

5. ORTHOGRAPHIC PROJECTION: 2D games use orthographic (not perspective) 
   projection where objects don't get smaller with distance.

=============================================================================
OPENGL CONCEPTS USED
=============================================================================

- VAO (Vertex Array Object): Stores the configuration of vertex attributes
- VBO (Vertex Buffer Object): Stores actual vertex data on the GPU
- Shaders: Programs that run on the GPU (vertex + fragment shaders)
- Uniforms: Variables passed from CPU to GPU (projection matrix, textures)
- Blending: For transparency/alpha in sprites
- Depth Testing: For proper layer ordering without manual sorting

=============================================================================
"""

import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from typing import Dict, List, Tuple, Optional
from PIL import Image

from .texture import Texture
from .sprite_batch import SpriteBatch
from ..shaders.sources import (
    VERTEX_SHADER, FRAGMENT_SHADER,
    SIMPLE_VERTEX_SHADER, SIMPLE_FRAGMENT_SHADER
)

# Default color constant - full white means "don't tint the texture"
WHITE = (255, 255, 255)


class OpenGLRenderer:
    """
    OpenGL renderer for tile maps (GLFW/PIL version)
    
    This class manages all OpenGL rendering state and provides methods
    for drawing tiles, shapes, and text efficiently.
    
    ==========================================================================
    WHY GLFW + PIL?
    ==========================================================================
    
    - GLFW: Cross-platform window/context creation library. Lighter than 
      alternatives like Pygame or PyQt. Gives us direct OpenGL access.
    
    - PIL (Pillow): Used for image loading and text rendering. We convert
      PIL images to OpenGL textures. This avoids dependencies on SDL_image
      or other libraries.
    
    ==========================================================================
    RENDERING PIPELINE
    ==========================================================================
    
    1. begin_frame() - Clear screen
    2. draw_batched_tiles() - Render all map tiles (main content)
    3. draw_lines() / draw_rects() - Render debug shapes (optional)
    4. draw_text_lines() - Render UI text overlay
    5. (GLFW swaps buffers externally)
    
    ==========================================================================
    """

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the OpenGL renderer.
        
        Parameters:
        -----------
        screen_width : int
            Initial window width in pixels
        screen_height : int
            Initial window height in pixels
            
        Note: OpenGL context must already be created before instantiating
        this class (GLFW handles this externally).
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize in specific order - shaders first, then buffers that use them
        self._init_shaders()
        self._init_buffers()
        self._init_state()
        
        # Debug info - useful for troubleshooting driver/version issues
        print(f"OpenGL Renderer: {glGetString(GL_VERSION).decode()}")

    # =========================================================================
    # INITIALIZATION METHODS
    # =========================================================================

    def _init_shaders(self):
        """
        Compile and link shader programs.
        
        =======================================================================
        WHY TWO SHADER PROGRAMS?
        =======================================================================
        
        1. shader_program (Main): For textured sprites (tiles, text)
           - Vertex shader: Transforms positions, passes UVs to fragment
           - Fragment shader: Samples texture, applies color
           
        2. simple_shader: For colored primitives (lines, rectangles)
           - No texture sampling needed
           - Just transforms positions and outputs solid colors
           
        Switching between shaders has a cost, but it's much cheaper than
        having conditional logic inside a single "do everything" shader.
        
        =======================================================================
        UNIFORM LOCATIONS
        =======================================================================
        
        Uniforms are like global variables for shaders. We cache their 
        locations (integer IDs) at startup because looking them up by 
        name string every frame would be wasteful.
        
        - projection: 4x4 matrix converting world coords to screen coords
        - texture0: Which texture unit to sample from (always 0 for us)
        """
        # Compile main shader for textured rendering
        # compileProgram links vertex + fragment shaders into a usable program
        self.shader_program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        
        # Compile simple shader for lines/shapes (no texture)
        self.simple_shader = compileProgram(
            compileShader(SIMPLE_VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(SIMPLE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        
        # Cache uniform locations for the main shader
        # These integers identify WHERE in the shader to send data
        self.proj_loc = glGetUniformLocation(self.shader_program, "projection")
        self.tex_loc = glGetUniformLocation(self.shader_program, "texture0")
        
        # Cache uniform location for simple shader
        self.simple_proj_loc = glGetUniformLocation(self.simple_shader, "projection")

    def _init_buffers(self):
        """
        Initialize vertex buffers for rendering.
        
        =======================================================================
        BUFFER STRATEGY
        =======================================================================
        
        1. SpriteBatch: Handles all textured sprite rendering (tiles, text).
           Internally manages its own VAO/VBO. Max 20,000 sprites per batch
           is a good balance - high enough to render most screens in one
           draw call, low enough to not waste GPU memory.
        
        2. simple_vao/vbo: For debug geometry (lines, rectangles).
           Pre-allocated 10MB buffer because:
           - Dynamic allocation per frame would be slow
           - 10MB is plenty for debug visualization
           - GL_DYNAMIC_DRAW hints to driver that data changes frequently
        
        =======================================================================
        VAO/VBO SETUP EXPLAINED
        =======================================================================
        
        VAO (Vertex Array Object):
        - Stores the FORMAT of vertex data (what attributes, what types)
        - Bind once at setup, then just bind the VAO when drawing
        
        VBO (Vertex Buffer Object):
        - Stores the actual vertex DATA on the GPU
        - We update this every frame with new geometry
        
        Vertex Format for simple_vbo (6 floats per vertex):
        [x, y, r, g, b, a]
         ^  ^  ^  ^  ^  ^
         |  |  |__|__|__|-- Color (4 floats, RGBA normalized 0-1)
         |__|-- Position (2 floats, screen coordinates)
        
        glVertexAttribPointer arguments:
        - attribute index (0=position, 1=color)
        - component count (2 for xy, 4 for rgba)
        - data type (GL_FLOAT)
        - normalized (GL_FALSE - we provide values directly)
        - stride (24 bytes = 6 floats * 4 bytes per float)
        - offset (0 for position, 8 for color = 2 floats * 4 bytes)
        """
        # Main sprite batch for tiles and textured quads
        # 20,000 sprites = 120,000 vertices = plenty for any screen
        self.batch = SpriteBatch(max_sprites=20000)
        
        # =====================================================================
        # Simple geometry VAO/VBO setup (for lines and rectangles)
        # =====================================================================
        
        # Generate and bind VAO - this "records" the following setup
        self.simple_vao = glGenVertexArrays(1)
        self.simple_vbo = glGenBuffers(1)
        glBindVertexArray(self.simple_vao)
        
        # Create VBO with pre-allocated space (10MB)
        # GL_DYNAMIC_DRAW = we'll update this frequently but also draw frequently
        glBindBuffer(GL_ARRAY_BUFFER, self.simple_vbo)
        glBufferData(GL_ARRAY_BUFFER, 10 * 1024 * 1024, None, GL_DYNAMIC_DRAW)
        
        # Attribute 0: Position (vec2 at offset 0)
        # Stride = 6 floats * 4 bytes = 24 bytes between vertices
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,                      # Attribute index
            2,                      # 2 components (x, y)
            GL_FLOAT,               # Data type
            GL_FALSE,               # Don't normalize
            6 * 4,                  # Stride: 24 bytes to next vertex
            ctypes.c_void_p(0)      # Offset: starts at byte 0
        )
        
        # Attribute 1: Color (vec4 at offset 8 bytes)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1,                      # Attribute index
            4,                      # 4 components (r, g, b, a)
            GL_FLOAT,               # Data type
            GL_FALSE,               # Don't normalize
            6 * 4,                  # Stride: 24 bytes
            ctypes.c_void_p(8)      # Offset: 8 bytes (after x,y)
        )
        
        # Unbind VAO to prevent accidental modification
        glBindVertexArray(0)

    def _init_state(self):
        """
        Initialize OpenGL rendering state and caches.
        
        =======================================================================
        OPENGL STATE MACHINE
        =======================================================================
        
        OpenGL is a state machine - you enable/configure things, and they
        stay that way until changed. We set up defaults here.
        
        BLENDING (for transparency):
        - GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = standard alpha blending
        - Formula: final_color = src_color * src_alpha + dst_color * (1 - src_alpha)
        - This gives us proper transparency for sprites with alpha channels
        
        DEPTH TESTING (for layer ordering):
        - Each tile has a depth value (z-coordinate)
        - GPU automatically discards pixels that are "behind" existing pixels
        - This means we don't need to sort tiles back-to-front!
        - GL_LESS = keep pixel if its depth is LESS than existing (closer to camera)
        
        CULLING (disabled):
        - Face culling removes triangles facing away from camera
        - Disabled because 2D sprites might be flipped/mirrored
        - In 3D games you'd enable this for performance
        
        =======================================================================
        CAMERA SYSTEM
        =======================================================================
        
        Simple 2D camera with position (x, y) and zoom factor.
        - camera_x/y: World position that maps to screen center
        - camera_zoom: Scale factor (1.0 = normal, 2.0 = zoomed in 2x)
        
        These are applied when rendering tiles in draw_batched_tiles().
        """
        # Initialize projection matrix (updated in update_projection)
        self.projection = np.eye(4, dtype=np.float32)
        self.update_projection()
        
        # Camera state - start at origin with no zoom
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_zoom = 1.0
        
        # ---------------------------------------------------------------------
        # Configure OpenGL state
        # ---------------------------------------------------------------------
        
        # Enable alpha blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable depth testing for automatic layer ordering
        # Range -10000 to 10000 gives plenty of depth precision
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)  # Closer objects (lower depth) win
        
        # Disable backface culling - not needed for 2D
        glDisable(GL_CULL_FACE)
        
        # ---------------------------------------------------------------------
        # Caches for performance
        # ---------------------------------------------------------------------
        
        # Texture cache: GID -> Texture object
        # Avoids re-uploading same texture to GPU multiple times
        self.texture_cache: Dict[int, Texture] = {}
        
        # Text rendering cache - avoid re-rendering unchanged text
        self._text_texture: Optional[Texture] = None
        self._text_cache_key = ""  # Hash of current text content

    # =========================================================================
    # PROJECTION AND CAMERA
    # =========================================================================

    def _ortho_matrix(self, left: float, right: float, bottom: float, 
                      top: float, near: float, far: float) -> np.ndarray:
        """
        Create orthographic projection matrix.
        
        =======================================================================
        ORTHOGRAPHIC vs PERSPECTIVE PROJECTION
        =======================================================================
        
        Perspective: Objects get smaller with distance (3D games)
        Orthographic: Objects stay same size regardless of distance (2D games)
        
        For 2D tile maps, orthographic is correct - a tile at the back of
        the map should be the same size as one at the front.
        
        =======================================================================
        MATRIX MATH
        =======================================================================
        
        This matrix transforms coordinates from "world space" to "clip space":
        - World space: Our game coordinates (pixels, typically)
        - Clip space: Normalized coordinates where visible = [-1, 1] on all axes
        
        The matrix scales and translates coordinates:
        - X: [left, right] -> [-1, 1]
        - Y: [bottom, top] -> [-1, 1]  (note: we flip Y so 0 is at top)
        - Z: [near, far] -> [-1, 1]
        
        Parameters:
        -----------
        left, right : float
            X coordinate range (usually 0 to screen_width)
        bottom, top : float
            Y coordinate range (screen_height to 0 for Y-down)
        near, far : float
            Z coordinate range (depth buffer range)
            
        Returns:
        --------
        4x4 numpy array (float32) - the projection matrix
        """
        mat = np.zeros((4, 4), dtype=np.float32)
        
        # Scale factors - map range to [-1, 1]
        mat[0, 0] = 2.0 / (right - left)      # X scale
        mat[1, 1] = 2.0 / (top - bottom)      # Y scale
        mat[2, 2] = -2.0 / (far - near)       # Z scale (negated for OpenGL convention)
        mat[3, 3] = 1.0                        # Homogeneous coordinate
        
        # Translation factors - center the range
        mat[0, 3] = -(right + left) / (right - left)    # X offset
        mat[1, 3] = -(top + bottom) / (top - bottom)    # Y offset
        mat[2, 3] = -(far + near) / (far - near)        # Z offset
        
        return mat

    def update_projection(self):
        """
        Update projection matrix for current screen size.
        
        Called on initialization and whenever window is resized.
        
        Note the parameter order:
        - left=0, right=screen_width (X increases rightward)
        - bottom=screen_height, top=0 (Y increases DOWNWARD - screen coords)
        
        This gives us a coordinate system where (0,0) is top-left,
        which is standard for 2D graphics and matches how TMX maps work.
        """
        self.projection = self._ortho_matrix(
            0, self.screen_width,           # X range: 0 to width
            self.screen_height, 0,          # Y range: height to 0 (Y-down!)
            -10000, 10000                   # Z range: large range for many layers
        )

    # =========================================================================
    # TEXTURE MANAGEMENT
    # =========================================================================

    def preload_texture(self, gid: int, image: Image.Image) -> Texture:
        """
        Pre-load texture with specific GID from PIL Image.
        
        =======================================================================
        TEXTURE CACHING STRATEGY
        =======================================================================
        
        TMX maps use GIDs (Global IDs) to identify tiles. Each tileset has
        a range of GIDs. We cache textures by GID to avoid:
        
        1. Re-loading the same image from disk
        2. Re-uploading the same texture to GPU
        3. Creating duplicate GPU resources
        
        In practice, most tiles share a small number of tileset textures,
        so this cache is very effective.
        
        Parameters:
        -----------
        gid : int
            Global ID for this texture (from TMX map)
        image : PIL.Image
            The image to convert to a texture
            
        Returns:
        --------
        Texture object (either cached or newly created)
        """
        if gid not in self.texture_cache:
            # Convert PIL image to OpenGL texture and cache it
            self.texture_cache[gid] = Texture.from_pil(image)
        return self.texture_cache[gid]

    # =========================================================================
    # CAMERA CONTROL
    # =========================================================================

    def set_camera(self, x: float, y: float, zoom: float):
        """
        Set camera position and zoom.
        
        Parameters:
        -----------
        x, y : float
            World position of camera (what world coord appears at screen origin)
        zoom : float
            Zoom factor (1.0 = normal, 2.0 = zoomed in, 0.5 = zoomed out)
            
        Note: Camera transform is applied in draw_batched_tiles(), not here.
        This just stores the values.
        """
        self.camera_x = x
        self.camera_y = y
        self.camera_zoom = zoom

    # =========================================================================
    # FRAME MANAGEMENT
    # =========================================================================

    def begin_frame(self):
        """
        Start a new frame - clear screen buffers.
        
        Must be called at the start of each frame before any drawing.
        
        We clear both:
        - Color buffer: The actual pixels (set to black)
        - Depth buffer: The per-pixel depth values (reset for new frame)
        
        If you forget to clear the depth buffer, tiles from previous frames
        might block new tiles from rendering (very confusing bug!).
        """
        # Set clear color to black (R=0, G=0, B=0, A=1)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Clear both color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # =========================================================================
    # TILE RENDERING (Main rendering function)
    # =========================================================================

    def draw_batched_tiles(self, tile_batches: Dict[Texture, List[Tuple]]):
        """
        Draw all tiles grouped by texture with depth.
        
        =======================================================================
        BATCHING EXPLAINED
        =======================================================================
        
        This is THE key optimization in the entire renderer!
        
        PROBLEM: Drawing tiles one-by-one is extremely slow because each
        draw call has significant CPU overhead (driver communication, etc.).
        A map with 10,000 tiles = 10,000 draw calls = terrible performance.
        
        SOLUTION: Group tiles by texture and draw all tiles with the same
        texture in a single draw call.
        
        RESULTS: 
        - 10 different textures = 10 draw calls total (regardless of tile count)
        - GPU is much better at processing large batches
        - Typical speedup: 100x or more!
        
        =======================================================================
        RENDERING PIPELINE
        =======================================================================
        
        1. Activate the textured shader program
        2. Send projection matrix to GPU
        3. For each texture:
           a. Bind the texture
           b. Begin batch accumulation
           c. For each tile using this texture:
              - Transform world coords to screen coords (apply camera)
              - Add to batch
              - If batch full, flush and continue
           d. Flush remaining sprites
        4. Deactivate shader
        
        =======================================================================
        CAMERA TRANSFORMATION
        =======================================================================
        
        Each tile's world position is transformed to screen position:
        - screen_pos = (world_pos - camera_pos) * zoom
        
        This centers the view on camera_x, camera_y and scales by zoom factor.
        
        Parameters:
        -----------
        tile_batches : Dict[Texture, List[Tuple]]
            Maps each texture to list of tiles using it.
            Each tile is a tuple: (x, y, width, height, depth)
            - x, y: World position
            - width, height: Tile size in world units
            - depth: Z-depth for layer ordering
        """
        # Activate textured shader program
        glUseProgram(self.shader_program)
        
        # Send projection matrix to GPU
        # Note: .T transposes because OpenGL expects column-major order
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection.T)
        
        # Tell shader to use texture unit 0
        glUniform1i(self.tex_loc, 0)

        # Render each texture batch
        for texture, tiles in tile_batches.items():
            if not tiles:
                continue  # Skip empty batches

            # Begin accumulating sprites for this texture
            # This binds the texture to unit 0
            self.batch.begin(texture)
            
            for x, y, w, h, depth in tiles:
                # ---------------------------------------------------------
                # CAMERA TRANSFORMATION
                # ---------------------------------------------------------
                # Convert world coordinates to screen coordinates:
                # 1. Subtract camera position (translate)
                # 2. Multiply by zoom (scale)
                screen_x = (x - self.camera_x) * self.camera_zoom
                screen_y = (y - self.camera_y) * self.camera_zoom
                screen_w = w * self.camera_zoom
                screen_h = h * self.camera_zoom

                # Add sprite to batch
                self.batch.add_sprite(screen_x, screen_y, screen_w, screen_h, depth)

                # ---------------------------------------------------------
                # BATCH OVERFLOW HANDLING
                # ---------------------------------------------------------
                # If batch is nearly full, flush it and start fresh.
                # The "-1" ensures we don't overflow on the next add.
                if self.batch.sprite_count >= self.batch.max_sprites - 1:
                    self.batch.flush()
                    self.batch.begin(texture)

            # Flush remaining sprites in this batch
            self.batch.flush()

        # Deactivate shader (good practice to avoid state leakage)
        glUseProgram(0)

    # =========================================================================
    # DEBUG RENDERING (Lines and Rectangles)
    # =========================================================================

    def draw_lines(self, lines: List[Tuple[float, float, float, float]], 
                   color: Tuple[int, int, int]):
        """
        Draw multiple lines in one call.
        
        =======================================================================
        USE CASES
        =======================================================================
        
        Useful for debug visualization:
        - Collision boundaries
        - Pathfinding paths
        - Grid lines
        - Selection indicators
        
        =======================================================================
        IMPLEMENTATION
        =======================================================================
        
        Uses GL_LINES primitive:
        - Every 2 vertices define one line segment
        - Vertices include position AND color (no texture)
        - Simple shader used (no texture sampling)
        
        Parameters:
        -----------
        lines : List[Tuple[float, float, float, float]]
            List of line segments, each as (x1, y1, x2, y2)
        color : Tuple[int, int, int]
            RGB color (0-255 range)
        """
        if not lines:
            return

        # Convert color from 0-255 to 0.0-1.0 (OpenGL convention)
        r, g, b = color[0]/255.0, color[1]/255.0, color[2]/255.0
        
        # Build vertex array
        # Format: [x1, y1, r, g, b, a, x2, y2, r, g, b, a, ...]
        vertices = []
        for x1, y1, x2, y2 in lines:
            # Start point
            vertices.extend([x1, y1, r, g, b, 1.0])
            # End point
            vertices.extend([x2, y2, r, g, b, 1.0])

        vertices = np.array(vertices, dtype=np.float32)

        # ---------------------------------------------------------------------
        # RENDER PIPELINE
        # ---------------------------------------------------------------------
        
        # 1. Activate simple (non-textured) shader
        glUseProgram(self.simple_shader)
        
        # 2. Send projection matrix
        glUniformMatrix4fv(self.simple_proj_loc, 1, GL_FALSE, self.projection.T)
        
        # 3. Upload vertex data to GPU
        # glBufferSubData updates part of an existing buffer (faster than recreating)
        glBindBuffer(GL_ARRAY_BUFFER, self.simple_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        
        # 4. Bind VAO (contains vertex format configuration)
        glBindVertexArray(self.simple_vao)
        
        # 5. Draw! 2 vertices per line
        glDrawArrays(GL_LINES, 0, len(lines) * 2)
        
        # 6. Cleanup
        glBindVertexArray(0)
        glUseProgram(0)

    def draw_rects(self, rects: List[Tuple[float, float, float, float]],
                   color: Tuple[int, int, int, int]):
        """
        Draw multiple filled rectangles with alpha.
        
        =======================================================================
        USE CASES
        =======================================================================
        
        - Selection highlights
        - Health bars
        - UI backgrounds
        - Debug collision boxes
        
        =======================================================================
        TRIANGLE DECOMPOSITION
        =======================================================================
        
        OpenGL doesn't have a "rectangle" primitive, so we draw each
        rectangle as 2 triangles:
        
            (x,y)------(x+w,y)
              |  \\   T1  |
              | T2 \\     |
              |      \\   |
           (x,y+h)----(x+w,y+h)
        
        Triangle 1: top-left, top-right, bottom-right
        Triangle 2: top-left, bottom-right, bottom-left
        
        6 vertices per rectangle (2 triangles * 3 vertices each)
        
        =======================================================================
        DEPTH TEST DISABLED
        =======================================================================
        
        We disable depth testing so rectangles always draw on top of
        the scene. This is correct for UI/debug overlays that should
        appear above game content regardless of depth values.
        
        Parameters:
        -----------
        rects : List[Tuple[float, float, float, float]]
            List of rectangles as (x, y, width, height)
        color : Tuple[int, int, int, int]
            RGBA color (0-255 range) - includes alpha for transparency
        """
        if not rects:
            return
        
        # Convert color from 0-255 to 0.0-1.0
        r, g, b, a = color[0]/255.0, color[1]/255.0, color[2]/255.0, color[3]/255.0
        
        vertices = []
        for x, y, w, h in rects:
            # Triangle 1: top-left -> top-right -> bottom-right
            vertices.extend([x, y, r, g, b, a])              # Top-left
            vertices.extend([x + w, y, r, g, b, a])          # Top-right
            vertices.extend([x + w, y + h, r, g, b, a])      # Bottom-right
            
            # Triangle 2: top-left -> bottom-right -> bottom-left
            vertices.extend([x, y, r, g, b, a])              # Top-left
            vertices.extend([x + w, y + h, r, g, b, a])      # Bottom-right
            vertices.extend([x, y + h, r, g, b, a])          # Bottom-left
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Render pipeline (same as draw_lines but with triangles)
        glUseProgram(self.simple_shader)
        glUniformMatrix4fv(self.simple_proj_loc, 1, GL_FALSE, self.projection.T)
        glBindBuffer(GL_ARRAY_BUFFER, self.simple_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glBindVertexArray(self.simple_vao)
        
        # IMPORTANT: Disable depth test so UI draws on top of everything
        glDisable(GL_DEPTH_TEST)
        
        # Draw! 6 vertices per rectangle
        glDrawArrays(GL_TRIANGLES, 0, len(rects) * 6)
        
        # Re-enable depth test for subsequent rendering
        glEnable(GL_DEPTH_TEST)
        
        glBindVertexArray(0)
        glUseProgram(0)

    # =========================================================================
    # TEXT RENDERING
    # =========================================================================

    def draw_text_lines(self, text_lines: List[str], x: int, y: int):
        """
        Draw text using PIL-rendered texture (optimized).
        
        =======================================================================
        WHY PIL FOR TEXT?
        =======================================================================
        
        OpenGL has no built-in text rendering. Common approaches:
        1. Bitmap fonts (pre-rendered character sprites)
        2. Signed distance field fonts (complex but scalable)
        3. System font rendering (PIL, FreeType, etc.)
        
        We use PIL because:
        - Simple to implement
        - No additional dependencies (PIL already used for images)
        - Good enough for debug/UI text
        - Handles any system font
        
        =======================================================================
        CACHING OPTIMIZATION
        =======================================================================
        
        Text rendering is expensive:
        1. PIL renders text to image (CPU)
        2. Image uploaded to GPU texture
        3. Texture drawn as quad
        
        If we did this every frame, performance would suffer.
        
        SOLUTION: Cache the rendered texture!
        - Only re-render if text content changes
        - FPS counter changes every frame, but we skip FPS lines when
          checking if cache is valid
        - Update FPS display every 6 frames (~10Hz) - fast enough to be
          useful, slow enough to not hurt performance
        
        _text_frame_counter tracks frames since last update.
        _text_cache_key stores hash of non-FPS text content.
        
        =======================================================================
        
        Parameters:
        -----------
        text_lines : List[str]
            Lines of text to render
        x, y : int
            Screen position (top-left of text block)
        """
        # ---------------------------------------------------------------------
        # CACHE INVALIDATION LOGIC
        # ---------------------------------------------------------------------
        
        # Separate FPS line (changes constantly) from other text (changes rarely)
        # FPS lines start with "FPS:" prefix
        base_lines = [l for l in text_lines if not l.startswith("FPS:")]
        cache_key = "|".join(base_lines)
        
        # Initialize frame counter if not exists
        self._text_frame_counter = getattr(self, '_text_frame_counter', 0) + 1
        
        # Decide if we need to re-render the text texture
        needs_update = (
            cache_key != self._text_cache_key or     # Text content changed
            self._text_texture is None or            # No cached texture
            self._text_frame_counter >= 6            # Time to update FPS (~10 Hz)
        )
        
        if needs_update:
            self._text_frame_counter = 0
            self._text_cache_key = cache_key
            
            # Import here to avoid circular imports and startup cost
            from PIL import ImageDraw, ImageFont
            
            # -----------------------------------------------------------------
            # FONT LOADING (cached)
            # -----------------------------------------------------------------
            # Loading fonts is slow, so we cache the font object.
            # Try to use DejaVu Sans Mono (common on Linux), fall back to default.
            if not hasattr(self, '_cached_font'):
                try:
                    self._cached_font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
                except:
                    # Fallback to PIL's built-in bitmap font
                    self._cached_font = ImageFont.load_default()
            
            # -----------------------------------------------------------------
            # TEXT IMAGE CREATION
            # -----------------------------------------------------------------
            line_height = 18  # Pixels per line
            max_width = 350   # Fixed width for text background
            total_height = len(text_lines) * line_height + 10  # +10 for padding
            
            # Create image with semi-transparent black background
            # RGBA mode: (R, G, B, A) where A=180 gives ~70% opacity
            img = Image.new('RGBA', (max_width, total_height), (0, 0, 0, 180))
            draw = ImageDraw.Draw(img)
            
            # Draw each line of text
            for i, line in enumerate(text_lines):
                draw.text(
                    (5, 5 + i * line_height),  # Position with 5px padding
                    line, 
                    font=self._cached_font, 
                    fill=(255, 255, 255, 255)  # White, fully opaque
                )
            
            # Convert PIL image to OpenGL texture
            # add_border=False because text doesn't need edge bleed prevention
            self._text_texture = Texture.from_pil(img, add_border=False)
        
        # ---------------------------------------------------------------------
        # RENDER TEXT TEXTURE
        # ---------------------------------------------------------------------
        
        if self._text_texture:
            glUseProgram(self.shader_program)
            glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection.T)
            glUniform1i(self.tex_loc, 0)
            
            # Disable depth test so text always appears on top
            glDisable(GL_DEPTH_TEST)
            
            # Draw as single sprite
            # depth=9999 is a high value (far from camera in our inverted system)
            # but it doesn't matter since depth testing is disabled
            # border=0 because text texture has no border padding
            self.batch.begin(self._text_texture)
            self.batch.add_sprite(
                x, y, 
                self._text_texture.width, self._text_texture.height, 
                depth=9999, 
                border=0
            )
            self.batch.flush()
            
            # Re-enable depth test
            glEnable(GL_DEPTH_TEST)
            glUseProgram(0)

    # =========================================================================
    # WINDOW MANAGEMENT
    # =========================================================================

    def resize(self, width: int, height: int):
        """
        Handle window resize.
        
        Called when the window is resized. Must update:
        1. Stored dimensions (for projection calculation)
        2. OpenGL viewport (tells OpenGL where to render)
        3. Projection matrix (maps world coords to new screen size)
        
        Parameters:
        -----------
        width, height : int
            New window dimensions in pixels
        """
        self.screen_width = width
        self.screen_height = height
        
        # Update OpenGL viewport to match new window size
        # Viewport defines the area of the window where OpenGL renders
        glViewport(0, 0, width, height)
        
        # Recalculate projection matrix for new dimensions
        self.update_projection()
