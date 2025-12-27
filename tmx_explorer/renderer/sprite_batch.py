"""
Batched sprite rendering for efficient tile drawing (GLFW version)

=============================================================================
WHAT IS SPRITE BATCHING?
=============================================================================

Sprite batching is THE fundamental optimization technique in 2D rendering.

THE PROBLEM:
------------
Without batching, to draw 10,000 tiles you would need:
- 10,000 draw calls (glDrawArrays/glDrawElements)
- Potentially 10,000 texture bindings
- Each draw call has ~1-5ms of CPU overhead

This would result in maybe 1-10 FPS at best!

THE SOLUTION:
-------------
With batching:
- Group sprites that share the same texture
- Combine all their vertex data into ONE buffer
- Issue ONE draw call for potentially thousands of sprites
- Result: 60+ FPS easily!

=============================================================================
HOW THIS CLASS WORKS
=============================================================================

1. Call begin(texture) - Sets which texture to use, resets counter
2. Call add_sprite() multiple times - Accumulates vertex data in CPU array
3. Call flush() - Uploads data to GPU and issues single draw call

The vertex data stays in a CPU-side numpy array until flush(), when it's
sent to the GPU all at once. This is more efficient than many small uploads.

=============================================================================
MEMORY LAYOUT
=============================================================================

Each sprite is a QUAD (rectangle) made of 4 vertices and 6 indices.

Vertices (4 per sprite):
    [0]---[1]     Indices define two triangles:
     |   / |      Triangle 1: 0, 1, 2
     |  /  |      Triangle 2: 0, 2, 3
     | /   |
    [3]---[2]

Each vertex has 9 floats (36 bytes):
    [x, y, u, v, r, g, b, a, depth]
     ^^^^  ^^^^  ^^^^^^^^^^  ^^^^^
     pos   tex   color       z-order

=============================================================================
"""

import ctypes
import numpy as np
from OpenGL.GL import *
from typing import Optional, Tuple
from .texture import Texture


class SpriteBatch:
    """
    Efficient batched sprite renderer.
    
    This class accumulates sprite geometry and renders it all in a single
    draw call, dramatically improving performance compared to drawing
    sprites one at a time.
    
    ==========================================================================
    USAGE PATTERN
    ==========================================================================
    
    ```python
    batch = SpriteBatch(max_sprites=10000)
    
    # In render loop:
    batch.begin(tileset_texture)
    for tile in visible_tiles:
        batch.add_sprite(tile.x, tile.y, tile.width, tile.height, tile.depth)
    batch.flush()  # Actually renders everything
    ```
    
    ==========================================================================
    DESIGN DECISIONS
    ==========================================================================
    
    1. FIXED MAXIMUM SIZE: Pre-allocate for max_sprites to avoid dynamic
       allocation during rendering. Memory is cheap, frame drops are not.
    
    2. CPU-SIDE VERTEX BUFFER: We keep vertices in a numpy array and only
       upload to GPU on flush(). This is faster than many small GPU writes.
    
    3. PRE-GENERATED INDICES: Index buffer never changes (always 0,1,2,0,2,3
       pattern), so we generate it once at startup.
    
    4. SINGLE TEXTURE PER BATCH: All sprites in a batch share one texture.
       The renderer groups tiles by texture before calling us.
    
    ==========================================================================
    """
    
    # =========================================================================
    # CONSTANTS - Define the vertex format
    # =========================================================================
    
    # How many float values per vertex
    # Layout: [x, y, u, v, r, g, b, a, depth] = 9 floats
    FLOATS_PER_VERTEX = 9
    
    # Each sprite (quad) has 4 corners
    VERTICES_PER_SPRITE = 4
    
    # Each sprite needs 6 indices (2 triangles × 3 vertices each)
    # But we reuse 2 vertices, so 4 vertices + 6 indices = efficient!
    INDICES_PER_SPRITE = 6
    
    def __init__(self, max_sprites: int = 20000):
        """
        Initialize the sprite batch.
        
        Parameters:
        -----------
        max_sprites : int
            Maximum number of sprites that can be batched before flush.
            Default 20,000 is enough for most screens:
            - 1920x1080 / 16x16 tiles = ~8,000 tiles max visible
            - 20,000 gives comfortable headroom
            
        Memory usage:
        - Vertex data: 20000 × 4 vertices × 9 floats × 4 bytes = 2.88 MB
        - Index data:  20000 × 6 indices × 4 bytes = 0.48 MB
        - Total: ~3.4 MB - very reasonable for modern systems
        """
        self.max_sprites = max_sprites
        self.sprite_count = 0  # Current number of sprites in batch
        
        # ---------------------------------------------------------------------
        # PRE-ALLOCATE VERTEX ARRAY (CPU-side)
        # ---------------------------------------------------------------------
        # This numpy array holds all vertex data before upload to GPU.
        # Pre-allocation avoids memory allocation during rendering.
        # 
        # Total floats = max_sprites × 4 vertices × 9 floats per vertex
        self.vertices = np.zeros(
            max_sprites * self.VERTICES_PER_SPRITE * self.FLOATS_PER_VERTEX,
            dtype=np.float32  # OpenGL expects 32-bit floats
        )
        
        # ---------------------------------------------------------------------
        # PRE-GENERATE INDEX ARRAY
        # ---------------------------------------------------------------------
        # Indices never change - they always follow the same pattern.
        # Generate once at startup, upload to GPU once, never touch again.
        self.indices = self._create_indices(max_sprites)
        
        # Currently bound texture (set in begin())
        self.current_texture: Optional[Texture] = None
        
        # Create OpenGL buffer objects
        self._setup_buffers()

    def _create_indices(self, max_sprites: int) -> np.ndarray:
        """
        Pre-generate index buffer for all possible sprites.
        
        =======================================================================
        WHY INDEXED DRAWING?
        =======================================================================
        
        A quad has 4 corners but needs 6 vertices to draw as triangles:
        
        Without indices (GL_TRIANGLES with 6 vertices):
            Triangle 1: v0, v1, v2  (top-left, top-right, bottom-right)
            Triangle 2: v3, v4, v5  (top-left, bottom-right, bottom-left)
            
            Notice v0==v3 and v2==v4 - we're duplicating data!
            6 vertices × 9 floats = 54 floats per sprite
        
        With indices (4 vertices + 6 indices):
            Vertices: v0, v1, v2, v3 (the 4 corners)
            Indices: 0, 1, 2, 0, 2, 3 (which vertices to use)
            
            4 vertices × 9 floats + 6 indices = 36 floats + 6 ints per sprite
            
        SAVINGS: ~25% less data to upload per sprite!
        
        =======================================================================
        INDEX PATTERN
        =======================================================================
        
        For each sprite, vertices are arranged:
        
            0-------1      Indices for triangles:
            |     / |      - Triangle 1: 0, 1, 2 (top-left, top-right, bottom-right)
            |   /   |      - Triangle 2: 0, 2, 3 (top-left, bottom-right, bottom-left)
            | /     |
            3-------2
        
        For sprite N, vertex indices start at N*4:
            Sprite 0: vertices 0,1,2,3  -> indices 0,1,2,0,2,3
            Sprite 1: vertices 4,5,6,7  -> indices 4,5,6,4,6,7
            Sprite 2: vertices 8,9,10,11 -> indices 8,9,10,8,10,11
            ...
        
        Parameters:
        -----------
        max_sprites : int
            Number of sprites to generate indices for
            
        Returns:
        --------
        numpy array of uint32 indices
        """
        indices = []
        for i in range(max_sprites):
            # Starting vertex index for this sprite
            offset = i * 4
            
            # Two triangles forming the quad
            indices.extend([
                offset + 0, offset + 1, offset + 2,  # Triangle 1
                offset + 0, offset + 2, offset + 3   # Triangle 2
            ])
        
        # uint32 because we might have > 65535 vertices (uint16 max)
        return np.array(indices, dtype=np.uint32)

    def _setup_buffers(self):
        """
        Initialize OpenGL buffers.
        
        =======================================================================
        OPENGL OBJECTS CREATED
        =======================================================================
        
        1. VAO (Vertex Array Object):
           - Stores the "format" of vertex data (what attributes, what types)
           - Bind VAO once, and all attribute setup is remembered
           
        2. VBO (Vertex Buffer Object):
           - Stores actual vertex DATA on GPU
           - GL_DYNAMIC_DRAW: We'll update this frequently (every frame)
           
        3. EBO (Element Buffer Object / Index Buffer):
           - Stores indices that reference vertices
           - GL_STATIC_DRAW: Generated once, never changes
        
        =======================================================================
        VERTEX ATTRIBUTE LAYOUT
        =======================================================================
        
        Each vertex is 9 floats (36 bytes):
        
        Offset (bytes):  0    8    16   20   24   28   32
                         |    |    |    |    |    |    |
        Data:           [x,y][u,v][r   ,g   ,b   ,a  ][depth]
                        └──┘ └──┘ └────────────────┘  └────┘
        Attribute:        0    1           2            3
        Components:       2    2           4            1
        
        Stride = 36 bytes (distance between same attribute in consecutive vertices)
        
        =======================================================================
        """
        # Generate OpenGL objects
        self.vao = glGenVertexArrays(1)  # Vertex Array Object
        self.vbo = glGenBuffers(1)        # Vertex Buffer Object
        self.ebo = glGenBuffers(1)        # Element Buffer Object (indices)

        # ---------------------------------------------------------------------
        # CONFIGURE VAO (this "records" the following attribute setup)
        # ---------------------------------------------------------------------
        glBindVertexArray(self.vao)
        
        # ---------------------------------------------------------------------
        # SETUP VBO (vertex data buffer)
        # ---------------------------------------------------------------------
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        
        # Allocate GPU memory but don't fill it yet (data=None)
        # GL_DYNAMIC_DRAW tells the driver we'll update this frequently
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, None, GL_DYNAMIC_DRAW)

        # Stride: bytes between consecutive vertices
        # 9 floats × 4 bytes per float = 36 bytes
        stride = self.FLOATS_PER_VERTEX * 4

        # -----------------------------------------------------------------
        # ATTRIBUTE 0: Position (vec2) - x, y
        # -----------------------------------------------------------------
        # Location: attribute 0 in vertex shader
        # Size: 2 components (x, y)
        # Offset: 0 bytes (starts at beginning)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,                      # Attribute index
            2,                      # Number of components (x, y)
            GL_FLOAT,               # Data type
            GL_FALSE,               # Don't normalize
            stride,                 # Bytes to next vertex's position
            ctypes.c_void_p(0)      # Offset from start of vertex
        )
        
        # -----------------------------------------------------------------
        # ATTRIBUTE 1: Texture Coordinates (vec2) - u, v
        # -----------------------------------------------------------------
        # Location: attribute 1 in vertex shader
        # Size: 2 components (u, v)
        # Offset: 8 bytes (after x, y = 2 floats × 4 bytes)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1,                      # Attribute index
            2,                      # Number of components (u, v)
            GL_FLOAT,               # Data type
            GL_FALSE,               # Don't normalize
            stride,                 # Bytes to next vertex's texcoord
            ctypes.c_void_p(8)      # Offset: 2 floats × 4 bytes = 8
        )
        
        # -----------------------------------------------------------------
        # ATTRIBUTE 2: Color (vec4) - r, g, b, a
        # -----------------------------------------------------------------
        # Location: attribute 2 in vertex shader
        # Size: 4 components (r, g, b, a)
        # Offset: 16 bytes (after x, y, u, v = 4 floats × 4 bytes)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(
            2,                      # Attribute index
            4,                      # Number of components (r, g, b, a)
            GL_FLOAT,               # Data type
            GL_FALSE,               # Don't normalize
            stride,                 # Bytes to next vertex's color
            ctypes.c_void_p(16)     # Offset: 4 floats × 4 bytes = 16
        )
        
        # -----------------------------------------------------------------
        # ATTRIBUTE 3: Depth (float) - z-order for layer sorting
        # -----------------------------------------------------------------
        # Location: attribute 3 in vertex shader
        # Size: 1 component (depth)
        # Offset: 32 bytes (after x, y, u, v, r, g, b, a = 8 floats × 4 bytes)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(
            3,                      # Attribute index
            1,                      # Number of components (depth)
            GL_FLOAT,               # Data type
            GL_FALSE,               # Don't normalize
            stride,                 # Bytes to next vertex's depth
            ctypes.c_void_p(32)     # Offset: 8 floats × 4 bytes = 32
        )

        # ---------------------------------------------------------------------
        # SETUP EBO (index buffer)
        # ---------------------------------------------------------------------
        # Bind to the VAO so it's remembered
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        
        # Upload index data - GL_STATIC_DRAW because indices never change
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, 
                     self.indices, GL_STATIC_DRAW)
        
        # Unbind VAO to prevent accidental modification
        glBindVertexArray(0)

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    def begin(self, texture: Texture):
        """
        Start a new batch with the given texture.
        
        This must be called before adding sprites. All sprites added until
        flush() will use this texture.
        
        Parameters:
        -----------
        texture : Texture
            The texture to use for all sprites in this batch.
            All sprites MUST share the same texture - this is the key
            constraint that enables batching.
            
        Note:
        -----
        If you need to draw sprites with different textures, you must:
        1. begin(texture_a)
        2. add sprites using texture_a
        3. flush()
        4. begin(texture_b)
        5. add sprites using texture_b
        6. flush()
        
        The renderer handles this grouping automatically.
        """
        self.sprite_count = 0  # Reset counter for new batch
        self.current_texture = texture

    def add_sprite(
        self,
        x: float, y: float,
        width: float, height: float,
        depth: float = 0.0,
        color: Tuple[float, float, float, float] = (1, 1, 1, 1),
        border: int = 1
    ) -> bool:
        """
        Add a sprite to the current batch.
        
        =======================================================================
        PARAMETERS EXPLAINED
        =======================================================================
        
        x, y : float
            Top-left corner position in screen coordinates.
            
        width, height : float
            Size of the sprite in pixels.
            
        depth : float
            Z-depth for layer ordering. Lower values = closer to camera.
            Used by depth buffer to automatically handle overlapping sprites.
            
        color : Tuple[float, float, float, float]
            RGBA color multiplier (0.0 to 1.0 range).
            Default (1,1,1,1) = white = no tinting.
            (1,0,0,1) = tint red, (0.5,0.5,0.5,1) = 50% darker, etc.
            
        border : int
            Pixel border added around tile for bleeding prevention.
            Default 1 = 1 pixel border on each side.
            See UV coordinate section for details.
            
        Returns:
        --------
        bool : True if sprite was added, False if batch is full.
        
        =======================================================================
        TILE BLEEDING FIX (the 'border' parameter)
        =======================================================================
        
        PROBLEM: When rendering tiles at non-integer positions or with
        scaling, the GPU may sample pixels from adjacent tiles due to
        floating-point imprecision. This causes ugly lines between tiles.
        
        SOLUTION: Each tile texture has a 1-pixel border that duplicates
        the edge pixels. Our UV coordinates skip this border, mapping only
        to the "real" tile content in the center.
        
        Example for a 16x16 tile with 1px border (total 18x18 texture):
        
            +------------------+
            |B B B B B B B B B |  B = border (copies of edge pixels)
            |B +------------+ B|
            |B |            | B|
            |B |   TILE     | B|  <- UV coordinates map to this area
            |B |  CONTENT   | B|
            |B |            | B|
            |B +------------+ B|
            |B B B B B B B B B |
            +------------------+
            
        UV range: u_min = 1/18, u_max = 17/18 (skipping border)
        
        =======================================================================
        VERTEX LAYOUT
        =======================================================================
        
        We add 4 vertices per sprite (one per corner):
        
            v0 (x,y)-------v1 (x+w, y)
               |               |
               |               |
               |               |
            v3 (x, y+h)----v2 (x+w, y+h)
        
        Each vertex has: [x, y, u, v, r, g, b, a, depth]
        
        =======================================================================
        UV COORDINATE FLIPPING
        =======================================================================
        
        NOTE: V coordinates are FLIPPED (v_max at top, v_min at bottom).
        
        This is because:
        1. OpenGL textures have (0,0) at BOTTOM-left
        2. Our screen coords have (0,0) at TOP-left
        3. The texture was flipped during loading to match
        4. So we flip V here to compensate
        
        Without this flip, tiles would appear upside-down!
        
        =======================================================================
        """
        # Check if batch is full
        if self.sprite_count >= self.max_sprites:
            return False

        # -----------------------------------------------------------------
        # CALCULATE UV COORDINATES (with border adjustment)
        # -----------------------------------------------------------------
        
        # Total texture size including border
        total_width = width + border * 2
        total_height = height + border * 2

        # UV coordinates that skip the border
        # Map [border, border+width] to [0, total_width] normalized
        u_min = border / total_width          # Left edge of content
        v_min = border / total_height         # Top edge of content
        u_max = (border + width) / total_width    # Right edge of content
        v_max = (border + height) / total_height  # Bottom edge of content

        # -----------------------------------------------------------------
        # CALCULATE BUFFER INDEX
        # -----------------------------------------------------------------
        
        # Where in the vertex array to write this sprite's data
        # Each sprite = 4 vertices × 9 floats = 36 floats
        idx = self.sprite_count * self.VERTICES_PER_SPRITE * self.FLOATS_PER_VERTEX
        
        # Unpack color for faster access
        r, g, b, a = color

        # -----------------------------------------------------------------
        # WRITE VERTEX DATA
        # -----------------------------------------------------------------
        # 4 vertices × 9 floats each = 36 consecutive floats
        #
        # IMPORTANT: v_max and v_min are SWAPPED!
        # Top vertices use v_max, bottom use v_min (flipped)
        # This corrects for OpenGL's bottom-up texture orientation.
        
        # Vertex 0: Top-left corner
        self.vertices[idx:idx+9] = [
            x, y,                    # Position
            u_min, v_max,            # Texture coords (note: v_max at TOP)
            r, g, b, a,              # Color
            depth                    # Depth for z-ordering
        ]
        
        # Vertex 1: Top-right corner
        self.vertices[idx+9:idx+18] = [
            x + width, y,            # Position
            u_max, v_max,            # Texture coords
            r, g, b, a,              # Color
            depth                    # Depth
        ]
        
        # Vertex 2: Bottom-right corner
        self.vertices[idx+18:idx+27] = [
            x + width, y + height,   # Position
            u_max, v_min,            # Texture coords (note: v_min at BOTTOM)
            r, g, b, a,              # Color
            depth                    # Depth
        ]
        
        # Vertex 3: Bottom-left corner
        self.vertices[idx+27:idx+36] = [
            x, y + height,           # Position
            u_min, v_min,            # Texture coords
            r, g, b, a,              # Color
            depth                    # Depth
        ]

        self.sprite_count += 1
        return True

    def flush(self):
        """
        Render all batched sprites.
        
        =======================================================================
        WHAT HAPPENS HERE
        =======================================================================
        
        1. Skip if nothing to draw (early exit optimization)
        2. Bind the texture to texture unit 0
        3. Upload vertex data from CPU (numpy array) to GPU (VBO)
        4. Bind VAO (which sets up all vertex attributes)
        5. Issue ONE draw call for ALL sprites
        6. Reset sprite count (ready for next batch)
        
        =======================================================================
        WHY glBufferSubData INSTEAD OF glBufferData?
        =======================================================================
        
        - glBufferData: Allocates AND uploads data. Would reallocate every frame!
        - glBufferSubData: Just uploads to existing allocation. Much faster!
        
        We allocated the buffer once in _setup_buffers() with glBufferData,
        now we just update its contents each frame with glBufferSubData.
        
        =======================================================================
        DRAW CALL EXPLANATION
        =======================================================================
        
        glDrawElements(GL_TRIANGLES, count, type, offset):
        - GL_TRIANGLES: Every 3 indices form one triangle
        - count: How many indices to use (6 per sprite × sprite_count)
        - GL_UNSIGNED_INT: Index data type (32-bit unsigned integers)
        - None: Start at index 0 (offset is null pointer)
        
        For 1000 sprites: ONE draw call renders 2000 triangles = 6000 indices
        
        =======================================================================
        """
        # Early exit if nothing to draw
        if self.sprite_count == 0:
            return
        
        # Bind texture to unit 0 (shader samples from unit 0)
        if self.current_texture:
            self.current_texture.bind(0)
        
        # ---------------------------------------------------------------------
        # UPLOAD VERTEX DATA TO GPU
        # ---------------------------------------------------------------------
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        
        # Calculate how many bytes of data we actually need to upload
        # Only upload what we've written, not the entire pre-allocated buffer
        # This is a micro-optimization: less data transfer = faster
        data_size = (self.sprite_count * 
                    self.VERTICES_PER_SPRITE * 
                    self.FLOATS_PER_VERTEX * 
                    4)  # 4 bytes per float
        
        # Upload vertex data (only the portion we've filled)
        # Slice the numpy array to only include active sprite data
        active_data = self.vertices[:self.sprite_count * 
                                     self.VERTICES_PER_SPRITE * 
                                     self.FLOATS_PER_VERTEX]
        glBufferSubData(GL_ARRAY_BUFFER, 0, data_size, active_data)
        
        # ---------------------------------------------------------------------
        # DRAW!
        # ---------------------------------------------------------------------
        
        # Bind VAO (restores all vertex attribute configurations)
        glBindVertexArray(self.vao)
        
        # Issue the draw call
        # This ONE call renders all sprites in the batch!
        glDrawElements(
            GL_TRIANGLES,                              # Primitive type
            self.sprite_count * self.INDICES_PER_SPRITE,  # Index count
            GL_UNSIGNED_INT,                           # Index type
            None                                       # Start at index 0
        )
        
        # Unbind VAO (good practice)
        glBindVertexArray(0)
        
        # Reset for next batch
        # Note: We don't zero the vertex array - just overwrite next time
        self.sprite_count = 0
