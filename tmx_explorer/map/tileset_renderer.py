"""
Tileset loading and tile texture management (GLFW version - uses PIL)

=============================================================================
WHAT IS A TILESET?
=============================================================================

A tileset is a collection of tile graphics used in a tile-based map.
Tiled Map Editor (TMX) supports two types of tilesets:

1. IMAGE-BASED TILESET (most common):
   A single large image containing all tiles arranged in a grid.
   
   Example: A 256x256 image with 16x16 tiles = 16×16 = 256 tiles
   
   +---+---+---+---+---+---+---+---+
   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |   <- Row 0: tiles 0-7
   +---+---+---+---+---+---+---+---+
   | 8 | 9 |10 |11 |12 |13 |14 |15 |   <- Row 1: tiles 8-15
   +---+---+---+---+---+---+---+---+
   |16 |17 |18 |... etc              |
   
   Advantages:
   - Single file to manage
   - Efficient loading (one disk read)
   - Common in retro/pixel art games

2. IMAGE COLLECTION TILESET:
   Each tile is a separate image file.
   
   tiles/
   ├── grass.png      <- tile 0
   ├── dirt.png       <- tile 1
   ├── water.png      <- tile 2
   └── ...
   
   Advantages:
   - Tiles can have different sizes
   - Easier to manage individual assets
   - Common for larger/varied tiles (isometric, etc.)

=============================================================================
GLOBAL TILE IDS (GIDs)
=============================================================================

TMX maps use Global IDs (GIDs) to reference tiles across ALL tilesets.

Example with two tilesets:
- Tileset A: firstgid=1, 100 tiles → GIDs 1-100
- Tileset B: firstgid=101, 50 tiles → GIDs 101-150

GID 0 is special: it means "empty tile" (no graphic)

The local tile_id within a tileset + firstgid = GID
  gid = tileset.firstgid + local_tile_id

=============================================================================
PRE-LOADING STRATEGY
=============================================================================

This renderer PRE-LOADS all tiles at startup rather than loading on-demand.

Why pre-load everything?
1. PREDICTABLE PERFORMANCE: No stuttering during gameplay
2. FASTER RENDERING: All textures already in GPU memory
3. SIMPLER CODE: No need for async loading or cache management
4. MEMORY IS CHEAP: A typical tileset uses only a few MB

Trade-off: Longer initial load time, but worth it for smooth gameplay.

=============================================================================
"""

from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from tmx_manager import TiledMap

# TYPE_CHECKING block: imports only used for type hints, not at runtime.
# This avoids circular import issues while still getting type checking benefits.
if TYPE_CHECKING:
    from ..renderer.opengl_renderer import OpenGLRenderer
    from ..renderer.texture import Texture


class TilesetRenderer:
    """
    Loads tilesets and manages tile textures with bleeding prevention.
    
    This class bridges TMX map data and OpenGL rendering by:
    1. Parsing tileset definitions from TMX
    2. Loading tile images from disk
    3. Creating GPU textures for each tile
    4. Providing fast GID → Texture lookup during rendering
    
    ==========================================================================
    ARCHITECTURE ROLE
    ==========================================================================
    
    TMX File → [TiledMap Parser] → [TilesetRenderer] → [OpenGLRenderer]
                                          ↓
                                   tile_texture_cache
                                   (GID → Texture mapping)
    
    The renderer queries us with a GID, we return the pre-loaded texture.
    No disk I/O or image processing during rendering!
    
    ==========================================================================
    BLEEDING PREVENTION
    ==========================================================================
    
    Each tile texture is created with a 1-pixel extruded border (handled by
    the Texture class). This prevents visual artifacts when tiles are
    rendered at non-integer positions or with zoom.
    
    See texture.py documentation for detailed explanation.
    
    ==========================================================================
    """

    def __init__(self, tmx_map: TiledMap, tmx_path: str, gl_renderer: 'OpenGLRenderer'):
        """
        Initialize tileset renderer and load all tilesets.
        
        Parameters:
        -----------
        tmx_map : TiledMap
            Parsed TMX map object (from tmx_manager library)
            Contains tileset definitions, layers, objects, etc.
            
        tmx_path : str
            Path to the .tmx file.
            Used to resolve relative paths to tileset images.
            Example: "maps/level1.tmx" → tileset at "maps/tiles/grass.png"
            
        gl_renderer : OpenGLRenderer
            Reference to the OpenGL renderer for texture creation.
            We use gl_renderer.preload_texture() to upload images to GPU.
            
        =======================================================================
        PATH HANDLING
        =======================================================================
        
        TMX files reference tilesets with RELATIVE paths. We need to:
        1. Get the directory containing the .tmx file
        2. Resolve tileset paths relative to that directory
        
        Example:
            tmx_path = "assets/maps/level1.tmx"
            tileset.image.source = "tilesets/terrain.png"
            actual_path = "assets/maps/tilesets/terrain.png"
        
        =======================================================================
        """
        self.tmx_map = tmx_map
        
        # Store the DIRECTORY containing the TMX file (not the file itself)
        # This is our base path for resolving relative tileset paths
        self.tmx_path = Path(tmx_path).parent
        
        self.gl_renderer = gl_renderer
        
        # =====================================================================
        # CACHES
        # =====================================================================
        
        # tile_size_cache: GID → (width, height) of the ORIGINAL tile
        # Note: This is the logical tile size, NOT including the border pixels
        # added by the Texture class. The renderer needs the original size
        # to position tiles correctly.
        self.tile_size_cache: Dict[int, Tuple[int, int]] = {}
        
        # tile_texture_cache: GID → Texture object
        # The actual GPU texture for each tile, ready for rendering.
        # This is the primary lookup used during rendering.
        self.tile_texture_cache: Dict[int, 'Texture'] = {}
        
        # Load all tilesets immediately
        # By the time __init__ returns, all tiles are in GPU memory
        self._load_tilesets()

    # =========================================================================
    # TILESET LOADING
    # =========================================================================

    def _load_tilesets(self):
        """
        Load all tilesets from the map.
        
        Iterates through all tilesets defined in the TMX map and loads them.
        TMX maps can have multiple tilesets, each with its own firstgid.
        
        =======================================================================
        TILESET TYPES DETECTION
        =======================================================================
        
        We detect tileset type by checking which properties are set:
        
        1. If tileset.image exists → Image-based tileset (single spritesheet)
        2. If tileset.tiles exists → Image collection (individual files)
        
        Each type needs different loading logic.
        
        =======================================================================
        EXTERNAL VS EMBEDDED TILESETS
        =======================================================================
        
        Tilesets can be:
        
        1. EMBEDDED: Tileset data is inside the .tmx file
           - tileset.source is None
           - Image paths are relative to .tmx file
        
        2. EXTERNAL: Tileset data is in a separate .tsx file
           - tileset.source = "path/to/tileset.tsx"
           - Image paths are relative to .tsx file, NOT .tmx file!
        
        This is why we recalculate tileset_base for external tilesets.
        """
        print("\n=== Loading Tilesets ===")
        
        for tileset in self.tmx_map.tilesets:
            # Determine base path for this tileset's images
            tileset_base = self.tmx_path
            
            # If tileset is external (.tsx file), images are relative to IT
            if tileset.source:
                # tileset.source = "tilesets/terrain.tsx"
                # tileset_base = tmx_path / "tilesets/"
                tileset_base = self.tmx_path / Path(tileset.source).parent

            # Dispatch to appropriate loader based on tileset type
            if tileset.image:
                # Single image containing all tiles (spritesheet)
                self._load_image_tileset(tileset, tileset_base)
            elif tileset.tiles:
                # Collection of individual tile images
                self._load_collection_tileset(tileset, tileset_base)

    def _load_image_tileset(self, tileset, tileset_base: Path):
        """
        Load a tileset based on a single image (spritesheet).
        
        This is the most common tileset type. A single large image contains
        all tiles arranged in a grid pattern.
        
        Parameters:
        -----------
        tileset : TMX Tileset object
            Contains metadata: tilewidth, tileheight, columns, tilecount, etc.
        tileset_base : Path
            Directory to resolve image path from
            
        =======================================================================
        PROCESS
        =======================================================================
        
        1. Load the full tileset image from disk
        2. Convert to RGBA (for transparency support)
        3. Pass to _preload_tileset_tiles() to extract individual tiles
        
        =======================================================================
        """
        # Resolve full path to tileset image
        image_path = tileset_base / tileset.image.source
        
        try:
            # Load image and ensure RGBA format for transparency
            image = Image.open(str(image_path)).convert('RGBA')
            print(f"Loaded tileset: {tileset.name} ({image.width}x{image.height})")
            
            # Extract and pre-load all individual tiles from this image
            self._preload_tileset_tiles(tileset, image)
            
        except Exception as e:
            # Don't crash if a tileset fails to load - just warn
            # The game might still be playable with missing tiles
            print(f"Warning: Could not load {image_path}: {e}")

    def _load_collection_tileset(self, tileset, tileset_base: Path):
        """
        Load an image collection tileset (individual tile files).
        
        Each tile is a separate image file. This type is common for:
        - Tiles with varying sizes
        - Isometric or large tiles
        - Tiles with complex shapes
        
        Parameters:
        -----------
        tileset : TMX Tileset object
            Contains tiles dict: {local_id: tile_data}
        tileset_base : Path
            Directory to resolve image paths from
            
        =======================================================================
        STRUCTURE OF COLLECTION TILESETS
        =======================================================================
        
        tileset.tiles is a dictionary:
        {
            0: Tile(image=Image(source="grass.png")),
            1: Tile(image=Image(source="dirt.png")),
            5: Tile(image=Image(source="water.png")),  # IDs can be sparse!
            ...
        }
        
        Note: Tile IDs might not be contiguous (0, 1, 5, 10, ...).
        We iterate over whatever tiles are defined.
        
        =======================================================================
        """
        print(f"Loading image collection: {tileset.name} ({len(tileset.tiles)} tiles)")
        
        # Iterate over all tiles defined in this tileset
        for tile_id, tile in tileset.tiles.items():
            if tile.image:  # Tile has an image defined
                # Resolve full path to this tile's image
                image_path = tileset_base / tile.image.source
                
                try:
                    # Load individual tile image
                    tile_img = Image.open(str(image_path)).convert('RGBA')
                    
                    # Calculate GID (Global ID) for this tile
                    # gid = firstgid + local_tile_id
                    gid = tileset.firstgid + tile_id
                    
                    # Cache tile dimensions (original size, no border)
                    self.tile_size_cache[gid] = (tile_img.width, tile_img.height)
                    
                    # Create GPU texture and cache it
                    self.tile_texture_cache[gid] = self.gl_renderer.preload_texture(
                        gid, tile_img
                    )
                    
                except Exception as e:
                    print(f"  Warning: {e}")

    def _preload_tileset_tiles(self, tileset, tileset_image: Image.Image):
        """
        Pre-load ALL tiles from a tileset image (spritesheet).
        
        Extracts each individual tile from the larger tileset image and
        creates a separate GPU texture for each one.
        
        Parameters:
        -----------
        tileset : TMX Tileset object
            Metadata about tile arrangement (size, columns, spacing, margin)
        tileset_image : PIL.Image
            The full tileset image to extract tiles from
            
        =======================================================================
        TILESET LAYOUT PARAMETERS
        =======================================================================
        
        TMX tilesets have several layout parameters:
        
        - tilewidth/tileheight: Size of each tile in pixels
        - columns: Number of tiles per row
        - tilecount: Total number of tiles
        - margin: Pixels around the EDGE of the entire tileset
        - spacing: Pixels BETWEEN tiles
        
        Example layout with margin=2, spacing=1, 16x16 tiles:
        
        +--+--+--+--+--+--+--+--+--+--+  <- 2px margin
        |  |  |  |  |  |  |  |  |  |  |
        +--+================+--+========+
        |  ||    TILE 0    || ||TILE 1 ||  <- 16x16 tile
        |  ||              || ||       ||
        +--+================+--+========+
        |  |              1px spacing
        +--+================+--+========+
        |  ||    TILE 2    || ||TILE 3 ||
        
        =======================================================================
        WHY INDIVIDUAL TEXTURES PER TILE?
        =======================================================================
        
        Alternative approaches:
        
        1. ONE texture for entire tileset, different UV coords per tile
           - Fewer texture switches
           - BUT: Can't add borders, causes tile bleeding!
        
        2. TEXTURE ATLAS with all tiles + borders (what some engines do)
           - Complex UV coordinate management
           - Must rebuild atlas if tiles change
        
        3. INDIVIDUAL texture per tile (our approach)
           - Simple: each tile is independent
           - Easy borders: Texture class adds them automatically
           - Slight overhead from texture switches, but batching helps
           - Best for correctness and simplicity
        
        =======================================================================
        """
        # Sanity check: need at least 1 column to extract tiles
        if tileset.columns <= 0:
            return

        tiles_loaded = 0
        
        # Cache tile dimensions for quick access
        tw = tileset.tilewidth   # Tile width in pixels
        th = tileset.tileheight  # Tile height in pixels
        
        # Process each tile in the tileset
        for tile_id in range(tileset.tilecount):
            # Calculate Global ID
            gid = tileset.firstgid + tile_id
            
            # -----------------------------------------------------------------
            # CALCULATE TILE POSITION IN TILESET IMAGE
            # -----------------------------------------------------------------
            # Convert linear tile_id to 2D grid position
            col = tile_id % tileset.columns   # Column (0 to columns-1)
            row = tile_id // tileset.columns  # Row (0 to rows-1)
            
            # Calculate pixel coordinates of tile's top-left corner
            # Formula accounts for margin (edge padding) and spacing (between tiles)
            #
            # tile_x = column * tile_width + margin + column * spacing
            #        = col * tw + margin + col * spacing
            #
            # Example: col=2, tw=16, margin=2, spacing=1
            # tile_x = 2*16 + 2 + 2*1 = 32 + 2 + 2 = 36
            tile_x = col * tw + tileset.margin + col * tileset.spacing
            tile_y = row * th + tileset.margin + row * tileset.spacing
            
            # -----------------------------------------------------------------
            # EXTRACT TILE FROM TILESET IMAGE
            # -----------------------------------------------------------------
            # PIL crop() takes (left, top, right, bottom) coordinates
            # Returns a new image containing just this tile
            tile_img = tileset_image.crop((
                tile_x,           # Left
                tile_y,           # Top
                tile_x + tw,      # Right
                tile_y + th       # Bottom
            ))
            
            # -----------------------------------------------------------------
            # CACHE TILE DATA
            # -----------------------------------------------------------------
            # Store original tile dimensions (without border)
            self.tile_size_cache[gid] = (tw, th)
            
            # Create GPU texture and cache it
            # preload_texture() handles border addition via Texture.from_pil()
            self.tile_texture_cache[gid] = self.gl_renderer.preload_texture(
                gid, tile_img
            )
            
            tiles_loaded += 1

        print(f"  Pre-loaded {tiles_loaded} tiles")

    # =========================================================================
    # PUBLIC LOOKUP METHODS
    # =========================================================================

    def get_tile_texture(self, gid: int) -> Optional['Texture']:
        """
        Get pre-loaded texture for tile GID.
        
        This is the primary method used during rendering to get the
        GPU texture for a specific tile.
        
        Parameters:
        -----------
        gid : int
            Global tile ID from the TMX map
            
        Returns:
        --------
        Texture or None
            The GPU texture for this tile, or None if:
            - GID is 0 (empty tile)
            - GID not found (missing/unloaded tile)
            
        =======================================================================
        PERFORMANCE NOTE
        =======================================================================
        
        This is called for EVERY visible tile EVERY frame, so it must be fast!
        
        dict.get() is O(1) average case - perfect for this use case.
        No disk I/O, no image processing - just a dictionary lookup.
        
        =======================================================================
        """
        # GID 0 = empty tile (no graphic to render)
        if gid == 0:
            return None
        
        # Lookup in cache (returns None if not found)
        return self.tile_texture_cache.get(gid)

    def get_tile_surface(self, gid: int) -> Optional[Tuple[int, int]]:
        """
        Get tile size (width, height) for GID.
        
        Returns the ORIGINAL tile dimensions, not including any border
        pixels added by the Texture class.
        
        Parameters:
        -----------
        gid : int
            Global tile ID
            
        Returns:
        --------
        Tuple[int, int] or None
            (width, height) in pixels, or None if GID is 0 or not found
            
        =======================================================================
        WHY IS THIS NEEDED?
        =======================================================================
        
        The renderer needs to know tile dimensions to:
        1. Position tiles correctly on screen
        2. Calculate tile boundaries for culling
        3. Handle tiles of different sizes (collection tilesets)
        
        Note: Most image-based tilesets have uniform tile sizes, but
        collection tilesets can have varying sizes per tile.
        
        =======================================================================
        NAMING NOTE
        =======================================================================
        
        "Surface" is terminology from Pygame/SDL. In this OpenGL renderer,
        we're returning just the size tuple, not a surface object.
        The name is kept for API compatibility with other renderers.
        
        =======================================================================
        """
        # GID 0 = empty tile
        if gid == 0:
            return None
        
        # Lookup in cache
        return self.tile_size_cache.get(gid)
