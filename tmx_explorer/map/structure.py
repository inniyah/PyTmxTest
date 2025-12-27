"""
3D structure representation of TMX maps with collision support

=============================================================================
CONCEPTUAL OVERVIEW: 2D MAPS WITH HEIGHT LEVELS
=============================================================================

TMX maps are inherently 2D (X, Y grid of tiles), but many games need the
concept of HEIGHT or ELEVATION (Z axis). Examples:

- Multi-story buildings (floor 1, floor 2, roof)
- Bridges over roads (bridge is higher than road beneath)
- Underground areas (negative Z levels)
- Flying/jumping mechanics

This class extends the 2D TMX map into a 3D structure by using a custom
layer property (Z, z, or level) to assign height levels to layers.

=============================================================================
COORDINATE SYSTEM
=============================================================================

We use a coordinate system suited for top-down 2D games with height:

         Y (Depth - "into screen" in top-down view)
         ^
        /
       /
      +-------> X (Width - horizontal)
      |
      |
      v
      Z (Height - stacking levels, NOT screen depth!)

IMPORTANT NAMING:
- X = Width (W) - horizontal tiles
- Y = Depth (D) - vertical tiles in 2D view (NOT height!)
- Z = Height (H) - elevation levels (floors, bridges, etc.)

This differs from typical 3D conventions where Y is up!
We chose this because in 2D top-down games:
- X/Y are the ground plane (what you see)
- Z represents "layers" of height above/below

=============================================================================
DATA STRUCTURE: 4D NUMPY ARRAY
=============================================================================

The map is stored as a 4D NumPy array: mapa[z, y, x, n]

Dimensions:
- z: Height level (0 to H-1, after offset adjustment)
- y: Depth/row (0 to D-1)
- x: Width/column (0 to W-1)
- n: Layer index (0 to N-1, multiple layers per level)

Why 4D instead of 3D?
- Multiple layers can exist at the same height level
- Example: "ground" layer and "decorations" layer both at Z=0
- The 4th dimension (n) preserves layer separation for rendering order

Example structure:
    Level Z=0 (ground):
        Layer 0: grass tiles
        Layer 1: path tiles (semi-transparent, over grass)
    Level Z=1 (bridge):
        Layer 2: bridge structure
        Layer 3: bridge railings

=============================================================================
COLLISION SYSTEM
=============================================================================

Parallel to the visual map, we maintain a CollisionMap that stores
walkability information. Tiles can be marked as "solid" in Tiled using
a custom property, and this class builds the collision data automatically.

The collision system enables:
- Player movement blocking
- NPC pathfinding
- Physics interactions

=============================================================================
"""

import numpy as np
from typing import List, Tuple, Dict
from tmx_manager import TiledMap, TileLayer, LayerGroup
from .collision import CollisionMap


class Map3DStructure:
    """
    3D representation of a TMX map.
    
    Converts a standard 2D TMX tile map into a 3D structure by interpreting
    layer properties as height levels. Also builds collision data.
    
    ==========================================================================
    USAGE EXAMPLE
    ==========================================================================
    
    ```python
    # Load TMX map
    tmx_map = TiledMap.load("level.tmx")
    
    # Create 3D structure
    map_3d = Map3DStructure(tmx_map)
    
    # Access tiles
    gid = map_3d.get_tile(x=5, y=10, z=0, layer=0)
    
    # Check collision
    if map_3d.is_walkable(pixel_x, pixel_y, z_level):
        player.move(...)
    ```
    
    ==========================================================================
    TMX LAYER SETUP (in Tiled Editor)
    ==========================================================================
    
    To use height levels, add a custom property to layers in Tiled:
    
    1. Select a layer
    2. Add custom property: Name="Z" (or "z" or "level"), Type=int
    3. Set value: 0 for ground, 1 for elevated, -1 for underground, etc.
    
    Layers without this property default to Z=0.
    
    ==========================================================================
    """

    def __init__(self, tmx_map: TiledMap):
        """
        Initialize 3D map structure from TMX map.
        
        Parameters:
        -----------
        tmx_map : TiledMap
            Parsed TMX map object containing layers, tilesets, etc.
            
        Process:
        --------
        1. Extract basic map dimensions
        2. Recursively find all tile layers (including nested groups)
        3. Determine height range from layer properties
        4. Allocate 4D array for tile storage
        5. Load tile GIDs into array
        6. Build collision map from tile properties
        """
        # =====================================================================
        # BASIC MAP DIMENSIONS
        # =====================================================================
        
        # Map dimensions in tiles (from TMX)
        self.map_width = tmx_map.width    # Number of tile columns
        self.map_height = tmx_map.height  # Number of tile rows
        
        # Tile dimensions in pixels (for coordinate conversion)
        self.tile_width = tmx_map.tilewidth    # Usually 16, 32, or 64
        self.tile_height = tmx_map.tileheight
        
        # Aliases for cleaner dimension access
        self.W = self.map_width   # Width (X dimension)
        self.D = self.map_height  # Depth (Y dimension) - NOT "height"!

        # =====================================================================
        # LAYER EXTRACTION
        # =====================================================================
        
        # layer_info stores tuples of (layer_object, z_level, full_name)
        # This intermediate list helps us process layers before array creation
        self.layer_info: List[Tuple[TileLayer, int, str]] = []
        self._extract_all_layers(tmx_map)

        # =====================================================================
        # HEIGHT LEVEL CALCULATION
        # =====================================================================
        
        # Collect all unique Z levels from layers
        # Using dict as ordered set (Python 3.7+ preserves insertion order)
        self.levels: Dict[int, bool] = {}
        for _, level, _ in self.layer_info:
            self.levels[level] = True

        # Calculate level range
        # Example: levels = {-1, 0, 1, 2} → min=-1, max=2, H=4
        self.min_level = min(self.levels.keys()) if self.levels else 0
        self.max_level = max(self.levels.keys()) if self.levels else 0
        
        # H = total number of height levels
        self.H = self.max_level - self.min_level + 1
        
        # level_offset converts TMX level values to array indices
        # Example: min_level=-1 → offset=1
        #          TMX level -1 → array index 0
        #          TMX level  0 → array index 1
        #          TMX level  1 → array index 2
        self.level_offset = -self.min_level
        
        # N = number of tile layers
        self.N = len(self.layer_info)

        # =====================================================================
        # 4D TILE ARRAY ALLOCATION
        # =====================================================================
        
        # Main data structure: 4D array of tile GIDs
        # Shape: [height_levels, depth, width, num_layers]
        # dtype=uint16: Supports GIDs up to 65535 (plenty for most maps)
        #               Uses 2 bytes per tile (memory efficient)
        self.mapa = np.zeros((self.H, self.D, self.W, self.N), dtype=np.uint16)
        
        # Parallel lists for layer metadata
        self.layer_names: List[str] = []   # Human-readable names
        self.layer_levels: List[int] = []  # Original Z values (not indices)

        # Debug output
        print(f"\n=== 3D Map Structure ===")
        print(f"Dimensions: W={self.W}, D={self.D}, H={self.H}, N={self.N}")

        # =====================================================================
        # LOAD TILE DATA
        # =====================================================================
        self._load_layers()
        
        # =====================================================================
        # COLLISION MAP
        # =====================================================================
        # Create collision map with same dimensions as tile map
        self.collision = CollisionMap(self.W, self.H, self.D)
        self._build_collision_map(tmx_map.tilesets)

    # =========================================================================
    # LAYER EXTRACTION
    # =========================================================================

    def _extract_all_layers(self, tmx_map: TiledMap):
        """
        Recursively extract all tile layers from map.
        
        TMX maps can have:
        - TileLayer: Actual tile data (what we want)
        - LayerGroup: Folder containing other layers (recursive)
        - ObjectLayer: Vector objects (not tile data, ignored here)
        - ImageLayer: Background images (not tile data, ignored here)
        
        This method finds ALL TileLayers, even nested in groups.
        
        =======================================================================
        LAYER GROUPS IN TILED
        =======================================================================
        
        Tiled allows organizing layers into folders (groups):
        
        Layers Panel:
        ├── Background (group)
        │   ├── Sky
        │   └── Mountains
        ├── Ground (group)
        │   ├── Terrain
        │   └── Decorations
        └── Foreground
        
        We flatten this hierarchy but preserve the path in names:
        - "Background/Sky"
        - "Background/Mountains"
        - "Ground/Terrain"
        etc.
        
        =======================================================================
        """
        
        def process_layer(layer, layer_name: str = ""):
            """
            Process a single layer (recursive for groups).
            
            Parameters:
            -----------
            layer : TileLayer | LayerGroup | other
                The layer to process
            layer_name : str
                Accumulated path name (e.g., "ParentGroup/SubGroup")
            """
            if isinstance(layer, TileLayer):
                # This is a tile layer - extract it!
                level = self._get_layer_level(layer)
                
                # Use layer's own name if no parent path
                full_name = layer_name or layer.name
                
                # Store for later processing
                self.layer_info.append((layer, level, full_name))
                
            elif isinstance(layer, LayerGroup):
                # This is a group - recurse into its children
                for sublayer in layer.layers:
                    # Build hierarchical name: "Parent/Child"
                    if layer_name:
                        subname = f"{layer_name}/{sublayer.name}"
                    else:
                        subname = sublayer.name
                    
                    # Recursive call
                    process_layer(sublayer, subname)
            
            # Other layer types (ObjectLayer, ImageLayer) are silently ignored

        # Process all top-level layers
        for layer in tmx_map.layers:
            process_layer(layer)

    def _get_layer_level(self, layer: TileLayer) -> int:
        """
        Extract Z level from layer properties.
        
        Looks for custom property named 'Z', 'z', or 'level' on the layer.
        Returns 0 if no such property exists.
        
        =======================================================================
        PROPERTY VALUE HANDLING
        =======================================================================
        
        Tiled can store property values as different types:
        - int: Direct use
        - float: Convert to int (truncate)
        - string: Parse as int (supports hex: "0x10" → 16)
        
        The string parsing with int(value, 0) auto-detects base:
        - "42" → 42 (decimal)
        - "0x2A" → 42 (hexadecimal)
        - "0b101010" → 42 (binary)
        
        =======================================================================
        """
        level = 0
        
        # Check for Z level property (try multiple common names)
        for prop_name in ('Z', 'z', 'level'):
            if prop_name in layer.properties:
                level = layer.properties[prop_name].value
                break  # Use first match
        
        # Convert to integer if necessary
        if isinstance(level, str):
            # Parse string to int (base 0 = auto-detect)
            level = int(level, 0)
        elif isinstance(level, float):
            # Truncate float to int
            level = int(level)
            
        return level

    # =========================================================================
    # TILE DATA LOADING
    # =========================================================================

    def _load_layers(self):
        """
        Load tile data into 4D array.
        
        Iterates through all extracted layers and copies tile GIDs into
        the appropriate positions in the 4D array.
        
        =======================================================================
        ARRAY INDEXING
        =======================================================================
        
        mapa[z, y, x, layer_idx] = gid
        
        - z: Height index (0 to H-1), converted from level using offset
        - y: Row in tile grid (0 to D-1)
        - x: Column in tile grid (0 to W-1)
        - layer_idx: Which layer (0 to N-1)
        - gid: Global tile ID (0 = empty, >0 = tile reference)
        
        =======================================================================
        BOUNDS CHECKING
        =======================================================================
        
        We use min(layer.dimension, map.dimension) to handle cases where
        a layer might be smaller than the map (Tiled allows this).
        Layers larger than the map are clipped to map bounds.
        
        =======================================================================
        """
        for layer_idx, (layer, level, layer_name) in enumerate(self.layer_info):
            # Convert TMX level value to array index
            z = level + self.level_offset
            
            # Skip layers outside our height range (shouldn't happen normally)
            if not (0 <= z < self.H):
                continue

            # Copy tile data from layer to 4D array
            for y in range(min(layer.height, self.D)):
                for x in range(min(layer.width, self.W)):
                    # Get tile GID from TMX layer
                    gid = layer.get_tile_gid(x, y)
                    
                    # Only store non-empty tiles (GID > 0)
                    # GID 0 means empty, and our array is zero-initialized
                    if gid > 0:
                        self.mapa[z, y, x, layer_idx] = gid

            # Store layer metadata
            self.layer_names.append(layer_name)
            self.layer_levels.append(level)  # Original level, not z index

    # =========================================================================
    # COLLISION MAP BUILDING
    # =========================================================================

    def _build_collision_map(self, tilesets):
        """
        Build collision map using the 'solid' property of tiles.
        
        Scans all tiles in all tilesets for a custom 'solid' property,
        then marks corresponding positions in the collision map.
        
        =======================================================================
        TILE PROPERTIES IN TILED
        =======================================================================
        
        In Tiled Editor, you can add custom properties to individual tiles:
        
        1. Open tileset editor
        2. Select a tile
        3. Add property: Name="solid", Type=bool, Value=true
        
        This marks the tile as blocking movement.
        
        Common patterns:
        - solid=true: Walls, obstacles, furniture
        - solid=false (or no property): Floor, paths, decorations
        
        =======================================================================
        TWO-PHASE APPROACH
        =======================================================================
        
        Phase 1: Build GID → solid lookup table
            Faster than checking tile properties during the grid scan.
            O(num_tiles_with_properties) preprocessing.
        
        Phase 2: Scan entire map grid
            For each position, check if any tile there is solid.
            Uses the prebuilt lookup for O(1) per tile.
        
        =======================================================================
        """
        print("\n=== Building Collision Map ===")
        
        # -----------------------------------------------------------------
        # PHASE 1: BUILD SOLID LOOKUP TABLE
        # -----------------------------------------------------------------
        # Maps GID → is_solid (bool)
        # Only tiles with explicit 'solid' property are included
        solid_lookup: Dict[int, bool] = {}
        
        for tileset in tilesets:
            # tileset.tiles is a dict of {local_id: tile_data}
            for tile_id, tile in tileset.tiles.items():
                # Calculate Global ID
                gid = tileset.firstgid + tile_id
                
                # Check for 'solid' property
                if 'solid' in tile.properties:
                    prop = tile.properties['solid']
                    
                    # Handle different property value types
                    if isinstance(prop.value, bool):
                        # Direct boolean
                        solid_lookup[gid] = prop.value
                    else:
                        # String: parse "true"/"false"
                        solid_lookup[gid] = str(prop.value).lower() == 'true'
        
        # Debug statistics
        print(f"Tiles with solid property: {len(solid_lookup)}")
        print(f"Marked as solid: {sum(1 for v in solid_lookup.values() if v)}")
        
        # -----------------------------------------------------------------
        # PHASE 2: SCAN MAP AND MARK COLLISIONS
        # -----------------------------------------------------------------
        # For each position in the 3D grid, check if any layer has a solid tile
        
        for z in range(self.H):
            for y in range(self.D):
                for x in range(self.W):
                    is_solid = False
                    
                    # Check all layers at this position
                    for n in range(self.N):
                        # Only check layers at this height level
                        # (Multiple layers can share a level)
                        if self.layer_levels[n] == self.get_level_value(z):
                            gid = self.mapa[z, y, x, n]
                            
                            # Check if this tile is solid
                            # Default to False if GID not in lookup
                            if gid != 0 and solid_lookup.get(gid, False):
                                is_solid = True
                                break  # No need to check more layers
                    
                    # Mark in collision map
                    if is_solid:
                        self.collision.set_flags(x, y, z, 1)  # 1 = solid
        
        # Print collision statistics
        stats = self.collision.get_stats()
        print(f"Solid tiles: {stats['solid_tiles']} ({stats['solid_percent']:.1f}%)")
        print(f"Empty tiles: {stats['empty_tiles']}")

    # =========================================================================
    # COORDINATE CONVERSION
    # =========================================================================

    def get_level_value(self, z: int) -> int:
        """
        Convert internal Z index to original level value.
        
        The internal array uses indices 0 to H-1, but TMX levels might
        be -2, -1, 0, 1, 2, etc. This method converts back.
        
        Parameters:
        -----------
        z : int
            Internal array index (0 to H-1)
            
        Returns:
        --------
        int : Original TMX level value
        
        Example:
        --------
        If min_level = -2, then level_offset = 2
        z=0 → level = 0 - 2 = -2
        z=1 → level = 1 - 2 = -1
        z=2 → level = 2 - 2 = 0
        etc.
        """
        return z - self.level_offset

    # =========================================================================
    # TILE ACCESS
    # =========================================================================

    def get_tile(self, x: int, y: int, z: int, layer: int) -> int:
        """
        Get tile GID at position.
        
        Parameters:
        -----------
        x : int
            Tile column (0 to W-1)
        y : int
            Tile row (0 to D-1)
        z : int
            Height index (0 to H-1) - NOT the original level value!
        layer : int
            Layer index (0 to N-1)
            
        Returns:
        --------
        int : Tile GID (0 if empty or out of bounds)
        
        Note:
        -----
        Bounds checking returns 0 for out-of-bounds access.
        This is safer than raising exceptions for game logic that might
        query positions outside the map (e.g., player at map edge).
        """
        if (0 <= x < self.W and 0 <= y < self.D and 
            0 <= z < self.H and 0 <= layer < self.N):
            return self.mapa[z, y, x, layer]
        return 0

    def set_tile(self, x: int, y: int, z: int, layer: int, gid: int):
        """
        Set tile GID at position.
        
        Parameters:
        -----------
        x, y, z, layer : int
            Position coordinates (see get_tile for details)
        gid : int
            New tile GID to set (0 to clear, >0 for tile reference)
            
        Note:
        -----
        Silently ignores out-of-bounds writes.
        Does NOT update collision map - call _build_collision_map()
        if tiles affecting collision are modified.
        """
        if (0 <= x < self.W and 0 <= y < self.D and 
            0 <= z < self.H and 0 <= layer < self.N):
            self.mapa[z, y, x, layer] = gid
    
    # =========================================================================
    # COLLISION CHECKING
    # =========================================================================

    def is_walkable(self, px: float, py: float, z: float) -> bool:
        """
        Check if a pixel position is walkable.
        
        Converts pixel coordinates to tile coordinates and checks
        the collision map.
        
        Parameters:
        -----------
        px : float
            X position in pixels
        py : float
            Y position in pixels
        z : float
            Height level (will be converted to int)
            
        Returns:
        --------
        bool : True if the position is walkable (not solid)
        
        Use Case:
        ---------
        Simple point collision - checks if a single point is blocked.
        Good for: projectiles, particles, simple entities.
        """
        return self.collision.can_move_to(
            px, py, z, 
            self.tile_width, self.tile_height
        )
    
    def is_walkable_with_size(self, px: float, py: float, z: float,
                               char_width: float, char_height: float) -> bool:
        """
        Check if a pixel position is walkable considering character size.
        
        Unlike is_walkable() which checks a single point, this method
        checks all tiles that a character's bounding box would overlap.
        
        Parameters:
        -----------
        px, py : float
            Character position in pixels (typically center or corner)
        z : float
            Height level
        char_width, char_height : float
            Character bounding box size in pixels
            
        Returns:
        --------
        bool : True if the character can occupy this position
        
        =======================================================================
        WHY SIZE MATTERS
        =======================================================================
        
        A 32x32 pixel character might span multiple tiles:
        
        +-----+-----+-----+
        |     |XXXXX|     |   X = Character bounding box
        |     |XXXXX|     |   
        +-----+-----+-----+
        |     |XXXXX|     |
        |     |XXXXX|     |
        +-----+-----+-----+
        
        Point collision would only check one tile.
        Size-aware collision checks ALL overlapped tiles.
        
        If ANY of those tiles is solid, movement is blocked.
        
        =======================================================================
        """
        return self.collision.can_move_to_with_size(
            px, py, z, 
            char_width, char_height,
            self.tile_width, self.tile_height
        )
