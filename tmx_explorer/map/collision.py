"""
Collision system using a parallel numpy array

=============================================================================
COLLISION DETECTION OVERVIEW
=============================================================================

This module implements tile-based collision detection for a 3D game world.
Instead of checking collisions against individual objects, we use a 3D grid
that mirrors the tile map structure.

WHY TILE-BASED COLLISION?
-------------------------
1. FAST: O(1) lookup - just index into array
2. SIMPLE: No complex geometry intersection math
3. MEMORY EFFICIENT: 2 bytes per tile (uint16 flags)
4. PREDICTABLE: Same performance regardless of map complexity

Alternative approaches and why we didn't use them:
- Per-pixel collision: Too slow, too much memory
- Polygon collision: Complex math, harder to debug
- Physics engine (Box2D, etc.): Overkill for tile games, adds dependencies

=============================================================================
WORLD UNITS AND SCALE
=============================================================================

This system uses a consistent unit scale:

    1 tile = 1 meter × 1 meter × 2 meters (width × depth × height)
    
Why this scale?
- 1 tile width/depth = 1 meter → Human ~0.5 tiles wide (50cm shoulders)
- 1 tile height = 2 meters → Human ~0.85 levels tall (1.7m)
- Makes character sizing intuitive
- Standard doorways fit in 1 tile width

Coordinate spaces:
- PIXEL coordinates: Screen/rendering space (float)
- TILE coordinates: Grid indices (int)
- LEVEL coordinates: Height/Z axis (float, but often integer levels)

=============================================================================
FLAG-BASED COLLISION DATA
=============================================================================

Each cell stores a uint16 (16-bit) value that can encode multiple flags:

    Bit 0: Solid (blocks movement)
    Bit 1-15: Reserved for future use

Future flag ideas:
- Bit 1: Water (swimming physics)
- Bit 2: Ladder (climbing allowed)
- Bit 3: Damage (lava, spikes)
- Bit 4: Slow (mud, sand)
- Bit 5: One-way platform (can jump through from below)

Currently we only use solid/empty, but the flag system allows expansion.

=============================================================================
"""

import numpy as np
from typing import Tuple, List


class CollisionMap:
    """
    Collision map parallel to the tile map.
    
    A 3D NumPy array storing collision flags for each tile position.
    Enables fast collision queries for game entities.
    
    ==========================================================================
    ARCHITECTURE
    ==========================================================================
    
    The CollisionMap mirrors the Map3DStructure but stores collision data
    instead of tile GIDs:
    
    Map3DStructure.mapa[z, y, x, layer] = GID (visual)
    CollisionMap.data[z, y, x] = flags (collision)
    
    Note: CollisionMap has no layer dimension - collision is per-position,
    not per-layer. Multiple solid tiles at the same position just mean
    "this position is solid."
    
    ==========================================================================
    COORDINATE CONVENTIONS
    ==========================================================================
    
    Array indexing: data[z, y, x]
    - z: Height level (0 = ground, 1 = elevated, etc.)
    - y: Depth in tiles (row, increases "into screen" in top-down view)
    - x: Width in tiles (column, increases rightward)
    
    This z-first ordering matches NumPy's convention for 3D arrays and
    enables efficient slicing by height level: data[z, :, :] = entire level
    
    ==========================================================================
    USAGE EXAMPLE
    ==========================================================================
    
    ```python
    # Create collision map
    collision = CollisionMap(width=100, height=3, depth=100)
    
    # Mark a tile as solid
    collision.set_flags(x=10, y=20, z=0, flags=1)
    
    # Check if player can move to position
    if collision.can_move_to_with_size(px, py, z, char_w, char_d, char_h, tw, th):
        player.move(...)
    ```
    
    ==========================================================================
    """
    
    def __init__(self, width: int, height: int, depth: int):
        """
        Initialize collision map with given dimensions.
        
        Parameters:
        -----------
        width : int
            Map width in tiles (X axis)
        height : int
            Number of height levels (Z axis) - NOT pixel height!
        depth : int
            Map depth in tiles (Y axis)
            
        Note on parameter naming:
        - 'height' here means vertical LEVELS, not the Y dimension
        - This matches the Map3DStructure convention (H = height levels)
        - Can be confusing! Remember: height = Z levels, depth = Y tiles
        """
        # Store dimensions with clear names
        self.W = width   # Width (X) in tiles
        self.H = height  # Height levels (Z)
        self.D = depth   # Depth (Y) in tiles
        
        # =====================================================================
        # ALLOCATE 3D COLLISION ARRAY
        # =====================================================================
        # Shape: [height_levels, depth, width]
        # dtype=uint16: 16 bits for flags (currently only using 1 bit)
        #               Could use uint8, but uint16 allows future expansion
        #               Memory: 100x100x3 map = 60KB (negligible)
        #
        # Initialized to zeros = all tiles walkable by default
        self.data = np.zeros((self.H, self.D, self.W), dtype=np.uint16)
        
        print(f"CollisionMap created: {self.W}x{self.D}x{self.H} (W x D x H)")
    
    # =========================================================================
    # BASIC FLAG OPERATIONS
    # =========================================================================
    
    def set_flags(self, x: int, y: int, z: int, flags: int):
        """
        Set collision flags at a position.
        
        Parameters:
        -----------
        x, y, z : int
            Tile coordinates
        flags : int
            Collision flags to set (0 = walkable, non-zero = solid)
            
        Note:
        -----
        Silently ignores out-of-bounds coordinates.
        This is intentional - building collision maps shouldn't crash
        if a tile is slightly outside expected bounds.
        """
        if self._in_bounds(x, y, z):
            self.data[z, y, x] = flags
    
    def get_flags(self, x: int, y: int, z: int) -> int:
        """
        Get collision flags at a position.
        
        Parameters:
        -----------
        x, y, z : int
            Tile coordinates
            
        Returns:
        --------
        int : Collision flags (0 = walkable, non-zero = solid)
        
        =======================================================================
        OUT-OF-BOUNDS BEHAVIOR
        =======================================================================
        
        Returns 1 (solid) for out-of-bounds coordinates.
        
        This is a CRITICAL design decision:
        - Prevents entities from walking off the map edge
        - No need for special boundary checking in movement code
        - "The void is solid" - you can't walk into nothingness
        
        Alternative approaches:
        - Return 0 (walkable): Entities could escape map bounds
        - Raise exception: Would crash game on edge cases
        - Return special value: Complicates calling code
        
        =======================================================================
        """
        if self._in_bounds(x, y, z):
            return int(self.data[z, y, x])
        return 1  # Out of bounds = solid (can't walk off map)
    
    def is_solid(self, x: int, y: int, z: int) -> bool:
        """
        Check if a tile is solid (blocks movement).
        
        Convenience method - equivalent to get_flags() != 0
        
        Returns:
        --------
        bool : True if tile blocks movement
        """
        return self.get_flags(x, y, z) != 0
    
    def is_walkable(self, x: int, y: int, z: int) -> bool:
        """
        Check if a tile is walkable (allows movement).
        
        Convenience method - equivalent to get_flags() == 0
        
        Returns:
        --------
        bool : True if tile allows movement
        """
        return self.get_flags(x, y, z) == 0
    
    def _in_bounds(self, x: int, y: int, z: int) -> bool:
        """
        Check if coordinates are within map bounds.
        
        Parameters:
        -----------
        x, y, z : int
            Tile coordinates to check
            
        Returns:
        --------
        bool : True if all coordinates are valid
        
        Note:
        -----
        Uses chained comparison (0 <= x < W) which is Pythonic and efficient.
        All three dimensions must be in bounds for the position to be valid.
        """
        return (0 <= x < self.W and 0 <= y < self.D and 0 <= z < self.H)
    
    # =========================================================================
    # COORDINATE CONVERSION
    # =========================================================================
    
    def pixel_to_tile(self, px: float, py: float, 
                      tile_width: int, tile_height: int) -> Tuple[int, int]:
        """
        Convert pixel coordinates to tile coordinates.
        
        Parameters:
        -----------
        px, py : float
            Position in pixels (can be fractional)
        tile_width, tile_height : int
            Size of tiles in pixels
            
        Returns:
        --------
        Tuple[int, int] : (tile_x, tile_y) coordinates
        
        =======================================================================
        CONVERSION MATH
        =======================================================================
        
        tile_x = floor(pixel_x / tile_width)
        
        Example with 32px tiles:
        - px=0   → tx=0 (start of tile 0)
        - px=31  → tx=0 (still in tile 0)
        - px=32  → tx=1 (start of tile 1)
        - px=63  → tx=1 (still in tile 1)
        - px=-1  → tx=-1 (off map to left)
        
        Using // (floor division) handles negative coordinates correctly:
        - -1 // 32 = -1 (not 0!)
        
        This is important for entities that might temporarily be at
        negative pixel positions during movement calculations.
        
        =======================================================================
        """
        tx = int(px // tile_width)
        ty = int(py // tile_height)
        return tx, ty
    
    # =========================================================================
    # HEIGHT/Z-LEVEL CALCULATIONS
    # =========================================================================
    
    def get_z_levels_to_check(self, z: float, char_height: float = 0.85) -> List[int]:
        """
        Get the height levels that a character occupies.
        
        Characters have vertical extent (height), so they may span multiple
        Z levels simultaneously. This method returns all levels to check.
        
        Parameters:
        -----------
        z : float
            Current Z position (character's feet level)
        char_height : float
            Character height in Z levels (default 0.85 ≈ 1.7m if 1 level = 2m)
            
        Returns:
        --------
        List[int] : Z levels the character occupies (usually 1-2 levels)
        
        =======================================================================
        HEIGHT SPANNING EXPLAINED
        =======================================================================
        
        A character standing at z=0.5 with height=0.85:
        - Feet at z=0.5
        - Head at z=0.5 + 0.85 = 1.35
        - Occupies levels 0 AND 1
        
              Level 2  +--------+
                       |        |
              Level 1  +---XX---+  <- Head at 1.35
                       |   XX   |
              Level 0  +---XX---+  <- Feet at 0.5
                       |        |
                       
        We need to check BOTH levels for collision!
        
        A character at z=0.0 with height=0.85:
        - Feet at z=0.0
        - Head at z=0.85
        - Occupies ONLY level 0 (doesn't quite reach level 1)
        
        =======================================================================
        UNIT SCALE REMINDER
        =======================================================================
        
        1 Z level = 2 meters (by our convention)
        char_height = 0.85 levels = 1.7 meters (average human height)
        
        This scale means:
        - Most characters fit within one level
        - Only when straddling level boundaries do we check multiple
        - Standard ceilings (1 level = 2m) comfortably fit characters
        
        =======================================================================
        """
        # Floor level (where feet are)
        z_floor = int(z)
        
        # Ceiling level (where head is)
        z_top = z + char_height
        z_ceil = int(z_top)
        
        # Build list of levels to check
        levels = []
        
        # Always include floor level (if valid)
        if 0 <= z_floor < self.H:
            levels.append(z_floor)
        
        # Include ceiling level if different from floor (and valid)
        # This handles the case where character spans two levels
        if z_ceil != z_floor and 0 <= z_ceil < self.H:
            levels.append(z_ceil)
        
        return levels
    
    # =========================================================================
    # MOVEMENT COLLISION CHECKING
    # =========================================================================
    
    def can_move_to_with_size(self, px: float, py: float, z: float,
                               char_width: float, char_depth: float, char_height: float,
                               tile_width: int, tile_height: int) -> bool:
        """
        Check collision considering character's full 3D bounding box.
        
        This is the PRIMARY collision check method for character movement.
        It accounts for the character's width, depth, and height.
        
        Parameters:
        -----------
        px, py : float
            Position in pixels (CENTER-BOTTOM of character)
            Center-bottom is common for top-down games: sprite's "feet" position
        z : float
            Height level (character's feet level)
        char_width : float
            Character width in TILES (e.g., 0.5 = half a tile = 50cm)
        char_depth : float
            Character depth in TILES (e.g., 0.5 = half a tile = 50cm)
        char_height : float
            Character height in Z LEVELS (e.g., 0.85 = 1.7m)
        tile_width, tile_height : int
            Tile size in pixels
            
        Returns:
        --------
        bool : True if character can occupy this position (no collision)
        
        =======================================================================
        BOUNDING BOX APPROACH
        =======================================================================
        
        Instead of checking every pixel the character occupies (expensive!),
        we check the 4 corners of the character's footprint:
        
            Top-left     Top-right
               +-------------+
               |             |
               |   (px,py)   |  <- Center point
               |      *      |
               |             |
               +-------------+
            Bottom-left  Bottom-right
        
        If ALL corners are in walkable tiles, the character can move there.
        
        This is a simplification that works well for tile-based games:
        - Characters smaller than tiles: corners cover all relevant tiles
        - Characters larger than tiles: might miss some tiles in the middle
          (could add edge midpoints for large characters if needed)
        
        =======================================================================
        CORNER CHECKING VS FULL OVERLAP
        =======================================================================
        
        Corner-only checking can miss collisions in rare cases:
        
               +-----+-----+
               |  X  |     |   X = Solid tile
               +-----+-----+   Character corners are in empty tiles,
               |     |     |   but character overlaps the solid!
               +-----+-----+
                  +-----+
                  |     |      <- Character straddles tiles
                  +-----+
        
        This only happens when:
        - Character is larger than a tile
        - Solid tile is completely inside character bounds
        
        For typical tile games (character smaller than tile), this is fine.
        For larger characters, add center and edge midpoint checks.
        
        =======================================================================
        """
        # -----------------------------------------------------------------
        # CONVERT CHARACTER SIZE FROM TILES TO PIXELS
        # -----------------------------------------------------------------
        # char_width is in tiles (e.g., 0.5 tiles)
        # We need pixels for bounding box calculation
        #
        # Half-sizes because position is at CENTER
        half_width_px = (char_width * tile_width) / 2
        half_depth_px = (char_depth * tile_height) / 2
        
        # -----------------------------------------------------------------
        # CALCULATE BOUNDING BOX CORNERS
        # -----------------------------------------------------------------
        # Position (px, py) is at center-bottom of character
        #
        #    left          px          right
        #      |           |            |
        #      v           v            v
        #   top -> +-------*-------+
        #          |   character   |
        # bottom-> +---------------+
        #
        left = px - half_width_px
        right = px + half_width_px
        top = py - half_depth_px      # "top" = smaller Y (top of screen)
        bottom = py + half_depth_px   # "bottom" = larger Y
        
        # -----------------------------------------------------------------
        # DEFINE CORNER POINTS TO CHECK
        # -----------------------------------------------------------------
        corners = [
            (left, top),      # Top-left corner
            (right, top),     # Top-right corner
            (left, bottom),   # Bottom-left corner
            (right, bottom),  # Bottom-right corner
        ]
        
        # -----------------------------------------------------------------
        # GET Z LEVELS TO CHECK (character may span multiple levels)
        # -----------------------------------------------------------------
        z_levels = self.get_z_levels_to_check(z, char_height)
        
        # -----------------------------------------------------------------
        # CHECK ALL CORNERS AT ALL Z LEVELS
        # -----------------------------------------------------------------
        # If ANY corner hits a solid tile at ANY occupied Z level,
        # the movement is blocked
        for cx, cy in corners:
            # Convert corner pixel position to tile coordinates
            tx, ty = self.pixel_to_tile(cx, cy, tile_width, tile_height)
            
            # Check each Z level the character occupies
            for tz in z_levels:
                if self.is_solid(tx, ty, tz):
                    return False  # Collision! Can't move here
        
        # All corners clear at all levels - movement allowed
        return True
    
    # =========================================================================
    # HEIGHT CHANGE COLLISION
    # =========================================================================
    
    def can_change_height(self, px: float, py: float, 
                          current_z: float, new_z: float,
                          char_width: float, char_depth: float, char_height: float,
                          tile_width: int, tile_height: int) -> bool:
        """
        Check if character can change height levels.
        
        Used for:
        - Climbing stairs/ladders
        - Jumping to higher platforms
        - Falling to lower levels
        - Using elevators
        
        Parameters:
        -----------
        px, py : float
            Current position in pixels
        current_z : float
            Current height level (not used in current implementation,
            but kept for potential future use like fall damage calculation)
        new_z : float
            Target height level
        char_width, char_depth, char_height : float
            Character dimensions
        tile_width, tile_height : int
            Tile dimensions in pixels
            
        Returns:
        --------
        bool : True if height change is allowed
        
        =======================================================================
        HEIGHT CHANGE RULES
        =======================================================================
        
        1. Can't go below ground (z < 0)
           - Prevents falling through the world
           - Ground level is always 0
        
        2. Can't go above map ceiling (z >= H)
           - Prevents escaping through the top
           - Maximum height is determined by map design
        
        3. Must have physical space at new height
           - Uses same collision check as horizontal movement
           - Prevents clipping through floors/ceilings
        
        =======================================================================
        POTENTIAL EXTENSIONS
        =======================================================================
        
        This method could be extended to support:
        
        - Gradual height changes (stairs): Check intermediate positions
        - Jump arcs: Verify entire trajectory is clear
        - Fall damage: Calculate height difference
        - One-way platforms: Allow upward but not downward
        - Ladders: Only allow height change at ladder tiles
        
        =======================================================================
        """
        # -----------------------------------------------------------------
        # BOUNDARY CHECKS
        # -----------------------------------------------------------------
        
        # Can't go below ground level
        if new_z < 0:
            return False
        
        # Can't go above map ceiling
        if new_z >= self.H:
            return False
        
        # -----------------------------------------------------------------
        # COLLISION CHECK AT NEW HEIGHT
        # -----------------------------------------------------------------
        # Reuse the same collision logic as horizontal movement
        # Just checking at the new Z level instead of current
        return self.can_move_to_with_size(
            px, py, new_z,
            char_width, char_depth, char_height,
            tile_width, tile_height
        )
    
    # =========================================================================
    # STATISTICS AND DEBUGGING
    # =========================================================================
    
    def get_stats(self) -> dict:
        """
        Get collision map statistics.
        
        Useful for:
        - Debugging (verify collision map was built correctly)
        - Level design validation (reasonable solid/empty ratio)
        - Performance profiling (very large maps might need optimization)
        
        Returns:
        --------
        dict : Statistics dictionary with keys:
            - total_tiles: Total number of tile positions
            - solid_tiles: Number of solid (blocking) tiles
            - empty_tiles: Number of empty (walkable) tiles
            - solid_percent: Percentage of map that is solid
            
        =======================================================================
        TYPICAL VALUES
        =======================================================================
        
        What to expect for different map types:
        
        - Open world: 5-15% solid (walls, obstacles)
        - Dungeon: 30-50% solid (lots of walls)
        - Maze: 40-60% solid (dense walls)
        - Platform level: 10-30% solid (platforms, walls)
        
        Unusual values might indicate:
        - 0% solid: Collision map not built correctly?
        - 90%+ solid: Wrong layer marked as solid?
        
        =======================================================================
        """
        # Total positions in the 3D grid
        total = self.W * self.D * self.H
        
        # Count non-zero entries (solid tiles)
        # np.count_nonzero is optimized C code, very fast
        solid = np.count_nonzero(self.data)
        
        # Empty = total - solid
        empty = total - solid
        
        return {
            'total_tiles': total,
            'solid_tiles': solid,
            'empty_tiles': empty,
            'solid_percent': (solid / total * 100) if total > 0 else 0
        }
