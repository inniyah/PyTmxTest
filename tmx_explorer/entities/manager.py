"""
Entity Manager - Manages all characters and NPCs

=============================================================================
ENTITY-COMPONENT PATTERN (Simplified)
=============================================================================

This module implements a simplified entity management system. In game
development, "entities" are game objects that exist in the world - characters,
NPCs, enemies, items, etc.

The EntityManager serves as a central registry and factory for all character
entities, providing:

1. CENTRALIZED CREATION: All characters are created through the manager
2. RESOURCE SHARING: Sprites are cached to save memory
3. UNIFIED UPDATES: All entities update in one place
4. SPATIAL QUERIES: Find entities near a position
5. RENDER COORDINATION: Collect and sort render data

=============================================================================
FACTORY PATTERN
=============================================================================

Instead of creating Character objects directly:
    player = Character(AnimatedSprite("hero.png"), ...)  # Direct creation

We use factory methods:
    player = entity_manager.create_character("hero.png", ...)  # Factory

Benefits:
- Automatic sprite caching (memory optimization)
- Consistent initialization (collision setup, bounds)
- Centralized tracking (all entities in one list)
- Simplified API (fewer parameters to remember)

=============================================================================
ARCHITECTURE DIAGRAM
=============================================================================

    EntityManager
    ├── characters: List[Character]     <- All entities
    ├── player: Character               <- Quick reference to player
    ├── _sprite_cache: Dict             <- Shared sprite resources
    │
    ├── create_character()  ─┐
    ├── create_npc()         │          <- Factory methods
    ├── create_npc_wanderer()│
    ├── create_npc_patrol()  │
    └── create_npc_follower()┘
    │
    ├── update()                        <- Game loop integration
    └── collect_render_data()           <- Render pipeline integration

=============================================================================
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
from .character import Character, NPCBehavior
from .sprite import AnimatedSprite

# TYPE_CHECKING: Import only for type hints, not at runtime
# This avoids circular imports while still enabling IDE autocomplete
if TYPE_CHECKING:
    from ..renderer.opengl_renderer import OpenGLRenderer


class EntityManager:
    """
    Manages all characters and entities in the game world.
    
    Central hub for entity lifecycle:
    - Creation (factory methods)
    - Updates (game loop)
    - Queries (find entities)
    - Rendering (collect render data)
    
    ==========================================================================
    USAGE EXAMPLE
    ==========================================================================
    
    ```python
    # Initialize
    manager = EntityManager(
        tile_width=32, 
        tile_height=32,
        collision_map=map_3d.collision,
        max_z=map_3d.H
    )
    
    # Create player
    player = manager.create_character(
        "sprites/hero.png",
        x=400, y=300, z=0,
        is_player=True
    )
    
    # Create NPCs
    manager.create_npc_wanderer("sprites/villager.png", x=500, y=400)
    manager.create_npc_patrol("sprites/guard.png", points=[(100,100), (200,200)])
    
    # In game loop
    def update(dt):
        manager.update(dt)  # Updates ALL entities
    
    def render():
        render_data = manager.collect_render_data(renderer, level_offset)
        # ... render all entities
    ```
    
    ==========================================================================
    """
    
    def __init__(self, tile_width: int = 32, tile_height: int = 32, 
                 collision_map = None, max_z: int = 10):
        """
        Initialize entity manager.
        
        Parameters:
        -----------
        tile_width : int
            Tile width in pixels (for coordinate conversion)
        tile_height : int
            Tile height in pixels (for coordinate conversion)
        collision_map : CollisionMap, optional
            Reference to collision system for movement blocking
        max_z : int
            Maximum height level (for Z bounds checking)
            
        =======================================================================
        DEPENDENCY INJECTION
        =======================================================================
        
        We pass in collision_map rather than creating it here. This is called
        "dependency injection" and has several benefits:
        
        1. TESTABILITY: Can pass mock collision for unit tests
        2. FLEXIBILITY: Manager doesn't need to know how collision works
        3. LOOSE COUPLING: Collision system can be changed independently
        
        =======================================================================
        """
        # -----------------------------------------------------------------
        # ENTITY STORAGE
        # -----------------------------------------------------------------
        
        # All characters (players, NPCs, enemies, etc.)
        # Using a simple list - good for small numbers of entities
        # For thousands of entities, consider spatial partitioning
        self.characters: List[Character] = []
        
        # Quick reference to the player character
        # Avoids searching the list every time we need the player
        self.player: Optional[Character] = None
        
        # -----------------------------------------------------------------
        # WORLD CONFIGURATION
        # -----------------------------------------------------------------
        
        # Tile dimensions for coordinate conversion
        self.tile_width = tile_width
        self.tile_height = tile_height
        
        # Collision system reference (can be None for no collision)
        self.collision_map = collision_map
        
        # Maximum height level (Z axis limit)
        self.max_z = max_z
        
        # -----------------------------------------------------------------
        # RESOURCE CACHE
        # -----------------------------------------------------------------
        
        # Sprite cache for memory optimization
        # Key: "path_width_height" → Value: AnimatedSprite (template)
        # New sprites sharing the same source use from_cached()
        self._sprite_cache = {}

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    def load_sprite(self, path: str, frame_width: int = None,
                    frame_height: int = None, animation_speed: float = 8.0) -> AnimatedSprite:
        """
        Load sprite with caching for memory efficiency.
        
        If the same spritesheet was loaded before, returns a new sprite
        instance that SHARES frame data with the cached one.
        
        Parameters:
        -----------
        path : str
            Path to spritesheet image
        frame_width : int, optional
            Frame width (auto-calculated if None)
        frame_height : int, optional
            Frame height (auto-calculated if None)
        animation_speed : float
            Animation playback speed
            
        Returns:
        --------
        AnimatedSprite : New sprite (possibly sharing data with cached)
        
        =======================================================================
        CACHING STRATEGY
        =======================================================================
        
        Cache key includes dimensions because the same image file could be
        used with different frame sizes:
        
        - "npc.png_32_32" → 32x32 frames (small NPCs)
        - "npc.png_64_64" → 64x64 frames (large NPCs)
        
        These are different cache entries despite same file path.
        
        =======================================================================
        MEMORY SAVINGS EXAMPLE
        =======================================================================
        
        Creating 50 villagers with same sprite:
        
        Without cache:
            50 × load_from_disk × cut_16_frames = 50 full sprite loads
            Memory: 50 × 256KB = 12.5 MB
        
        With cache:
            1 × load_from_disk + 49 × from_cached()
            Memory: 1 × 256KB + 49 × tiny_state ≈ 0.3 MB
        
        =======================================================================
        """
        # Build cache key from path and dimensions
        # Using f-string for readable, consistent key format
        cache_key = f"{path}_{frame_width}_{frame_height}"
        
        if cache_key in self._sprite_cache:
            # Cache HIT: Create new instance sharing frames with cached sprite
            cached = self._sprite_cache[cache_key]
            return AnimatedSprite.from_cached(cached, animation_speed)
        else:
            # Cache MISS: Load sprite from disk and cache it
            sprite = AnimatedSprite(path, frame_width, frame_height, animation_speed)
            self._sprite_cache[cache_key] = sprite
            return sprite

    def _setup_character(self, character: Character):
        """
        Configure collision and bounds references for a character.
        
        Called automatically by all create_* factory methods.
        Ensures consistent initialization for all characters.
        
        Parameters:
        -----------
        character : Character
            The character to configure
            
        =======================================================================
        WHY A SEPARATE METHOD?
        =======================================================================
        
        This setup logic is needed by ALL factory methods:
        - create_character()
        - create_npc()
        - create_npc_wanderer()
        - etc.
        
        Instead of duplicating code in each method, we extract it here.
        This is the DRY principle: Don't Repeat Yourself.
        
        If we later add more setup (e.g., AI system registration),
        we only change it in ONE place.
        
        =======================================================================
        """
        # Give character access to collision system
        character.collision_map = self.collision_map
        
        # Set tile dimensions for coordinate conversion
        character.tile_width = self.tile_width
        
        # Set Z axis bounds
        # Characters can't go below ground (z=0) or above map ceiling
        character.min_z = 0.0
        character.max_z = float(self.max_z - 1)  # -1 because levels are 0-indexed

    # =========================================================================
    # FACTORY METHODS - Character Creation
    # =========================================================================

    def create_character(self, spritesheet_path: str,
                         x: float = 0.0, y: float = 0.0, z: float = 0.0,
                         speed: float = 100.0, frame_width: int = None,
                         frame_height: int = None, is_player: bool = False,
                         collision_width: float = None,
                         collision_depth: float = None,
                         collision_height: float = None) -> Character:
        """
        Create a player or generic character.
        
        Parameters:
        -----------
        spritesheet_path : str
            Path to character spritesheet
        x, y : float
            Initial position in pixels
        z : float
            Initial height level
        speed : float
            Movement speed in pixels per second
        frame_width, frame_height : int, optional
            Frame dimensions (auto-calculated if None)
        is_player : bool
            If True, registers this as THE player character
        collision_width : float, optional
            Collision box width in TILES (default 0.5 = 50cm)
        collision_depth : float, optional
            Collision box depth in TILES (default 0.5 = 50cm)
        collision_height : float, optional
            Collision box height in Z LEVELS (default 0.85 = 1.7m)
            
        Returns:
        --------
        Character : The created character
        
        =======================================================================
        COLLISION SIZE UNITS
        =======================================================================
        
        Collision dimensions use TILE UNITS (not pixels):
        
        - collision_width = 0.5 means half a tile wide
        - If tile = 1 meter, then 0.5 tiles = 50 centimeters
        - A human character might be 0.5 × 0.5 tiles (50cm × 50cm)
        
        Height uses Z LEVELS:
        - collision_height = 0.85 means 85% of one level
        - If 1 level = 2 meters, then 0.85 = 1.7 meters (human height)
        
        This unit system makes it easy to reason about:
        "Does this character fit through a 1-tile doorway?" → Yes if width < 1.0
        
        =======================================================================
        """
        # Load sprite (may use cached version)
        sprite = self.load_sprite(spritesheet_path, frame_width, frame_height)
        
        # Create character instance
        character = Character(
            sprite, x, y, z, speed,
            tile_height=self.tile_height,
            is_npc=False,  # Player/generic character, not NPC
            collision_width=collision_width,
            collision_depth=collision_depth,
            collision_height=collision_height
        )
        
        # Configure collision and bounds
        self._setup_character(character)
        
        # Register in entity list
        self.characters.append(character)
        
        # If this is the player, keep a direct reference
        if is_player:
            self.player = character
        
        return character

    def create_npc(self, spritesheet_path: str,
                   x: float = 0.0, y: float = 0.0, z: float = 0.0,
                   speed: float = 80.0,
                   behavior: NPCBehavior = NPCBehavior.WANDER,
                   frame_width: int = None,
                   frame_height: int = None,
                   collision_width: float = None,
                   collision_depth: float = None,
                   collision_height: float = None) -> Character:
        """
        Create NPC (Non-Player Character) with AI behavior.
        
        This is the base NPC factory method. Specialized NPC types
        (wanderer, patrol, follower) call this internally.
        
        Parameters:
        -----------
        spritesheet_path : str
            Path to NPC spritesheet
        x, y, z : float
            Initial position
        speed : float
            Movement speed (default 80, slightly slower than player)
        behavior : NPCBehavior
            AI behavior type (IDLE, WANDER, PATROL, FOLLOW)
        collision_width, collision_depth, collision_height : float, optional
            Collision box dimensions
            
        Returns:
        --------
        Character : The created NPC
        
        =======================================================================
        NPC vs PLAYER
        =======================================================================
        
        The main differences:
        - is_npc=True: Enables AI behavior system
        - behavior: Determines what the NPC does autonomously
        - Default speed: 80 (slower than player's 100)
        
        NPCs are still Character objects - they share the same
        movement, animation, and collision systems.
        
        =======================================================================
        """
        sprite = self.load_sprite(spritesheet_path, frame_width, frame_height)
        
        npc = Character(
            sprite, x, y, z, speed,
            tile_height=self.tile_height,
            is_npc=True,        # Enable NPC behaviors
            behavior=behavior,  # Set AI behavior type
            collision_width=collision_width,
            collision_depth=collision_depth,
            collision_height=collision_height
        )
        
        self._setup_character(npc)
        self.characters.append(npc)
        
        return npc

    # =========================================================================
    # FACTORY METHODS - Specialized NPC Types
    # =========================================================================

    def create_npc_wanderer(self, spritesheet_path: str,
                            x: float, y: float, z: float = 0.0,
                            radius: float = 200.0, speed: float = 60.0,
                            collision_size: Tuple[float, float, float] = None) -> Character:
        """
        Create NPC that wanders randomly around a home point.
        
        The NPC moves to random positions within a circular area,
        simulating idle wandering behavior (villagers, animals, etc.)
        
        Parameters:
        -----------
        spritesheet_path : str
            Path to NPC spritesheet
        x, y : float
            Home position (center of wander area)
        z : float
            Height level
        radius : float
            Wander radius in pixels (how far from home the NPC can go)
        speed : float
            Movement speed (default 60, leisurely pace)
        collision_size : Tuple[float, float, float], optional
            Collision (width, depth, height) in tile units
            
        Returns:
        --------
        Character : The wandering NPC
        
        =======================================================================
        WANDER BEHAVIOR
        =======================================================================
        
        The AI works like this:
        1. NPC reaches its current target position
        2. Pick random point within radius of home position
        3. Walk to that point
        4. Wait briefly (idle animation)
        5. Repeat
        
        This creates natural-looking "aimless" movement.
        
        Example: A chicken wandering around a farm
        - home = center of chicken coop area
        - radius = 150 pixels
        - Chicken wanders within 150px of coop, never strays too far
        
        =======================================================================
        """
        # Unpack collision size tuple (or use defaults)
        cw, cd, ch = collision_size if collision_size else (None, None, None)
        
        # Create base NPC with WANDER behavior
        npc = self.create_npc(
            spritesheet_path, x, y, z, speed, NPCBehavior.WANDER,
            collision_width=cw, collision_depth=cd, collision_height=ch
        )
        
        # Configure wander-specific properties
        npc.wander_radius = radius  # How far to wander
        npc.home_x = x              # Remember starting position
        npc.home_y = y              # (wander center)
        
        return npc

    def create_npc_patrol(self, spritesheet_path: str,
                          points: List[Tuple[float, float]] = None,
                          patrol_points: List[Tuple[float, float]] = None,
                          z: float = 0.0, speed: float = 70.0,
                          collision_size: Tuple[float, float, float] = None) -> Character:
        """
        Create NPC that patrols between waypoints.
        
        The NPC walks in a loop through the specified points,
        useful for guards, sentries, or any scheduled movement.
        
        Parameters:
        -----------
        spritesheet_path : str
            Path to NPC spritesheet
        points : List[Tuple[float, float]]
            List of (x, y) waypoints to patrol between
        patrol_points : List[Tuple[float, float]]
            Alias for 'points' (for API compatibility)
        z : float
            Height level
        speed : float
            Movement speed (default 70, purposeful pace)
        collision_size : Tuple[float, float, float], optional
            Collision dimensions
            
        Returns:
        --------
        Character : The patrolling NPC
        
        =======================================================================
        PATROL BEHAVIOR
        =======================================================================
        
        The AI works like this:
        1. Start at first waypoint
        2. Walk to next waypoint
        3. When reached, advance to next waypoint index
        4. After last waypoint, loop back to first
        5. Repeat forever
        
        Example patrol path (guard route):
        
            [1]─────────[2]
             │           │
             │   AREA    │
             │           │
            [4]─────────[3]
        
        Points: [(100,100), (300,100), (300,300), (100,300)]
        Guard walks: 1→2→3→4→1→2→3→4→...
        
        =======================================================================
        PARAMETER FLEXIBILITY
        =======================================================================
        
        We accept both 'points' and 'patrol_points' parameters.
        This is for API compatibility - different calling code might
        use different naming conventions. We handle both.
        
        =======================================================================
        """
        # Accept either parameter name for flexibility
        actual_points = points or patrol_points
        
        if not actual_points:
            raise ValueError("Patrol points cannot be empty")
        
        # Unpack collision size
        cw, cd, ch = collision_size if collision_size else (None, None, None)
        
        # Create NPC starting at first patrol point
        npc = self.create_npc(
            spritesheet_path, 
            actual_points[0][0],  # Start X = first point's X
            actual_points[0][1],  # Start Y = first point's Y
            z, speed, NPCBehavior.PATROL,
            collision_width=cw, collision_depth=cd, collision_height=ch
        )
        
        # Configure patrol waypoints
        npc.set_patrol_points(actual_points)
        
        return npc

    def create_npc_follower(self, spritesheet_path: str,
                            target: Character,
                            x: float = 0.0, y: float = 0.0, z: float = 0.0,
                            speed: float = 90.0,
                            collision_size: Tuple[float, float, float] = None) -> Character:
        """
        Create NPC that follows a target character.
        
        Useful for companions, pets, or enemies that chase the player.
        
        Parameters:
        -----------
        spritesheet_path : str
            Path to NPC spritesheet
        target : Character
            The character to follow (usually the player)
        x, y, z : float
            Initial position
        speed : float
            Movement speed (default 90, slightly slower than player)
        collision_size : Tuple[float, float, float], optional
            Collision dimensions
            
        Returns:
        --------
        Character : The follower NPC
        
        =======================================================================
        FOLLOW BEHAVIOR
        =======================================================================
        
        The AI works like this:
        1. Calculate distance to target
        2. If distance > follow_distance: Walk toward target
        3. If distance < stop_distance: Stop (don't crowd the target)
        4. Update facing direction toward target
        
        Speed considerations:
        - Slightly slower than player (90 vs 100): Player can outrun if needed
        - Slightly faster could work for aggressive enemies
        - Equal speed for companions that stay alongside
        
        =======================================================================
        COMMON USE CASES
        =======================================================================
        
        Companion NPC:
            companion = manager.create_npc_follower(
                "pet_dog.png", target=player, speed=95)
        
        Enemy that chases:
            enemy = manager.create_npc_follower(
                "zombie.png", target=player, speed=60)  # Slow zombie
        
        Party member:
            ally = manager.create_npc_follower(
                "knight.png", target=player, speed=100)  # Match player speed
        
        =======================================================================
        """
        cw, cd, ch = collision_size if collision_size else (None, None, None)
        
        npc = self.create_npc(
            spritesheet_path, x, y, z, speed, NPCBehavior.FOLLOW,
            collision_width=cw, collision_depth=cd, collision_height=ch
        )
        
        # Set the target to follow
        npc.target = target
        
        return npc

    # =========================================================================
    # GAME LOOP INTEGRATION
    # =========================================================================

    def update(self, dt: float):
        """
        Update all entities.
        
        Call this once per frame from your game loop.
        
        Parameters:
        -----------
        dt : float
            Delta time in seconds since last update
            
        =======================================================================
        WHAT GETS UPDATED
        =======================================================================
        
        For each character, update() handles:
        - AI behavior (NPC movement decisions)
        - Physics/movement (position changes)
        - Animation (frame advancement)
        - Collision checking (movement blocking)
        
        Order matters! All entities update together to prevent
        "first mover advantage" in collision detection.
        
        =======================================================================
        PERFORMANCE NOTE
        =======================================================================
        
        Simple iteration works fine for ~100 entities.
        For thousands of entities, consider:
        - Spatial partitioning (only update nearby entities)
        - Level-of-detail updates (distant entities update less often)
        - Multithreading (update groups in parallel)
        
        =======================================================================
        """
        for character in self.characters:
            character.update(dt)

    # =========================================================================
    # RENDERING INTEGRATION
    # =========================================================================

    def collect_render_data(self, renderer: 'OpenGLRenderer', 
                            level_height_offset: int) -> List[Tuple]:
        """
        Collect and sort render data for all entities.
        
        Prepares a list of render instructions sorted by depth,
        ready for the rendering pipeline.
        
        Parameters:
        -----------
        renderer : OpenGLRenderer
            The renderer (used to get textures from sprites)
        level_height_offset : int
            Pixels per height level (for Y position adjustment)
            
        Returns:
        --------
        List[Tuple] : Sorted render data, each tuple contains:
            (texture, x, y, width, height, depth)
        
        =======================================================================
        HEIGHT LEVEL RENDERING
        =======================================================================
        
        In our 3D tile system, higher Z levels appear "above" lower ones.
        To create this illusion in 2D rendering, we:
        
        1. Offset Y position based on Z level
           - Higher Z = rendered higher on screen (lower Y value)
           - This creates the "elevation" effect
        
        2. Adjust depth value for proper sorting
           - Characters at higher Z should render in front of
             same-Y characters at lower Z
        
        Example with level_height_offset = 16:
        - Character at Z=0: Y unchanged
        - Character at Z=1: Y -= 16 (appears 16 pixels higher)
        - Character at Z=2: Y -= 32 (appears 32 pixels higher)
        
        =======================================================================
        DEPTH SORTING
        =======================================================================
        
        We sort by depth value (ascending) to ensure proper draw order:
        - Lower depth = rendered first (background)
        - Higher depth = rendered last (foreground)
        
        For top-down games, depth is typically based on Y position:
        - Objects lower on screen (higher Y) appear in front
        - This creates the illusion of 3D depth
        
        =======================================================================
        """
        render_data = []
        
        for character in self.characters:
            # Get GPU texture for current animation frame
            texture = character.get_texture(renderer)
            
            # Get render position (may differ from logical position)
            rx, ry = character.get_render_position()
            
            # Adjust Y position for height level
            # Higher Z = subtract from Y = appears higher on screen
            level_y_offset = character.z * level_height_offset
            ry -= level_y_offset
            
            # Get depth value for sorting
            depth = character.get_depth()
            
            # Collect render data tuple
            render_data.append((
                texture,
                rx, ry,                          # Screen position
                character.width, character.height,  # Sprite size
                depth                            # Sort key
            ))
        
        # Sort by depth (ascending: low depth = rendered first)
        render_data.sort(key=lambda x: x[5])
        
        return render_data

    # =========================================================================
    # SPATIAL QUERIES
    # =========================================================================

    def get_characters_at(self, x: float, y: float, radius: float = 32.0) -> List[Character]:
        """
        Find all characters within a radius of a point.
        
        Useful for:
        - Interaction detection (who can the player talk to?)
        - Area-of-effect abilities (who gets hit by explosion?)
        - Proximity triggers (NPC notices player approaching)
        
        Parameters:
        -----------
        x, y : float
            Center point in world coordinates
        radius : float
            Search radius in pixels
            
        Returns:
        --------
        List[Character] : All characters within the radius
        
        =======================================================================
        DISTANCE CALCULATION
        =======================================================================
        
        We use squared distance comparison to avoid sqrt():
        
        Normal distance: sqrt((dx² + dy²)) <= radius
        Optimized:       dx² + dy² <= radius²
        
        Since sqrt is expensive and we only need comparison,
        comparing squared values gives the same result faster.
        
        =======================================================================
        PERFORMANCE NOTE
        =======================================================================
        
        This is O(n) - checks every character. Fine for ~100 entities.
        
        For thousands of entities, use spatial partitioning:
        - Grid-based: Divide world into cells, only check nearby cells
        - Quadtree: Hierarchical spatial subdivision
        - R-tree: For complex queries
        
        =======================================================================
        """
        result = []
        
        for char in self.characters:
            # Calculate distance components
            dx = char.x - x
            dy = char.y - y
            
            # Compare squared distance (avoids expensive sqrt)
            if dx*dx + dy*dy <= radius*radius:
                result.append(char)
        
        return result

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def npc_count(self) -> int:
        """
        Number of NPCs (non-player characters).
        
        Useful for:
        - Debug display
        - Spawn limits ("max 50 NPCs in this area")
        - Statistics
        """
        return sum(1 for c in self.characters if c.is_npc)
    
    @property
    def character_count(self) -> int:
        """
        Total number of characters (including player).
        
        Useful for:
        - Performance monitoring
        - Debug display
        - Entity limits
        """
        return len(self.characters)
