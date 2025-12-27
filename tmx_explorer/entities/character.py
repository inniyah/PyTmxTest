"""
Character entity - Player and NPC support

=============================================================================
CHARACTER ENTITY OVERVIEW
=============================================================================

This module implements the Character class - the core entity for all
characters in the game, including the player and NPCs.

UNIFIED DESIGN DECISION:
------------------------
Instead of having separate Player and NPC classes, we use ONE Character class
that can behave as either. This simplifies:
- Code maintenance (one movement system, one animation system)
- Feature parity (NPCs get same capabilities as player)
- Interaction code (everything is just a Character)

The `is_npc` flag and `behavior` enum control NPC-specific features.

=============================================================================
COORDINATE SYSTEM AND PHYSICS
=============================================================================

Characters exist in 3D space:
- X: Horizontal position (pixels)
- Y: Vertical position (pixels) - increases downward
- Z: Height level (floating point for smooth transitions)

Movement uses velocity-based physics:
- velocity_x/y/z: Current speed in each axis
- Position updated each frame: pos += velocity * dt
- Velocity set by input (player) or AI (NPC)

This separation of velocity and position enables:
- Smooth acceleration/deceleration
- Collision response (stop velocity, keep position)
- AI behaviors that set velocity, not position

=============================================================================
COLLISION SYSTEM
=============================================================================

Characters have a 3D collision box defined in TILE UNITS:
- collision_width: How wide (X axis)
- collision_depth: How deep (Y axis)  
- collision_height: How tall (Z axis)

Using tile units (not pixels) makes reasoning easier:
- 0.5 tiles = half a tile = 50cm if tile = 1m
- A human is about 0.5 wide, 0.5 deep, 0.85 tall

=============================================================================
"""

import random
import math
from typing import Dict, Tuple, TYPE_CHECKING, Optional
from enum import Enum
from .sprite import AnimatedSprite, Direction

# TYPE_CHECKING: Imports only for type hints (avoids circular imports)
if TYPE_CHECKING:
    from ..renderer.texture import Texture
    from ..renderer.opengl_renderer import OpenGLRenderer


class NPCBehavior(Enum):
    """
    NPC behavior types.
    
    Each behavior implements a different AI pattern:
    
    IDLE:   Stand still, do nothing
            Use: Shopkeepers, stationary NPCs
            
    WANDER: Move randomly within an area
            Use: Villagers, animals, ambient NPCs
            
    PATROL: Walk between waypoints in order
            Use: Guards, sentries, scheduled movement
            
    FOLLOW: Move toward a target character
            Use: Companions, pets, chasing enemies
    
    ==========================================================================
    EXTENDING BEHAVIORS
    ==========================================================================
    
    To add new behaviors:
    1. Add enum value here
    2. Create _behavior_newname() method in Character
    3. Add case in _update_npc_behavior()
    
    Potential additions:
    - FLEE: Run away from target
    - GUARD: Stay near a point, chase nearby threats
    - SCHEDULE: Different behaviors at different times
    
    ==========================================================================
    """
    IDLE = "idle"       # Stand still
    WANDER = "wander"   # Random movement in area
    PATROL = "patrol"   # Walk between points
    FOLLOW = "follow"   # Chase a target


class Character:
    """
    Character entity - Player or NPC.
    
    Handles:
    - Position and movement (velocity-based)
    - Collision detection (tile-based)
    - Animation (sprite management)
    - NPC AI behaviors (if is_npc=True)
    
    ==========================================================================
    DEFAULT COLLISION SIZE
    ==========================================================================
    
    Default dimensions represent an average human:
    - Width: 0.5 tiles = 50cm (shoulder width)
    - Depth: 0.5 tiles = 50cm (front to back)
    - Height: 0.85 levels = 1.7m (average human height)
    
    Scale assumptions:
    - 1 tile = 1 meter × 1 meter
    - 1 Z level = 2 meters
    
    These defaults allow characters to:
    - Walk through 1-tile-wide doorways
    - Fit under 1-level-high ceilings
    - Stand next to each other in a 1-tile corridor
    
    ==========================================================================
    """
    
    # =========================================================================
    # DEFAULT COLLISION DIMENSIONS (in tile units)
    # =========================================================================
    
    DEFAULT_COLLISION_WIDTH = 0.5   # 50cm wide (shoulder width)
    DEFAULT_COLLISION_DEPTH = 0.5   # 50cm deep (front to back)
    DEFAULT_COLLISION_HEIGHT = 0.85 # 1.7m tall (if 1 level = 2m)
    
    def __init__(self, sprite: AnimatedSprite,
                 x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 speed: float = 100.0,
                 tile_height: int = 32,
                 is_npc: bool = False,
                 behavior: NPCBehavior = NPCBehavior.IDLE,
                 collision_width: float = None,
                 collision_depth: float = None,
                 collision_height: float = None):
        """
        Initialize character entity.
        
        Parameters:
        -----------
        sprite : AnimatedSprite
            The character's animated sprite
        x, y : float
            Initial position in PIXELS
        z : float
            Initial height LEVEL (0 = ground, 1 = one level up)
        speed : float
            Movement speed in PIXELS PER SECOND
        tile_height : int
            Tile height in pixels (for coordinate conversion)
        is_npc : bool
            True for NPC (enables AI), False for player
        behavior : NPCBehavior
            AI behavior type (only used if is_npc=True)
        collision_width, collision_depth, collision_height : float, optional
            Custom collision box dimensions in TILE UNITS
        """
        # -----------------------------------------------------------------
        # SPRITE AND ANIMATION
        # -----------------------------------------------------------------
        self.sprite = sprite
        
        # -----------------------------------------------------------------
        # POSITION (in world coordinates)
        # -----------------------------------------------------------------
        # x, y are in PIXELS
        # z is in HEIGHT LEVELS (can be fractional for smooth transitions)
        self.x = x
        self.y = y
        self.z = z
        
        # -----------------------------------------------------------------
        # MOVEMENT PARAMETERS
        # -----------------------------------------------------------------
        # Speed: pixels per second for horizontal movement
        self.speed = speed
        
        # Tile height for depth calculations
        self.tile_height = tile_height
        
        # -----------------------------------------------------------------
        # VELOCITY (current movement speed in each axis)
        # -----------------------------------------------------------------
        # These are set by move() (player) or AI behaviors (NPC)
        # Position is updated by: pos += velocity * dt
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        
        # -----------------------------------------------------------------
        # COLLISION BOX (in tile units)
        # -----------------------------------------------------------------
        # Use provided values or fall back to class defaults
        # The "or" pattern: if value is None (or 0), use default
        self.collision_width = collision_width or self.DEFAULT_COLLISION_WIDTH
        self.collision_depth = collision_depth or self.DEFAULT_COLLISION_DEPTH
        self.collision_height = collision_height or self.DEFAULT_COLLISION_HEIGHT
        
        # -----------------------------------------------------------------
        # EXTERNAL REFERENCES (set by EntityManager)
        # -----------------------------------------------------------------
        # These are None until EntityManager configures them
        self.collision_map = None  # Reference to CollisionMap
        self.tile_width = 32       # Tile width in pixels
        
        # Height bounds (prevents falling through world / flying too high)
        self.min_z = 0.0   # Usually 0 (ground level)
        self.max_z = 10.0  # Set by EntityManager based on map height
        
        # -----------------------------------------------------------------
        # NPC CONFIGURATION
        # -----------------------------------------------------------------
        self.is_npc = is_npc
        self.behavior = behavior
        
        # Target for FOLLOW behavior (reference to another Character)
        self.target: Optional['Character'] = None
        
        # -----------------------------------------------------------------
        # PATROL BEHAVIOR STATE
        # -----------------------------------------------------------------
        # List of (x, y) waypoints to visit in order
        self.patrol_points: list[Tuple[float, float]] = []
        # Current waypoint index (loops back to 0 after last)
        self.patrol_index = 0
        
        # -----------------------------------------------------------------
        # WANDER BEHAVIOR STATE
        # -----------------------------------------------------------------
        # Timer until next direction change
        self.wander_timer = 0.0
        # How often to pick new direction (seconds)
        self.wander_interval = 2.0
        # Maximum distance from home position (pixels)
        self.wander_radius = 200.0
        # Home position (center of wander area)
        self.home_x = x
        self.home_y = y
        
        # -----------------------------------------------------------------
        # IDLE PAUSE STATE (used by WANDER behavior)
        # -----------------------------------------------------------------
        # Timer for idle pause duration
        self.idle_timer = 0.0
        # Whether currently in idle pause
        self.is_idle_pausing = False
        
        # -----------------------------------------------------------------
        # TEXTURE CACHE
        # -----------------------------------------------------------------
        # Maps (direction, frame_index) -> GPU Texture
        # Pre-caches all frame textures for efficient rendering
        self._texture_cache: Dict[Tuple[Direction, int], 'Texture'] = {}
        self._textures_initialized = False

    # =========================================================================
    # COLLISION CONFIGURATION
    # =========================================================================

    def set_collision_size(self, width: float, depth: float, height: float):
        """
        Configure character's collision box size.
        
        Parameters:
        -----------
        width : float
            Collision width in TILES (e.g., 0.5 = half tile = 50cm)
        depth : float
            Collision depth in TILES
        height : float
            Collision height in Z LEVELS (e.g., 0.85 = 1.7m)
            
        Use cases:
        - Smaller collision for children NPCs
        - Larger collision for big creatures
        - Tall collision for giants
        - Short collision for crouching
        """
        self.collision_width = width
        self.collision_depth = depth
        self.collision_height = height

    # =========================================================================
    # TEXTURE INITIALIZATION
    # =========================================================================

    def _init_textures(self, renderer: 'OpenGLRenderer'):
        """
        Initialize GPU textures for all animation frames.
        
        Called lazily on first render (not in __init__) because:
        1. OpenGL context might not be ready during character creation
        2. Renderer reference might not be available yet
        3. Avoids loading textures for characters that are never rendered
        
        Creates 16 textures total (4 directions × 4 frames each).
        
        =======================================================================
        LAZY INITIALIZATION PATTERN
        =======================================================================
        
        Instead of:
            def __init__(self):
                self._init_textures()  # Might fail if GL not ready
        
        We use:
            def get_texture(self):
                if not self._textures_initialized:
                    self._init_textures()  # Called when actually needed
                return self._texture_cache[key]
        
        This is called "lazy initialization" - defer expensive work until
        it's actually needed.
        
        =======================================================================
        """
        from ..renderer.texture import Texture
        
        # Create texture for each direction and frame combination
        for direction in Direction:
            for frame_idx in range(4):  # 4 frames per direction
                # Get PIL Image from sprite
                frame_image = self.sprite.get_frame(direction, frame_idx)
                # Convert to GPU texture and cache
                self._texture_cache[(direction, frame_idx)] = Texture.from_pil(frame_image)
        
        self._textures_initialized = True

    # =========================================================================
    # MAIN UPDATE LOOP
    # =========================================================================

    def update(self, dt: float):
        """
        Update character state for this frame.
        
        Parameters:
        -----------
        dt : float
            Delta time in seconds since last update
            
        =======================================================================
        UPDATE ORDER
        =======================================================================
        
        1. NPC AI (if is_npc): Decides velocity based on behavior
        2. Movement: Apply velocity to position with collision
        3. Animation: Update sprite based on movement state
        
        This order ensures:
        - AI makes decisions before movement happens
        - Animation reflects current movement state
        - Everything is synchronized for this frame
        
        =======================================================================
        """
        # -----------------------------------------------------------------
        # STEP 1: NPC AI BEHAVIOR
        # -----------------------------------------------------------------
        # If this is an NPC, run AI to decide movement
        if self.is_npc:
            self._update_npc_behavior(dt)
        
        # -----------------------------------------------------------------
        # STEP 2: HORIZONTAL MOVEMENT (X and Y)
        # -----------------------------------------------------------------
        is_moving = (self.velocity_x != 0 or self.velocity_y != 0)
        
        if is_moving:
            # Calculate new positions
            new_x = self.x + self.velocity_x * dt
            new_y = self.y + self.velocity_y * dt
            
            # Apply collision if collision map exists
            if self.collision_map is not None:
                # SEPARATE AXIS COLLISION
                # =========================
                # We check X and Y movement SEPARATELY.
                # This allows "sliding" along walls:
                #
                # Without separate checks:
                #   Moving diagonally into a wall blocks ALL movement
                #
                # With separate checks:
                #   Moving diagonally into vertical wall:
                #   - X blocked (wall)
                #   - Y allowed (can slide along wall)
                #
                # This feels much better to play!
                
                # Try X movement
                if self._can_move_to(new_x, self.y, self.z):
                    self.x = new_x
                
                # Try Y movement (independently)
                if self._can_move_to(self.x, new_y, self.z):
                    self.y = new_y
            else:
                # No collision - just move
                self.x = new_x
                self.y = new_y
            
            # Update sprite direction based on velocity
            self._update_direction()
        
        # -----------------------------------------------------------------
        # STEP 3: VERTICAL MOVEMENT (Z / Height)
        # -----------------------------------------------------------------
        if self.velocity_z != 0:
            new_z = self.z + self.velocity_z * dt
            
            # Clamp to valid height range
            new_z = max(self.min_z, min(self.max_z, new_z))
            
            # Check if height change is allowed (no ceiling collision)
            if self.collision_map is not None:
                if self._can_change_height(new_z):
                    self.z = new_z
            else:
                self.z = new_z
        
        # -----------------------------------------------------------------
        # STEP 4: ANIMATION UPDATE
        # -----------------------------------------------------------------
        # Tell sprite whether we're moving (affects animation)
        self.sprite.set_walking(is_moving)
        # Advance animation frame based on time
        self.sprite.update(dt)
    
    # =========================================================================
    # COLLISION CHECKING
    # =========================================================================
    
    def _can_move_to(self, px: float, py: float, z: float) -> bool:
        """
        Check if character can move to a position.
        
        Parameters:
        -----------
        px, py : float
            Target position in pixels
        z : float
            Current height level
            
        Returns:
        --------
        bool : True if movement is allowed (no collision)
        """
        if self.collision_map is None:
            return True  # No collision system = everything allowed
        
        # Delegate to collision map with character's collision box size
        return self.collision_map.can_move_to_with_size(
            px, py, z,
            self.collision_width,
            self.collision_depth,
            self.collision_height,
            self.tile_width, self.tile_height
        )
    
    def _can_change_height(self, new_z: float) -> bool:
        """
        Check if character can change to a new height level.
        
        Used for stairs, jumping, ladders, etc.
        
        Parameters:
        -----------
        new_z : float
            Target height level
            
        Returns:
        --------
        bool : True if height change is allowed
        """
        if self.collision_map is None:
            return True
        
        return self.collision_map.can_change_height(
            self.x, self.y,
            self.z, new_z,
            self.collision_width,
            self.collision_depth,
            self.collision_height,
            self.tile_width, self.tile_height
        )

    # =========================================================================
    # NPC AI BEHAVIORS
    # =========================================================================

    def _update_npc_behavior(self, dt: float):
        """
        Dispatch to appropriate behavior method based on behavior type.
        
        This is a simple state machine / strategy pattern:
        - Current behavior determines which method runs
        - Each method sets velocity_x, velocity_y
        - Main update() then applies the velocity
        """
        if self.behavior == NPCBehavior.IDLE:
            self._behavior_idle()
        elif self.behavior == NPCBehavior.WANDER:
            self._behavior_wander(dt)
        elif self.behavior == NPCBehavior.PATROL:
            self._behavior_patrol(dt)
        elif self.behavior == NPCBehavior.FOLLOW:
            self._behavior_follow(dt)

    def _behavior_idle(self):
        """
        IDLE behavior: Stand still, do nothing.
        
        Simplest behavior - just stop moving.
        Used for stationary NPCs like shopkeepers.
        """
        self.velocity_x = 0
        self.velocity_y = 0

    def _behavior_wander(self, dt: float):
        """
        WANDER behavior: Move randomly within an area.
        
        =======================================================================
        ALGORITHM
        =======================================================================
        
        1. If in idle pause: Count down timer, stay still
        2. When wander_timer expires:
           a. 30% chance: Start idle pause (1-3 seconds)
           b. 70% chance: Pick random direction and speed
        3. If too far from home: Walk back toward home
        
        This creates natural-looking "aimless" movement:
        - Random directions (not grid-aligned)
        - Variable speeds (30-70% of max)
        - Occasional pauses (looks like stopping to look around)
        - Stays within designated area (wander_radius)
        
        =======================================================================
        IDLE PAUSES
        =======================================================================
        
        Without pauses, NPCs look like they're frantically pacing.
        With random pauses, they look like they're:
        - Stopping to look at something
        - Thinking about where to go next
        - Just hanging out
        
        Much more natural!
        
        =======================================================================
        """
        # -----------------------------------------------------------------
        # IDLE PAUSE HANDLING
        # -----------------------------------------------------------------
        if self.is_idle_pausing:
            self.idle_timer -= dt
            if self.idle_timer <= 0:
                # Pause ended
                self.is_idle_pausing = False
            else:
                # Still pausing - stand still
                self.velocity_x = 0
                self.velocity_y = 0
                return
        
        # -----------------------------------------------------------------
        # WANDER TIMER
        # -----------------------------------------------------------------
        self.wander_timer -= dt
        
        if self.wander_timer <= 0:
            # Time to make a new decision
            
            if random.random() < 0.3:
                # 30% chance: Start idle pause
                self.is_idle_pausing = True
                self.idle_timer = random.uniform(1.0, 3.0)  # 1-3 seconds
                self.velocity_x = 0
                self.velocity_y = 0
            else:
                # 70% chance: Pick new random direction
                
                # Random angle (0 to 2π radians = full circle)
                angle = random.uniform(0, 2 * math.pi)
                
                # Random speed (30-70% of max speed)
                # This variation makes movement look more natural
                speed = self.speed * random.uniform(0.3, 0.7)
                
                # Convert angle to velocity components
                # cos(angle) = X component, sin(angle) = Y component
                self.velocity_x = math.cos(angle) * speed
                self.velocity_y = math.sin(angle) * speed
            
            # Reset timer with random interval
            self.wander_timer = random.uniform(1.0, self.wander_interval)
        
        # -----------------------------------------------------------------
        # HOME BOUNDARY CHECK
        # -----------------------------------------------------------------
        # If NPC has wandered too far, walk back toward home
        dist_from_home = math.sqrt(
            (self.x - self.home_x)**2 + 
            (self.y - self.home_y)**2
        )
        
        if dist_from_home > self.wander_radius:
            # Calculate direction toward home
            dx = self.home_x - self.x
            dy = self.home_y - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                # Normalize and apply speed (half speed for gentle return)
                self.velocity_x = (dx / dist) * self.speed * 0.5
                self.velocity_y = (dy / dist) * self.speed * 0.5

    def _behavior_patrol(self, dt: float):
        """
        PATROL behavior: Walk between waypoints in order.
        
        =======================================================================
        ALGORITHM
        =======================================================================
        
        1. Get current target waypoint from patrol_points[patrol_index]
        2. Calculate distance to target
        3. If close enough (< 10 pixels): Advance to next waypoint
        4. Otherwise: Walk toward current waypoint
        
        Waypoints loop: After reaching last point, go back to first.
        
        =======================================================================
        ARRIVAL THRESHOLD
        =======================================================================
        
        We use 10 pixels as the "close enough" threshold.
        
        Why not 0? Because:
        - Floating point imprecision might never hit exactly 0
        - Character might overshoot slightly
        - 10 pixels is close enough that it looks like arrival
        
        =======================================================================
        """
        # No waypoints = fall back to idle
        if not self.patrol_points:
            self._behavior_idle()
            return
        
        # Get current target waypoint
        target_x, target_y = self.patrol_points[self.patrol_index]
        
        # Calculate vector to target
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Check if we've arrived (within threshold)
        if dist < 10:  # 10 pixels = arrived
            # Advance to next waypoint (wrap around to 0)
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
        else:
            # Walk toward waypoint at half speed (patrol pace, not rushing)
            speed = self.speed * 0.5
            
            # Normalize direction and apply speed
            self.velocity_x = (dx / dist) * speed
            self.velocity_y = (dy / dist) * speed

    def _behavior_follow(self, dt: float):
        """
        FOLLOW behavior: Move toward a target character.
        
        =======================================================================
        ALGORITHM
        =======================================================================
        
        1. Calculate distance to target
        2. If far enough (> min_distance): Walk toward target
        3. If close enough (<= min_distance): Stop (don't crowd)
        
        =======================================================================
        MINIMUM DISTANCE
        =======================================================================
        
        We stop at 80 pixels from the target. This prevents:
        - Follower standing ON TOP of target (looks weird)
        - Follower pushing target around (if collision enabled between chars)
        - Follower constantly jittering trying to reach exact position
        
        80 pixels ≈ 2.5 tiles away, a comfortable "following" distance.
        
        =======================================================================
        SPEED FACTOR
        =======================================================================
        
        We use 60% speed (0.6), slightly slower than the target.
        This prevents:
        - Follower overtaking and getting ahead
        - Follower appearing to chase aggressively
        
        For enemies that chase, you might use 100% or higher speed.
        
        =======================================================================
        """
        # No target = fall back to idle
        if not self.target:
            self._behavior_idle()
            return
        
        # Calculate vector to target
        dx = self.target.x - self.x
        dy = self.target.y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Minimum following distance (don't get too close)
        min_distance = 80  # pixels
        
        if dist > min_distance:
            # Too far - move toward target
            speed = self.speed * 0.6  # 60% speed
            
            # Normalize direction and apply speed
            self.velocity_x = (dx / dist) * speed
            self.velocity_y = (dy / dist) * speed
        else:
            # Close enough - stop
            self.velocity_x = 0
            self.velocity_y = 0

    # =========================================================================
    # DIRECTION HANDLING
    # =========================================================================

    def _update_direction(self):
        """
        Update sprite facing direction based on velocity.
        
        =======================================================================
        FOUR-DIRECTIONAL SPRITES
        =======================================================================
        
        Our sprites only have 4 directions (DOWN, LEFT, RIGHT, UP).
        But movement can be in any direction (diagonal, any angle).
        
        We pick the direction that best matches the velocity:
        
        - If moving more vertically than horizontally:
            Use UP or DOWN based on velocity_y sign
        - Otherwise (more horizontal):
            Use LEFT or RIGHT based on velocity_x sign
        
        This creates natural-looking direction changes:
        - Walking up-right? Shows UP sprite (more vertical)
        - Walking right-up? Shows RIGHT sprite (more horizontal)
        
        =======================================================================
        ABS() FOR MAGNITUDE COMPARISON
        =======================================================================
        
        We compare abs(velocity_y) vs abs(velocity_x) to check MAGNITUDE,
        not direction. Whether moving up or down, we want to know
        "is vertical movement dominant?"
        
        =======================================================================
        """
        # Compare magnitudes to determine dominant direction
        if abs(self.velocity_y) > abs(self.velocity_x):
            # Moving more vertically
            if self.velocity_y > 0:
                self.sprite.set_direction(Direction.DOWN)   # +Y = down on screen
            else:
                self.sprite.set_direction(Direction.UP)     # -Y = up on screen
        elif self.velocity_x != 0:
            # Moving more horizontally (or equally)
            if self.velocity_x > 0:
                self.sprite.set_direction(Direction.RIGHT)  # +X = right
            else:
                self.sprite.set_direction(Direction.LEFT)   # -X = left

    # =========================================================================
    # PLAYER INPUT
    # =========================================================================

    def move(self, dx: float, dy: float, dz: float = 0.0):
        """
        Set movement direction (called by input handling).
        
        Parameters:
        -----------
        dx : float
            Horizontal direction (-1 = left, 0 = none, 1 = right)
        dy : float
            Vertical direction (-1 = up, 0 = none, 1 = down)
        dz : float
            Height direction (-1 = descend, 0 = none, 1 = ascend)
            
        =======================================================================
        DIAGONAL MOVEMENT NORMALIZATION
        =======================================================================
        
        Problem: Moving diagonally would be faster than cardinal directions!
        
        Cardinal: velocity = 1 * speed = 100 px/s
        Diagonal:  velocity = sqrt(1² + 1²) * speed = 1.414 * 100 = 141 px/s
        
        That's 41% faster diagonally - feels wrong!
        
        Solution: Multiply by 0.7071 (≈ 1/√2 ≈ 0.707)
        
        Diagonal with fix: velocity = sqrt(0.7² + 0.7²) * speed
                                     = sqrt(0.98) * 100 ≈ 99 px/s
        
        Now diagonal is approximately the same speed as cardinal.
        
        =======================================================================
        Z MOVEMENT SPEED
        =======================================================================
        
        Z movement uses 1% of horizontal speed (speed * 0.01).
        This makes height changes slow and deliberate.
        
        Why so slow?
        - Height changes are significant (floor to floor)
        - Fast Z movement would feel unnatural
        - Gives time for visual feedback (animation, etc.)
        
        =======================================================================
        """
        if dx == 0 and dy == 0:
            # No input = stop moving
            self.velocity_x = 0
            self.velocity_y = 0
        else:
            # Apply diagonal normalization if moving diagonally
            if dx != 0 and dy != 0:
                # 0.7071 ≈ 1/√2 - normalizes diagonal to same speed as cardinal
                factor = 0.7071
                dx *= factor
                dy *= factor
            
            # Convert direction to velocity
            self.velocity_x = dx * self.speed
            self.velocity_y = dy * self.speed
        
        # Z velocity (much slower than horizontal)
        self.velocity_z = dz * self.speed * 0.01

    # =========================================================================
    # PATROL CONFIGURATION
    # =========================================================================

    def set_patrol_points(self, points: list[Tuple[float, float]]):
        """
        Set waypoints for patrol behavior.
        
        Parameters:
        -----------
        points : list[Tuple[float, float]]
            List of (x, y) positions to patrol between.
            NPC will visit them in order, then loop back to start.
        """
        self.patrol_points = points
        self.patrol_index = 0  # Start at first point

    # =========================================================================
    # RENDERING SUPPORT
    # =========================================================================

    def get_render_position(self) -> tuple:
        """
        Get position for rendering the sprite.
        
        Returns screen position where sprite's TOP-LEFT corner should be drawn.
        
        =======================================================================
        POSITION OFFSET EXPLANATION
        =======================================================================
        
        Character position (self.x, self.y) represents the character's FEET
        (center-bottom of the sprite).
        
        But sprites are drawn from their TOP-LEFT corner.
        
            Sprite image:
            +--------+ <- render_y (top-left)
            | HEAD   |
            | BODY   |
            | FEET   | <- self.y (character position)
            +--------+
            ^
            |
            render_x (but we want center, not left edge)
        
        So we offset:
        - X: Subtract half width (center -> left edge)
        - Y: Subtract full height (bottom -> top)
        
        =======================================================================
        WHY FEET-CENTERED POSITION?
        =======================================================================
        
        Positioning by feet (center-bottom) is common in top-down games:
        - Collision box is naturally at feet level
        - Depth sorting based on Y position works correctly
        - Characters of different heights align at ground level
        
        =======================================================================
        """
        render_x = self.x - self.sprite.width / 2   # Center horizontally
        render_y = self.y - self.sprite.height      # Feet at bottom of sprite
        return render_x, render_y

    def get_depth(self) -> float:
        """
        Get depth value for render sorting.
        
        =======================================================================
        DEPTH CALCULATION
        =======================================================================
        
        In top-down 2D games, objects lower on screen (higher Y) should
        appear IN FRONT of objects higher on screen (lower Y).
        
        Our depth formula: base_offset + tile_y + z + 0.5
        
        Components:
        - base_offset (-1000): Ensures characters sort after background tiles
        - tile_y: Y position in tiles (higher Y = higher depth = in front)
        - z: Height level adjustment
        - 0.5: Character offset (characters slightly in front of same-Y tiles)
        
        =======================================================================
        DEPTH BUFFER INTERACTION
        =======================================================================
        
        Lower depth = rendered first (background)
        Higher depth = rendered later (foreground)
        
        Example depths:
        - Background tile at Y=100: -900 + 3.125 = -896.875
        - Character at Y=100, Z=0: -1000 + 3.125 + 0 + 0.5 = -996.375
        
        Wait, that would put character BEHIND the tile!
        
        The actual tile rendering uses its own depth calculation.
        Characters should sort among themselves and relative to
        objects at similar Y positions.
        
        =======================================================================
        """
        # Base offset to separate from tile depths
        base_offset = -1000.0
        
        # Convert Y position to tile units
        tile_y = self.y / self.tile_height
        
        # Final depth: lower Y = lower depth = background
        return base_offset + tile_y + self.z + 0.5

    def get_texture(self, renderer: 'OpenGLRenderer') -> 'Texture':
        """
        Get GPU texture for current animation frame.
        
        Initializes textures on first call (lazy initialization).
        
        Parameters:
        -----------
        renderer : OpenGLRenderer
            The renderer (needed for first-time initialization)
            
        Returns:
        --------
        Texture : GPU texture for the current frame
        """
        # Lazy initialization - create textures on first access
        if not self._textures_initialized:
            self._init_textures(renderer)
        
        # Look up cached texture for current direction and frame
        key = (self.sprite.direction, self.sprite.current_frame)
        return self._texture_cache[key]

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def width(self) -> int:
        """Sprite width in pixels (for rendering calculations)."""
        return self.sprite.width
    
    @property
    def height(self) -> int:
        """Sprite height in pixels (for rendering calculations)."""
        return self.sprite.height
