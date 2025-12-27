"""
Animated sprite system using PIL

=============================================================================
SPRITE ANIMATION OVERVIEW
=============================================================================

Sprite animation creates the illusion of movement by displaying a sequence
of images (frames) in rapid succession, like a flipbook.

This system is designed for CHARACTER sprites in RPG-style games, using
a standard 4x4 spritesheet layout:

    +-------+-------+-------+-------+
    | Down  | Down  | Down  | Down  |   <- Row 0: Facing down
    | Idle  | Walk1 | Idle  | Walk2 |
    +-------+-------+-------+-------+
    | Left  | Left  | Left  | Left  |   <- Row 1: Facing left
    | Idle  | Walk1 | Idle  | Walk2 |
    +-------+-------+-------+-------+
    | Right | Right | Right | Right |   <- Row 2: Facing right
    | Idle  | Walk1 | Idle  | Walk2 |
    +-------+-------+-------+-------+
    | Up    | Up    | Up    | Up    |   <- Row 3: Facing up
    | Idle  | Walk1 | Idle  | Walk2 |
    +-------+-------+-------+-------+
      Col 0   Col 1   Col 2   Col 3

This is a VERY common spritesheet format, used by:
- RPG Maker
- Many asset packs on itch.io
- OpenGameArt character sprites

=============================================================================
ANIMATION TIMING
=============================================================================

Animation uses TIME-BASED updates (not frame-based):

Frame-based (BAD):
    every_render_frame: current_frame++
    Problem: Animation speed depends on framerate!
    At 60 FPS: 60 frames/second = too fast!
    At 30 FPS: 30 frames/second = half speed!

Time-based (GOOD):
    update(delta_time): timer += dt * speed
    Problem solved: Animation runs at same speed regardless of FPS
    At 60 FPS: dt ≈ 0.016s per frame
    At 30 FPS: dt ≈ 0.033s per frame
    Same animation speed in both cases!

=============================================================================
MEMORY OPTIMIZATION: FRAME SHARING
=============================================================================

If you have 100 NPCs using the same spritesheet, you don't want 100 copies
of the image data in memory!

Solution: from_cached() class method creates new sprite instances that
SHARE the frame images with a "template" sprite. Each instance has its own
animation state (current frame, direction, etc.) but references the same
underlying image data.

Memory comparison for 100 NPCs with 64x64 sprites (16 frames):
- Without sharing: 100 × 16 × 64 × 64 × 4 bytes = 26 MB
- With sharing:    1 × 16 × 64 × 64 × 4 bytes = 0.26 MB
- Savings: ~99%!

=============================================================================
"""

from PIL import Image
from enum import IntEnum
from typing import List, Dict, Optional
from pathlib import Path


class Direction(IntEnum):
    """
    Character facing direction.
    
    Values match spritesheet row order (industry standard):
    - Row 0 = Down (toward camera in top-down view)
    - Row 1 = Left
    - Row 2 = Right
    - Row 3 = Up (away from camera)
    
    Using IntEnum allows:
    - Direct use as array index: frames[direction] works!
    - Comparison: direction == Direction.DOWN
    - Iteration: for d in Direction
    
    ==========================================================================
    WHY THIS ORDER?
    ==========================================================================
    
    Down-Left-Right-Up is the most common convention because:
    1. "Down" (facing camera) is the default/idle direction in many RPGs
    2. Left-Right are horizontally adjacent (might be mirrored)
    3. "Up" (back turned) is often least important
    
    Some spritesheets use Down-Up-Left-Right or other orders.
    If yours differs, just change the IntEnum values to match.
    
    ==========================================================================
    """
    DOWN = 0   # Facing toward camera (south)
    LEFT = 1   # Facing left (west)
    RIGHT = 2  # Facing right (east)
    UP = 3     # Facing away from camera (north)


class AnimationState(IntEnum):
    """
    Animation states for character sprites.
    
    This simple system has only two states:
    - IDLE: Character standing still (shows frame 0)
    - WALKING: Character moving (cycles through walk frames)
    
    ==========================================================================
    EXTENDING THE STATE SYSTEM
    ==========================================================================
    
    For more complex games, you might add:
    - RUNNING = 2    (faster walk cycle or different frames)
    - ATTACKING = 3  (combat animation)
    - HURT = 4       (damage reaction)
    - DYING = 5      (death sequence)
    - SWIMMING = 6   (water movement)
    
    Each state would need corresponding frames in the spritesheet
    (likely additional rows).
    
    ==========================================================================
    """
    IDLE = 0      # Standing still
    WALKING = 1   # Moving


class AnimatedSprite:
    """
    Animated sprite from a 4x4 spritesheet.
    
    Handles loading, frame management, and time-based animation updates
    for character sprites in RPG-style games.
    
    ==========================================================================
    USAGE EXAMPLE
    ==========================================================================
    
    ```python
    # Load sprite
    player_sprite = AnimatedSprite("characters/hero.png")
    
    # In game loop:
    def update(dt):
        # Set direction based on input
        if moving_left:
            player_sprite.set_direction(Direction.LEFT)
            player_sprite.set_walking(True)
        elif not moving:
            player_sprite.set_walking(False)
        
        # Update animation timing
        player_sprite.update(dt)
    
    def render():
        frame = player_sprite.get_current_frame()
        # Draw frame at player position...
    ```
    
    ==========================================================================
    SPRITESHEET REQUIREMENTS
    ==========================================================================
    
    - Format: 4 columns × 4 rows grid
    - Each cell: One animation frame
    - Row order: Down, Left, Right, Up (Direction enum values)
    - Column order: Idle, Walk1, Idle/Walk2, Walk3 (varies by asset)
    - File format: Any PIL-supported format (PNG recommended for transparency)
    
    ==========================================================================
    """
    
    # =========================================================================
    # SPRITESHEET LAYOUT CONSTANTS
    # =========================================================================
    
    COLS = 4  # Frames per direction (columns in spritesheet)
    ROWS = 4  # Number of directions (rows in spritesheet)
    
    def __init__(self, spritesheet_path: str,
                 frame_width: int = None, frame_height: int = None,
                 animation_speed: float = 8.0):
        """
        Load and initialize animated sprite from spritesheet.
        
        Parameters:
        -----------
        spritesheet_path : str
            Path to spritesheet image file
            
        frame_width : int, optional
            Width of each frame in pixels.
            If None, auto-calculated as sheet_width / 4
            
        frame_height : int, optional
            Height of each frame in pixels.
            If None, auto-calculated as sheet_height / 4
            
        animation_speed : float
            Animation playback speed.
            Higher = faster animation.
            8.0 = cycle through ~8 frames per second
            
        =======================================================================
        AUTO-SIZING EXPLAINED
        =======================================================================
        
        Most 4x4 spritesheets are evenly divided, so we can auto-calculate:
        
        Example: 128x128 spritesheet
        - frame_width = 128 / 4 = 32 pixels
        - frame_height = 128 / 4 = 32 pixels
        
        Override these if your spritesheet has:
        - Padding/margins between frames
        - Non-square frames
        - Irregular layout
        
        =======================================================================
        """
        # -----------------------------------------------------------------
        # LOAD SPRITESHEET
        # -----------------------------------------------------------------
        # Convert to RGBA ensures consistent format with alpha channel
        # (transparency support for character sprites)
        self.spritesheet = Image.open(spritesheet_path).convert('RGBA')
        
        # Get spritesheet dimensions
        sheet_w = self.spritesheet.width
        sheet_h = self.spritesheet.height
        
        # -----------------------------------------------------------------
        # CALCULATE FRAME DIMENSIONS
        # -----------------------------------------------------------------
        # Use provided dimensions or auto-calculate from spritesheet size
        # The "or" pattern: if frame_width is None (or 0), use calculated value
        self.frame_width = frame_width or (sheet_w // self.COLS)
        self.frame_height = frame_height or (sheet_h // self.ROWS)
        
        # -----------------------------------------------------------------
        # ANIMATION STATE
        # -----------------------------------------------------------------
        
        # Animation speed: frames per second
        # Higher value = faster animation
        self.animation_speed = animation_speed
        
        # Timer accumulates delta time until it reaches 1.0 (next frame)
        # This allows smooth timing independent of framerate
        self.animation_timer = 0.0
        
        # Current frame index (0-3 for 4-frame animation)
        self.current_frame = 0
        
        # -----------------------------------------------------------------
        # SPRITE STATE
        # -----------------------------------------------------------------
        
        # Current facing direction (determines which row of frames to use)
        self.direction = Direction.DOWN
        
        # Current animation state (IDLE or WALKING)
        self.state = AnimationState.IDLE
        
        # -----------------------------------------------------------------
        # PRE-CUT FRAMES
        # -----------------------------------------------------------------
        # Dictionary mapping Direction -> list of frame images
        # Pre-cutting is an optimization - see _cut_frames() for details
        self.frames: Dict[Direction, List[Image.Image]] = {}
        self._cut_frames()
        
        # Debug output
        print(f"Loaded spritesheet: {Path(spritesheet_path).name} "
              f"({self.frame_width}x{self.frame_height} per frame)")

    @classmethod
    def from_cached(cls, cached_sprite: 'AnimatedSprite', 
                    animation_speed: float = 8.0) -> 'AnimatedSprite':
        """
        Create new instance sharing frames with cached sprite.
        
        This is a MEMORY OPTIMIZATION for multiple characters using
        the same spritesheet. Instead of loading and cutting the
        spritesheet again, we reuse the existing frame images.
        
        Parameters:
        -----------
        cached_sprite : AnimatedSprite
            An existing sprite to share frames with (the "template")
        animation_speed : float
            Animation speed for this instance (can differ from template)
            
        Returns:
        --------
        AnimatedSprite : New instance with shared frame data
        
        =======================================================================
        WHY SHARE FRAMES?
        =======================================================================
        
        PIL Image objects are relatively large in memory:
        - 64x64 RGBA image = 64 × 64 × 4 bytes = 16 KB per frame
        - 16 frames per spritesheet = 256 KB per character
        - 100 NPCs = 25 MB just for sprites!
        
        By sharing frame references:
        - 1 spritesheet loaded = 256 KB (once)
        - 100 NPCs = same 256 KB + tiny state objects
        - Python reference counting handles cleanup automatically
        
        =======================================================================
        HOW IT WORKS
        =======================================================================
        
        ```python
        # Create template (loads and cuts spritesheet)
        npc_template = AnimatedSprite("npc.png")
        
        # Create many NPCs sharing the same frames
        npcs = [AnimatedSprite.from_cached(npc_template) for _ in range(100)]
        
        # Each NPC can have independent animation state:
        npcs[0].set_direction(Direction.LEFT)
        npcs[1].set_direction(Direction.RIGHT)
        # But they all reference the same underlying image data!
        ```
        
        =======================================================================
        OBJECT CREATION WITHOUT __init__
        =======================================================================
        
        We use object.__new__(cls) to create the instance WITHOUT calling
        __init__. This avoids:
        - Reloading the spritesheet from disk
        - Re-cutting all frames
        - Duplicate memory allocation
        
        Then we manually set all attributes.
        
        =======================================================================
        """
        # Create instance without calling __init__
        # object.__new__() allocates memory but skips initialization
        instance = object.__new__(cls)
        
        # -----------------------------------------------------------------
        # SHARED DATA (references, not copies!)
        # -----------------------------------------------------------------
        # These point to the SAME objects as cached_sprite
        # No memory duplication - just reference copying
        instance.frames = cached_sprite.frames          # THE big memory saver!
        instance.frame_width = cached_sprite.frame_width
        instance.frame_height = cached_sprite.frame_height
        instance.spritesheet = cached_sprite.spritesheet  # Keep reference
        
        # -----------------------------------------------------------------
        # INSTANCE-SPECIFIC STATE
        # -----------------------------------------------------------------
        # Each character needs its own animation state
        # (different NPCs can be in different animation frames)
        instance.animation_speed = animation_speed
        instance.animation_timer = 0.0
        instance.current_frame = 0
        instance.direction = Direction.DOWN
        instance.state = AnimationState.IDLE
        
        return instance

    def _cut_frames(self):
        """
        Pre-cut all frames from spritesheet.
        
        =======================================================================
        WHY PRE-CUT?
        =======================================================================
        
        Alternative approach - cut on demand:
        ```python
        def get_frame(self, direction, index):
            x = index * self.frame_width
            y = direction * self.frame_height
            return self.spritesheet.crop((x, y, x+w, y+h))
        ```
        
        Problem: crop() creates a new Image object EVERY call!
        - 60 FPS = 60 crop operations per second per sprite
        - Memory allocation + copying every frame
        - Garbage collector overhead
        
        Solution: Pre-cut all frames once at load time
        - 16 crop operations total (once)
        - get_frame() is just a dictionary lookup - O(1), no allocation
        - Much better performance in the game loop
        
        =======================================================================
        MEMORY LAYOUT AFTER CUTTING
        =======================================================================
        
        self.frames = {
            Direction.DOWN:  [frame0, frame1, frame2, frame3],
            Direction.LEFT:  [frame0, frame1, frame2, frame3],
            Direction.RIGHT: [frame0, frame1, frame2, frame3],
            Direction.UP:    [frame0, frame1, frame2, frame3],
        }
        
        Each frame is an independent PIL Image object.
        
        =======================================================================
        """
        for direction in Direction:
            # Initialize list for this direction
            self.frames[direction] = []
            
            # Row in spritesheet (Direction enum value = row index)
            row = direction.value
            
            # Cut each frame in this row
            for col in range(self.COLS):
                # Calculate pixel coordinates for this frame
                x = col * self.frame_width
                y = row * self.frame_height
                
                # Crop frame from spritesheet
                # crop() takes (left, top, right, bottom) box
                frame = self.spritesheet.crop((
                    x, y,                                    # Top-left
                    x + self.frame_width, y + self.frame_height  # Bottom-right
                ))
                
                # Add to frames list
                self.frames[direction].append(frame)

    # =========================================================================
    # ANIMATION UPDATE
    # =========================================================================

    def update(self, dt: float):
        """
        Update animation state based on elapsed time.
        
        Call this every frame with the time since last frame (delta time).
        
        Parameters:
        -----------
        dt : float
            Delta time in seconds since last update
            Typically 1/60 ≈ 0.0167 at 60 FPS
            
        =======================================================================
        TIME-BASED ANIMATION MATH
        =======================================================================
        
        The formula: timer += dt * speed
        
        Example with speed = 8.0 (8 frames per second):
        - Each frame at 60 FPS: dt ≈ 0.0167
        - Timer increment: 0.0167 × 8.0 ≈ 0.133
        - After ~7.5 render frames: timer reaches 1.0 → advance animation
        - Result: ~8 animation frames per second, regardless of FPS!
        
        =======================================================================
        FRAME ADVANCEMENT LOGIC
        =======================================================================
        
        When timer >= 1.0:
        1. Calculate how many frames to advance (usually 1)
        2. Subtract that amount from timer (keep fractional remainder)
        3. Advance current_frame by that amount
        4. Wrap around if past frame 3 (cycle animation)
        
        The fractional remainder prevents "drift" - if we just reset to 0,
        we'd lose the extra accumulated time.
        
        =======================================================================
        WALK CYCLE FRAMES
        =======================================================================
        
        For walking animation, we use frames 1-3 (skipping frame 0):
        - Frame 0: Idle pose (standing still)
        - Frame 1: Walk step A
        - Frame 2: Passing pose (might be same as idle)
        - Frame 3: Walk step B
        
        The cycle is: 1 → 2 → 3 → 1 → 2 → 3 → ...
        
        When walking starts, we jump to frame 1 (first walk frame).
        The "while current_frame > 3" wraps 4→1, 5→2, etc.
        
        Note: This specific cycle (1-2-3, not 0-1-2-3) is designed for
        spritesheets where frame 0 is a distinct "standing" pose.
        
        =======================================================================
        """
        if self.state == AnimationState.WALKING:
            # Accumulate time, scaled by animation speed
            self.animation_timer += dt * self.animation_speed
            
            # Check if we should advance one or more frames
            frames_to_advance = int(self.animation_timer)
            
            if frames_to_advance > 0:
                # Remove whole frames from timer (keep fractional part)
                self.animation_timer -= frames_to_advance
                
                # Advance frame counter
                self.current_frame += frames_to_advance
                
                # Wrap around: cycle through frames 1, 2, 3 (not 0)
                # Frame 0 is reserved for idle pose
                while self.current_frame > 3:
                    self.current_frame -= 3  # 4→1, 5→2, 6→3, 7→1, etc.
        else:
            # IDLE state: always show frame 0, reset timer
            self.current_frame = 0
            self.animation_timer = 0.0

    # =========================================================================
    # STATE CONTROL
    # =========================================================================

    def set_direction(self, direction: Direction):
        """
        Set character facing direction.
        
        Parameters:
        -----------
        direction : Direction
            New facing direction (DOWN, LEFT, RIGHT, UP)
            
        Note: This changes which ROW of frames is used.
        Does not reset animation frame - direction changes are seamless.
        """
        self.direction = direction

    def set_walking(self, walking: bool):
        """
        Set whether character is walking or idle.
        
        Parameters:
        -----------
        walking : bool
            True to start walking animation, False for idle
            
        =======================================================================
        STATE TRANSITION HANDLING
        =======================================================================
        
        When changing states, we need to handle the transition properly:
        
        IDLE → WALKING:
        - Set state to WALKING
        - Jump to frame 1 (first walk frame, skip idle frame 0)
        - Reset timer to start fresh
        
        WALKING → IDLE:
        - Set state to IDLE
        - Jump to frame 0 (idle pose)
        - Reset timer
        
        We only do this when state ACTUALLY changes (check current state first).
        This prevents resetting the animation on every frame when walking
        continuously.
        
        =======================================================================
        """
        if walking:
            # Only act if we're not already walking
            if self.state != AnimationState.WALKING:
                self.state = AnimationState.WALKING
                self.current_frame = 1  # Start on first walk frame
                self.animation_timer = 0.0
        else:
            # Only act if we're not already idle
            if self.state != AnimationState.IDLE:
                self.state = AnimationState.IDLE
                self.current_frame = 0  # Return to idle pose
                self.animation_timer = 0.0

    # =========================================================================
    # FRAME ACCESS
    # =========================================================================

    def get_current_frame(self) -> Image.Image:
        """
        Get the current animation frame image.
        
        Returns the appropriate frame based on current direction and
        animation frame index.
        
        Returns:
        --------
        PIL.Image : The current frame image (RGBA)
        
        Usage:
        ------
        ```python
        frame = sprite.get_current_frame()
        texture = Texture.from_pil(frame)
        # or draw directly with PIL
        ```
        """
        return self.frames[self.direction][self.current_frame]

    def get_frame(self, direction: Direction, frame_index: int) -> Image.Image:
        """
        Get a specific frame by direction and index.
        
        Useful for:
        - UI previews (show character facing specific direction)
        - Manual animation control
        - Debug visualization
        
        Parameters:
        -----------
        direction : Direction
            Which direction's frames to access
        frame_index : int
            Which frame (0-3)
            
        Returns:
        --------
        PIL.Image : The requested frame image
        """
        return self.frames[direction][frame_index]

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def width(self) -> int:
        """
        Frame width in pixels.
        
        Use for positioning and collision calculations.
        """
        return self.frame_width
    
    @property
    def height(self) -> int:
        """
        Frame height in pixels.
        
        Use for positioning and collision calculations.
        """
        return self.frame_height
