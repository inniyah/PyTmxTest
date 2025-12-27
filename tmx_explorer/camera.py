"""
Camera system for 2D map navigation

=============================================================================
WHAT IS A CAMERA IN 2D GAMES?
=============================================================================

A 2D camera is a "virtual window" into the game world. It determines:
- WHAT part of the world is visible (position)
- HOW MUCH of the world is visible (zoom)

The camera doesn't actually "see" anything - it's a mathematical transform
that converts between two coordinate systems:

1. WORLD COORDINATES: Where things actually are in the game
   - Fixed, absolute positions
   - Example: Player is at (1500, 2300) pixels in the world
   
2. SCREEN COORDINATES: Where things appear on screen
   - Relative to the viewport (window)
   - Example: Player appears at (400, 300) on screen

=============================================================================
CAMERA TRANSFORM VISUALIZATION
=============================================================================

Imagine the game world as a huge piece of paper, and the camera as a
magnifying glass you move around:

    WORLD (large)                        SCREEN (fixed size)
    +----------------------------------+
    |                                  |     +----------+
    |    +----------+                  |     |          |
    |    | VISIBLE  |  <- Camera       | --> | What you |
    |    | AREA     |     viewport     |     | see      |
    |    +----------+                  |     +----------+
    |                                  |
    +----------------------------------+

The camera has:
- Position (x, y): Top-left corner of visible area in world coords
- Zoom: Magnification factor
- Viewport size (width, height): Screen dimensions

=============================================================================
COORDINATE TRANSFORMATION MATH
=============================================================================

World to Screen:
    screen_x = (world_x - camera_x) * zoom
    screen_y = (world_y - camera_y) * zoom

Screen to World:
    world_x = camera_x + screen_x / zoom
    world_y = camera_y + screen_y / zoom

Example:
- Camera at (100, 50), zoom = 2.0
- World point (150, 75)
- Screen position: ((150-100)*2, (75-50)*2) = (100, 50)

=============================================================================
"""


class Camera:
    """
    2D camera with pan and zoom capabilities.
    
    Manages the view into the game world, supporting:
    - Panning (moving the view around)
    - Zooming (magnifying or shrinking the view)
    - Coordinate conversion (world ↔ screen)
    - Viewport bounds calculation (for culling)
    
    ==========================================================================
    USAGE EXAMPLE
    ==========================================================================
    
    ```python
    # Create camera matching window size
    camera = Camera(width=800, height=600)
    
    # In input handling:
    if dragging:
        camera.move(dx, dy)
    if scroll_wheel:
        camera.zoom_by(1.1 if scroll_up else 0.9)
    
    # In rendering:
    renderer.set_camera(camera.x, camera.y, camera.zoom)
    
    # For mouse picking:
    world_pos = camera.screen_to_world(mouse_x, mouse_y)
    ```
    
    ==========================================================================
    COORDINATE SYSTEM
    ==========================================================================
    
    This camera uses a Y-down coordinate system (standard for 2D games):
    
        (0,0) -----> +X
          |
          |
          v
         +Y
    
    Camera (x, y) represents the TOP-LEFT corner of the visible area.
    This matches OpenGL/screen conventions where Y increases downward.
    
    ==========================================================================
    """
    
    # =========================================================================
    # ZOOM LIMITS
    # =========================================================================
    
    # Minimum zoom level (zoomed out maximum)
    # 0.1 = world appears at 10% size, see 10x more area
    # Lower values would make tiles too small to see
    MIN_ZOOM = 0.1
    
    # Maximum zoom level (zoomed in maximum)
    # 5.0 = world appears at 500% size, see 1/5 of normal area
    # Higher values would make individual pixels huge
    MAX_ZOOM = 5.0
    
    def __init__(self, width: int, height: int):
        """
        Initialize camera with viewport dimensions.
        
        Parameters:
        -----------
        width : int
            Viewport width in pixels (typically window width)
        height : int
            Viewport height in pixels (typically window height)
            
        Initial state:
        - Position (0, 0): Looking at world origin
        - Zoom 1.0: 1:1 pixel mapping (1 world pixel = 1 screen pixel)
        """
        # -----------------------------------------------------------------
        # CAMERA POSITION
        # -----------------------------------------------------------------
        # (x, y) = top-left corner of visible area in WORLD coordinates
        # 
        # Think of it as: "Where in the world is the camera pointed?"
        # These are floats to allow smooth sub-pixel camera movement
        self.x = 0.0
        self.y = 0.0
        
        # -----------------------------------------------------------------
        # ZOOM LEVEL
        # -----------------------------------------------------------------
        # zoom = scale factor between world and screen
        # 
        # zoom > 1.0: Zoomed IN (world appears larger, see less area)
        # zoom = 1.0: Normal (1 world pixel = 1 screen pixel)
        # zoom < 1.0: Zoomed OUT (world appears smaller, see more area)
        #
        # Example: zoom = 2.0 means a 16x16 tile appears as 32x32 on screen
        self.zoom = 1.0
        
        # -----------------------------------------------------------------
        # VIEWPORT SIZE
        # -----------------------------------------------------------------
        # Size of the "window" into the world (in screen pixels)
        # Determines how much of the world is visible at zoom = 1.0
        self.width = width
        self.height = height

    # =========================================================================
    # CAMERA MOVEMENT (PANNING)
    # =========================================================================

    def move(self, dx: float, dy: float):
        """
        Move camera by delta, adjusted for zoom.
        
        Parameters:
        -----------
        dx : float
            Horizontal movement in SCREEN pixels
        dy : float
            Vertical movement in SCREEN pixels
            
        =======================================================================
        WHY DIVIDE BY ZOOM?
        =======================================================================
        
        The delta comes from user input (e.g., mouse drag distance in pixels).
        We want consistent "feel" regardless of zoom level.
        
        Without zoom adjustment:
        - At zoom 2.0: Drag 100px → camera moves 100 world units
        - At zoom 0.5: Drag 100px → camera moves 100 world units
        - Problem: At zoom 2.0, 100 world units = 200 screen pixels of movement!
                   The world would appear to move 2x faster when zoomed in.
        
        With zoom adjustment (dx / zoom):
        - At zoom 2.0: Drag 100px → camera moves 50 world units → 100px on screen
        - At zoom 0.5: Drag 100px → camera moves 200 world units → 100px on screen
        - Result: Mouse drag always matches world movement on screen!
        
        This is called "zoom-compensated panning" and is essential for
        natural-feeling map navigation.
        
        =======================================================================
        """
        self.x += dx / self.zoom
        self.y += dy / self.zoom

    # =========================================================================
    # ZOOM CONTROL
    # =========================================================================

    def set_zoom(self, zoom: float):
        """
        Set zoom level, maintaining center point.
        
        Parameters:
        -----------
        zoom : float
            New zoom level (will be clamped to MIN_ZOOM..MAX_ZOOM)
            
        =======================================================================
        CENTER-POINT ZOOM EXPLAINED
        =======================================================================
        
        Naive zoom would just change the zoom value:
            self.zoom = new_zoom
        
        Problem: This zooms from the TOP-LEFT corner, not the center!
        
            Before (zoom 1.0):          After naive zoom 2.0:
            +--------+                  +--------+
            |   X    |                  | X      |  <- X moved!
            |        |                  |        |
            +--------+                  +--------+
            
        The point in the center moves toward the top-left corner.
        This feels WRONG to users - they expect to zoom into the center.
        
        Solution: Calculate the CENTER point before zoom, then reposition
        the camera so the center stays in the same place after zoom.
        
            Before:                     After center-preserving zoom:
            +--------+                  +--------+
            |   X    |                  |   X    |  <- X stays centered!
            |        |                  |        |
            +--------+                  +--------+
        
        =======================================================================
        THE MATH
        =======================================================================
        
        1. Find current center in WORLD coordinates:
           center_x = camera_x + (viewport_width / 2) / zoom
           
           Why divide viewport by zoom? Because viewport is in screen pixels,
           but we need world coordinates.
        
        2. Apply new zoom (with clamping)
        
        3. Reposition camera so center stays at same world position:
           new_camera_x = center_x - (viewport_width / 2) / new_zoom
        
        =======================================================================
        """
        # Step 1: Calculate current center point in WORLD coordinates
        # The center of the viewport is at (width/2, height/2) in screen coords
        # Convert to world: screen_pos / zoom + camera_pos
        center_x = self.x + self.width / (2 * self.zoom)
        center_y = self.y + self.height / (2 * self.zoom)
        
        # Step 2: Apply new zoom level with clamping
        # max(MIN, min(MAX, value)) ensures value is within [MIN, MAX]
        self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
        
        # Step 3: Reposition camera to maintain center point
        # Reverse the center calculation with new zoom
        self.x = center_x - self.width / (2 * self.zoom)
        self.y = center_y - self.height / (2 * self.zoom)

    def zoom_by(self, factor: float):
        """
        Multiply current zoom by factor.
        
        Convenience method for relative zoom changes.
        
        Parameters:
        -----------
        factor : float
            Zoom multiplier
            - factor > 1.0: Zoom IN (e.g., 1.1 = 10% closer)
            - factor < 1.0: Zoom OUT (e.g., 0.9 = 10% farther)
            
        =======================================================================
        WHY MULTIPLICATIVE ZOOM?
        =======================================================================
        
        Zoom should feel "even" at all levels. Additive zoom doesn't work:
        
        Additive (zoom += 0.1):
        - From 0.5 to 0.6: 20% increase (big change!)
        - From 2.0 to 2.1: 5% increase (tiny change)
        
        Multiplicative (zoom *= 1.1):
        - From 0.5 to 0.55: 10% increase
        - From 2.0 to 2.2: 10% increase
        
        Multiplicative feels consistent because human perception of scale
        is logarithmic, not linear.
        
        =======================================================================
        COMMON USAGE
        =======================================================================
        
        ```python
        # Scroll wheel zoom
        if scroll_up:
            camera.zoom_by(1.1)   # 10% zoom in
        elif scroll_down:
            camera.zoom_by(0.9)   # ~10% zoom out (1/1.1 ≈ 0.909)
        
        # For smoother zoom, use smaller factors:
        camera.zoom_by(1.05)     # 5% zoom in
        ```
        
        =======================================================================
        """
        self.set_zoom(self.zoom * factor)

    def reset(self, map_width: int, map_height: int, 
              tile_width: int, tile_height: int):
        """
        Reset camera to fit map in view.
        
        Positions camera at origin and calculates zoom to show entire map.
        Useful for:
        - Initial view when loading a map
        - "Fit to window" button
        - Overview/minimap mode
        
        Parameters:
        -----------
        map_width : int
            Map width in TILES
        map_height : int
            Map height in TILES
        tile_width : int
            Tile width in pixels
        tile_height : int
            Tile height in pixels
            
        =======================================================================
        FIT-TO-VIEW ALGORITHM
        =======================================================================
        
        Calculate the zoom needed to fit the entire map in the viewport:
        
        1. Calculate zoom needed to fit WIDTH:
           zoom_x = viewport_width / (map_width * tile_width)
           
        2. Calculate zoom needed to fit HEIGHT:
           zoom_y = viewport_height / (map_height * tile_height)
           
        3. Use the SMALLER zoom to ensure BOTH dimensions fit:
           zoom = min(zoom_x, zoom_y)
           
        4. Don't zoom in past 1.0 (no magnification for small maps)
           zoom = min(zoom, 1.0)
        
        Example:
        - Viewport: 800x600
        - Map: 50x50 tiles at 16px each = 800x800 pixels
        - zoom_x = 800/800 = 1.0
        - zoom_y = 600/800 = 0.75
        - Final zoom = min(1.0, 0.75, 1.0) = 0.75
        
        =======================================================================
        """
        # Position camera at world origin (top-left of map)
        self.x = 0
        self.y = 0
        
        # Calculate map size in pixels
        map_pixel_width = map_width * tile_width
        map_pixel_height = map_height * tile_height
        
        # Calculate zoom needed for each axis
        zoom_x = self.width / map_pixel_width
        zoom_y = self.height / map_pixel_height
        
        # Use smaller zoom (ensures both dimensions fit)
        # Cap at 1.0 (don't zoom in for small maps)
        self.zoom = min(zoom_x, zoom_y, 1.0)

    # =========================================================================
    # COORDINATE CONVERSION
    # =========================================================================

    def screen_to_world(self, screen_x: float, screen_y: float) -> tuple:
        """
        Convert screen coordinates to world coordinates.
        
        Parameters:
        -----------
        screen_x : float
            X position on screen (0 = left edge of viewport)
        screen_y : float
            Y position on screen (0 = top edge of viewport)
            
        Returns:
        --------
        tuple : (world_x, world_y) position in world coordinates
        
        =======================================================================
        USE CASES
        =======================================================================
        
        1. MOUSE PICKING: Click at screen position → which tile/entity?
           ```python
           world_x, world_y = camera.screen_to_world(mouse_x, mouse_y)
           tile_x = int(world_x // tile_width)
           tile_y = int(world_y // tile_height)
           ```
        
        2. UI INTERACTION: Where in the world did the user click?
        
        3. DEBUG DISPLAY: Show world coords at mouse position
        
        =======================================================================
        THE MATH
        =======================================================================
        
        The camera shows a portion of the world. Given a screen position,
        we need to find the corresponding world position.
        
        screen_pos = (world_pos - camera_pos) * zoom    [world_to_screen]
        
        Solving for world_pos:
        screen_pos / zoom = world_pos - camera_pos
        world_pos = camera_pos + screen_pos / zoom      [screen_to_world]
        
        =======================================================================
        """
        world_x = self.x + screen_x / self.zoom
        world_y = self.y + screen_y / self.zoom
        return world_x, world_y

    def world_to_screen(self, world_x: float, world_y: float) -> tuple:
        """
        Convert world coordinates to screen coordinates.
        
        Parameters:
        -----------
        world_x : float
            X position in world coordinates
        world_y : float
            Y position in world coordinates
            
        Returns:
        --------
        tuple : (screen_x, screen_y) position in screen coordinates
        
        =======================================================================
        USE CASES
        =======================================================================
        
        1. RENDERING: Where to draw a game object on screen?
           ```python
           screen_x, screen_y = camera.world_to_screen(entity.x, entity.y)
           draw_sprite(sprite, screen_x, screen_y)
           ```
        
        2. UI OVERLAYS: Position a health bar above an entity
        
        3. VISIBILITY CHECK: Is this world position on screen?
           ```python
           sx, sy = camera.world_to_screen(obj.x, obj.y)
           on_screen = (0 <= sx < width and 0 <= sy < height)
           ```
        
        =======================================================================
        THE MATH
        =======================================================================
        
        1. Translate: Subtract camera position to get relative position
           relative = world_pos - camera_pos
           
        2. Scale: Multiply by zoom to get screen position
           screen = relative * zoom
        
        Combined:
           screen_pos = (world_pos - camera_pos) * zoom
        
        =======================================================================
        """
        screen_x = (world_x - self.x) * self.zoom
        screen_y = (world_y - self.y) * self.zoom
        return screen_x, screen_y

    # =========================================================================
    # VIEWPORT BOUNDS
    # =========================================================================

    def get_visible_bounds(self) -> tuple:
        """
        Get visible world bounds (left, top, right, bottom).
        
        Returns the rectangle of world coordinates currently visible
        on screen.
        
        Returns:
        --------
        tuple : (left, top, right, bottom) in world coordinates
        
        =======================================================================
        USE CASES
        =======================================================================
        
        1. CULLING: Only render objects within visible bounds
           ```python
           left, top, right, bottom = camera.get_visible_bounds()
           for entity in entities:
               if left <= entity.x <= right and top <= entity.y <= bottom:
                   entity.render()  # Only render visible entities
           ```
        
        2. TILE RANGE: Which tiles are visible?
           ```python
           left, top, right, bottom = camera.get_visible_bounds()
           start_tile_x = max(0, int(left // tile_width))
           end_tile_x = min(map_width, int(right // tile_width) + 1)
           # Only iterate visible tiles, not entire map!
           ```
        
        3. CHUNK LOADING: Which map chunks need to be loaded?
        
        =======================================================================
        THE MATH
        =======================================================================
        
        The visible area in world coordinates:
        - Left edge: camera.x (already in world coords)
        - Top edge: camera.y
        - Right edge: camera.x + (viewport_width / zoom)
        - Bottom edge: camera.y + (viewport_height / zoom)
        
        Note: We divide viewport size by zoom because:
        - Higher zoom = see less world area
        - zoom 2.0 + 800px viewport = 400 world units visible
        
        =======================================================================
        CULLING OPTIMIZATION
        =======================================================================
        
        Culling is CRITICAL for performance in large maps:
        
        Without culling (render everything):
        - 1000x1000 tile map = 1,000,000 tiles to process
        - Most are off-screen = wasted work
        
        With culling (render visible only):
        - 800x600 viewport at 32px tiles = ~25x19 = ~475 tiles
        - Only process what's actually visible!
        
        This can be the difference between 5 FPS and 60 FPS.
        
        =======================================================================
        """
        # Left and top are just the camera position
        left = self.x
        top = self.y
        
        # Right and bottom extend by viewport size, adjusted for zoom
        # Higher zoom = smaller visible area in world coordinates
        right = self.x + self.width / self.zoom
        bottom = self.y + self.height / self.zoom
        
        return left, top, right, bottom
