"""
Gamepad/Joystick support using GLFW

=============================================================================
GAMEPAD INPUT OVERVIEW
=============================================================================

This module provides gamepad/joystick support for the game engine, handling:
- Controller detection and connection/disconnection
- Standard gamepad mapping (Xbox-style layout)
- Fallback for generic/unmapped joysticks
- Analog stick deadzone handling
- D-pad and hat switch support
- Button press detection (including "just pressed" events)

=============================================================================
GLFW GAMEPAD vs JOYSTICK
=============================================================================

GLFW distinguishes between two modes:

1. GAMEPAD MODE (glfw.joystick_is_gamepad() = True):
   - Controller has a known mapping in SDL_GameControllerDB
   - Buttons/axes have standardized names (A, B, X, Y, left stick, etc.)
   - Uses glfw.get_gamepad_state() for consistent layout
   - Works like Xbox controller regardless of actual hardware

2. JOYSTICK MODE (fallback):
   - Unknown/unmapped controller
   - Raw axes and buttons by index (axis 0, button 3, etc.)
   - Layout varies by manufacturer
   - We guess common layouts (Xbox-like)

=============================================================================
SDL_GAMECONTROLLERDB
=============================================================================

The SDL GameController database is a community-maintained file mapping
thousands of controllers to a standard Xbox-like layout.

Download from: https://github.com/gabomdq/SDL_GameControllerDB

Format example:
    030000004c050000c405000000010000,PS4 Controller,a:b1,b:b2,back:b8,...

Each line maps a hardware GUID to button/axis assignments.

=============================================================================
DEADZONE HANDLING
=============================================================================

Analog sticks rarely rest exactly at 0.0 due to hardware imprecision.
A "deadzone" ignores small values near center:

    Raw value:      -0.05  (stick slightly off-center)
    With deadzone:   0.0   (treated as centered)

We also rescale the remaining range so the usable range is still 0.0 to 1.0:

    Raw range:      [-1.0 ... -0.15] [DEADZONE] [0.15 ... 1.0]
    Output range:   [-1.0 ......... 0.0 ........ 0.0 ......... 1.0]

=============================================================================
"""

import glfw
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


def load_gamepad_mappings(filepath: str = None) -> int:
    """
    Load gamepad mappings from an SDL_GameControllerDB file.
    
    This function searches for the gamecontrollerdb.txt file and loads
    controller mappings into GLFW. With mappings loaded, more controllers
    will be recognized as "gamepads" with standardized button layouts.
    
    Parameters:
    -----------
    filepath : str, optional
        Explicit path to gamecontrollerdb.txt.
        If None, searches common locations.
        
    Returns:
    --------
    int : Number of mappings loaded, or -1 if no file found
    
    ==========================================================================
    SEARCH PATHS
    ==========================================================================
    
    We search multiple locations to make the file easy to place:
    
    1. Explicit path (if provided)
    2. Current working directory
    3. assets/ subdirectory
    4. Same directory as this module
    5. Parent directory of this module
    6. User's .config directory (Linux)
    
    This flexibility helps with different project structures.
    
    ==========================================================================
    """
    search_paths = []
    
    # User-specified path first
    if filepath:
        search_paths.append(Path(filepath))
    
    # Common locations
    search_paths.extend([
        Path("gamecontrollerdb.txt"),                      # Current directory
        Path("assets/gamecontrollerdb.txt"),               # Assets folder
        Path(__file__).parent / "gamecontrollerdb.txt",    # Module directory
        Path(__file__).parent.parent / "gamecontrollerdb.txt",  # Parent dir
        Path.home() / ".config/gamecontrollerdb.txt",      # User config (Linux)
    ])
    
    # Try each path
    for path in search_paths:
        if path.exists():
            try:
                # Read the entire file
                content = path.read_text(encoding='utf-8')
                
                # Load into GLFW
                # Returns True on success
                result = glfw.update_gamepad_mappings(content)
                
                if result:
                    # Count non-comment, non-empty lines
                    count = sum(1 for line in content.splitlines() 
                               if line.strip() and not line.startswith('#'))
                    print(f"Gamepad mappings loaded: {count} from {path}")
                    return count
                    
            except Exception as e:
                print(f"Error loading mappings from {path}: {e}")
    
    print("gamecontrollerdb.txt not found")
    return -1


# =============================================================================
# GAMEPAD STATE DATA CLASS
# =============================================================================

@dataclass
class GamepadState:
    """
    Current state of all gamepad inputs.
    
    This dataclass holds the state of every button and axis on a standard
    Xbox-style gamepad. Values are updated each frame by GamepadManager.
    
    ==========================================================================
    XBOX-STYLE LAYOUT REFERENCE
    ==========================================================================
    
                    [LB]                              [RB]
                    [LT]                              [RT]
    
                     (Y)                              
               (X)       (B)                    [RIGHT STICK]
                     (A)
    
        [LEFT STICK]            [BACK] [GUIDE] [START]
    
                               [D-PAD]
    
    ==========================================================================
    AXIS VALUES
    ==========================================================================
    
    Sticks: -1.0 to 1.0
        left_x:  -1.0 = left,  1.0 = right
        left_y:  -1.0 = up,    1.0 = down
        right_x: -1.0 = left,  1.0 = right
        right_y: -1.0 = up,    1.0 = down
    
    Triggers: 0.0 to 1.0
        0.0 = not pressed
        1.0 = fully pressed
    
    ==========================================================================
    BUTTON VALUES
    ==========================================================================
    
    All buttons are boolean: True = pressed, False = released
    
    ==========================================================================
    """
    # -------------------------------------------------------------------------
    # ANALOG STICKS (-1.0 to 1.0, with deadzone applied)
    # -------------------------------------------------------------------------
    left_x: float = 0.0      # Left stick horizontal
    left_y: float = 0.0      # Left stick vertical
    right_x: float = 0.0     # Right stick horizontal
    right_y: float = 0.0     # Right stick vertical
    
    # -------------------------------------------------------------------------
    # TRIGGERS (0.0 to 1.0)
    # -------------------------------------------------------------------------
    left_trigger: float = 0.0    # LT (left trigger)
    right_trigger: float = 0.0   # RT (right trigger)
    
    # -------------------------------------------------------------------------
    # FACE BUTTONS (A, B, X, Y)
    # -------------------------------------------------------------------------
    a: bool = False    # A button (bottom)
    b: bool = False    # B button (right)
    x: bool = False    # X button (left)
    y: bool = False    # Y button (top)
    
    # -------------------------------------------------------------------------
    # SHOULDER BUTTONS (Bumpers)
    # -------------------------------------------------------------------------
    left_bumper: bool = False    # LB (left bumper)
    right_bumper: bool = False   # RB (right bumper)
    
    # -------------------------------------------------------------------------
    # MENU BUTTONS
    # -------------------------------------------------------------------------
    back: bool = False     # Back/Select/Share
    start: bool = False    # Start/Menu/Options
    guide: bool = False    # Xbox/PS/Home button
    
    # -------------------------------------------------------------------------
    # STICK BUTTONS (clicking the sticks)
    # -------------------------------------------------------------------------
    left_stick: bool = False     # L3 (left stick click)
    right_stick: bool = False    # R3 (right stick click)
    
    # -------------------------------------------------------------------------
    # D-PAD (directional pad)
    # -------------------------------------------------------------------------
    dpad_up: bool = False
    dpad_right: bool = False
    dpad_down: bool = False
    dpad_left: bool = False


# =============================================================================
# GAMEPAD MANAGER CLASS
# =============================================================================

class GamepadManager:
    """
    Manages gamepad input detection and state updates.
    
    This class handles:
    - Finding connected controllers
    - Detecting connection/disconnection
    - Reading input state each frame
    - Converting raw values (deadzone, trigger normalization)
    - Supporting both mapped gamepads and generic joysticks
    
    ==========================================================================
    USAGE
    ==========================================================================
    
    ```python
    # Initialize (usually at game startup)
    gamepad = GamepadManager()
    
    # Each frame:
    gamepad.update()
    
    # Get movement input
    dx, dy = gamepad.get_movement()
    player.move(dx, dy)
    
    # Check button presses
    if gamepad.button_just_pressed('a'):
        player.jump()
    
    # Check current button state
    if gamepad.state.b:
        player.run()
    ```
    
    ==========================================================================
    """
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Deadzone threshold for analog sticks
    # Values below this (in absolute terms) are treated as 0
    # 0.15 = 15% of full range
    DEADZONE = 0.15
    
    def __init__(self, mappings_file: str = None):
        """
        Initialize gamepad manager.
        
        Parameters:
        -----------
        mappings_file : str, optional
            Path to gamecontrollerdb.txt for additional controller mappings.
            If None, searches default locations.
        """
        # -----------------------------------------------------------------
        # CONNECTION STATE
        # -----------------------------------------------------------------
        
        # Currently connected gamepad ID (GLFW joystick slot)
        # None = no gamepad connected
        self.connected_gamepad: Optional[int] = None
        
        # Whether the connected device has a standard mapping
        # True = use gamepad API, False = use raw joystick API
        self.is_standard_gamepad = False
        
        # -----------------------------------------------------------------
        # INPUT STATE
        # -----------------------------------------------------------------
        
        # Current frame's input state
        self.state = GamepadState()
        
        # Previous frame's input state (for detecting "just pressed")
        self.previous_state = GamepadState()
        
        # -----------------------------------------------------------------
        # JOYSTICK INFO (for unmapped controllers)
        # -----------------------------------------------------------------
        
        # Number of axes/buttons on connected joystick
        # Used for fallback joystick mode
        self.num_axes = 0
        self.num_buttons = 0
        
        # -----------------------------------------------------------------
        # INITIALIZATION
        # -----------------------------------------------------------------
        
        # Load controller mappings from SDL_GameControllerDB
        # This increases the chance that controllers will be recognized
        load_gamepad_mappings(mappings_file)
        
        # Search for a connected gamepad
        self._find_gamepad()
    
    # =========================================================================
    # GAMEPAD DETECTION
    # =========================================================================
    
    def _find_gamepad(self):
        """
        Search for a connected gamepad/joystick.
        
        GLFW supports up to 16 joysticks (JOYSTICK_1 through JOYSTICK_LAST).
        We scan all slots looking for the first connected device.
        
        Priority:
        1. Standard gamepads (with mapping) - preferred
        2. Generic joysticks (fallback)
        
        Note: We only track ONE gamepad. For multiplayer, you'd need
        to manage multiple GamepadManager instances or extend this class.
        """
        # Scan all joystick slots
        for jid in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
            if glfw.joystick_present(jid):
                # Get device name
                name = glfw.get_joystick_name(jid)
                
                # GLFW returns bytes, convert to string
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Check if it's a mapped gamepad
                if glfw.joystick_is_gamepad(jid):
                    # -------------------------------------------------
                    # MAPPED GAMEPAD
                    # -------------------------------------------------
                    # This controller has a known mapping
                    # We'll get standardized button/axis layout
                    
                    gp_name = glfw.get_gamepad_name(jid)
                    if isinstance(gp_name, bytes):
                        gp_name = gp_name.decode('utf-8')
                    
                    print(f"Gamepad found: {gp_name} (ID: {jid})")
                    self.connected_gamepad = jid
                    self.is_standard_gamepad = True
                    return
                else:
                    # -------------------------------------------------
                    # UNMAPPED JOYSTICK
                    # -------------------------------------------------
                    # Unknown controller, we'll use raw input
                    # and guess a common layout
                    
                    print(f"Joystick found: {name} (ID: {jid})")
                    self.connected_gamepad = jid
                    self.is_standard_gamepad = False
                    self._detect_joystick_layout(jid)
                    return
        
        # No controller found
        if self.connected_gamepad is None:
            print("No joystick/gamepad found")
    
    def _detect_joystick_layout(self, jid: int):
        """
        Detect the layout of an unmapped joystick.
        
        For generic joysticks, we need to know how many axes and buttons
        are available so we can map them appropriately.
        
        Parameters:
        -----------
        jid : int
            GLFW joystick ID
            
        =======================================================================
        GLFW API QUIRKS
        =======================================================================
        
        GLFW's joystick functions return data in different formats depending
        on the Python binding version:
        
        - Old style: (array, count) tuple
        - New style: Direct list/array
        - Error: None
        
        We handle all cases defensively.
        
        =======================================================================
        """
        # Get axes info
        axes_result = glfw.get_joystick_axes(jid)
        buttons_result = glfw.get_joystick_buttons(jid)
        
        # Parse axes count (handle different return formats)
        if axes_result is None:
            self.num_axes = 0
        elif isinstance(axes_result, tuple):
            # Old format: (array, count)
            self.num_axes = axes_result[1] if len(axes_result) > 1 else 0
        else:
            # New format: direct array
            try:
                self.num_axes = len(axes_result)
            except:
                self.num_axes = 0
        
        # Parse buttons count
        if buttons_result is None:
            self.num_buttons = 0
        elif isinstance(buttons_result, tuple):
            self.num_buttons = buttons_result[1] if len(buttons_result) > 1 else 0
        else:
            try:
                self.num_buttons = len(buttons_result)
            except:
                self.num_buttons = 0
        
        print(f"  Axes: {self.num_axes}, Buttons: {self.num_buttons}")
    
    # =========================================================================
    # STATE UPDATE
    # =========================================================================
    
    def update(self):
        """
        Update gamepad state. Call this once per frame.
        
        This method:
        1. Saves previous state (for "just pressed" detection)
        2. Checks if controller is still connected
        3. Reads current input state
        4. Applies deadzones and normalization
        
        =======================================================================
        DISCONNECTION HANDLING
        =======================================================================
        
        Controllers can be disconnected at any time. We check each frame
        and gracefully handle disconnection by:
        - Printing a message
        - Clearing the connected_gamepad reference
        - Resetting state to defaults (all zeros/false)
        
        =======================================================================
        """
        # -----------------------------------------------------------------
        # SAVE PREVIOUS STATE
        # -----------------------------------------------------------------
        # We only copy the values we need for "just pressed" detection
        # This is cheaper than deep-copying the entire state
        self.previous_state = GamepadState(
            left_x=self.state.left_x,
            left_y=self.state.left_y,
            a=self.state.a,
            b=self.state.b,
            start=self.state.start,
            back=self.state.back,
        )
        
        # -----------------------------------------------------------------
        # CHECK CONNECTION
        # -----------------------------------------------------------------
        if self.connected_gamepad is not None:
            if not glfw.joystick_present(self.connected_gamepad):
                # Controller was disconnected
                print("Joystick disconnected")
                self.connected_gamepad = None
                self.state = GamepadState()  # Reset to defaults
                return
        else:
            # No controller connected
            return
        
        # -----------------------------------------------------------------
        # READ INPUT STATE
        # -----------------------------------------------------------------
        if self.is_standard_gamepad and glfw.joystick_is_gamepad(self.connected_gamepad):
            # Use standard gamepad API (mapped controller)
            state = glfw.get_gamepad_state(self.connected_gamepad)
            if state:
                self._parse_gamepad_state(state)
        else:
            # Use raw joystick API (unmapped controller)
            self._parse_joystick_state()
    
    def _parse_gamepad_state(self, state):
        """
        Parse input from a mapped gamepad (standard layout).
        
        Parameters:
        -----------
        state : glfw.GamepadState
            GLFW gamepad state object with axes[] and buttons[] arrays
            
        =======================================================================
        STANDARD GAMEPAD LAYOUT
        =======================================================================
        
        GLFW provides constants for each input:
        
        Axes:
        - GAMEPAD_AXIS_LEFT_X, GAMEPAD_AXIS_LEFT_Y
        - GAMEPAD_AXIS_RIGHT_X, GAMEPAD_AXIS_RIGHT_Y
        - GAMEPAD_AXIS_LEFT_TRIGGER, GAMEPAD_AXIS_RIGHT_TRIGGER
        
        Buttons:
        - GAMEPAD_BUTTON_A, B, X, Y
        - GAMEPAD_BUTTON_LEFT_BUMPER, RIGHT_BUMPER
        - GAMEPAD_BUTTON_BACK, START, GUIDE
        - GAMEPAD_BUTTON_LEFT_THUMB, RIGHT_THUMB (stick clicks)
        - GAMEPAD_BUTTON_DPAD_UP, DOWN, LEFT, RIGHT
        
        =======================================================================
        TRIGGER AXIS RANGE
        =======================================================================
        
        Triggers report -1.0 (released) to 1.0 (pressed).
        We convert to 0.0 to 1.0 for easier use:
        
            normalized = (raw + 1) / 2
            
            raw = -1.0 → normalized = 0.0
            raw =  0.0 → normalized = 0.5
            raw =  1.0 → normalized = 1.0
        
        =======================================================================
        """
        axes = state.axes
        buttons = state.buttons
        
        # -----------------------------------------------------------------
        # ANALOG STICKS (with deadzone)
        # -----------------------------------------------------------------
        self.state.left_x = self._apply_deadzone(axes[glfw.GAMEPAD_AXIS_LEFT_X])
        self.state.left_y = self._apply_deadzone(axes[glfw.GAMEPAD_AXIS_LEFT_Y])
        self.state.right_x = self._apply_deadzone(axes[glfw.GAMEPAD_AXIS_RIGHT_X])
        self.state.right_y = self._apply_deadzone(axes[glfw.GAMEPAD_AXIS_RIGHT_Y])
        
        # -----------------------------------------------------------------
        # TRIGGERS (normalized to 0-1)
        # -----------------------------------------------------------------
        # Raw range is -1 to 1, convert to 0 to 1
        self.state.left_trigger = (axes[glfw.GAMEPAD_AXIS_LEFT_TRIGGER] + 1) / 2
        self.state.right_trigger = (axes[glfw.GAMEPAD_AXIS_RIGHT_TRIGGER] + 1) / 2
        
        # -----------------------------------------------------------------
        # FACE BUTTONS
        # -----------------------------------------------------------------
        self.state.a = buttons[glfw.GAMEPAD_BUTTON_A] == glfw.PRESS
        self.state.b = buttons[glfw.GAMEPAD_BUTTON_B] == glfw.PRESS
        self.state.x = buttons[glfw.GAMEPAD_BUTTON_X] == glfw.PRESS
        self.state.y = buttons[glfw.GAMEPAD_BUTTON_Y] == glfw.PRESS
        
        # -----------------------------------------------------------------
        # SHOULDER BUTTONS
        # -----------------------------------------------------------------
        self.state.left_bumper = buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER] == glfw.PRESS
        self.state.right_bumper = buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER] == glfw.PRESS
        
        # -----------------------------------------------------------------
        # MENU BUTTONS
        # -----------------------------------------------------------------
        self.state.back = buttons[glfw.GAMEPAD_BUTTON_BACK] == glfw.PRESS
        self.state.start = buttons[glfw.GAMEPAD_BUTTON_START] == glfw.PRESS
        self.state.guide = buttons[glfw.GAMEPAD_BUTTON_GUIDE] == glfw.PRESS
        
        # -----------------------------------------------------------------
        # STICK BUTTONS
        # -----------------------------------------------------------------
        self.state.left_stick = buttons[glfw.GAMEPAD_BUTTON_LEFT_THUMB] == glfw.PRESS
        self.state.right_stick = buttons[glfw.GAMEPAD_BUTTON_RIGHT_THUMB] == glfw.PRESS
        
        # -----------------------------------------------------------------
        # D-PAD
        # -----------------------------------------------------------------
        self.state.dpad_up = buttons[glfw.GAMEPAD_BUTTON_DPAD_UP] == glfw.PRESS
        self.state.dpad_right = buttons[glfw.GAMEPAD_BUTTON_DPAD_RIGHT] == glfw.PRESS
        self.state.dpad_down = buttons[glfw.GAMEPAD_BUTTON_DPAD_DOWN] == glfw.PRESS
        self.state.dpad_left = buttons[glfw.GAMEPAD_BUTTON_DPAD_LEFT] == glfw.PRESS
    
    def _parse_joystick_state(self):
        """
        Parse input from a generic/unmapped joystick.
        
        For unknown controllers, we read raw axes and buttons by index
        and guess a common layout. Most modern controllers follow a
        similar pattern to Xbox controllers.
        
        =======================================================================
        ASSUMED LAYOUT (Xbox-like)
        =======================================================================
        
        Axes:
        - 0, 1: Left stick X, Y
        - 2, 3: Right stick X, Y
        - 4, 5: Left trigger, Right trigger
        
        Buttons (common mapping):
        - 0: A    - 4: LB    - 8: L3
        - 1: B    - 5: RB    - 9: R3
        - 2: X    - 6: Back  - 10-13: D-pad
        - 3: Y    - 7: Start
        
        D-Pad might be:
        - Buttons 10-13 (as discrete buttons)
        - A "hat" switch (bitfield)
        
        =======================================================================
        DEFENSIVE CODING
        =======================================================================
        
        We check array lengths before accessing indices because:
        - Different controllers have different numbers of inputs
        - We don't want to crash on an unusual controller
        
        =======================================================================
        """
        # -----------------------------------------------------------------
        # READ RAW AXES
        # -----------------------------------------------------------------
        result = glfw.get_joystick_axes(self.connected_gamepad)
        buttons_result = glfw.get_joystick_buttons(self.connected_gamepad)
        
        # Parse axes (handle different GLFW return formats)
        if result is None:
            axes = []
        elif isinstance(result, tuple):
            # Old format: (array, count)
            axes = [result[0][i] for i in range(result[1])] if result[1] > 0 else []
        else:
            # New format: direct array
            try:
                axes = [result[i] for i in range(len(result))]
            except:
                axes = []
        
        # Parse buttons
        if buttons_result is None:
            buttons = []
        elif isinstance(buttons_result, tuple):
            buttons = [buttons_result[0][i] for i in range(buttons_result[1])] if buttons_result[1] > 0 else []
        else:
            try:
                buttons = [buttons_result[i] for i in range(len(buttons_result))]
            except:
                buttons = []
        
        # -----------------------------------------------------------------
        # MAP AXES TO STATE
        # -----------------------------------------------------------------
        if axes:
            # Left stick (axes 0, 1)
            if len(axes) >= 2:
                self.state.left_x = self._apply_deadzone(float(axes[0]))
                self.state.left_y = self._apply_deadzone(float(axes[1]))
            
            # Right stick (axes 2, 3)
            if len(axes) >= 4:
                self.state.right_x = self._apply_deadzone(float(axes[2]))
                self.state.right_y = self._apply_deadzone(float(axes[3]))
            
            # Triggers (axes 4, 5) - normalize from -1..1 to 0..1
            if len(axes) >= 6:
                self.state.left_trigger = (float(axes[4]) + 1) / 2
                self.state.right_trigger = (float(axes[5]) + 1) / 2
        
        # -----------------------------------------------------------------
        # MAP BUTTONS TO STATE
        # -----------------------------------------------------------------
        if buttons:
            # Face buttons (0-3)
            self.state.a = int(buttons[0]) == 1 if len(buttons) > 0 else False
            self.state.b = int(buttons[1]) == 1 if len(buttons) > 1 else False
            self.state.x = int(buttons[2]) == 1 if len(buttons) > 2 else False
            self.state.y = int(buttons[3]) == 1 if len(buttons) > 3 else False
            
            # Shoulder buttons (4-5)
            self.state.left_bumper = int(buttons[4]) == 1 if len(buttons) > 4 else False
            self.state.right_bumper = int(buttons[5]) == 1 if len(buttons) > 5 else False
            
            # Menu buttons (6-7)
            self.state.back = int(buttons[6]) == 1 if len(buttons) > 6 else False
            self.state.start = int(buttons[7]) == 1 if len(buttons) > 7 else False
            
            # Stick buttons (8-9)
            self.state.left_stick = int(buttons[8]) == 1 if len(buttons) > 8 else False
            self.state.right_stick = int(buttons[9]) == 1 if len(buttons) > 9 else False
            
            # D-pad as buttons (10-13)
            if len(buttons) > 13:
                self.state.dpad_up = int(buttons[10]) == 1
                self.state.dpad_right = int(buttons[11]) == 1
                self.state.dpad_down = int(buttons[12]) == 1
                self.state.dpad_left = int(buttons[13]) == 1
        
        # -----------------------------------------------------------------
        # D-PAD AS HAT SWITCH
        # -----------------------------------------------------------------
        # Some controllers report D-pad as a "hat" (bitfield) instead of
        # individual buttons. The hat is a single value with direction bits.
        #
        # HAT_UP    = 0x01 (bit 0)
        # HAT_RIGHT = 0x02 (bit 1)
        # HAT_DOWN  = 0x04 (bit 2)
        # HAT_LEFT  = 0x08 (bit 3)
        #
        # Diagonal = multiple bits set (e.g., UP+RIGHT = 0x03)
        
        hats_result = glfw.get_joystick_hats(self.connected_gamepad)
        if hats_result is not None:
            try:
                # Parse hats array (same format handling as axes)
                if isinstance(hats_result, tuple):
                    hats = [hats_result[0][i] for i in range(hats_result[1])] if hats_result[1] > 0 else []
                else:
                    hats = [hats_result[i] for i in range(len(hats_result))]
                
                if len(hats) > 0:
                    # First hat (usually the D-pad)
                    hat = int(hats[0])
                    
                    # Extract direction bits
                    self.state.dpad_up = bool(hat & glfw.HAT_UP)
                    self.state.dpad_right = bool(hat & glfw.HAT_RIGHT)
                    self.state.dpad_down = bool(hat & glfw.HAT_DOWN)
                    self.state.dpad_left = bool(hat & glfw.HAT_LEFT)
            except:
                pass  # Ignore hat parsing errors
    
    # =========================================================================
    # DEADZONE HANDLING
    # =========================================================================
    
    def _apply_deadzone(self, value: float) -> float:
        """
        Apply deadzone to an analog axis value.
        
        Parameters:
        -----------
        value : float
            Raw axis value (-1.0 to 1.0)
            
        Returns:
        --------
        float : Processed value with deadzone applied
        
        =======================================================================
        DEADZONE ALGORITHM
        =======================================================================
        
        1. If |value| < DEADZONE, return 0
           (Small values near center are ignored)
        
        2. Otherwise, rescale the remaining range to 0-1:
           
           output = sign(value) × (|value| - DEADZONE) / (1.0 - DEADZONE)
        
        Example with DEADZONE = 0.15:
        
        Input    Output
        -----    ------
         0.00     0.00   (in deadzone)
         0.10     0.00   (in deadzone)
         0.15     0.00   (edge of deadzone)
         0.20     0.06   (rescaled)
         0.50     0.41   (rescaled)
         1.00     1.00   (max)
        -0.50    -0.41   (negative works too)
        
        =======================================================================
        WHY RESCALE?
        =======================================================================
        
        Without rescaling, the usable range would be 0.15 to 1.0, meaning
        the player could never reach exactly 0.0 or would have a "jump"
        from 0.0 to 0.15 as soon as they leave the deadzone.
        
        Rescaling provides a smooth 0.0 to 1.0 range for the entire
        usable stick movement.
        
        =======================================================================
        """
        # Check if in deadzone
        if abs(value) < self.DEADZONE:
            return 0.0
        
        # Preserve sign
        sign = 1 if value > 0 else -1
        
        # Rescale: map [DEADZONE, 1.0] to [0.0, 1.0]
        return sign * (abs(value) - self.DEADZONE) / (1.0 - self.DEADZONE)
    
    # =========================================================================
    # HIGH-LEVEL INPUT METHODS
    # =========================================================================
    
    def get_movement(self) -> Tuple[float, float]:
        """
        Get movement input from left stick or D-pad.
        
        Returns:
        --------
        Tuple[float, float] : (dx, dy) movement vector
            - dx: -1.0 = left, 0.0 = neutral, 1.0 = right
            - dy: -1.0 = up, 0.0 = neutral, 1.0 = down
        
        =======================================================================
        D-PAD OVERRIDE
        =======================================================================
        
        If D-pad is pressed, it overrides the analog stick value.
        This allows players to use either input method, with D-pad
        taking priority for precise digital control.
        
        The analog stick provides gradual values (0.0 to 1.0).
        The D-pad provides only -1.0, 0.0, or 1.0.
        
        =======================================================================
        """
        # Start with analog stick values
        dx = self.state.left_x
        dy = self.state.left_y
        
        # D-pad overrides (digital input)
        if self.state.dpad_left:
            dx = -1.0
        elif self.state.dpad_right:
            dx = 1.0
        
        if self.state.dpad_up:
            dy = -1.0
        elif self.state.dpad_down:
            dy = 1.0
        
        return dx, dy
    
    def get_height_change(self) -> float:
        """
        Get height change input from bumpers.
        
        Used for changing Z level in the game:
        - LB (left bumper): Descend
        - RB (right bumper): Ascend
        
        Returns:
        --------
        float : -1.0 (descend), 0.0 (no change), or 1.0 (ascend)
        """
        dz = 0.0
        if self.state.left_bumper:
            dz = -1.0
        elif self.state.right_bumper:
            dz = 1.0
        return dz
    
    def is_connected(self) -> bool:
        """
        Check if a gamepad is currently connected.
        
        Returns:
        --------
        bool : True if a gamepad is connected
        """
        return self.connected_gamepad is not None
    
    def button_just_pressed(self, button: str) -> bool:
        """
        Detect if a button was JUST pressed this frame.
        
        Returns True only on the frame the button transitions from
        released to pressed. Useful for one-shot actions like jumping
        or menu selection.
        
        Parameters:
        -----------
        button : str
            Button name as a string matching GamepadState attributes:
            'a', 'b', 'x', 'y', 'start', 'back', etc.
            
        Returns:
        --------
        bool : True if button was just pressed this frame
        
        =======================================================================
        EDGE DETECTION
        =======================================================================
        
        "Just pressed" = current frame pressed AND previous frame released
        
        Frame 1: released → current=False, previous=False → False
        Frame 2: pressed  → current=True,  previous=False → TRUE
        Frame 3: pressed  → current=True,  previous=True  → False
        Frame 4: pressed  → current=True,  previous=True  → False
        Frame 5: released → current=False, previous=True  → False
        
        Only Frame 2 returns True - the moment of pressing.
        
        =======================================================================
        """
        # Get current and previous state using attribute name
        current = getattr(self.state, button, False)
        previous = getattr(self.previous_state, button, False)
        
        # True only if pressed now but wasn't pressed before
        return current and not previous
