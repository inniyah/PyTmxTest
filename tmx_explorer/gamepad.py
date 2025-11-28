"""
Gamepad/Joystick support using GLFW
"""

import glfw
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


def load_gamepad_mappings(filepath: str = None) -> int:
    """
    Carga mappings de gamepad desde un archivo SDL_GameControllerDB.
    
    Args:
        filepath: Ruta al archivo gamecontrollerdb.txt
                  Si es None, busca en ubicaciones comunes
    
    Returns:
        Número de mappings cargados, o -1 si falla
    """
    search_paths = []
    
    if filepath:
        search_paths.append(Path(filepath))
    
    # Buscar en ubicaciones comunes
    search_paths.extend([
        Path("gamecontrollerdb.txt"),
        Path("assets/gamecontrollerdb.txt"),
        Path(__file__).parent / "gamecontrollerdb.txt",
        Path(__file__).parent.parent / "gamecontrollerdb.txt",
        Path.home() / ".config/gamecontrollerdb.txt",
    ])
    
    for path in search_paths:
        if path.exists():
            try:
                content = path.read_text(encoding='utf-8')
                result = glfw.update_gamepad_mappings(content)
                if result:
                    count = sum(1 for line in content.splitlines() 
                               if line.strip() and not line.startswith('#'))
                    print(f"Gamepad mappings cargados: {count} desde {path}")
                    return count
            except Exception as e:
                print(f"Error cargando mappings desde {path}: {e}")
    
    print("No se encontró gamecontrollerdb.txt")
    return -1


@dataclass
class GamepadState:
    """Estado actual del gamepad"""
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    
    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    
    left_bumper: bool = False
    right_bumper: bool = False
    
    back: bool = False
    start: bool = False
    guide: bool = False
    
    left_stick: bool = False
    right_stick: bool = False
    
    dpad_up: bool = False
    dpad_right: bool = False
    dpad_down: bool = False
    dpad_left: bool = False


class GamepadManager:
    """Manages gamepad input"""
    
    DEADZONE = 0.15
    
    def __init__(self, mappings_file: str = None):
        self.connected_gamepad: Optional[int] = None
        self.is_standard_gamepad = False
        self.state = GamepadState()
        self.previous_state = GamepadState()
        
        self.num_axes = 0
        self.num_buttons = 0
        
        # Cargar mappings de SDL_GameControllerDB
        load_gamepad_mappings(mappings_file)
        
        self._find_gamepad()
    
    def _find_gamepad(self):
        """Busca un gamepad conectado"""
        for jid in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
            if glfw.joystick_present(jid):
                name = glfw.get_joystick_name(jid)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                if glfw.joystick_is_gamepad(jid):
                    gp_name = glfw.get_gamepad_name(jid)
                    if isinstance(gp_name, bytes):
                        gp_name = gp_name.decode('utf-8')
                    print(f"Gamepad encontrado: {gp_name} (ID: {jid})")
                    self.connected_gamepad = jid
                    self.is_standard_gamepad = True
                    return
                else:
                    print(f"Joystick encontrado: {name} (ID: {jid})")
                    self.connected_gamepad = jid
                    self.is_standard_gamepad = False
                    self._detect_joystick_layout(jid)
                    return
        
        if self.connected_gamepad is None:
            print("No se encontró ningún joystick/gamepad")
    
    def _detect_joystick_layout(self, jid: int):
        """Detecta la configuración del joystick"""
        axes_result = glfw.get_joystick_axes(jid)
        buttons_result = glfw.get_joystick_buttons(jid)
        
        if axes_result is None:
            self.num_axes = 0
        elif isinstance(axes_result, tuple):
            self.num_axes = axes_result[1] if len(axes_result) > 1 else 0
        else:
            try:
                self.num_axes = len(axes_result)
            except:
                self.num_axes = 0
        
        if buttons_result is None:
            self.num_buttons = 0
        elif isinstance(buttons_result, tuple):
            self.num_buttons = buttons_result[1] if len(buttons_result) > 1 else 0
        else:
            try:
                self.num_buttons = len(buttons_result)
            except:
                self.num_buttons = 0
        
        print(f"  Ejes: {self.num_axes}, Botones: {self.num_buttons}")
    
    def update(self):
        """Actualiza el estado del gamepad"""
        self.previous_state = GamepadState(
            left_x=self.state.left_x,
            left_y=self.state.left_y,
            a=self.state.a,
            b=self.state.b,
            start=self.state.start,
            back=self.state.back,
        )
        
        if self.connected_gamepad is not None:
            if not glfw.joystick_present(self.connected_gamepad):
                print("Joystick desconectado")
                self.connected_gamepad = None
                self.state = GamepadState()
                return
        else:
            return
        
        if self.is_standard_gamepad and glfw.joystick_is_gamepad(self.connected_gamepad):
            state = glfw.get_gamepad_state(self.connected_gamepad)
            if state:
                self._parse_gamepad_state(state)
        else:
            self._parse_joystick_state()
    
    def _parse_gamepad_state(self, state):
        """Parsea el estado del gamepad (con mapping estándar)"""
        axes = state.axes
        buttons = state.buttons
        
        self.state.left_x = self._apply_deadzone(axes[glfw.GAMEPAD_AXIS_LEFT_X])
        self.state.left_y = self._apply_deadzone(axes[glfw.GAMEPAD_AXIS_LEFT_Y])
        self.state.right_x = self._apply_deadzone(axes[glfw.GAMEPAD_AXIS_RIGHT_X])
        self.state.right_y = self._apply_deadzone(axes[glfw.GAMEPAD_AXIS_RIGHT_Y])
        
        self.state.left_trigger = (axes[glfw.GAMEPAD_AXIS_LEFT_TRIGGER] + 1) / 2
        self.state.right_trigger = (axes[glfw.GAMEPAD_AXIS_RIGHT_TRIGGER] + 1) / 2
        
        self.state.a = buttons[glfw.GAMEPAD_BUTTON_A] == glfw.PRESS
        self.state.b = buttons[glfw.GAMEPAD_BUTTON_B] == glfw.PRESS
        self.state.x = buttons[glfw.GAMEPAD_BUTTON_X] == glfw.PRESS
        self.state.y = buttons[glfw.GAMEPAD_BUTTON_Y] == glfw.PRESS
        
        self.state.left_bumper = buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER] == glfw.PRESS
        self.state.right_bumper = buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER] == glfw.PRESS
        
        self.state.back = buttons[glfw.GAMEPAD_BUTTON_BACK] == glfw.PRESS
        self.state.start = buttons[glfw.GAMEPAD_BUTTON_START] == glfw.PRESS
        self.state.guide = buttons[glfw.GAMEPAD_BUTTON_GUIDE] == glfw.PRESS
        
        self.state.left_stick = buttons[glfw.GAMEPAD_BUTTON_LEFT_THUMB] == glfw.PRESS
        self.state.right_stick = buttons[glfw.GAMEPAD_BUTTON_RIGHT_THUMB] == glfw.PRESS
        
        self.state.dpad_up = buttons[glfw.GAMEPAD_BUTTON_DPAD_UP] == glfw.PRESS
        self.state.dpad_right = buttons[glfw.GAMEPAD_BUTTON_DPAD_RIGHT] == glfw.PRESS
        self.state.dpad_down = buttons[glfw.GAMEPAD_BUTTON_DPAD_DOWN] == glfw.PRESS
        self.state.dpad_left = buttons[glfw.GAMEPAD_BUTTON_DPAD_LEFT] == glfw.PRESS
    
    def _parse_joystick_state(self):
        """Parsea joystick genérico"""
        result = glfw.get_joystick_axes(self.connected_gamepad)
        buttons_result = glfw.get_joystick_buttons(self.connected_gamepad)
        
        if result is None:
            axes = []
        elif isinstance(result, tuple):
            axes = [result[0][i] for i in range(result[1])] if result[1] > 0 else []
        else:
            try:
                axes = [result[i] for i in range(len(result))]
            except:
                axes = []
        
        if buttons_result is None:
            buttons = []
        elif isinstance(buttons_result, tuple):
            buttons = [buttons_result[0][i] for i in range(buttons_result[1])] if buttons_result[1] > 0 else []
        else:
            try:
                buttons = [buttons_result[i] for i in range(len(buttons_result))]
            except:
                buttons = []
        
        if axes:
            if len(axes) >= 2:
                self.state.left_x = self._apply_deadzone(float(axes[0]))
                self.state.left_y = self._apply_deadzone(float(axes[1]))
            
            if len(axes) >= 4:
                self.state.right_x = self._apply_deadzone(float(axes[2]))
                self.state.right_y = self._apply_deadzone(float(axes[3]))
            
            if len(axes) >= 6:
                self.state.left_trigger = (float(axes[4]) + 1) / 2
                self.state.right_trigger = (float(axes[5]) + 1) / 2
        
        if buttons:
            self.state.a = int(buttons[0]) == 1 if len(buttons) > 0 else False
            self.state.b = int(buttons[1]) == 1 if len(buttons) > 1 else False
            self.state.x = int(buttons[2]) == 1 if len(buttons) > 2 else False
            self.state.y = int(buttons[3]) == 1 if len(buttons) > 3 else False
            
            self.state.left_bumper = int(buttons[4]) == 1 if len(buttons) > 4 else False
            self.state.right_bumper = int(buttons[5]) == 1 if len(buttons) > 5 else False
            
            self.state.back = int(buttons[6]) == 1 if len(buttons) > 6 else False
            self.state.start = int(buttons[7]) == 1 if len(buttons) > 7 else False
            
            self.state.left_stick = int(buttons[8]) == 1 if len(buttons) > 8 else False
            self.state.right_stick = int(buttons[9]) == 1 if len(buttons) > 9 else False
            
            if len(buttons) > 13:
                self.state.dpad_up = int(buttons[10]) == 1
                self.state.dpad_right = int(buttons[11]) == 1
                self.state.dpad_down = int(buttons[12]) == 1
                self.state.dpad_left = int(buttons[13]) == 1
        
        hats_result = glfw.get_joystick_hats(self.connected_gamepad)
        if hats_result is not None:
            try:
                if isinstance(hats_result, tuple):
                    hats = [hats_result[0][i] for i in range(hats_result[1])] if hats_result[1] > 0 else []
                else:
                    hats = [hats_result[i] for i in range(len(hats_result))]
                
                if len(hats) > 0:
                    hat = int(hats[0])
                    self.state.dpad_up = bool(hat & glfw.HAT_UP)
                    self.state.dpad_right = bool(hat & glfw.HAT_RIGHT)
                    self.state.dpad_down = bool(hat & glfw.HAT_DOWN)
                    self.state.dpad_left = bool(hat & glfw.HAT_LEFT)
            except:
                pass
    
    def _apply_deadzone(self, value: float) -> float:
        """Aplica deadzone al valor del eje"""
        if abs(value) < self.DEADZONE:
            return 0.0
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.DEADZONE) / (1.0 - self.DEADZONE)
    
    def get_movement(self) -> Tuple[float, float]:
        """Obtiene el movimiento del stick izquierdo o D-pad"""
        dx = self.state.left_x
        dy = self.state.left_y
        
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
        """Obtiene cambio de altura (bumpers)"""
        dz = 0.0
        if self.state.left_bumper:
            dz = -1.0
        elif self.state.right_bumper:
            dz = 1.0
        return dz
    
    def is_connected(self) -> bool:
        return self.connected_gamepad is not None
    
    def button_just_pressed(self, button: str) -> bool:
        """Detecta si un botón acaba de ser presionado"""
        current = getattr(self.state, button, False)
        previous = getattr(self.previous_state, button, False)
        return current and not previous
