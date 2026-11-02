"""Mando/joystick vía GLFW.

Si GLFW reconoce el dispositivo como gamepad (gracias a gamecontrollerdb.txt) se usa
el layout estándar tipo Xbox; si no, se lee como joystick genérico con una asignación
de botones aproximada. Los mandos se detectan también si se conectan en caliente.
"""

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import glfw

from . import config

log = logging.getLogger(__name__)


def load_gamepad_mappings(filepath: Optional[str] = None) -> int:
    """Carga mapeos SDL_GameControllerDB. Devuelve cuántos, o -1 si no hay archivo."""
    candidates = [Path(filepath)] if filepath else []
    candidates += [
        config.GAMEPAD_DB,
        Path("gamecontrollerdb.txt"),
        Path.home() / ".config/gamecontrollerdb.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("No se pudo leer %s: %s", path, e)
            continue
        if glfw.update_gamepad_mappings(content):
            count = sum(1 for line in content.splitlines()
                        if line.strip() and not line.startswith("#"))
            log.info("Mapeos de mando: %d desde %s", count, path)
            return count
    log.info("gamecontrollerdb.txt no encontrado")
    return -1


@dataclass
class GamepadState:
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


# Botones del layout estándar de GLFW -> campo de GamepadState
_GAMEPAD_BUTTONS = {
    "a": glfw.GAMEPAD_BUTTON_A,
    "b": glfw.GAMEPAD_BUTTON_B,
    "x": glfw.GAMEPAD_BUTTON_X,
    "y": glfw.GAMEPAD_BUTTON_Y,
    "left_bumper": glfw.GAMEPAD_BUTTON_LEFT_BUMPER,
    "right_bumper": glfw.GAMEPAD_BUTTON_RIGHT_BUMPER,
    "back": glfw.GAMEPAD_BUTTON_BACK,
    "start": glfw.GAMEPAD_BUTTON_START,
    "guide": glfw.GAMEPAD_BUTTON_GUIDE,
    "left_stick": glfw.GAMEPAD_BUTTON_LEFT_THUMB,
    "right_stick": glfw.GAMEPAD_BUTTON_RIGHT_THUMB,
    "dpad_up": glfw.GAMEPAD_BUTTON_DPAD_UP,
    "dpad_right": glfw.GAMEPAD_BUTTON_DPAD_RIGHT,
    "dpad_down": glfw.GAMEPAD_BUTTON_DPAD_DOWN,
    "dpad_left": glfw.GAMEPAD_BUTTON_DPAD_LEFT,
}

# Joystick genérico: índice de botón -> campo (asignación típica de mandos USB)
_JOYSTICK_BUTTONS = ("a", "b", "x", "y", "left_bumper", "right_bumper",
                     "back", "start", "left_stick", "right_stick")
_JOYSTICK_DPAD_BUTTONS = ("dpad_up", "dpad_right", "dpad_down", "dpad_left")  # 10..13


def _to_list(result) -> List:
    """Normaliza lo que devuelve pyGLFW: None, (puntero, n) o secuencia."""
    if result is None:
        return []
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        data, count = result
        return [data[i] for i in range(count)]
    return list(result)


def _decode(name) -> str:
    return name.decode("utf-8") if isinstance(name, bytes) else (name or "")


class GamepadManager:
    DEADZONE = config.GAMEPAD_DEADZONE

    def __init__(self, mappings_file: Optional[str] = None):
        self.connected_gamepad: Optional[int] = None
        self.is_standard_gamepad = False
        self.state = GamepadState()
        self.previous_state = GamepadState()
        self.num_axes = 0
        self.num_buttons = 0
        self._next_scan = 0.0

        load_gamepad_mappings(mappings_file)
        if not self._find_gamepad():
            log.info("No hay mando conectado")

    # --- Detección ---

    def _find_gamepad(self) -> bool:
        for jid in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
            if not glfw.joystick_present(jid):
                continue
            self.connected_gamepad = jid
            self.is_standard_gamepad = bool(glfw.joystick_is_gamepad(jid))
            if self.is_standard_gamepad:
                log.info("Mando: %s (id %d)", _decode(glfw.get_gamepad_name(jid)), jid)
            else:
                self.num_axes = len(_to_list(glfw.get_joystick_axes(jid)))
                self.num_buttons = len(_to_list(glfw.get_joystick_buttons(jid)))
                log.info("Joystick: %s (id %d), %d ejes, %d botones",
                         _decode(glfw.get_joystick_name(jid)), jid,
                         self.num_axes, self.num_buttons)
            return True
        return False

    # --- Por frame ---

    def update(self):
        self.previous_state = dataclasses.replace(self.state)

        if self.connected_gamepad is None:
            # Sin mando: buscar uno de vez en cuando (conexión en caliente)
            now = time.monotonic()
            if now < self._next_scan:
                return
            self._next_scan = now + config.GAMEPAD_RESCAN_SECONDS
            if not self._find_gamepad():
                return

        if not glfw.joystick_present(self.connected_gamepad):
            log.info("Mando desconectado")
            self.connected_gamepad = None
            self.state = GamepadState()
            return

        if self.is_standard_gamepad and glfw.joystick_is_gamepad(self.connected_gamepad):
            gp = glfw.get_gamepad_state(self.connected_gamepad)
            if gp:
                self._read_gamepad(gp)
        else:
            self._read_joystick()

    def _read_gamepad(self, gp):
        s, axes, dz = self.state, gp.axes, self._apply_deadzone
        s.left_x = dz(axes[glfw.GAMEPAD_AXIS_LEFT_X])
        s.left_y = dz(axes[glfw.GAMEPAD_AXIS_LEFT_Y])
        s.right_x = dz(axes[glfw.GAMEPAD_AXIS_RIGHT_X])
        s.right_y = dz(axes[glfw.GAMEPAD_AXIS_RIGHT_Y])
        s.left_trigger = (axes[glfw.GAMEPAD_AXIS_LEFT_TRIGGER] + 1) / 2
        s.right_trigger = (axes[glfw.GAMEPAD_AXIS_RIGHT_TRIGGER] + 1) / 2
        for field, index in _GAMEPAD_BUTTONS.items():
            setattr(s, field, gp.buttons[index] == glfw.PRESS)

    def _read_joystick(self):
        jid, s, dz = self.connected_gamepad, self.state, self._apply_deadzone
        axes = [float(a) for a in _to_list(glfw.get_joystick_axes(jid))]
        buttons = [int(b) == glfw.PRESS for b in _to_list(glfw.get_joystick_buttons(jid))]

        if len(axes) >= 2:
            s.left_x, s.left_y = dz(axes[0]), dz(axes[1])
        if len(axes) >= 4:
            s.right_x, s.right_y = dz(axes[2]), dz(axes[3])
        if len(axes) >= 6:
            s.left_trigger, s.right_trigger = (axes[4] + 1) / 2, (axes[5] + 1) / 2

        if buttons:
            for index, field in enumerate(_JOYSTICK_BUTTONS):
                setattr(s, field, index < len(buttons) and buttons[index])
            if len(buttons) > 13:
                for index, field in enumerate(_JOYSTICK_DPAD_BUTTONS, start=10):
                    setattr(s, field, buttons[index])

        hats = _to_list(glfw.get_joystick_hats(jid))
        if hats:   # el hat, si existe, manda sobre los botones de cruceta
            hat = int(hats[0])
            s.dpad_up = bool(hat & glfw.HAT_UP)
            s.dpad_right = bool(hat & glfw.HAT_RIGHT)
            s.dpad_down = bool(hat & glfw.HAT_DOWN)
            s.dpad_left = bool(hat & glfw.HAT_LEFT)

    def _apply_deadzone(self, value: float) -> float:
        """0 dentro de la zona muerta; fuera, se reescala para empezar suave desde 0."""
        if abs(value) < self.DEADZONE:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.DEADZONE) / (1.0 - self.DEADZONE)

    # --- Consultas ---

    def get_movement(self) -> Tuple[float, float]:
        """Stick izquierdo; la cruceta tiene prioridad."""
        s = self.state
        dx = -1.0 if s.dpad_left else 1.0 if s.dpad_right else s.left_x
        dy = -1.0 if s.dpad_up else 1.0 if s.dpad_down else s.left_y
        return dx, dy

    def get_height_change(self) -> float:
        if self.state.left_bumper:
            return -1.0
        if self.state.right_bumper:
            return 1.0
        return 0.0

    def is_connected(self) -> bool:
        return self.connected_gamepad is not None

    def button_just_pressed(self, button: str) -> bool:
        return getattr(self.state, button, False) and not getattr(self.previous_state, button, False)
