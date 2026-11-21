"""Mando: aquí no hay ninguno físico, así que se simula sustituyendo las funciones de GLFW que lee."""
from types import SimpleNamespace

import glfw
import numpy as np
import pytest

import engine.input.gamepad as gp
from engine.input import GamepadManager, load_gamepad_mappings


def test_the_controller_database_loads():
    """GLFW (3.4) solo la acepta inicializado: en el visor, el mando se crea después de la ventana."""
    if not glfw.init():
        pytest.skip("sin pantalla para GLFW")
    assert load_gamepad_mappings() > 1000                     # assets/input/gamecontrollerdb.txt


class FakeGLFW:
    """Un mando en el id 0: estándar (gamepad) o joystick genérico."""

    def __init__(self, monkeypatch, standard=True):
        self.present, self.standard = True, standard
        self.axes = [0.0] * 6
        self.buttons = [glfw.RELEASE] * 15
        self.hat = 0
        m = gp.glfw
        monkeypatch.setattr(m, "update_gamepad_mappings", lambda _c: True)
        monkeypatch.setattr(m, "joystick_present", lambda jid: self.present and jid == 0)
        monkeypatch.setattr(m, "joystick_is_gamepad", lambda jid: self.standard)
        monkeypatch.setattr(m, "get_gamepad_name", lambda jid: b"Mando de prueba")
        monkeypatch.setattr(m, "get_joystick_name", lambda jid: b"Joystick de prueba")
        monkeypatch.setattr(m, "get_gamepad_state", lambda jid: SimpleNamespace(axes=self.axes, buttons=self.buttons))
        monkeypatch.setattr(m, "get_joystick_axes", lambda jid: list(self.axes))
        monkeypatch.setattr(m, "get_joystick_buttons", lambda jid: list(self.buttons))
        monkeypatch.setattr(m, "get_joystick_hats", lambda jid: [self.hat])

    def press(self, index, down=True):
        self.buttons[index] = glfw.PRESS if down else glfw.RELEASE


def test_standard_gamepad_sticks_buttons_and_dpad(monkeypatch):
    fake = FakeGLFW(monkeypatch)
    pad = GamepadManager()
    assert pad.is_connected() and pad.is_standard_gamepad
    fake.axes[glfw.GAMEPAD_AXIS_LEFT_X], fake.axes[glfw.GAMEPAD_AXIS_LEFT_Y] = 0.1, -1.0   # x en la zona muerta
    fake.press(glfw.GAMEPAD_BUTTON_A)
    pad.update()
    assert pad.get_movement() == (0.0, -1.0)
    assert pad.button_just_pressed("a")
    pad.update()
    assert pad.state.a and not pad.button_just_pressed("a")                  # solo el primer frame
    fake.axes[glfw.GAMEPAD_AXIS_LEFT_X] = 0.6                                  # fuera de la zona muerta:
    pad.update()                                                              # empieza suave desde 0
    assert pad.get_movement()[0] == pytest.approx((0.6 - gp.GAMEPAD_DEADZONE) / (1 - gp.GAMEPAD_DEADZONE))
    fake.press(glfw.GAMEPAD_BUTTON_DPAD_RIGHT)
    pad.update()
    assert pad.get_movement()[0] == 1.0                                        # la cruceta manda
    fake.press(glfw.GAMEPAD_BUTTON_RIGHT_BUMPER)
    fake.axes[glfw.GAMEPAD_AXIS_RIGHT_X] = -1.0
    pad.update()
    assert pad.get_zoom() == 1.0 and pad.get_look() == (-1.0, 0.0)


def test_hot_plug_and_unplug(monkeypatch):
    fake = FakeGLFW(monkeypatch)
    fake.present = False
    clock = [100.0]
    monkeypatch.setattr(gp.time, "monotonic", lambda: clock[0])
    pad = GamepadManager()
    assert not pad.is_connected()
    fake.present = True
    pad.update()
    assert pad.is_connected()                                                 # lo encuentra al buscar
    fake.press(glfw.GAMEPAD_BUTTON_X)
    pad.update()
    assert pad.state.x
    fake.present = False
    pad.update()
    assert not pad.is_connected() and not pad.state.x                         # desconectado: en reposo
    fake.present = True
    pad.update()
    assert not pad.is_connected()                                             # aún no toca buscar
    clock[0] += gp.GAMEPAD_RESCAN_SECONDS + 0.1
    pad.update()
    assert pad.is_connected()


def test_generic_joystick(monkeypatch):
    fake = FakeGLFW(monkeypatch, standard=False)
    pad = GamepadManager()
    assert pad.is_connected() and not pad.is_standard_gamepad
    fake.axes[0], fake.axes[3] = -1.0, 1.0
    fake.press(0)                                                              # el primer botón: A
    fake.hat = glfw.HAT_UP
    pad.update()
    assert pad.state.left_x == -1.0 and pad.state.right_y == 1.0 and pad.state.a
    assert pad.get_movement() == (-1.0, -1.0)                                  # cruceta arriba (norte)


class FakePad:
    """Un mando para el visor, sin GLFW."""

    def __init__(self):
        self.move, self.look, self.zoom, self.pressed = (0.0, 0.0), (0.0, 0.0), 0.0, set()

    def update(self):
        pass

    def is_connected(self):
        return True

    def get_movement(self):
        return self.move

    def get_look(self):
        return self.look

    def get_zoom(self):
        return self.zoom

    def button_just_pressed(self, name):
        return name in self.pressed


@pytest.fixture
def viewer(tmp_path_factory, shared_tileset):
    glfw_mod = pytest.importorskip("glfw")
    if not glfw_mod.init():
        pytest.skip("sin pantalla para OpenGL")
    from engine.compiler import compile_tmx
    from engine.viewer import Viewer
    from tests.conftest import retrying_viewer
    from tools.pieces.levels import build_levels
    d = tmp_path_factory.mktemp("pad")
    build_levels(d / "l.tmx", shared_tileset)
    assert compile_tmx(d / "l.tmx", d / "l.cworld", nav=False).ok
    v = retrying_viewer(Viewer)(d / "l.cworld", 320, 200, visible=False, spawn=(6.5, 13.5), spawn_level=0)
    v.gamepad = FakePad()
    yield v
    v.close()


def test_viewer_walks_runs_jumps_zooms_and_looks_with_the_pad(viewer):
    v, pad = viewer, viewer.gamepad
    t, row = v.table, v.table.row(v.player_handle)

    def speed_with(move):
        pad.move = move
        for _ in range(60):
            v.update(1 / 60)
        return float(np.hypot(t.vel[row, 0], t.vel[row, 2]))
    walk, run = speed_with((0.4, 0.0)), speed_with((1.0, 0.0))
    assert 0.5 < walk < 0.6 * run and t.vel[row, 0] > 0                       # analógico: anda / corre, al este
    pad.move = (0.0, 0.0)
    for _ in range(30):
        v.update(1 / 60)
    pad.pressed = {"a"}
    v.update(1 / 60)
    pad.pressed = set()
    assert t.vel[row, 1] > 1.0                                                 # A: salta
    zoom = v.camera.zoom
    pad.zoom = 1.0
    for _ in range(30):
        v.update(1 / 60)
    assert v.camera.zoom > zoom * 1.2                                          # RB: acerca
    pad.zoom, pad.look = 0.0, (1.0, 0.0)
    x = v.camera.x
    v.update(1 / 60)
    assert not v.follow and v.camera.x > x                                     # stick derecho: la vista
    pad.look, pad.pressed = (0.0, 0.0), {"right_stick"}
    v.update(1 / 60)
    assert v.follow                                                            # clic: vuelve a seguirle
