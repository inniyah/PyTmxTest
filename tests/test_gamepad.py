import dataclasses

from tmx_explorer.gamepad import GamepadManager, GamepadState


def make_manager():
    gm = object.__new__(GamepadManager)       # sin tocar GLFW
    gm.state, gm.previous_state = GamepadState(), GamepadState()
    return gm


def test_just_pressed_only_on_first_frame_for_every_button():
    gm = make_manager()
    for field in (f.name for f in dataclasses.fields(GamepadState)):
        if not isinstance(getattr(gm.state, field), bool):
            continue
        gm.state, gm.previous_state = GamepadState(), GamepadState()
        setattr(gm.state, field, True)
        assert gm.button_just_pressed(field), field
        gm.previous_state = dataclasses.replace(gm.state)   # lo que hace update()
        assert not gm.button_just_pressed(field), field


def test_deadzone():
    gm = make_manager()
    assert gm._apply_deadzone(0.1) == 0.0
    assert gm._apply_deadzone(1.0) == 1.0
    assert 0 < gm._apply_deadzone(0.5) < 0.5
