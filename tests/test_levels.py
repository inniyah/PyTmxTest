"""Mapa de dos niveles: suelos con aire debajo (puente, edificio de dos plantas, balcón)."""
import numpy as np
import pytest

from engine.compiler import compile_tmx
from engine.physics import WalkerPhysics, Walkers
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.build import build_tileset
from tools.pieces.levels import build_levels

CW = 2.0


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    out = tmp_path_factory.mktemp("levels")
    tsx = build_tileset(out / "assets")
    build_levels(out / "levels.tmx", tsx)
    report = compile_tmx(out / "levels.tmx", out / "levels.cworld")
    assert report.ok and not report.warnings, report.summary()
    return out / "levels.cworld"


@pytest.fixture(scope="module")
def phys(world):
    return WalkerPhysics(WorldView(SQLiteWorldStore(world)))


def walk(phys, start, moves, **kw):
    start = list(start)
    start[1] = float(phys.support([start[0]], [start[2]], [start[1] + 0.3])[0])      # apoyado
    w = Walkers.create([start], **kw)
    w.grounded[:] = True
    track = []
    for wish, seconds in moves:
        for _ in range(int(seconds * 60)):
            w.wish[:] = wish
            phys.step(w, 1 / 60)
            track.append(w.pos[0].copy())
    return w, np.array(track)


def test_columns_keep_air_between_floors(world):
    view = WorldView(SQLiteWorldStore(world))
    assert view.column(6, 8) == (0, 2)                    # bajo el puente: suelo, y tablero arriba
    assert view.cells(6, 8, 0) != 0 and view.cells(6, 8, 1) != 0


def test_walk_under_the_bridge(phys):
    w, track = walk(phys, [13.0, 0, 26.0], [((0, -1), 4)])
    assert w.pos[0, 2] < 14 and (track[:, 1] == 0).all()   # pasa por debajo sin subir


def test_cross_the_bridge(phys):
    w, track = walk(phys, [3.0, 0, 17.0], [((1, 0), 6)])
    assert track[:, 1].max() == pytest.approx(3.0) and w.pos[0, 0] > 24 and w.grounded[0]


def test_climb_the_inner_stairs_and_walk_upstairs(phys):
    w, track = walk(phys, [41.0, 0, 19.0], [((0, -1), 1.6), ((-1, 0), 2.0)])
    assert w.grounded[0] and w.pos[0, 1] == pytest.approx(3.0) and w.pos[0, 0] < 36


def test_railing_around_the_stairwell(phys):
    w, _ = walk(phys, [41.0, 3.0, 19.0], [((0, -1), 1.0)])
    assert w.pos[0, 1] == pytest.approx(3.0) and w.pos[0, 2] == pytest.approx(18.25, abs=0.01)


def test_fall_through_the_hole(phys):
    w, track = walk(phys, [33.0, 3.0, 8.0], [((0, 1), 1.0), ((0, 0), 1.0)])
    assert w.grounded[0] and w.pos[0, 1] == 0.0 and track[:, 1].max() == pytest.approx(3.0)


def test_fall_off_the_balcony(phys):
    w, track = walk(phys, [33.0, 3.0, 17.0], [((0, 1), 2.0)])
    assert w.pos[0, 1] == 0.0 and w.pos[0, 2] > 22 and (track[:, 1] == 3.0).any()


@pytest.mark.parametrize("start,wish", [([30.0, 0, 9.0], (0, 0)), ([30.0, 0, 9.0], (1, 0))])
def test_head_stops_at_the_upper_floor(phys, start, wish):
    w = Walkers.create([start], jump_speed=10.0)          # salto de sobra para llegar al forjado
    w.grounded[:] = True
    w.jump[:] = True
    heads = []
    for _ in range(120):
        w.wish[:] = wish
        phys.step(w, 1 / 60)
        heads.append(w.pos[0, 1] + w.height[0])
    assert max(heads) == pytest.approx(3.0, abs=1e-6) and w.grounded[0]


def test_stress_on_two_levels(phys):
    rng = np.random.default_rng(7)
    n = 150
    pos = np.c_[rng.uniform(1, 51, n), np.zeros(n), rng.uniform(1, 35, n)]
    pos[:, 1] = phys.support(pos[:, 0], pos[:, 2], np.where(rng.random(n) < 0.5, 2.9, 5.9), search=6)
    w = Walkers.create(pos)
    w.grounded[:] = True
    wish = rng.uniform(-1, 1, (n, 2))
    worst = 0.0
    levels = set()
    for step in range(600):
        if step % 30 == 0:
            wish = rng.uniform(-1, 1, (n, 2))
        w.wish[:] = wish
        w.jump[:] = rng.random(n) < 0.03
        phys.step(w, 1 / 60)
        x, y, z = w.pos.T
        worst = max(worst, float((phys.support(x, z, y + w.height) - y).max()))
        levels |= set(np.floor((y + 0.05) / 3).astype(int).tolist())
    assert worst < 0.06 and levels == {0, 1}


# --- Ocultar lo que hay encima (con GPU) -------------------------------------------------

@pytest.fixture(scope="module")
def viewer_factory():
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("sin pantalla para OpenGL")
    glfw.window_hint(glfw.VISIBLE, 0)
    probe = glfw.create_window(16, 16, "sonda", None, None)
    if not probe:
        glfw.terminate()
        pytest.skip("no se puede crear una ventana OpenGL")
    glfw.destroy_window(probe)
    from engine.viewer import Viewer
    yield Viewer


def _skin_pixels(path):
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB")).astype(int)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return int(((r > 200) & (g > 130) & (g < 215) & (b > 90) & (b < 190) & (r - b > 40)).sum())


def test_covered_detection(world, viewer_factory):
    v = viewer_factory(world, 320, 200, visible=False, spawn=(17.5, 6.5), spawn_level=0)
    try:
        assert v.covered()                                  # planta baja del edificio
        v.player.pos[0] = [17.5 * CW, 3.0, 6.5 * CW]       # planta 1: nada encima
        assert not v.covered()
        v.player.pos[0] = [3.0 * CW, 0.0, 14.5 * CW]       # calle abierta
        assert not v.covered()
        v.player.pos[0] = [6.5 * CW, 0.0, 8.5 * CW]        # bajo el puente
        assert v.covered()
    finally:
        v.close()


@pytest.mark.parametrize("mode,visible", [("no", False), ("círculo", True), ("planta", True)])
def test_player_inside_the_building(world, viewer_factory, tmp_path, mode, visible):
    from engine.viewer import CUT_MODES
    v = viewer_factory(world, 480, 320, visible=False, spawn=(17.5, 6.5), spawn_level=0)
    try:
        v.renderer.lighting.enabled = False
        v.cut_mode = CUT_MODES.index(mode)
        v._center_on_player(1.0)
        v.screenshot(tmp_path / "inside.png")
        skin = _skin_pixels(tmp_path / "inside.png")
        assert (skin > 200) if visible else (skin < 20), f"{skin} píxeles de piel con modo {mode}"
    finally:
        v.close()
