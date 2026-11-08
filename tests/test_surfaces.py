"""Capa de superficie: metal, rugosidad, brillo propio con halo y pulso."""
import numpy as np
import pytest
from PIL import Image

from engine.compiler import compile_tmx
from engine.render.world_renderer import mip_chains
from tools.pieces.levels import build_levels


def test_mip_chain_keeps_the_pulse_as_data():
    """El canal A de la superficie es el pulso, no la opacidad: al reducir se promedia."""
    color = np.zeros((4, 4, 4), np.uint8)
    color[..., 3] = 255
    surface = np.zeros((4, 4, 4), np.uint8)
    surface[...] = (255, 40, 200, 100)
    chains = mip_chains({"color": color, "surface": surface}, 2)
    assert chains["surface"][1].shape == (2, 2, 4)
    assert (chains["surface"][1] == (255, 40, 200, 100)).all() and (chains["surface"][2] == (255, 40, 200, 100)).all()


@pytest.fixture(scope="module")
def levels(tmp_path_factory, shared_tileset):
    d = tmp_path_factory.mktemp("surf")
    build_levels(d / "l.tmx", shared_tileset)
    assert compile_tmx(d / "l.tmx", d / "l.cworld", nav=False).ok
    return d / "l.cworld"


@pytest.fixture
def viewer(levels):
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("sin pantalla para OpenGL")
    from engine.viewer import Viewer
    from tests.conftest import retrying_viewer
    made = []

    def make(spawn, night=False):
        v = retrying_viewer(Viewer)(levels, 520, 360, visible=False, spawn=spawn, spawn_level=0)
        v.set_night(night)
        v.silhouette = False
        made.append(v)
        return v
    yield make
    for v in made:
        v.close()


def _shot(v, tmp_path, name="s"):
    v.screenshot(tmp_path / f"{name}.png")
    return np.asarray(Image.open(tmp_path / f"{name}.png").convert("RGB")).astype(float)


def _cyan(img):
    return (img[..., 2] > 170) & (img[..., 1] > 170) & (img[..., 0] < 140)


def test_glow_lines_shine_at_night_and_the_halo_spreads(viewer, tmp_path):
    v = viewer((5.5, 16.5), night=True)
    v.lantern = False                                # sin farol: solo el brillo propio
    v.halo = False
    plain = _shot(v, tmp_path, "plain")
    assert _cyan(plain).sum() > 500                  # las líneas brillan a oscuras
    v.halo = True
    halo = _shot(v, tmp_path, "halo")
    around = ~_cyan(plain)
    assert (halo[around] - plain[around]).sum(-1).mean() > 2.0      # el halo aclara alrededor
    v.renderer.use_surface = False
    assert _cyan(_shot(v, tmp_path, "off")).sum() < _cyan(plain).sum() * 0.2


def test_pulse_changes_with_time(viewer, tmp_path):
    v = viewer((5.5, 16.5), night=True)
    v.halo = False
    bright = []
    for t in (0.0, 0.5, 1.0, 1.5):
        v.clock = t
        v.renderer.time = t
        img = _shot(v, tmp_path, f"p{t}")
        bright.append(img[_cyan(img)].mean() if _cyan(img).any() else 0.0)
    assert max(bright) - min(bright) > 8.0


def test_polished_floor_reflects_the_lamps(viewer, tmp_path):
    v = viewer((17.5, 6.5), night=True)
    v.halo = False
    with_surface = _shot(v, tmp_path, "on")
    v.renderer.use_surface = False
    without = _shot(v, tmp_path, "off")
    assert np.percentile(with_surface.sum(-1), 99.5) > np.percentile(without.sum(-1), 99.5) + 30


def test_sprite_sheet_surface_makes_it_glow(viewer, tmp_path):
    """Una hoja con capa de superficie (hoja.s.png) junto a ella: de noche, sin farol, el
    personaje brilla con su propio color."""
    from engine.render.sprites import SpriteSheet
    from engine.viewer import ASSETS
    v = viewer((5.5, 12.5), night=True)
    v.lantern = False
    v.halo = False
    t = v.table
    path = v.types[t.type_names[t.type[t.row(v.player_handle)]]].sheet
    plain = _shot(v, tmp_path, "plain")
    sheet = Image.open(ASSETS / path).convert("RGBA")
    sheet.save(tmp_path / "hoja.png")
    surface = np.zeros((sheet.height, sheet.width, 4), np.uint8)
    surface[np.asarray(sheet)[..., 3] > 0] = (0, 255, 255, 0)          # todo lo opaco brilla
    Image.fromarray(surface).save(tmp_path / "hoja.s.png")
    loaded = SpriteSheet.load(tmp_path / "hoja.png")
    assert loaded.surface != 0
    v.sheets[path] = loaded
    glowing = _shot(v, tmp_path, "glow")
    h, w = plain.shape[:2]
    box = (slice(h // 2 - 60, h // 2 + 20), slice(w // 2 - 25, w // 2 + 25))   # el personaje
    assert glowing[box].sum(-1).max() > plain[box].sum(-1).max() + 60
