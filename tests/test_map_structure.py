import numpy as np

from conftest import MAP01
from tmx_explorer.config import GID_FLIP_H, GID_FLIP_V
from tmx_explorer.map.structure import Map3DStructure
from tmx_manager import TiledMap


def load():
    return Map3DStructure(TiledMap.load(MAP01))


def test_dimensions_and_levels():
    m = load()
    assert (m.W, m.D, m.H, m.N) == (30, 300, 2, 2)
    assert m.layer_levels == [0, 1]
    # cada capa solo tiene datos en su propio nivel
    assert not m.mapa[1, :, :, 0].any() and not m.mapa[0, :, :, 1].any()


def test_is_walkable_methods_work():
    m = load()
    solid = np.argwhere(m.collision.data[0] != 0)[0]
    free = np.argwhere(m.collision.data[0] == 0)[0]
    to_px = lambda ty, tx: ((tx + 0.5) * m.tile_width, (ty + 0.5) * m.tile_height)
    assert not m.is_walkable(*to_px(*solid), 0)
    assert m.is_walkable(*to_px(*free), 0)
    assert isinstance(m.is_walkable_with_size(*to_px(*free), 0), bool)


def test_flip_flags_are_separated_from_gid():
    tmx = TiledMap.load(MAP01)
    layer = tmx.layers[0]
    gid = layer.get_tile_gid(0, 0)
    assert gid
    layer.set_tile_gid(0, 0, gid | GID_FLIP_H | GID_FLIP_V)
    m = Map3DStructure(tmx)
    assert m.mapa[0, 0, 0, 0] == gid
    assert m.flips[0, 0, 0, 0] == (GID_FLIP_H | GID_FLIP_V) >> 28


def test_out_of_bounds_is_solid():
    c = load().collision
    assert c.is_solid(-1, 0, 0) and c.is_solid(0, 0, 99)
