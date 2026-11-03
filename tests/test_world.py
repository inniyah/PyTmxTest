"""Mundo integrado: capas de Tiled -> tiles[z, y, x, hueco], roles y celdas."""
import logging

import pytest

import tmx_manager as tm
from conftest import MAP01, ROOT
from tmx_explorer.config import GID_FLIP_H, GID_FLIP_V, GID_MASK
from tmx_explorer.map import cells
from tmx_explorer.map.world import EMPTY_SOURCE, SLOTS, LayerRole, World


def make_map(w=4, h=3, tilecount=8, tile_props=None, tile_classes=None):
    m = tm.create_empty_map(w, h, 16, 16)
    ts = tm.Tileset(firstgid=1, name="t", tilewidth=16, tileheight=16, tilecount=tilecount, columns=4)
    for tid, props in (tile_props or {}).items():
        ts.tiles.setdefault(tid, tm.Tile(id=tid)).properties.update(
            {k: tm.Property(k, "bool" if isinstance(v, bool) else "string", v) for k, v in props.items()})
    for tid, klass in (tile_classes or {}).items():
        ts.tiles.setdefault(tid, tm.Tile(id=tid)).type = klass
    m.tilesets.append(ts)
    return m


def layer(name, fill, w=4, h=3, klass="", z=None, **attrs):
    l = tm.create_layer(name, w, h)
    l.klass = klass
    if z is not None:
        l.properties["Z"] = tm.Property("Z", "int", z)
    for (x, y), gid in fill.items():
        l.set_tile_gid(x, y, gid)
    for k, v in attrs.items():
        setattr(l, k, v)
    return l


def test_example_map_loads_like_before():
    w = World(tm.TiledMap.load(MAP01))
    assert (w.W, w.D, w.H) == (30, 300, 2)
    assert [l.z for l in w.layers] == [0, 1]
    # una capa por nivel: todo en el hueco 0
    assert not w.tiles[..., 1:].any()
    assert w.collision_stats()["solid"] > 0


def test_layers_of_a_level_stack_into_slots_in_tiled_order():
    m = make_map()
    m.layers += [layer("a", {(0, 0): 1, (1, 0): 1}), layer("b", {(0, 0): 2}),
                 layer("c", {(0, 0): 3, (1, 0): 3})]
    w = World(m)
    assert list(w.tiles[0, 0, 0] & GID_MASK) == [1, 2, 3, 0]
    assert list(w.tiles[0, 0, 1] & GID_MASK) == [1, 3, 0, 0]          # sin huecos vacíos
    assert list(w.source[0, 0, 1]) == [0, 2, EMPTY_SOURCE, EMPTY_SOURCE]


def test_overflow_keeps_lowest_tiles_and_warns(caplog):
    m = make_map()
    m.layers += [layer(f"l{i}", {(2, 1): i + 1}) for i in range(SLOTS + 2)]
    with caplog.at_level(logging.WARNING):
        w = World(m)
    assert list(w.tiles[0, 1, 2] & GID_MASK) == [1, 2, 3, 4]
    assert w.dropped_tiles == 2
    assert "más de 4 tiles" in caplog.text and "x=2, y=1" in caplog.text


def test_group_level_role_opacity_and_visibility_are_inherited():
    m = make_map()
    g = tm.LayerGroup(name="Planta 1", klass="overhead", opacity=0.5,
                      properties={"Z": tm.Property("Z", "int", 1)})
    g.layers = [layer("suelo", {(0, 0): 1}, klass="floor"),
                layer("techo", {(0, 0): 2}, opacity=0.5),
                layer("oculta", {(0, 0): 3}, visible=False)]
    m.layers += [layer("calle", {(0, 0): 4}, z=0), g]
    w = World(m)
    names = [l.name for l in w.layers]
    assert names == ["calle", "Planta 1/suelo", "Planta 1/techo"]        # la oculta no se carga
    suelo, techo = w.layers[1], w.layers[2]
    assert (suelo.z, suelo.role, suelo.opacity) == (1, LayerRole.FLOOR, 0.5)
    assert (techo.z, techo.role, techo.opacity) == (1, LayerRole.OVERHEAD, 0.25)
    assert [int(w.role_of(k)[1, 0, 0]) for k in range(2)] == [LayerRole.FLOOR, LayerRole.OVERHEAD]


def test_tile_class_overrides_layer_role():
    m = make_map(tile_classes={1: "object"})                 # GID 2 es siempre OBJECT
    m.layers.append(layer("suelo", {(0, 0): 1, (1, 0): 2}, klass="floor"))
    w = World(m)
    assert int(w.role_of(0)[0, 0, 0]) == LayerRole.FLOOR
    assert int(w.role_of(0)[0, 0, 1]) == LayerRole.OBJECT


def test_negative_levels_and_level_height_property():
    m = make_map()
    m.properties["level_height"] = tm.Property("level_height", "int", 92)
    m.layers += [layer("sotano", {(0, 0): 1}, z=-1), layer("calle", {(0, 0): 2}, z=0)]
    w = World(m)
    assert (w.H, w.get_level_value(0), w.level_height) == (2, -1, 92)
    assert w.get_tile(0, 0, 0, 0) == 1 and w.get_tile(0, 0, 1, 0) == 2


def test_flip_bits_are_kept_in_the_gid():
    m = make_map()
    m.layers.append(layer("l", {(0, 0): 3 | GID_FLIP_H | GID_FLIP_V}))
    w = World(m)
    assert int(w.tiles[0, 0, 0, 0]) == 3 | GID_FLIP_H | GID_FLIP_V
    assert w.get_tile(0, 0, 0, 0) == 3


def test_cells_from_tile_properties_topmost_geometry_wins():
    m = make_map(tile_props={0: {"solid": True}, 1: {"solid": False}})
    m.layers += [layer("a", {(0, 0): 1, (1, 0): 2}), layer("b", {(1, 0): 1})]
    w = World(m)
    assert w.get_cell(0, 0, 0) == cells.BLOCK
    assert w.get_cell(1, 0, 0) == cells.BLOCK         # el hueco de arriba es sólido
    assert w.get_cell(2, 0, 0) == cells.EMPTY
    assert w.collision.is_solid(0, 0, 0) and not w.collision.is_solid(2, 0, 0)
    assert w.collision.data is w.cells                # mismo array, sin copias


def test_infinite_map_uses_chunk_bounds_as_origin():
    w = World(tm.TiledMap.load(ROOT / "tests" / "data" / "infinite.tmx"))
    assert (w.origin_x, w.origin_y, w.W, w.D) == (-16, 0, 16, 16)
    assert (w.tiles[0, :, :, 0] & GID_MASK == 1).all()


def test_cell_format():
    assert cells.encode(cells.BLOCK) == 1                     # compatible con el antiguo "sólido"
    v = cells.encode(cells.SHAPES["ramp"], cells.VARIANTS["east"], top=0.5, base=0.25)
    assert cells.describe(v) == {"shape": "ramp", "variant": 1, "top": 0.5, "base": 0.25, "flags": 0}
    assert cells.is_solid(v) and not cells.is_solid(cells.EMPTY)
    with pytest.raises(ValueError):
        cells.encode(cells.BLOCK, top=0.3)
