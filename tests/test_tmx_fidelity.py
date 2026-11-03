"""Ida y vuelta sin pérdidas de tmx_manager."""
import xml.etree.ElementTree as ET

import pytest

import tmx_manager as tm
from conftest import MAP01, MAP02, ROOT

DATA = ROOT / "tests" / "data"


def signature(path):
    """Todos los elementos y atributos (con valor) del XML, salvo los datos de tiles."""
    out = set()
    for e in ET.parse(path).getroot().iter():
        if e.tag == "tile" and set(e.attrib) <= {"gid"}:
            continue                                    # formato XML antiguo de datos
        out.add(e.tag)
        out.update(f"{e.tag}@{k}={v}" for k, v in e.attrib.items())
        if e.tag in ("property", "text") and e.text and e.text.strip():
            out.add(f"{e.tag}#{e.text.strip()}")
    return out


def tile_data(m):
    return [(list(l.data.tiles), [(c.x, c.y, list(c.tiles)) for c in l.data.chunks])
            for l in m.iter_layers() if isinstance(l, tm.TileLayer)]


@pytest.mark.parametrize("path", [DATA / "full_features.tmx", DATA / "infinite.tmx", MAP01, MAP02])
def test_roundtrip_loses_nothing(tmp_path, path):
    m = tm.TiledMap.load(path)
    out = tmp_path / "out.tmx"
    m.save(out)
    lost = {s for s in signature(path) - signature(out) if not s.endswith("offsetx=0")}
    assert not lost
    assert tile_data(tm.TiledMap.load(out)) == tile_data(m)


def test_unmodelled_features_survive():
    m = tm.TiledMap.load(DATA / "full_features.tmx")
    assert [type(l).__name__ for l in m.layers] == ["TileLayer", "RawLayer", "ObjectGroup", "LayerGroup"]
    assert m.extra_attrib["nextobjectid"] == "5"
    tile0, tile1 = m.tilesets[0].tiles[0], m.tilesets[0].tiles[1]
    assert [c.tag for c in tile0.extra_children] == ["objectgroup"]
    assert [c.tag for c in tile1.extra_children] == ["animation"]
    assert [o.shape for o in m.layers[2].objects] == ["polygon", "text", "point"]


def test_bool_and_special_properties_are_written_as_tiled_expects(tmp_path):
    m = tm.create_empty_map(1, 1, 16, 16)
    m.properties = {
        "a": tm.Property("a", "bool", True),
        "b": tm.Property("b", "string", "línea 1\nlínea 2"),
        "c": tm.Property("c", "class", {"x": tm.Property("x", "int", 3)}, propertytype="Punto"),
        "d": tm.Property("d", "float", 2.0),
    }
    m.save(tmp_path / "p.tmx")
    text = (tmp_path / "p.tmx").read_text(encoding="utf-8")
    assert 'value="true"' in text and "True" not in text
    back = tm.TiledMap.load(tmp_path / "p.tmx").properties
    assert back["a"].value is True
    assert back["b"].value == "línea 1\nlínea 2"
    assert back["c"].value["x"].value == 3 and back["c"].propertytype == "Punto"
    assert back["d"].value == 2.0 and 'value="2"' in text


def test_tileset_for_flipped_gid():
    m = tm.TiledMap.load(MAP01)
    assert m.get_tileset_for_gid(5 | 0x80000000).name == "VTiles1A"
    assert m.get_tileset_for_gid(177 | 0x40000000).name == "VTiles1B"
    assert m.get_tileset_for_gid(0) is None


def test_infinite_layer_access():
    m = tm.TiledMap.load(DATA / "infinite.tmx")
    layer = m.layers[0]
    assert layer.is_infinite and layer.bounds() == (-16, 0, 0, 16)
    assert layer.get_tile_gid(-16, 0) == 1 and layer.get_tile_gid(0, 0) == 0
    layer.set_tile_gid(20, 3, 7)                        # crea un chunk nuevo
    assert layer.get_tile_gid(20, 3) == 7 and layer.bounds() == (-16, 0, 32, 16)
    dense = layer.dense_tiles(-16, 0, 48, 16)
    assert dense[3 * 48 + 36] == 7 and dense[0] == 1
