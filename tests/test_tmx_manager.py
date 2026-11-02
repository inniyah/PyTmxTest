import pytest

from conftest import MAP01, MAP02
from tmx_manager import TiledMap, TileLayer


def tile_layers(m):
    return [l for l in m.get_all_layers_flat() if isinstance(l, TileLayer)]


@pytest.mark.parametrize("path", [MAP01, MAP02])
def test_load_example_maps(path):
    m = TiledMap.load(path)
    assert m.width > 0 and m.height > 0 and m.tilesets
    for layer in tile_layers(m):
        assert len(layer.data.tiles) == layer.width * layer.height


@pytest.mark.parametrize("encoding,compression", [
    ("csv", None), ("base64", None), ("base64", "zlib"), ("base64", "gzip"), ("base64", "zstd"),
])
def test_save_roundtrip(tmp_path, encoding, compression):
    if compression == "zstd":
        pytest.importorskip("zstandard")
    original = TiledMap.load(MAP01)
    out = tmp_path / "copy.tmx"
    original.save(out, encoding=encoding, compression=compression)
    reloaded = TiledMap.load(out)
    for a, b in zip(tile_layers(original), tile_layers(reloaded)):
        assert list(a.data.tiles) == list(b.data.tiles)


def test_unknown_compression_is_rejected(tmp_path):
    m = TiledMap.load(MAP01)
    with pytest.raises(ValueError):
        m.save(tmp_path / "x.tmx", encoding="base64", compression="lz4")
