"""Matemática de render sin GPU: UV con flips y recolección vectorizada de tiles."""
import numpy as np
import pytest
from PIL import Image

from conftest import MAP01
from tmx_explorer.config import DEPTH_BASE, DEPTH_LAYER_STEP, GID_FLIP_D, GID_FLIP_H, GID_FLIP_V
from tmx_explorer.map.structure import Map3DStructure
from tmx_explorer.map.tileset_renderer import TileLookup, load_tileset_images
from tmx_explorer.map.visible_tiles import collect_visible_tiles
from tmx_explorer.renderer.atlas import layout_atlas
from tmx_explorer.renderer.sprite_batch import quad_vertices
from tmx_manager import TiledMap

H, V, D = GID_FLIP_H >> 28, GID_FLIP_V >> 28, GID_FLIP_D >> 28


def sample(img: np.ndarray, u: float, v: float):
    """Muestreo NEAREST de una textura subida volteada (v = 1 arriba)."""
    h, w = img.shape[:2]
    return img[min(h - 1, int((1 - v) * h)), min(w - 1, int(u * w))]


@pytest.mark.parametrize("flip,transform", [
    (0, []), (H, [Image.FLIP_LEFT_RIGHT]), (V, [Image.FLIP_TOP_BOTTOM]),
    (H | V, [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]),
    (D, [Image.TRANSPOSE]), (D | H, [Image.TRANSPOSE, Image.FLIP_LEFT_RIGHT]),
    (D | V, [Image.TRANSPOSE, Image.FLIP_TOP_BOTTOM]),
    (D | H | V, [Image.TRANSPOSE, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]),
])
def test_flip_uvs_match_tiled_spec(flip, transform):
    rng = np.random.default_rng(2)
    src = rng.integers(0, 255, (6, 4, 3), dtype=np.uint8)       # 4x6, asimétrico
    expected = Image.fromarray(src)
    for t in transform:                                          # diagonal, luego H, luego V
        expected = expected.transpose(t)
    expected = np.asarray(expected)
    eh, ew = expected.shape[:2]

    verts = quad_vertices(0, 0, ew, eh, 0, 0, 0, 1, 1, flips=np.array([flip]))
    uv = verts[:, 2:4]          # UV de las esquinas: arriba-izq, arriba-der, abajo-der, abajo-izq
    # interpolación bilineal de las UV en el centro de cada píxel mostrado
    for y in range(eh):
        for x in range(ew):
            s, t = (x + 0.5) / ew, (y + 0.5) / eh
            top = uv[0] + (uv[1] - uv[0]) * s
            bottom = uv[3] + (uv[2] - uv[3]) * s
            u, v = top + (bottom - top) * t
            np.testing.assert_array_equal(sample(src, u, v), expected[y, x])


def reference_tiles(m, lookup, x0, x1, y0, y1, max_z, offset):
    """La versión con bucles de la fase 2, como referencia."""
    out = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            for z in range(max_z + 1):
                for n in range(m.N):
                    if m.layer_levels[n] != m.get_level_value(z):
                        continue
                    gid = int(m.mapa[z, y, x, n])
                    if gid == 0 or lookup.page[gid] < 0:
                        continue
                    w, h = lookup.size[gid]
                    out.append((x * m.tile_width, (y + 1) * m.tile_height - h - z * offset,
                                w, h, DEPTH_BASE + y + z + n * DEPTH_LAYER_STEP))
    return np.array(out, dtype=np.float32)


@pytest.fixture(scope="module")
def map_and_lookup():
    tmx = TiledMap.load(MAP01)
    _, regions = layout_atlas(load_tileset_images(tmx, MAP01), 4096)
    return Map3DStructure(tmx), TileLookup.from_regions(regions)


@pytest.mark.parametrize("window", [(0, 30, 0, 300), (3, 17, 120, 140), (10, 11, 50, 51)])
@pytest.mark.parametrize("max_z", [0, 1])
def test_vectorized_collection_matches_reference(map_and_lookup, window, max_z):
    m, lookup = map_and_lookup
    offset = 128
    batches = collect_visible_tiles(m, lookup, *window, max_z, offset, [True] * m.N)
    verts = np.concatenate(list(batches.values())).reshape(-1, 4, 9)
    # x, y de la esquina superior izquierda; w, h de la diagonal; depth
    got = np.column_stack([verts[:, 0, 0], verts[:, 0, 1],
                           verts[:, 2, 0] - verts[:, 0, 0], verts[:, 2, 1] - verts[:, 0, 1],
                           verts[:, 0, 8]])
    expected = reference_tiles(m, lookup, *window, max_z, offset)
    key = lambda a: a[np.lexsort(a.T[::-1])]
    np.testing.assert_allclose(key(got), key(expected), atol=1e-3)


def test_hidden_layers_are_skipped(map_and_lookup):
    m, lookup = map_and_lookup
    all_on = collect_visible_tiles(m, lookup, 0, 30, 0, 300, 1, 128, [True, True])
    only_0 = collect_visible_tiles(m, lookup, 0, 30, 0, 300, 1, 128, [True, False])
    count = lambda b: sum(len(v) for v in b.values()) // 4
    assert 0 < count(only_0) < count(all_on)


def test_empty_window():
    m = Map3DStructure(TiledMap.load(MAP01))
    assert collect_visible_tiles(m, None, 5, 5, 0, 10, 1, 128, [True] * m.N) == {}
