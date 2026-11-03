import numpy as np
import pytest
from PIL import Image

from tmx_explorer.renderer.atlas import layout_atlas, pack_shelves


def overlaps(a, b):
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


@pytest.mark.parametrize("max_size", [64, 256, 4096])
def test_pack_shelves_no_overlap_and_inside_pages(max_size):
    rng = np.random.default_rng(0)
    sizes = [tuple(int(v) for v in rng.integers(4, 60, 2)) for _ in range(120)]
    placements, pages = pack_shelves(sizes, max_size)
    for page, (pw, ph) in enumerate(pages):
        assert pw <= max_size and ph <= max_size
        rects = [(p.x, p.y, w, h) for p, (w, h) in zip(placements, sizes) if p.page == page]
        for x, y, w, h in rects:
            assert x + w <= pw and y + h <= ph
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                assert not overlaps(rects[i], rects[j])
    if max_size == 64:
        assert len(pages) > 1                       # no cabe en una sola página


def test_too_big_image_is_rejected():
    with pytest.raises(ValueError):
        pack_shelves([(100, 10)], 64)


def test_layout_atlas_regions_point_to_original_pixels():
    rng = np.random.default_rng(1)
    images = {k: Image.fromarray(rng.integers(0, 255, (h, w, 4), dtype=np.uint8))
              for k, (w, h) in enumerate([(16, 24), (32, 8), (5, 7), (64, 64)])}
    pages, regions = layout_atlas(images, 128, border=1)
    for key, img in images.items():
        r = regions[key]
        page = pages[r.page]
        ph, pw = page.shape[:2]
        left, top = round(r.u0 * pw), round((1 - r.v1) * ph)     # v1 = borde superior
        assert (r.width, r.height) == img.size
        np.testing.assert_array_equal(page[top:top + r.height, left:left + r.width],
                                      np.asarray(img))
        # el borde extruido repite el píxel del contorno
        np.testing.assert_array_equal(page[top - 1, left:left + r.width], np.asarray(img)[0])
