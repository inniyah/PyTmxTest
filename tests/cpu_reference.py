"""Composición de referencia en la CPU para el dibujo del mundo: mismas reglas que el shader."""
import io, numpy as np
from PIL import Image
from engine import png16
from engine.render.geometry import chunk_quads
from engine.render.shaders import LEVEL_BIAS

def cpu_render(renderer, camera, width, height):
    g, view = renderer.geo, renderer.view
    pages_c = {p: np.asarray(Image.open(io.BytesIO(view.store.atlas_page(p, "color"))).convert("RGBA")).astype(float)
               for p in renderer.layout.page_sizes}
    pages_p = {p: png16.decode_rgba16(view.store.atlas_page(p, "position")).astype(float) / 65535
               for p in renderer.layout.page_sizes}
    color = np.zeros((height, width, 3)); color[:] = (0.07 * 255, 0.07 * 255, 0.09 * 255)
    depth = np.ones((height, width))
    pmin, pmax = np.array(g.pos_min), np.array(g.pos_max)
    for key in renderer.visible_chunks(camera):
        chunk = view.chunk(*key)
        if chunk is None: continue
        for page, verts in chunk_quads(chunk, g, renderer.layout).items():
            pw, ph = renderer.layout.page_sizes[page]
            for q in verts.reshape(-1, 4, 8):
                l, t = q[0, 0], q[0, 1]; w, h = q[1, 0] - l, q[2, 1] - t
                u0, u1, v0 = q[0, 2], q[1, 2], q[0, 3]
                sx, sy = int(round((l - camera.x) * camera.zoom)), int(round((t - camera.y) * camera.zoom))
                ax, ay = int(round(min(u0, u1) * pw)), int(round(v0 * ph))
                iw, ih = int(round(w)), int(round(h))
                src_c = pages_c[page][ay:ay + ih, ax:ax + iw]
                src_p = pages_p[page][ay:ay + ih, ax:ax + iw]
                if u0 > u1: src_c, src_p = src_c[:, ::-1], src_p[:, ::-1]
                x0, y0 = max(sx, 0), max(sy, 0); x1, y1 = min(sx + iw, width), min(sy + ih, height)
                if x0 >= x1 or y0 >= y1: continue
                c = src_c[y0 - sy:y1 - sy, x0 - sx:x1 - sx]; p = src_p[y0 - sy:y1 - sy, x0 - sx:x1 - sx]
                y_m = q[0, 5] + pmin[1] + p[..., 1] * (pmax[1] - pmin[1])      # altura del píxel
                rows = np.arange(y0, y1)[:, None] + 0.5
                screen_y = camera.y + rows / camera.zoom                 # píxeles de mundo
                close = (y_m * (g.sz + g.sy ** 2 / g.sz) + screen_y * g.sy / g.sz
                         + LEVEL_BIAS * np.floor(q[0, 5] / g.cell_h + 0.5))
                d = np.clip(1 - (close - g.closeness_min) / (g.closeness_max - g.closeness_min), 0, 1)
                win = (c[..., 3] * q[0, 7] >= 0.01 * 255) & (d <= depth[y0:y1, x0:x1])
                depth[y0:y1, x0:x1][win] = d[win]
                color[y0:y1, x0:x1][win] = c[..., :3][win]
    return color.astype(np.uint8), depth
