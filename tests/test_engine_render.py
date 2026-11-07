"""Dibujo del mundo compilado: geometría (sin GPU) y comparación GPU/CPU (con pantalla)."""
import numpy as np
import pytest

from engine.compiler import compile_tmx
from engine.render.geometry import FLOATS_PER_VERTEX, AtlasLayout, WorldGeometry, chunk_quads
from engine.world import visual as V
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.build import build_gallery, build_tileset
from tools.pieces.course import build_course


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    out = tmp_path_factory.mktemp("render")
    tsx = build_tileset(out / "assets")
    build_gallery(out / "maps" / "gallery.tmx", tsx)
    build_course(out / "maps" / "course.tmx", tsx)
    for name in ("gallery", "course"):
        report = compile_tmx(out / "maps" / f"{name}.tmx", out / f"{name}.cworld")
        assert report.ok and not report.warnings, report.summary()
    return {name: out / f"{name}.cworld" for name in ("gallery", "course")}


def open_world(path):
    view = WorldView(SQLiteWorldStore(path))
    return view, WorldGeometry.from_meta(view.meta), AtlasLayout.from_store(view.store)


def test_geometry_from_meta(worlds):
    _, geo, _ = open_world(worlds["course"])
    assert (geo.tile_w, geo.row_h, geo.level_h) == (128, 92, 133)
    assert np.isclose(geo.sz, 46) and np.isclose(geo.sy, 133 / 3)
    assert geo.closeness_min < 0 < geo.closeness_max


def test_one_quad_per_visual_slot_anchored_to_its_cell(worlds):
    view, geo, layout = open_world(worlds["course"])
    for key in [(0, 0)]:
        chunk = view.chunk(*key)
        quads = chunk_quads(chunk, geo, layout)
        verts = np.concatenate(list(quads.values())).reshape(-1, 4, FLOATS_PER_VERTEX)
        assert len(verts) == int((V.region(chunk.visual) != 0).sum())
        # El borde inferior de cada quad está en el borde sur de su celda, subido por su planta
        x_cell = verts[:, 0, 4] / geo.cell_w
        level = np.round(verts[:, 0, 5] / geo.cell_h)
        y_cell = verts[:, 0, 6] / geo.cell_w
        assert np.allclose(verts[:, 0, 0], x_cell * geo.tile_w)
        assert np.allclose(verts[:, 2, 1], (y_cell + 1) * geo.row_h - level * geo.level_h)
        assert set(np.unique(level)) == {0, 1, 2}
        # De lejos a cerca
        close = geo.closeness(level * geo.cell_h, (y_cell + 1) * geo.cell_w)
        assert (np.diff(close) >= -1e-6).all()
        # UV dentro de la página
        assert verts[..., 2:4].min() >= 0 and verts[..., 2:4].max() <= 1


def test_horizontal_flip_swaps_u(worlds, tmp_path):
    import tmx_manager as tm
    view, geo, layout = open_world(worlds["course"])
    # Recompilar el recorrido con una rampa volteada y comparar su quad
    src = tm.TiledMap.load(worlds["course"].parent / "maps" / "course.tmx")
    layer = next(l for l in src.iter_layers() if isinstance(l, tm.TileLayer) and l.name == "suelo")
    layer.set_tile_gid(0, 0, layer.get_tile_gid(0, 0) | 0x80000000)
    src.save(worlds["course"].parent / "maps" / "flip.tmx")
    compile_tmx(worlds["course"].parent / "maps" / "flip.tmx", tmp_path / "flip.cworld")
    a = chunk_quads(view.chunk(0, 0), geo, layout)[0].reshape(-1, 4, FLOATS_PER_VERTEX)
    v2 = WorldView(SQLiteWorldStore(tmp_path / "flip.cworld"))
    b = chunk_quads(v2.chunk(0, 0), geo, AtlasLayout.from_store(v2.store))[0].reshape(-1, 4, FLOATS_PER_VERTEX)
    qa = a[(a[:, 0, 4] == 0) & (a[:, 0, 6] == 0)][0]
    qb = b[(b[:, 0, 4] == 0) & (b[:, 0, 6] == 0)][0]
    assert np.allclose(qa[:, 2], qb[[1, 0, 3, 2], 2]) and np.allclose(qa[:, :2], qb[:, :2])


# --- Con GPU: el shader de profundidad por píxel frente a la referencia en CPU --------

@pytest.fixture(scope="module")
def gl_viewer_factory():
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


@pytest.mark.parametrize("world,center", [("gallery", (9, 15)), ("gallery", (9, 33)),
                                          ("course", (6, 6)), ("course", (21, 7))])
def test_gpu_matches_cpu_reference_pixel_by_pixel(worlds, gl_viewer_factory, world, center):
    from PIL import Image

    from cpu_reference import cpu_render
    W, H = 640, 400
    viewer = gl_viewer_factory(worlds[world], W, H, visible=False)
    try:
        viewer.renderer.lighting.enabled = False          # la referencia compara la geometría y la oclusión
        g = viewer.renderer.geo
        viewer.camera.center_on(center[0] * g.tile_w, center[1] * g.row_h)
        viewer.camera.x, viewer.camera.y = round(viewer.camera.x), round(viewer.camera.y)
        path = worlds[world].parent / "gpu.png"
        viewer.screenshot(path)
        gpu = np.asarray(Image.open(path).convert("RGB")).astype(int)
        cpu, _ = cpu_render(viewer.renderer, viewer.camera, W, H)
        import glfw
        bad = np.abs(gpu - cpu.astype(int)).max(axis=-1) > 2
        ys, xs = np.nonzero(bad)
        assert not bad.any(), (f"{bad.sum()} píxeles distintos en filas {ys.min()}-{ys.max()}, columnas "
                               f"{xs.min()}-{xs.max()}; cámara {viewer.camera.width}x{viewer.camera.height}, "
                               f"ventana {glfw.get_window_size(viewer.window)}")
    finally:
        viewer.close()


def test_character_half_hidden_by_the_doorway(worlds, gl_viewer_factory, tmp_path):
    """En la puerta de la habitación, pegado a la pared sur: la pared tapa su mitad izquierda."""
    from PIL import Image
    viewer = gl_viewer_factory(worlds["course"], 700, 460, visible=False, spawn=(14.0, 7.8))
    try:
        viewer.renderer.lighting.enabled = False
        viewer.camera.zoom = 2.5
        viewer._center_on_player(1.0)
        viewer.screenshot(tmp_path / "door.png")
        img = np.asarray(Image.open(tmp_path / "door.png").convert("RGB")).astype(int)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        skin = (r > 200) & (g > 130) & (g < 215) & (b > 90) & (b < 190) & (r - b > 40)
        geo = viewer.renderer.geo
        wall_edge = (14 * geo.cell_w * geo.tile_w / geo.cell_w - viewer.camera.x) * viewer.camera.zoom
        ys, xs = np.nonzero(skin)
        import glfw
        info = (f"cámara {viewer.camera.width}x{viewer.camera.height} en ({viewer.camera.x:.1f}, "
                f"{viewer.camera.y:.1f}), ventana {glfw.get_window_size(viewer.window)}, "
                f"framebuffer {glfw.get_framebuffer_size(viewer.window)}, borde {wall_edge:.1f}")
        assert len(xs) > 300, f"{len(xs)} píxeles de piel; {info}"   # se ve una buena parte del cuerpo
        assert (xs >= wall_edge - 1).all(), f"piel desde x={xs.min()}; {info}"   # nada a la izquierda
    finally:
        viewer.close()


def test_facing_is_stable_along_a_diagonal_and_follows_new_input():
    from engine.render.sprites import facing_index
    f = 2                                                   # oeste
    for t in range(50):                                     # resbalando a 45° con algo de ruido
        a = np.radians(135 + 4 * np.sin(t))
        f = facing_index(np.cos(a), np.sin(a), f)
        assert f == 2
    assert facing_index(0, -1, 2) == 3                      # pulsar arriba: norte
    assert facing_index(0, 0, 3) == 3                       # sin pulsar: se conserva
    assert facing_index(1, 0, 3) == 0                       # este
    assert facing_index(1, -1, 0, directions=8) == 7        # 8 direcciones: 45° ya es el noreste
    assert facing_index(1, -0.6, 0, directions=8) == 0      # 31°: dentro del margen del este, se mantiene
    assert facing_index(0.2, -1, 0, directions=8) == 6      # casi norte: cambia


def test_walk_cycle_follows_the_sheet_order():
    from engine.render.sprites import SpriteSheet
    sheet = SpriteSheet(texture=0, width=320, height=420)
    seq = [sheet.walk_col(t / 8 + 1e-6, 8.0) for t in range(9)]
    assert sheet.walk_col(0.0, 8.0) == 0                        # quieto: de pie
    assert seq[:8] == [1, 2, 3, 0, 1, 2, 3, 0]                  # paso, de pie, otro paso, de pie…


# --- Iluminación ------------------------------------------------------------------------

def test_lights_are_compiled_from_tiled(worlds):
    store = SQLiteWorldStore(worlds["course"])
    lights = np.array(store.lights())
    assert len(lights) == 8
    assert (lights[:, 6] > 0).all() and (lights[:, 7] > 0).all()              # radio, intensidad
    assert set(np.round(lights[:, 1] / 3.0 - 2.2 / 3.0).astype(int)) == {0, 1, 2}   # en tres plantas


def test_light_defaults_and_colors():
    from engine.compiler.compile import LIGHT_DEFAULTS, _parse_color
    assert _parse_color("#ffffff") == (1.0, 1.0, 1.0)
    assert _parse_color("#ff000000") == (0.0, 0.0, 0.0)                       # #AARRGGBB de Tiled
    r, g, b = _parse_color(LIGHT_DEFAULTS["color"])
    assert r > g > b > 0                                                      # cálida, en lineal


def test_flipped_tiles_flip_their_normals(gl_viewer_factory, tmp_path):
    """Una luz al este: la cara de una rampa volteada (mira al este) brilla más que la normal."""
    from pathlib import Path

    from PIL import Image

    from tools.pieces.catalog import build_catalog
    from tools.pieces.course import CourseBuilder
    from tools.pieces.render import DEFAULT_PROJECTION
    tsx = build_tileset(tmp_path / "assets")
    b = CourseBuilder(8, 5, build_catalog())
    for y in range(5):
        for x in range(8):
            if (x, y) not in ((2, 2), (5, 2)):
                b.put(0, x, y, "floor_flat_000")
    b.put(0, 2, 2, "floor_ramp_E_050")
    b.put(0, 5, 2, "floor_ramp_E_050", flip=True)
    for x in (3.6, 6.6):
        b.light(0, x, 2.5, "#ffffff", radius=4.0, intensity=3.0, height=1.5)
    b.to_map(Path(tsx), tmp_path / "f.tmx", DEFAULT_PROJECTION).save(tmp_path / "f.tmx")
    assert compile_tmx(tmp_path / "f.tmx", tmp_path / "f.cworld").ok
    v = gl_viewer_factory(tmp_path / "f.cworld", 1000, 500, visible=False, player=False)
    try:
        v.set_night(True)
        v.camera.center_on(4 * 128, 2.5 * 92)
        v.screenshot(tmp_path / "f.png")
        img = np.asarray(Image.open(tmp_path / "f.png").convert("L")).astype(float)
        g = v.renderer.geo

        def face(cx):
            X, Y, Z = (cx + 0.5) * 2, 2.25, 5.0
            sx = int((X * 64 - v.camera.x) * v.camera.zoom)
            sy = int((Z * g.sz - Y * g.sy - v.camera.y) * v.camera.zoom)
            return img[sy - 3:sy + 4, sx - 3:sx + 4].mean()
        assert face(5) > 1.5 * face(2)
    finally:
        v.close()


def test_lantern_lights_around_the_player(worlds, gl_viewer_factory, tmp_path):
    from PIL import Image
    v = gl_viewer_factory(worlds["course"], 400, 300, visible=False, spawn=(10.5, 13.0))
    try:
        v.set_night(True)
        v._center_on_player(1.0)
        means = []
        for lantern in (True, False):
            v.lantern = lantern
            v.screenshot(tmp_path / "l.png")
            img = np.asarray(Image.open(tmp_path / "l.png").convert("L")).astype(float)
            means.append(img[110:190, 140:260].mean())
        assert means[0] > means[1] + 15
    finally:
        v.close()
