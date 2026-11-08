"""Terreno tipo Populous: alturas en las esquinas -> piezas, y mundo grande con niveles de detalle."""
import itertools

import numpy as np
import pytest

from engine.compiler import compile_tmx
from engine.entities import BRAIN_WANDER, ActorTable, load_types
from engine.entities.world_sim import ActorSim
from engine.world import cell as C
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.catalog import build_catalog
from tools.pieces.terrain import build_terrain, choose_plots, flatten_plots, make_heights, tile_pieces

CATALOG = {p.name: p for p in build_catalog()}


def _neighbours_ok(H):
    return np.abs(np.diff(H, axis=0)).max() <= 1 and np.abs(np.diff(H, axis=1)).max() <= 1


# --- Alturas -------------------------------------------------------------------------------

def test_heights_respect_the_neighbour_rule_and_plots_are_flat():
    rng = np.random.default_rng(11)
    H = make_heights(128, 96, rng)
    assert _neighbours_ok(H) and H.min() >= 0
    plots = choose_plots(H, 6, rng=rng)
    F = flatten_plots(H, plots)
    assert _neighbours_ok(F)
    for x, y, w, d in plots:
        area = F[y:y + d + 1, x:x + w + 1]
        assert np.ptp(area) == 0 and area[0, 0] % 4 == 0                  # llano, a una planta exacta


# --- Casillas -> piezas: exhaustivo -------------------------------------------------------

def _all_tiles():
    """Todas las casillas posibles (esquinas vecinas a 1 como mucho) a varias alturas."""
    for nw, ne, sw, se in itertools.product(range(3), repeat=4):
        if max(abs(nw - ne), abs(nw - sw), abs(ne - se), abs(sw - se)) > 1:
            continue
        for base in (0, 1, 2, 3, 4, 5, 7):
            yield nw + base, ne + base, sw + base, se + base


def _expected(nw, ne, sw, se, u, v, split_backslash):
    """Altura (en cuartos) del terreno en (u, v): plano del triángulo que lo contiene."""
    if split_backslash:                                 # diagonal NO-SE
        if u >= v:
            return nw + (ne - nw) * u + (se - ne) * v
        return nw + (se - sw) * u + (sw - nw) * v
    if u + v <= 1:                                      # diagonal SO-NE
        return nw + (ne - nw) * u + (sw - nw) * v
    return ne + sw - se + (se - sw) * u + (se - ne) * v


def test_every_possible_tile_is_reproduced_exactly():
    rng = np.random.default_rng(0)
    uv = rng.random((40, 2))
    count = 0
    for nw, ne, sw, se in _all_tiles():
        pieces = tile_pieces(nw, ne, sw, se)
        cells = {}
        for level, name in pieces:
            assert name in CATALOG, name
            spec = CATALOG[name].spec
            cells[level] = cells[level].merge(spec) if level in cells else spec
        values = {lv: spec.encode() for lv, spec in cells.items()}
        splits = {bool(C.split(v)) for v in values.values() if C.floor_code(v, 1) != C.floor_code(v, 2)}
        split_backslash = splits.pop() if splits else abs(nw - se) <= abs(ne - sw)
        for u, v in uv:
            # La superficie más alta entre las plantas de la casilla, en ese punto
            tops = [(lv + float(C.surface_at(val, u, v))) * 4 for lv, val in values.items()
                    if np.isfinite(C.surface_at(val, u, v))]
            assert tops, (nw, ne, sw, se, u, v)
            assert max(tops) == pytest.approx(_expected(nw, ne, sw, se, u, v, split_backslash), abs=1e-9)
        count += 1
    assert count == 35 * 7                                      # 35 formas de casilla a 7 alturas


# --- Mundo grande ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def terrain(tmp_path_factory, shared_tileset):
    out = tmp_path_factory.mktemp("terrain")
    build_terrain(out / "t.tmx", shared_tileset, 128, 128, houses=6, seed=5)
    report = compile_tmx(out / "t.tmx", out / "t.cworld", chunk_size=16)
    assert report.ok and not report.warnings, report.summary()
    return out / "t.cworld"


def test_spawn_points_are_never_buried(terrain):
    view = WorldView(SQLiteWorldStore(terrain))
    sim = ActorSim(view, ActorTable(load_types()), np.random.default_rng(1))
    pts = sim.spawn_points(300, np.random.default_rng(2), levels=list(range(view.level_min, view.level_max + 1)))
    x, y, z = pts.T
    assert (sim.physics.ceiling(x, z, y) >= y + 1.8).all()
    top = sim.physics.support(x, z, np.full(len(pts), 99.0), search=40)
    assert (np.abs(top - y) < 1e-6).mean() > 0.9             # casi todos sobre el terreno


def test_crowd_walks_the_terrain(terrain):
    view = WorldView(SQLiteWorldStore(terrain))
    table = ActorTable(load_types())
    rng = np.random.default_rng(3)
    sim = ActorSim(view, table, rng)
    for p in sim.spawn_points(120, rng, levels=list(range(view.level_min, view.level_max + 1))):
        table.spawn("persona", p, brain=BRAIN_WANDER)
    start = table.pos[table.rows()].copy()
    for _ in range(15):
        sim.step(1.0)
        rows = table.rows()
        x, y, z = table.pos[rows].T
        assert (sim.physics.support(x, z, y + table.height[rows]) - y < 0.06).all()
    moved = np.hypot(x - start[:, 0], z - start[:, 2])
    assert np.median(moved) > 4.0 and (moved < 1).mean() < 0.15
    assert np.ptp(y - start[:, 1]) > 1.0                       # suben y bajan laderas


def test_far_actors_are_frozen_and_near_ones_move(terrain):
    view = WorldView(SQLiteWorldStore(terrain))
    table = ActorTable(load_types())
    rng = np.random.default_rng(4)
    sim = ActorSim(view, table, rng, near=30.0, far=60.0)
    for p in sim.spawn_points(200, rng, levels=list(range(view.level_min, view.level_max + 1))):
        table.spawn("persona", p, brain=BRAIN_WANDER)
    focus = (128.0, 128.0)
    rows = table.rows()
    d = np.hypot(table.pos[rows, 0] - focus[0], table.pos[rows, 2] - focus[1])
    start = table.pos[rows].copy()
    sim.step(4.0, focus)
    moved = np.hypot(*(table.pos[rows][:, [0, 2]] - start[:, [0, 2]]).T)
    far, near = d > 70.0, d < 25.0                               # con margen por si alguien cruza
    assert (moved[far] == 0).all() and np.median(moved[near]) > 1.0
    assert sum(sim.tier_counts) == len(table)


def test_streaming_keeps_memory_bounded(terrain):
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
    v = Viewer(terrain, 640, 360, visible=False, spawn=(10, 10), npcs=0)
    try:
        v.sim.far = 20.0
        loaded = []
        for c in np.linspace(10, 118, 6):
            v.player.pos = v._spawn_point((c, c))
            v._center_on_player(1.0)
            for _ in range(130):
                v.update(1 / 60)
            v.frame()
            loaded.append(len(v.view.loaded))
        total = int(np.ceil(128 / v.view.chunk_size)) ** 2
        assert max(loaded) < total / 2, loaded                   # nunca el mapa entero en memoria
    finally:
        v.close()


def test_rising_terrain_is_not_a_roof_but_a_house_is(terrain):
    """El terreno que sube a la planta siguiente no activa la ocultación de lo de encima (es macizo
    hasta su superficie); el forjado de una casa, que tiene aire debajo, sí."""
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
    rng = np.random.default_rng(5)
    H = make_heights(128, 128, rng)
    plots = choose_plots(H, 6, rng=rng)
    F = flatten_plots(H, plots)

    def in_house(cx, cy):
        return any(X + 1 <= cx < X + w - 1 and Y + 1 <= cy < Y + d - 1 for X, Y, w, d in plots)
    v = Viewer(terrain, 320, 200, visible=False, spawn=(10, 10))
    try:
        crossings = 0
        for cx, cy in np.random.default_rng(6).uniform(3, 125, (250, 2)):
            if in_house(cx, cy):
                continue
            v.player.pos = v._spawn_point((cx, cy))
            lv = int(np.floor((v.player.pos[1] + 0.05) / 3))
            crossings += any(int(np.floor((v._spawn_point((cx + dx, cy + dy))[1] + 0.05) / 3)) != lv
                             for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            assert not v.covered(), (cx, cy)
        assert crossings > 10                                         # se ha probado el caso de verdad
        for X, Y, _w, _d in plots:
            v.player.pos = v._spawn_point((X + 2.5, Y + 2.5), level=int(F[Y, X]) // 4)
            assert v.covered()
    finally:
        v.close()


def test_only_visible_actors_are_drawn(terrain):
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
    v = Viewer(terrain, 480, 320, visible=False, spawn=(64, 64), npcs=400, seed=1)
    try:
        v.camera.zoom = 1.0
        v._center_on_player(1.0)
        v.frame()
        drawn = set(v.visible_actors().tolist())
        assert 0 < v.drawn_actors < len(v.table) / 4             # solo los de alrededor
        assert v.table.row(v.player_handle) in drawn
        # Los que no se dibujan están de verdad fuera de la vista
        g = v.renderer.geo
        left, top, right, bottom = v.camera.bounds()
        for row in v.table.rows():
            if row in drawn:
                continue
            X, Y, Z = v.table.pos[row]
            sx, sy = X * g.tile_w / g.cell_w, Z * g.sz - Y * g.sy
            assert not (left < sx < right and top < sy < bottom)
    finally:
        v.close()


def test_gpu_timer_reports_each_pass(terrain):
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
    v = Viewer(terrain, 320, 200, visible=False, spawn=(64, 64), npcs=50, seed=1)
    try:
        for _ in range(12):
            v.frame()
            v.gpu_timer.frame_done()
            glfw.swap_buffers(v.window)
        assert {"mundo", "sombras", "personajes"} <= set(v.gpu_timer.ms)
        assert all(ms >= 0 for ms in v.gpu_timer.ms.values())
        assert "GPU" in v.timing_text() and "CPU dibujo" in v.timing_text()
    finally:
        v.close()


def shot_without(v):
    """Captura sin luces puntuales (para comprobar que la prueba de luces no es trivial)."""
    from PIL import Image
    saved = v.renderer.world_lights
    v.renderer.world_lights = saved[:0]
    path = "/tmp/_sin_luces.png"
    v.screenshot(path)
    v.renderer.world_lights = saved
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


@pytest.mark.parametrize("spacing", [10.0, 4.0], ids=["en las listas", "demasiadas: todas"])
def test_per_chunk_light_lists_match_all_lights(terrain, tmp_path, spacing):
    """Con chunks pequeños (muchos bordes) y de noche, asignar las luces por chunk da la misma imagen
    que recorrer todas las luces de la vista en cada píxel: ninguna luz se pierde junto a un borde,
    tampoco en un chunk con más luces de las que caben en su lista."""
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("sin pantalla para OpenGL")
    glfw.window_hint(glfw.VISIBLE, 0)
    probe = glfw.create_window(16, 16, "sonda", None, None)
    if not probe:
        glfw.terminate()
        pytest.skip("no se puede crear una ventana OpenGL")
    glfw.destroy_window(probe)
    from PIL import Image

    from engine.render.shaders import MAX_CHUNK_LIGHTS
    from engine.viewer import Viewer
    v = Viewer(terrain, 640, 400, visible=False, spawn=(64, 64), npcs=0)
    try:
        v.set_night(True)
        v.lantern = False
        v.camera.zoom = 0.18
        v._center_on_player(1.0)
        # Más luces: una rejilla por todo el mapa, también sobre los bordes de los chunks
        g = v.renderer.geo
        xs, zs = np.meshgrid(np.arange(4, 256, spacing), np.arange(4, 256, spacing))
        extra = np.zeros((xs.size, 8), np.float32)
        extra[:, 0], extra[:, 2] = xs.ravel(), zs.ravel()
        extra[:, 1] = v.sim.physics.support(extra[:, 0], extra[:, 2], np.full(xs.size, 99.0), search=40) + 2.0
        extra[:, 3:6], extra[:, 6], extra[:, 7] = (1.0, 0.7, 0.4), 7.0, 1.2
        v.renderer.world_lights = np.concatenate([v.renderer.world_lights, extra]).astype(np.float32)

        def shot(per_chunk):
            v.renderer.per_chunk_lights = per_chunk
            v.screenshot(tmp_path / "s.png")
            return np.asarray(Image.open(tmp_path / "s.png").convert("RGB")).astype(int)
        per_chunk, everything = shot(True), shot(False)
        assert len(v.renderer.frame_lights) > 32                 # más que el antiguo tope
        entries = list(v.renderer.chunk_lights.values())
        if spacing >= 10:
            assert all(e is not None and e[0] <= MAX_CHUNK_LIGHTS for e in entries)
        else:
            assert any(e is None for e in entries)                # se prueba la reserva
        assert np.abs(per_chunk - everything).max() <= 2
        assert np.abs(per_chunk - shot_without(v)).mean() > 1      # las luces de verdad alumbran
        del g
    finally:
        v.close()


# --- Niveles reducidos del atlas (mipmaps) ------------------------------------------------

def test_mip_levels_average_the_upward_opaque_texels():
    from engine.render.world_renderer import mip_chains
    color = np.zeros((4, 4, 4), np.uint8)
    normal = np.zeros((4, 4, 4), np.uint8)
    up, south = (128, 255, 128), (128, 128, 255)
    color[0, 0], normal[0, 0, :3] = (200, 100, 0, 255), up        # cara superior
    color[0, 1], normal[0, 1, :3] = (100, 50, 50, 255), up        # cara superior
    color[1, 1], normal[1, 1, :3] = (10, 10, 10, 255), south      # cara sur (suele quedar tapada)
    color[1, 0] = (255, 255, 255, 0)                              # transparente: no cuenta
    normal[..., 3] = color[..., 3]
    chains = mip_chains({"color": color, "normal": normal}, 2)
    assert [lv.shape[:2] for lv in chains["color"]] == [(4, 4), (2, 2), (1, 1)]
    assert tuple(chains["color"][1][0, 0]) == (150, 75, 25, 255)   # sin la cara sur
    assert chains["color"][1][1, 1, 3] == 0                        # bloque sin nada opaco
    only = mip_chains({"color": color}, 1)["color"][1][0, 0]       # sin normales: todos los opacos
    assert tuple(only[:3]) == (103, 53, 20)


def test_atlas_regions_are_aligned_with_wide_borders(terrain):
    from engine.compiler.compile import ATLAS_BORDER
    rows = SQLiteWorldStore(terrain).atlas_regions()
    assert ATLAS_BORDER == 16
    assert all(b == ATLAS_BORDER and x % ATLAS_BORDER == 0 and y % ATLAS_BORDER == 0 for _r, _p, x, y, _w, _h, b in rows)


def _gl_or_skip():
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("sin pantalla para OpenGL")
    glfw.window_hint(glfw.VISIBLE, 0)
    probe = glfw.create_window(16, 16, "sonda", None, None)
    if not probe:
        glfw.terminate()
        pytest.skip("no se puede crear una ventana OpenGL")
    glfw.destroy_window(probe)


@pytest.mark.parametrize("zoom", [1.0, 0.3, 0.1])
def test_mipmaps_never_change_the_depth(terrain, tmp_path, zoom):
    """La posición se lee siempre del atlas completo: con o sin niveles reducidos, la profundidad
    (y la silueta) de cada píxel es idéntica; solo cambia el color, y solo de lejos."""
    _gl_or_skip()
    from PIL import Image

    from engine.viewer import Viewer
    v = Viewer(terrain, 480, 320, visible=False, spawn=(64, 64), npcs=0)
    try:
        v.camera.zoom = zoom
        v._center_on_player(1.0)
        assert v.renderer.atlas.levels == 4

        def shot(mips, depth):
            v.renderer.use_mipmaps, v.renderer.show_depth = mips, depth
            v.screenshot(tmp_path / "s.png")
            return np.asarray(Image.open(tmp_path / "s.png").convert("RGB")).astype(int)
        assert np.array_equal(shot(True, True), shot(False, True))
        with_mips, without = shot(True, False), shot(False, False)
        colour = np.abs(with_mips - without).mean()
        assert colour == 0 if zoom >= 0.5 else colour < 3
        # Ninguna fila de pantalla se oscurece de forma sistemática (trazos a lo largo de las filas
        # de tiles, por meter en las aristas el color de caras que quedan tapadas)
        rows = (with_mips.sum(-1) - without.sum(-1)).mean(axis=1)
        assert rows.min() > -6
    finally:
        v.close()


def test_worlds_with_narrow_borders_load_without_mipmaps(tmp_path, shared_tileset, monkeypatch):
    _gl_or_skip()
    import engine.compiler.compile as comp
    monkeypatch.setattr(comp, "ATLAS_BORDER", 1)
    build_terrain(tmp_path / "t.tmx", shared_tileset, 32, 32, houses=1, seed=2)
    assert compile_tmx(tmp_path / "t.tmx", tmp_path / "t.cworld").ok
    from engine.viewer import Viewer
    v = Viewer(tmp_path / "t.cworld", 320, 200, visible=False, spawn=(16, 16), npcs=0)
    try:
        assert v.renderer.atlas.levels == 0
        v.camera.zoom = 0.1
        assert v.renderer.lod(v.camera) == 0
        v.frame()
    finally:
        v.close()
