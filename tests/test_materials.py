"""Materiales del suelo: orden de peor a mejor, el peor manda en cada celda, marcadores sin huecos."""
import json

import numpy as np
import pytest

from engine.compiler import compile_tmx
from engine.world import cell as C
from engine.world.materials import Materials, load_materials, material_of
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.render import DEFAULT_PROJECTION
from tools.pieces.terrain import GridBuilder, build_terrain


def test_materials_are_ordered_worst_to_best():
    m = load_materials()
    assert m.names[0] == "barro" and m.names[-1] == "adoquin"
    assert m.code("barro") == 1 and m.code("camino") > m.code("hierba")      # mejor = código mayor
    assert m.costs[m.code("camino")] < m.costs[m.code("hierba")] < m.costs[m.code("barro")]
    assert Materials.from_json(m.to_json()).names == m.names
    with pytest.raises(KeyError):
        m.code("lava")
    with pytest.raises(ValueError):
        Materials.from_json(json.dumps({"materiales": [{"name": f"m{i}", "cost": 1} for i in range(16)]}))
    with pytest.raises(ValueError):
        Materials.from_json(json.dumps({"materiales": [{"name": "a", "cost": 1}, {"name": "a", "cost": 2}]}))


def _small_map(tmp_path, tsx, extra):
    b = GridBuilder(4, 4)
    for y in range(4):
        for x in range(4):
            b.put(0, x, y, "floor_flat_000")
            b.put(0, x, y, "material_hierba")
    extra(b)
    tmx = tmp_path / "m.tmx"
    b.to_map(tsx, tmx, DEFAULT_PROJECTION).save(tmx)
    return tmx


def test_the_worst_material_wins_and_markers_take_no_slots(tmp_path, shared_tileset):
    def extra(b):
        b.put(0, 1, 1, "material_barro")          # hierba + barro (con transparencia) -> barro
        b.put(0, 2, 2, "material_camino")         # hierba + camino -> hierba (el peor)
        b.put(1, 3, 3, "material_camino")         # marcador sin suelo: no crea nada
        for wall in ("wall_N", "wall_W", "wall_S"):   # 4 tiles reales + 2 marcadores en (0, 0)
            b.put(0, 0, 0, wall)
    tmx = _small_map(tmp_path, shared_tileset, extra)
    report = compile_tmx(tmx, tmp_path / "m.cworld")
    assert report.ok and not report.warnings, report.summary()
    store = SQLiteWorldStore(tmp_path / "m.cworld")
    mats = Materials.from_json(store.meta()["materials"])
    view = WorldView(store)
    ys, xs = np.mgrid[0:4, 0:4]
    got = material_of(view.cells(xs, ys, 0))
    assert mats.name(int(got[1, 1])) == "barro" and mats.name(int(got[2, 2])) == "hierba"
    assert mats.name(int(got[3, 0])) == "hierba"
    assert int(view.cells(3, 3, 1)) == 0
    # (0, 0): suelo y paredes N y O intactos pese a los marcadores
    v = int(view.cells(0, 0, 0))
    assert C.wall_n(v) and C.wall_w(v) and C.floor_code(v, 1) != C.FLOOR_NONE
    assert report.stats["marcadores_de_material"] == 4 * 4 + 3


def test_unknown_material_is_an_error(tmp_path, shared_tileset):
    tmx = _small_map(tmp_path, shared_tileset, lambda b: None)
    data = json.loads(load_materials().to_json())
    data["materiales"] = [m for m in data["materiales"] if m["name"] != "hierba"]
    report = compile_tmx(tmx, tmp_path / "m.cworld", materials=Materials.from_json(json.dumps(data)))
    assert not report.ok and any("hierba" in e for e in report.errors)


def test_generated_terrain_has_roads_grass_and_house_floors(tmp_path, shared_tileset):
    info = build_terrain(tmp_path / "t.tmx", shared_tileset, 96, 96, houses=4, seed=3)
    assert compile_tmx(tmp_path / "t.tmx", tmp_path / "t.cworld", chunk_size=16).ok
    store = SQLiteWorldStore(tmp_path / "t.cworld")
    mats = Materials.from_json(store.meta()["materials"])
    view = WorldView(store)

    def top_material(x, y):
        start, length = view.column(x, y)
        for lv in range(start + length - 1, start - 1, -1):
            code = int(material_of(view.cells(x, y, lv)))
            if code:
                return mats.name(code)
        return None
    road = sorted(info.road)[:: max(1, len(info.road) // 40)]
    assert road and all(top_material(x, y) == "camino" for x, y in road)
    X, Y, w, d = info.plots[0]
    assert top_material(X + 2, Y + 2) == "suelo"
    rng = np.random.default_rng(0)
    inside = {(cx, cy) for X, Y, w, d in info.plots for cx in range(X, X + w) for cy in range(Y, Y + d)}
    grass = [(int(x), int(y)) for x, y in rng.integers(0, 96, (60, 2))
             if (int(x), int(y)) not in info.road and (int(x), int(y)) not in inside]
    assert all(top_material(x, y) == "hierba" for x, y in grass)


def test_material_tint_only_when_enabled(tmp_path, shared_tileset):
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

    from engine.viewer import Viewer
    tmx = _small_map(tmp_path, shared_tileset, lambda b: b.put(0, 1, 1, "material_barro"))
    assert compile_tmx(tmx, tmp_path / "m.cworld").ok
    v = Viewer(tmp_path / "m.cworld", 320, 240, visible=False, player=False)
    try:
        assert v.renderer.materials is not None
        shots = []
        for tint in (False, True):
            v.renderer.material_tint = tint
            v.screenshot(tmp_path / "s.png")
            shots.append(np.asarray(Image.open(tmp_path / "s.png").convert("RGB")).astype(int))
        assert np.abs(shots[0] - shots[1]).mean() > 3
    finally:
        v.close()
