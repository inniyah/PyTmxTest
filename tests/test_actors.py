"""Personajes generados (tools.actors) y hojas con disposición, normales y posición por píxel."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from engine.render.sprites import SpriteRenderer, SpriteSheet, facing_indices, read_layout
from tools.actors.body import standing_height
from tools.actors.poses import POSES
from tools.actors.render import render_frame
from tools.actors.sheet import build_sheet, write_sheet
from tools.pieces.render import DEFAULT_PROJECTION as P

ASSETS_SHEET = "actors/sheets/maniqui.png"


@pytest.fixture(scope="module")
def idle_south():
    return render_frame(POSES["quieto"][0], 90, 1.7, standing_height())


def test_every_pixel_is_the_projection_of_its_point(idle_south):
    """Cada píxel tocado es la proyección de su punto 3D con la cámara del motor: el sprite
    encaja con los tiles en tamaño, perspectiva y profundidad."""
    f = idle_south
    ys, xs = np.where(f.hit)
    pts = f.point[f.hit]
    sx = 48 + pts[:, 0] * P.sx                                  # anclaje por defecto (48, 124)
    sy = 124 - (pts[:, 1] * P.sy - pts[:, 2] * P.sz)
    assert np.abs(sx - (xs + 0.5)).max() < 1e-6
    assert np.abs(sy - (ys + 0.5)).max() < 1e-6


def test_feet_on_the_ground_and_the_height_in_metres(idle_south):
    """Desde arriba no se ven ni la planta del pie ni lo más alto de la cabeza, pero casi."""
    from tools.actors.body import HEAD_TOP, build
    _, low, _ = build(POSES["quieto"][0])
    assert np.isclose(HEAD_TOP - low, standing_height())
    y = idle_south.point[idle_south.hit][:, 1]
    assert 0.0 <= y.min() < 0.06                                # el empeine, sobre el suelo
    assert 1.6 < y.max() < 1.71                                 # cerca de lo alto de la cabeza


def test_eyes_only_when_looking_towards_the_camera():
    model_h = standing_height()
    assert len(render_frame(POSES["quieto"][0], 90, 1.7, model_h).eyes) == 2      # sur
    assert render_frame(POSES["quieto"][0], 270, 1.7, model_h).eyes == []         # norte


def test_facing_east_puts_the_front_to_the_east():
    """Mirando al este, la pierna que avanza (la derecha en la primera pose de andar) está al
    este de los pies."""
    f = render_frame(POSES["andar"][0], 0, 1.7, standing_height())
    low = f.hit & (f.point[..., 1] < 0.15)
    assert f.point[low][:, 0].max() > 0.15


@pytest.fixture(scope="module")
def small_sheet(tmp_path_factory):
    desc = {"altura": 1.2, "materiales": {"m": {"luz": "#ffffff", "sombra": "#888888"}},
            "contorno": "#000000", "ojos": "#000000",
            "partes": dict.fromkeys(("cabeza", "oreja_d", "oreja_i", "torso", "brazo_d", "brazo_i",
                                     "pierna_d", "pierna_i"), "m"), "animaciones": []}
    out = tmp_path_factory.mktemp("hoja")
    layers, layout = build_sheet(desc)
    write_sheet(layers, layout, out, "mini")
    return out / "mini.png", layout


def test_sheet_files_and_layout(small_sheet):
    path, layout = small_sheet
    for suffix in (".png", ".a.png", ".n.png", ".p.png", ".json"):
        assert path.with_name("mini" + suffix).exists()
    w, h = layout["frame"]
    assert Image.open(path).size == (w * layout["cols"], h * 8)
    read = read_layout(path)
    assert read["directions"] == 8 and read["direction_rows"] == tuple(range(8))
    assert read["anchor"] == tuple(layout["anchor"]) and read["model_height"] == 1.2
    assert read["idle_col"] == 0 and "walk_cycle" not in read
    assert read["animations"] == {"quieto": ((0,), 8.0, True)}
    sheet = SpriteSheet(texture=0, width=w * layout["cols"], height=h * 8, **read)
    assert sheet.frame_size == (w, h)


def test_position_map_round_trip(small_sheet):
    """El mapa de posición de 16 bits devuelve el punto de cada píxel con precisión de mm."""
    from engine import png16
    path, layout = small_sheet
    pos = png16.read_rgba16(path.with_name("mini.p.png")).astype(float)
    lo, hi = np.array(layout["position_range"]).T
    opaque = pos[..., 3] > 0
    pts = lo + pos[..., :3][opaque] / 65535 * (hi - lo)
    assert pts[:, 1].min() > -0.05 and pts[:, 1].max() < 1.3       # entre los pies y la cabeza
    assert (hi - lo).max() / 65535 < 1e-4


def test_sheet_without_json_keeps_the_old_layout(tmp_path):
    Image.new("RGBA", (64, 64)).save(tmp_path / "vieja.png")
    assert read_layout(tmp_path / "vieja.png") == {}
    sheet = SpriteSheet(texture=0, width=64, height=64)
    assert list(sheet.row_of([0, 1, 2, 3])) == [2, 0, 1, 3]          # este, sur, oeste, norte


def test_walk_or_run_by_speed():
    sheet = SpriteSheet(texture=0, width=9, height=8, cols=9, rows=8, directions=8,
                        direction_rows=tuple(range(8)), walk_cycle=(1, 2, 3, 4), walk_fps=8.0,
                        walk_speed=1.4, run_cycle=(5, 6, 7, 8), run_fps=10.0, run_speed=4.0)
    t = np.array([0.0, 0.01, 0.01, 0.01])
    assert list(sheet.frame_cols(t, [0.0, 1.4, 2.5, 4.0])) == [0, 1, 1, 5]   # quieto, andar, andar, correr
    assert sheet.frame_cols([0.26], [1.4])[0] == 3                           # 0.26 s a 8 fps: tercer paso
    assert list(sheet.row_of([0, 2, 6])) == [0, 2, 6]


def test_facing_with_four_and_eight_directions_at_once():
    """Cada personaje con las direcciones de su hoja: en diagonal, el de 8 cambia y el de 4 no."""
    f = facing_indices([1.0, 1.0], [1.0, 1.0], [0, 0], np.array([4, 8]))
    assert list(f) == [0, 1]                                    # 4: sigue al este; 8: sureste


def test_sprite_quad_at_the_generated_scale(small_sheet):
    """Con la misma proyección, un píxel de la hoja es un píxel de mundo, y los pies caen en
    su píxel; un personaje más alto que el modelo se escala en proporción."""
    path, layout = small_sheet
    w, h = layout["frame"]
    ax, ay = layout["anchor"]
    sheet = SpriteSheet(texture=0, width=w * layout["cols"], height=h * 8, **read_layout(path))
    geo = SimpleNamespace(tile_w=P.tile_w, cell_w=P.cell.width, sz=P.sz, sy=P.sy, cell_h=P.cell.height)
    renderer = SimpleNamespace(geo=geo)
    feet = [(2.0, 0.0, 3.0), (2.0, 0.0, 3.0)]
    v = SpriteRenderer.sprite_vertices(renderer, sheet, feet, [1.2, 2.4], [0, 0], [0, 0],
                                       np.ones((2, 3)), [0, 0]).reshape(2, 6, -1)
    x, s = 2.0 * P.sx, 3.0 * P.sz
    left, top = v[0, 0, 0], v[0, 0, 1]
    assert np.isclose(v[0, 1, 0] - left, w) and np.isclose(v[0, 2, 1] - top, h)
    assert np.isclose(x - left, ax) and np.isclose(s - top, ay)
    assert np.isclose(v[1, 1, 0] - v[1, 0, 0], 2 * w)                        # el doble de alto
    assert np.allclose(v[0, :, 11:14], (0.0, 3.0, 1.0)) and np.allclose(v[1, :, 13], 2.0)


@pytest.mark.parametrize("name", ["maniqui", "soldado", "enemigo"])
def test_committed_sheets_are_up_to_date(name):
    """Cada hoja del repositorio corresponde a su descripción (altura, animaciones, superficie)
    y su tipo la usa."""
    from engine.entities import load_types
    from engine.viewer import ASSETS
    layout = json.loads((ASSETS / f"actors/sheets/{name}.json").read_text(encoding="utf-8"))
    desc = json.loads((ASSETS / f"actors/models/{name}.json").read_text(encoding="utf-8"))
    assert layout["height"] == desc["altura"]
    assert set(layout["animations"]) == {"quieto", *desc["animaciones"]}
    assert layout["projection"]["sx"] == P.sx and layout["projection"]["sy"] == P.sy
    from tools.actors.sheet import pitch_of
    assert layout["pitch"] == pitch_of(desc) and len(layout["columns"]) == layout["cols"]
    metal = any(m.get("metal", 0) > 0 for m in desc["materiales"].values())
    assert (ASSETS / f"actors/sheets/{name}.s.png").exists() == metal
    assert load_types()[name].sheet == f"actors/sheets/{name}.png"
    assert load_types()[name].height == desc["altura"]


# --- Soldado: proporciones, equipo, poses -----------------------------------------------------

def _soldier():
    from engine.viewer import ASSETS
    from tools.actors.body import Proportions
    from tools.actors.sheet import equipment_of
    desc = json.loads((ASSETS / "actors/models/soldado.json").read_text(encoding="utf-8"))
    return desc, Proportions.from_desc(desc), equipment_of(desc)


def test_soldier_proportions_and_equipment():
    from tools.actors.body import Proportions, build, pose
    desc, prop, eq = _soldier()
    names = [n for n, _ in build(pose(aim=True), prop, eq)[0]]
    assert {"casco", "mochila", "fusil", "mano_d", "bota_i"} <= set(names)
    assert standing_height(prop) < standing_height(Proportions()) * 1.05      # cabeza pequeña, piernas largas
    f = render_frame(pose(aim=True), 90, 1.75, standing_height(prop), (160, 176), (80, 132), prop=prop, equipment=eq)
    head = f.hit & (f.group == names.index("cabeza"))
    helmet = f.hit & (f.group == names.index("casco"))
    assert helmet.sum() > head.sum() * 0.5                                   # el casco cubre buena parte
    assert np.where(f.hit)[0].min() > 132 - 1.75 * P.sy - 0.4 * P.sz         # no más alto que su altura


def test_aiming_rifle_points_where_the_soldier_faces():
    """Con el fusil apuntando, la boca del cañón queda delante: mirando al este, al este."""
    from tools.actors.body import build, pose
    desc, prop, eq = _soldier()
    parts, _, _ = build(pose(aim=True), prop, eq)
    f = render_frame(pose(aim=True), 0, 1.75, standing_height(prop), (200, 176), (100, 132),
                     prop=prop, equipment=eq)
    rifle = f.hit & (f.group == [n for n, _ in parts].index("fusil"))
    assert rifle.any() and f.point[rifle][:, 0].max() > 0.55


def test_crouching_is_lower():
    desc, prop, eq = _soldier()
    h = standing_height(prop)
    tops = {}
    for name in ("quieto", "agachado"):
        f = render_frame(POSES[name][0], 90, 1.75, h,
                         (160, 176), (80, 132), prop=prop, equipment=eq)
        tops[name] = f.point[f.hit][:, 1].max()
    assert tops["agachado"] < 0.85 * tops["quieto"]                         # rodilla en tierra


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_dying_ends_lying_across_the_screen(angle):
    """Tendido al final de morir: bajo y alargado de este a oeste (de norte a sur, en esta
    cámara, parecería de pie)."""
    from tools.actors.poses import DIE
    desc, prop, eq = _soldier()
    f = render_frame(DIE[-1], angle, 1.75, standing_height(prop), (288, 176), (144, 132),
                     prop=prop, equipment=eq)
    pts = f.point[f.hit]
    assert pts[:, 1].max() < 0.5                                             # nada a más de medio metro
    assert np.ptp(pts[:, 0]) > 1.2 and np.ptp(pts[:, 0]) > 1.5 * np.ptp(pts[:, 2])


def test_named_animations_loop_or_hold():
    sheet = SpriteSheet(texture=0, width=1, height=1, animations={
        "andar": ((1, 2, 3, 4), 8.0, True), "morir": ((5, 6, 7, 8), 8.0, False)})
    assert list(sheet.anim_cols("andar", [0.0, 0.13, 0.5])) == [1, 2, 1]
    assert list(sheet.anim_cols("morir", [0.0, 0.13, 0.5, 9.0])) == [5, 6, 8, 8]       # se queda tendido
    assert sheet.anim_duration("morir") == 0.5


# --- En el motor ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def levels(tmp_path_factory, shared_tileset):
    from engine.compiler import compile_tmx
    from tools.pieces.levels import build_levels
    d = tmp_path_factory.mktemp("act")
    build_levels(d / "l.tmx", shared_tileset)
    assert compile_tmx(d / "l.tmx", d / "l.cworld", nav=False).ok
    return d / "l.cworld"


def _side_light(v, side):
    """Sustituye el farol por una luz al este (+1) o al oeste (-1) del jugador."""
    def update():
        X, Y, Z = v.player.pos
        v.renderer.dynamic_lights = [(X + 1.2 * side, Y + 1.0, Z + 0.3, 1.0, 1.0, 1.0, 4.0, 3.0)]
    v.update_lights = update


RELATIVE_CONTRAST = 0.06       # ganancia de cada mitad al pasar la luz a su lado, sobre el brillo medio
                               # (maniquí con ropa ≈ 0.09; plantilla plana ≈ 0.03)


@pytest.mark.parametrize("player_type, sheet, lit_by_side", [
    ("maniqui", None, True),
    ("persona", None, True),                       # el personaje de base: el maniquí
    ("persona", "people/guy.png", False),          # una plantilla plana, sin normales
])
def test_normals_light_the_side_facing_the_light(levels, tmp_path, player_type, sheet, lit_by_side):
    """De noche, con una luz a un lado: con normales por píxel, la mitad del personaje que da
    a la luz brilla más que la otra; con el sprite plano de siempre, apenas hay diferencia."""
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("sin pantalla para OpenGL")
    from engine.viewer import Viewer
    from tests.conftest import retrying_viewer
    v = retrying_viewer(Viewer)(levels, 400, 400, visible=False, spawn=(5.5, 12.5), spawn_level=0,
                                player_type=player_type)
    try:
        if sheet is not None:
            import dataclasses
            v.types[player_type] = dataclasses.replace(v.types[player_type], sheet=sheet)
        # Con los colores de la hoja (no los de su ropa al azar): el contraste no depende de ellos
        v.table.set_outfit(v.player_handle, **{part: None for part in v.table.outfit_names})
        v.set_night(True)
        v.silhouette = False
        v.camera.zoom = 3.0
        v._center_on_player(1.0)
        shots = {}
        for name, side in (("este", 1), ("oeste", -1)):
            _side_light(v, side)
            v.screenshot(tmp_path / f"{name}.png")
            shots[name] = np.asarray(Image.open(tmp_path / f"{name}.png").convert("RGB")).astype(float).sum(-1)
        feet = v.player.pos.copy()
        v.player.pos = v._spawn_point((9.5, 12.5), 0)          # fuera: queda el fondo
        _side_light(v, 1)
        v.update_lights = lambda: setattr(v.renderer, "dynamic_lights",
                                          [(feet[0] + 1.2, feet[1] + 1.0, feet[2] + 0.3, 1, 1, 1, 4.0, 3.0)])
        v.screenshot(tmp_path / "fondo.png")
        empty = np.asarray(Image.open(tmp_path / "fondo.png").convert("RGB")).astype(float).sum(-1)
    finally:
        v.close()
    body = np.abs(shots["este"] - empty) > 30
    ys, xs = np.where(body)
    assert len(xs) > 300
    body[int(np.percentile(ys, 85)):] = False                    # sin los pies ni la sombra
    mid = np.median(xs)
    east, west = body.copy(), body.copy()
    east[:, :int(mid)] = False
    west[:, int(mid):] = False
    gain_east = shots["este"][east].mean() - shots["oeste"][east].mean()   # la mitad este, al cambiar la luz
    gain_west = shots["oeste"][west].mean() - shots["este"][west].mean()
    # Relativo al brillo del cuerpo: no depende de lo oscura que sea la ropa
    level = (shots["este"][body].mean() + shots["oeste"][body].mean()) / 2
    contrast = min(gain_east, gain_west) / level
    if lit_by_side:
        assert contrast > RELATIVE_CONTRAST, f"{gain_east:.0f} {gain_west:.0f} / {level:.0f}"
    else:
        assert contrast < RELATIVE_CONTRAST, f"{gain_east:.0f} {gain_west:.0f} / {level:.0f}"


def _walk_and_count_silhouette(levels, tmp_path, player_type, walk):
    """Píxeles que cambian al quitar la silueta del jugador, tras andar ``walk``."""
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("sin pantalla para OpenGL")
    from engine.viewer import PHYSICS_HZ, Viewer
    from tests.conftest import retrying_viewer
    v = retrying_viewer(Viewer)(levels, 600, 420, visible=False, player_type=player_type)
    try:
        v.camera.zoom = 2.5
        for dx, dz, secs in walk:
            for _ in range(int(secs * PHYSICS_HZ)):
                v.update(1 / PHYSICS_HZ, wish=(dx, dz))
        v._center_on_player(1.0)
        v.screenshot(tmp_path / "con.png")
        v.silhouette = False
        v.screenshot(tmp_path / "sin.png")
    finally:
        v.close()
    a, b = (np.asarray(Image.open(tmp_path / f).convert("RGB")).astype(int) for f in ("con.png", "sin.png"))
    return int((np.abs(a - b).max(-1) > 8).sum())


@pytest.mark.parametrize("player_type, walk", [
    ("soldado", [(0, 1, 0.4), (-1, 0, 2.3), (-0.6, -1, 0.5)]),   # el fusil contra la pared de la rampa
    ("maniqui", [(-1, 0, 1.0), (-1, -0.2, 0.55)]),                # la mano y el pie contra la rampa
])
def test_touching_the_scenery_shows_no_silhouette(levels, tmp_path, player_type, walk):
    """Lo que sobresale del cilindro (el fusil, una mano) y roza o entra un poco en una pared
    no se ve azul: lo de detrás del cilindro se acerca hasta él y los roces tienen margen."""
    assert _walk_and_count_silhouette(levels, tmp_path, player_type, walk) == 0



# --- Cámara configurable y fotogramas por animación ----------------------------------------

def test_camera_projection_keeps_the_scale_and_sets_the_pitch():
    from tools.actors.sheet import camera_projection
    from tools.pieces.render import DEFAULT_PROJECTION as P
    for pitch in (20.0, 30.0, 46.0, 60.0):
        q = camera_projection(pitch)
        assert q.sx == pytest.approx(P.sx) and q.sy == pytest.approx(P.sy)       # un metro, lo mismo
        assert np.degrees(np.arctan2(q.sz, q.sy)) == pytest.approx(pitch)
    world = np.degrees(np.arctan2(P.sz, P.sy))                                   # la del mundo
    assert camera_projection(world).sz == pytest.approx(P.sz)


def test_pitch_from_the_command_line_the_model_or_the_default():
    from tools.actors.sheet import DEFAULT_PITCH, pitch_of
    assert pitch_of({}) == DEFAULT_PITCH == 30.0
    assert pitch_of({"camara": {"inclinacion": 38}}) == 38.0
    assert pitch_of({"camara": {"inclinacion": 38}}, 46) == 46.0


@pytest.fixture(scope="module")
def walk_and_die():
    from tools.actors.sheet import build_sheet
    from engine.viewer import ASSETS
    desc = json.loads((ASSETS / "actors/models/maniqui.json").read_text(encoding="utf-8"))
    desc["animaciones"] = ["andar", "morir"]               # el maniquí, más rápido de generar
    return build_sheet(desc, pitch=30)


def test_each_animation_is_cropped_to_its_frames(walk_and_die):
    layers, layout = walk_and_die
    cols = layout["columns"]
    assert len(cols) == layout["cols"] and layout["pitch"] == 30.0
    walk = [cols[c] for c in layout["animations"]["andar"]["cols"]]
    die = [cols[c] for c in layout["animations"]["morir"]["cols"]]
    assert all(c["w"] == walk[0]["w"] for c in walk) and die[0]["w"] > walk[0]["w"] + 40   # tendido, más ancho
    assert walk[0]["w"] <= 120 and walk[0]["h"] <= 110                                     # tamaño razonable
    assert layers["alpha"].shape == (8 * max(c["h"] for c in cols), sum(c["w"] for c in cols))
    x = 0
    for c in cols:                                       # contiguas, y los pies en su ancla
        assert c["x"] == x
        x += c["w"]
    a = layers["alpha"]
    for c in walk[:1] + die[:1]:
        frame = a[:c["h"], c["x"]:c["x"] + c["w"]]
        ys, xs = np.nonzero(frame)
        assert abs(ys.max() - c["anchor"][1]) <= 3                   # los pies, en su fila
        assert xs.min() < c["anchor"][0] < xs.max()


def test_sprite_quads_use_each_column_size_and_anchor(walk_and_die):
    _layers, layout = walk_and_die
    meta = dict(layout)
    cols = meta["columns"]
    read = {"columns": tuple((c["x"], c["w"], c["h"], *c["anchor"]) for c in cols),
            "anchor": tuple(cols[0]["anchor"]), "px_per_m": meta["px_per_m"], "model_height": meta["height"]}
    width, height = sum(c["w"] for c in cols), 8 * max(c["h"] for c in cols)
    sheet = SpriteSheet(texture=0, width=width, height=height, cols=meta["cols"], rows=8, directions=8,
                        direction_rows=tuple(range(8)), **read)
    r = SimpleNamespace(geo=SimpleNamespace(tile_w=P.tile_w, cell_w=P.cell.width, sz=P.sz, sy=P.sy,
                                            cell_h=P.cell.height))
    walk_col, die_col = layout["animations"]["andar"]["cols"][0], layout["animations"]["morir"]["cols"][-1]
    h = meta["height"]                                     # a la altura del modelo: escala 1
    q = SpriteRenderer.sprite_vertices(r, sheet, [[10.0, 0.0, 10.0]] * 2, [h, h], [3, 3], [walk_col, die_col],
                          np.ones((2, 3)), [0, 0]).reshape(2, 6, -1)
    for k, c in enumerate((cols[walk_col], cols[die_col])):
        xs, us, vs = q[k, :, 0], q[k, :, 2], q[k, :, 3]
        assert xs.max() - xs.min() == pytest.approx(c["w"], rel=1e-3)        # a la escala generada
        assert us.min() == pytest.approx(c["x"] / width) and us.max() == pytest.approx((c["x"] + c["w"]) / width)
        assert vs.min() == pytest.approx(3 * c["h"] / height)                 # fila 3, de su alto
        feet_x = 10.0 * r.geo.tile_w / r.geo.cell_w
        assert feet_x - xs.min() == pytest.approx(c["anchor"][0], rel=1e-3)   # los pies, en su ancla


# --- Pelo y ropa recoloreable ----------------------------------------------------------------

def _maniqui_layers():
    from engine.viewer import ASSETS
    sheets = ASSETS / "actors/sheets"
    layout = json.loads((sheets / "maniqui.json").read_text(encoding="utf-8"))
    albedo = np.asarray(Image.open(sheets / "maniqui.a.png").convert("RGBA"))
    color = np.asarray(Image.open(sheets / "maniqui.png").convert("RGBA"))
    return layout, albedo, color


def test_maniqui_sheet_marks_its_clothes_in_the_albedo_alpha():
    layout, albedo, color = _maniqui_layers()
    assert layout["ropa"] == ["zapatos", "pantalones", "chaleco", "mangas", "pelo"]
    values = set(np.unique(albedo[..., 3]).tolist())
    parts = {255 - k for k in range(1, 6)}
    assert values <= {0, 255} | parts and parts <= values          # las cinco partes, y nada más
    assert ((albedo[..., 3] > 0) == (color[..., 3] > 0)).all()       # la misma silueta que el color


def test_hair_covers_the_head_but_not_the_face():
    """De frente: lo alto de la cabeza es pelo, y debajo, en la cabeza, se ve la cara (piel)."""
    from engine.viewer import ASSETS
    desc = json.loads((ASSETS / "actors/models/maniqui.json").read_text(encoding="utf-8"))
    layout, albedo, _color = _maniqui_layers()
    col = layout["columns"][0]                                       # quieto
    south = layout["direction_rows"].index(2) if "direction_rows" in layout else 2
    frame = albedo[south * col["h"]:(south + 1) * col["h"], col["x"]:col["x"] + col["w"]]
    hair = frame[..., 3] == 250
    skin_rgb = [int(desc["materiales"]["piel"]["luz"][i:i + 2], 16) for i in (1, 3, 5)]
    skin = (frame[..., 3] == 255) & (frame[..., :3] == skin_rgb).all(-1)
    top = np.nonzero(frame[..., 3])[0].min()
    head = slice(top, top + 18)                                      # la cabeza, de frente
    assert hair[top:top + 4].any()                                   # lo alto de la cabeza, pelo
    face_rows = np.nonzero(skin[head].any(axis=1))[0] + top
    assert face_rows.size >= 4 and face_rows.min() > np.nonzero(hair)[0].min()   # la cara, bajo el pelo


def test_outfits_come_from_each_type_palette():
    from engine.entities import ActorTable, load_types
    from engine.entities.table import pack_rgb
    types = load_types()
    t = ActorTable(types, rng=np.random.default_rng(5))
    people = [t.spawn("persona", [0.0, 0.0, 0.0]) for _ in range(40)]
    k = t.outfit_index("chaleco")
    vests = {float(t.outfit[t.row(h), k]) for h in people}
    assert vests <= {pack_rgb(c) for c in types["persona"].ropa["chaleco"]} and len(vests) > 3   # variados
    guard = t.spawn("guardia", [0.0, 0.0, 0.0])
    assert t.outfit[t.row(guard), k] == pack_rgb("#2c3e66")                                     # uniforme
    t.set_outfit(guard, chaleco="#ff0000", pelo=None)
    assert t.outfit[t.row(guard), k] == pack_rgb("#ff0000") and t.outfit[t.row(guard), t.outfit_index("pelo")] == -1
    assert pack_rgb("#0a0b0c") == float(0x0A0B0C) and pack_rgb((1, 2, 3)) == float(0x010203)


def test_a_green_vest_paints_the_vest_green(levels, tmp_path):
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("sin pantalla para OpenGL")
    from engine.viewer import Viewer
    from tests.conftest import retrying_viewer
    v = retrying_viewer(Viewer)(levels, 400, 400, visible=False, spawn=(5.5, 12.5), spawn_level=0,
                                player_type="persona")
    try:
        v.silhouette = False
        v.camera.zoom = 3.0
        v._center_on_player(1.0)
        counts = {}
        for lighting in (True, False):
            v.renderer.lighting.enabled = lighting
            for vest in ("#00ff00", None):
                v.table.set_outfit(v.player_handle, chaleco=vest, mangas="#ffffff", pelo="#202020")
                v.screenshot(tmp_path / "v.png")
                img = np.asarray(Image.open(tmp_path / "v.png").convert("RGB")).astype(int)
                counts[lighting, vest] = int(((img[..., 1] > 120) & (img[..., 1] > img[..., 0] * 1.8)
                                              & (img[..., 1] > img[..., 2] * 1.8)).sum())
        for lighting in (True, False):                    # con y sin iluminación
            assert counts[lighting, "#00ff00"] > 300 and counts[lighting, None] < 20
    finally:
        v.close()
