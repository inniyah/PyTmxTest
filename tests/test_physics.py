"""Física de andar sobre el mapa de recorrido compilado."""
import numpy as np
import pytest

from engine.compiler import compile_tmx
from engine.physics import WalkerPhysics, Walkers
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.course import build_course

CW, DT = 2.0, 1 / 120


@pytest.fixture(scope="module")
def phys(tmp_path_factory, shared_tileset):
    out = tmp_path_factory.mktemp("phys")
    tsx = shared_tileset
    build_course(out / "course.tmx", tsx)
    assert compile_tmx(out / "course.tmx", out / "course.cworld").ok
    return WalkerPhysics(WorldView(SQLiteWorldStore(out / "course.cworld")))


def walker(cx, cy, y=0.0, **kw):
    w = Walkers.create([[(cx + 0.5) * CW, y, (cy + 0.5) * CW]], **kw)
    w.grounded[:] = True
    return w


def run(phys, w, wish, seconds, jump_at=None):
    track = []
    for i in range(int(seconds / DT)):
        w.wish[:] = wish
        if jump_at is not None and i == jump_at:
            w.jump[:] = True
        phys.step(w, DT)
        track.append(w.pos[0].copy())
    return np.array(track)


def test_standing_still_on_flat_ground(phys):
    w = walker(10, 10)
    run(phys, w, (0, 0), 0.5)
    assert np.allclose(w.pos[0], (21, 0, 21)) and w.grounded[0]


def test_thin_wall_stops_at_one_radius(phys):
    w = walker(14, 5)                                   # dentro de la habitación
    run(phys, w, (0, -1), 2)                            # hacia la pared norte (z = 6)
    assert abs(w.pos[0, 2] - (6 + 0.25)) < 0.01 and abs(w.pos[0, 0] - 29) < 1e-6


def test_corner_of_the_room(phys):
    w = walker(14, 5)
    run(phys, w, (-1, -1), 3)
    assert np.allclose(w.pos[0, [0, 2]], (24.25, 6.25), atol=0.01)


def test_slides_along_the_diagonal_wall(phys):
    w = walker(15, 13)                                  # al SE de la diagonal X + Z = 54
    start = w.pos[0].copy()
    track = run(phys, w, (-1, 0), 2.2)                  # empujando hacia el oeste
    on_wall = track[:, 0] > 24 + 0.25                   # el muro acaba en X = 24
    assert (track[on_wall, 0] + track[on_wall, 2] >= 54 + 0.25 * np.sqrt(2) - 0.01).all()   # no lo atraviesa
    assert w.pos[0, 2] > start[2] + 2                                           # resbala hacia el SO
    # Ya pegado al muro: se mueve en la dirección de la diagonal (-1, 1) a 4·cos 45° ≈ 2.83 m/s
    v = (track[-1] - track[-31])[[0, 2]] / (30 * DT)
    assert np.allclose(v / np.linalg.norm(v), [-np.sqrt(0.5), np.sqrt(0.5)], atol=0.02)
    assert abs(np.linalg.norm(v) - 4 * np.sqrt(0.5)) < 0.1


def test_climbs_the_stairs_to_the_platform(phys):
    w = walker(5, 11)
    track = run(phys, w, (0, -1), 3)
    assert w.grounded[0] and np.isclose(w.pos[0, 1], 3.0) and w.pos[0, 2] < 16
    assert (np.diff(track[:, 1]) >= -1e-9).all()        # sube sin bajones


def test_falls_off_the_cliff(phys):
    w = walker(4, 5, y=3.0)                             # sobre la plataforma
    track = run(phys, w, (-1, 0), 2)
    assert w.grounded[0] and w.pos[0, 1] == 0.0 and w.pos[0, 0] < 6
    assert track[:, 1].min() >= 0 and (track[:, 1] < 3).any()


def test_tunnel_blocks_a_person_but_not_a_dog(phys):
    person = walker(1, 13)
    run(phys, person, (1, 0), 3)
    assert person.pos[0, 0] < 6                         # no entra: techo a 1.5 m
    dog = walker(1, 13, radius=0.2, height=0.6)
    run(phys, dog, (1, 0), 6)
    assert dog.pos[0, 0] > 18                           # atraviesa el túnel entero


@pytest.mark.parametrize("x", [40.3, 42.0, 44.0, 46.0, 47.7])
def test_wide_ramp_can_be_entered_anywhere_along_its_low_edge(phys, x):
    w = Walkers.create([[x, 0.0, 25.0]])
    w.grounded[:] = True
    track = run(phys, w, (0, -1), 2.5)
    assert w.grounded[0] and np.isclose(w.pos[0, 1], 3.0)          # en la terraza (planta 1)
    assert (np.diff(track[:, 1]) >= -1e-9).all()


@pytest.fixture(scope="module")
def slope(tmp_path_factory, shared_tileset):
    """Un campo de 3 × 3 rampas diagonales hacia el NE sobre suelo plano (bloques debajo)."""
    from tools.pieces.catalog import build_catalog
    from tools.pieces.course import CourseBuilder
    out = tmp_path_factory.mktemp("slope")
    tsx = shared_tileset
    b = CourseBuilder(10, 10, build_catalog())
    field = {(3 + i, 6 - j): 0.25 * (i + j) for i in range(3) for j in range(3)}
    for y in range(10):
        for x in range(10):
            if (x, y) not in field:
                b.put(0, x, y, "floor_flat_000")
    for (x, y), h in field.items():
        level, base = int(h), h - int(h)
        for lv in range(level):
            b.put(lv, x, y, "fill_solid")
        b.put(level, x, y, f"floor_ramp_NE_{int(round(base * 100)):03d}")
    m = b.to_map(tsx, out / "slope.tmx", __import__("tools.pieces.render", fromlist=["x"]).DEFAULT_PROJECTION)
    m.save(out / "slope.tmx")
    assert compile_tmx(out / "slope.tmx", out / "slope.cworld").ok
    return WalkerPhysics(WorldView(SQLiteWorldStore(out / "slope.cworld")))


def test_diagonal_ramps_climbed_along_the_diagonal(slope):
    w = Walkers.create([[5.0, 0.0, 15.0]])            # al SO de la esquina baja (6, 14)
    w.grounded[:] = True
    track = run(slope, w, (1, -1), 1.5)
    assert track[:, 1].max() > 2.0                      # sube por el campo (¼ de planta por celda)
    on = (track[:, 0] > 6.3) & (track[:, 0] < 11.7) & (track[:, 2] > 8.3) & (track[:, 2] < 13.7)
    x, z = track[on, 0], track[on, 2]
    expected = 0.25 * ((x - 6) / CW + (14 - z) / CW) * 3.0       # el plano del campo, en metros
    assert np.allclose(track[on, 1], expected, atol=0.02)


def test_diagonal_ramps_have_too_high_edges_from_the_side(slope):
    w = Walkers.create([[9.0, 0.0, 17.0]])            # al sur del centro del borde sur
    w.grounded[:] = True
    run(slope, w, (0, -1), 1.5)
    assert w.pos[0, 1] == 0.0 and w.pos[0, 2] >= 14 + 0.0  # el borde sube 0.75 m: no se entra


def test_jump_and_land(phys):
    w = walker(10, 10)
    track = run(phys, w, (0, 0), 1.5, jump_at=0)
    assert track[:, 1].max() > 0.8                      # v²/2g = 6.5² / 40 ≈ 1.06 m
    assert w.grounded[0] and w.pos[0, 1] == 0.0


def test_tower_blocks(phys):
    w = walker(22, 5)
    run(phys, w, (0, -1), 2)
    assert abs(w.pos[0, 2] - (8 + 0.25)) < 0.01


def test_many_walkers_at_once(phys):
    rng = np.random.default_rng(0)
    cells = [(x, y) for x in range(0, 26) for y in (0, 1, 15, 16)]
    pos = [[(x + 0.5) * CW, 0.0, (y + 0.5) * CW] for x, y in cells]
    w = Walkers.create(pos)
    w.grounded[:] = True
    for _ in range(120):
        w.wish[:] = rng.uniform(-1, 1, (len(pos), 2))
        phys.step(w, DT)
    inside = (w.pos[:, 0] > 0.25 - 1e-6) & (w.pos[:, 0] < 52 - 0.25 + 1e-6)
    assert inside.all() and (w.pos[:, 1] == 0).all() and w.grounded.all()


def test_jumping_while_walking_up_the_ramp(phys):
    w = Walkers.create([[44.0, 0.75, 23.0]])
    w.grounded[:] = True
    run(phys, w, (0, -1), 0.1)
    track = run(phys, w, (0, -1), 1.8, jump_at=0)        # la terraza mide 8 m de fondo
    on_ramp = (track[:, 2] > 20) & (track[:, 2] < 24)
    surface = np.clip((24 - track[:, 2]) * 0.75, 0, 3)
    assert (track[on_ramp, 1] >= surface[on_ramp] - 0.05).all()     # nunca dentro (margen del aire)
    assert w.grounded[0] and np.isclose(w.pos[0, 1], 3.0)            # y acaba en la terraza


def _floor_intrusion(phys, x, z, y, h, margin, radius_fraction=0.9, samples=24):
    """Máxima altura de un volumen de suelo por encima de los pies + margen, en el contorno."""
    from engine.world import cell as C
    view, ch = phys.world, phys.p.cell_h
    worst = np.full(len(x), -9.0)
    for a in np.radians(np.arange(samples) * 360 / samples):
        ex, ez = x + 0.25 * radius_fraction * np.cos(a), z + 0.25 * radius_fraction * np.sin(a)
        cx, cy = np.floor(ex / CW).astype(int), np.floor(ez / CW).astype(int)
        u, v = ex / CW - cx, ez / CW - cy
        for dl in range(-1, 3):
            lv = np.floor(y / ch).astype(int) + dl
            top = (lv + C.surface_at(view.cells(cx, cy, lv), u, v)) * ch
            inside = np.isfinite(top) & (lv * ch < y + h - 1e-6)
            worst = np.maximum(worst, np.where(inside, top - (y + margin), -9))
    return worst


def test_stress_random_walking_and_jumping_never_enters_the_ground(phys):
    """200 personajes al azar 10 s: ni el centro ni el contorno se meten en suelos o rampas."""
    rng = np.random.default_rng(42)
    free = [(x, y) for x in range(26) for y in range(17)
            if not (3 <= x <= 8 and 3 <= y <= 7) and not (19 <= x <= 24 and 6 <= y <= 9)
            and not (22 <= x <= 23 and 2 <= y <= 3) and (x, y) not in {(12 + i, 14 - i) for i in range(6)}]
    picks = rng.choice(len(free), 200)
    pos = np.array([[(free[i][0] + 0.5) * CW, 0.0, (free[i][1] + 0.5) * CW] for i in picks])
    pos[:, 1] = phys.support(pos[:, 0], pos[:, 2], np.full(len(pos), 20.0), search=8)   # apoyados en su celda
    w = Walkers.create(pos)
    w.grounded[:] = True
    wish = rng.uniform(-1, 1, (200, 2))
    worst_center, worst_edge = 0.0, -9.0
    for step in range(600):
        if step % 30 == 0:
            wish = rng.uniform(-1, 1, (200, 2))
        w.wish[:] = wish
        w.jump[:] = rng.random(200) < 0.02
        phys.step(w, 1 / 60)
        x, y, z = w.pos.T
        worst_center = max(worst_center, float((phys.support(x, z, y + w.height) - y).max()))
        if step % 5 == 0:
            # Margen = el escalón también en el aire: al descender, lo que queda a menos de un
            # escalón bajo el borde es algo sobre lo que se aterriza; más que eso, se sale
            worst_edge = max(worst_edge, float(_floor_intrusion(phys, x, z, y, w.height, w.step).max()))
        assert (x > 0.24).all() and (x < 52 - 0.24).all() and (z > 0.24).all() and (z < 34 - 0.24).all()
    assert worst_center < 0.06, f"en el centro, una superficie llegó a {worst_center:.3f} m sobre los pies"
    # En el contorno (a 0.9 r): solo la imprecisión de comprobar 8 puntos sobre un canto inclinado
    assert worst_edge < 0.10, f"en el contorno, un suelo llegó a {worst_edge:.3f} m sobre el margen"
    for _ in range(90):                                    # sin saltar: nadie se queda en el aire
        w.wish[:] = wish
        phys.step(w, 1 / 60)
    assert w.grounded.all()


def test_body_stops_at_one_radius_from_floor_edges(phys):
    """El lateral de la rampa y la entrada del túnel paran el cilindro a un radio, no en su centro."""
    for start, edge in (([38.5, 0.0, 22.5], 40.0), ([3.0, 0.0, 27.0], 6.0)):
        w = Walkers.create([start])
        w.grounded[:] = True
        run(phys, w, (1, 0), 2)
        assert 0.24 < edge - w.pos[0, 0] < 0.27
