"""Personajes: tipos, tabla con identificadores, contactos entre cilindros y multitudes."""
import json

import numpy as np
import pytest

from engine.compiler import compile_tmx
from engine.entities import BRAIN_WANDER, ActorTable, load_types
from engine.entities.world_sim import ActorSim
from engine.physics import WalkerPhysics, Walkers
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.course import build_course
from tools.pieces.levels import build_levels


@pytest.fixture(scope="module")
def worlds(tmp_path_factory, shared_tileset):
    out = tmp_path_factory.mktemp("crowd")
    tsx = shared_tileset
    build_course(out / "course.tmx", tsx)
    build_levels(out / "levels.tmx", tsx)
    for name in ("course", "levels"):
        assert compile_tmx(out / f"{name}.tmx", out / f"{name}.cworld").ok
    return {name: WorldView(SQLiteWorldStore(out / f"{name}.cworld")) for name in ("course", "levels")}


# --- Tipos y tabla ---------------------------------------------------------------------

def test_types_merge_defaults(tmp_path):
    types = load_types()
    assert {"persona", "niño", "perro", "guardia"} <= set(types)
    assert types["perro"].height == 0.6 and types["perro"].step == types["persona"].step   # heredado
    bad = tmp_path / "t.json"
    bad.write_text(json.dumps({"defecto": {}, "raro": {"alas": 2}}))
    with pytest.raises(ValueError):
        load_types(bad)


def test_handles_survive_reuse_of_rows():
    t = ActorTable(load_types(), capacity=2)
    a = t.spawn("persona", [0, 0, 0])
    b = t.spawn("perro", [1, 0, 1])
    c = t.spawn("guardia", [2, 0, 2])                        # crece la capacidad
    assert t.capacity >= 3 and len(t) == 3
    t.despawn(b)
    d = t.spawn("niño", [3, 0, 3])
    assert t.row(d) == (b & 0xFFFFFFFF)                      # misma fila…
    assert not t.valid(b) and t.valid(d) and t.valid(a) and t.valid(c)   # …otro identificador
    with pytest.raises(KeyError):
        t.get(b)
    actor = t.get(d)
    assert actor.type_name == "niño" and actor.mass == 30.0
    actor.pos = [5, 0, 5]
    assert np.allclose(t.pos[t.row(d)], [5, 0, 5])


# --- Contactos -------------------------------------------------------------------------

def _pair(phys, masses, wishes, start=(20.0, 24.0), seconds=2.0):
    w = Walkers.create([[start[0], 0, 21.0], [start[1], 0, 21.0]])
    w.mass[:] = masses
    w.grounded[:] = True
    for _ in range(int(seconds * 60)):
        w.wish[:] = wishes
        phys.step(w, 1 / 60)
    return w


def test_head_on_collision_stops_at_the_sum_of_radii(worlds):
    w = _pair(WalkerPhysics(worlds["course"]), [70, 70], [(1, 0), (-1, 0)])
    assert w.pos[1, 0] - w.pos[0, 0] == pytest.approx(0.5, abs=0.01)


def test_heavier_pushes_lighter(worlds):
    phys = WalkerPhysics(worlds["course"])
    light = _pair(phys, [140, 70], [(1, 0), (0, 0)], start=(20.0, 20.6), seconds=1.5)
    heavy = _pair(phys, [70, 140], [(1, 0), (0, 0)], start=(20.0, 20.6), seconds=1.5)
    assert light.pos[1, 0] - 20.6 > heavy.pos[1, 0] - 20.6 + 0.3     # el ligero cede más


def test_infinite_mass_does_not_move(worlds):
    w = _pair(WalkerPhysics(worlds["course"]), [70, np.inf], [(1, 0), (0, 0)], start=(20.0, 20.6))
    assert w.pos[1, 0] == pytest.approx(20.6) and w.pos[1, 0] - w.pos[0, 0] == pytest.approx(0.5, abs=0.01)


def test_cylinders_on_different_floors_do_not_touch(worlds):
    phys = WalkerPhysics(worlds["levels"])
    # Uno en la planta baja del edificio y otro justo encima, en la planta 1
    w = Walkers.create([[33.0, 0.0, 15.0], [33.1, 3.0, 15.0]])
    w.grounded[:] = True
    phys.step(w, 1 / 60)
    assert np.allclose(w.pos[:, [0, 2]], [[33.0, 15.0], [33.1, 15.0]])


def test_neighbour_pairs_match_brute_force():
    rng = np.random.default_rng(0)
    x, z = rng.uniform(0, 20, 300), rng.uniform(0, 20, 300)
    i, j = WalkerPhysics._neighbour_pairs(x, z, 1.0)
    found = {(a, b) for a, b in zip(i.tolist(), j.tolist())}
    d = np.hypot(x[:, None] - x[None], z[:, None] - z[None])
    close = {(a, b) for a, b in zip(*np.nonzero(d < 1.0)) if a < b}
    assert close <= found                                    # ningún par cercano se pierde


# --- Multitudes ------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["course", "levels"])
def test_crowd_wanders_without_overlaps_or_falls(worlds, name):
    view = worlds[name]
    table = ActorTable(load_types())
    rng = np.random.default_rng(5)
    sim = ActorSim(view, table, rng)
    points = sim.spawn_points(150, rng, levels=list(range(view.level_min, view.level_max + 1)))
    kinds = rng.choice(["persona", "niño", "perro", "guardia"], len(points), p=[0.55, 0.15, 0.2, 0.1])
    for p, kind in zip(points, kinds):
        table.spawn(kind, p, brain=BRAIN_WANDER)
    start = table.pos[table.rows()].copy()
    start_level = np.floor((start[:, 1] + 0.05) / 3)
    for _ in range(20):                                       # 20 s, comprobando cada segundo
        sim.step(1.0)
        rows = table.rows()
        x, y, z = table.pos[rows].T
        r, h = table.radius[rows], table.height[rows]
        body = sim.physics.support(x, z, y + h)
        assert (body - y < 0.06).all()                        # nadie dentro del suelo
        d = np.hypot(x[:, None] - x[None], z[:, None] - z[None]) - (r[:, None] + r[None])
        same_floor = (y[:, None] < y[None] + h[None]) & (y[None] < y[:, None] + h[:, None])
        np.fill_diagonal(same_floor, False)
        assert d[same_floor].min() > -0.05                    # sin solaparse (aglomeraciones contra paredes: 5 cm)
        assert (x > 0).all() and (z > 0).all()
    moved = np.hypot(x - start[:, 0], z - start[:, 2])
    assert np.median(moved) > 2.0                             # de verdad pasean
    end_level = np.floor((y + 0.05) / 3)
    assert (end_level == start_level).mean() > 0.9            # casi nadie se cae de su planta


def test_max_steps_slows_down_instead_of_spiralling(worlds):
    """Con un frame muy largo, la simulación da como mucho ``max_steps`` pasos y descarta el resto
    del tiempo (el juego va más lento un momento) en lugar de acumular cada vez más pasos."""
    table = ActorTable(load_types())
    sim = ActorSim(worlds["course"], table, np.random.default_rng(0))
    table.spawn("persona", [20.0, 0.0, 21.0], brain=BRAIN_WANDER)
    assert sim.step(1.0, max_steps=4) == 4
    assert sim.step(1 / 60, max_steps=4) == 1                  # no arrastra el tiempo descartado
    assert sim.step(0.5) == 30                                  # sin límite, como siempre
