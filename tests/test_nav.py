"""Grafo de navegación: lo que dice coincide con lo que la física deja hacer."""
import numpy as np
import pytest

from engine.compiler import compile_tmx
from engine.nav import DIRS, NavGraph
from engine.physics import WalkerPhysics, Walkers
from engine.world import cell as C
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.course import build_course
from tools.pieces.levels import build_levels
from tools.pieces.terrain import build_terrain


@pytest.fixture(scope="module")
def worlds(tmp_path_factory, shared_tileset):
    out = tmp_path_factory.mktemp("nav")
    build_course(out / "course.tmx", shared_tileset)
    build_levels(out / "levels.tmx", shared_tileset)
    build_terrain(out / "terrain.tmx", shared_tileset, 96, 96, houses=4, seed=3)
    result = {}
    for name in ("course", "levels", "terrain"):
        report = compile_tmx(out / f"{name}.tmx", out / f"{name}.cworld", chunk_size=16)
        assert report.ok and report.stats["nav_nodos"] > 0, report.summary()
        store = SQLiteWorldStore(out / f"{name}.cworld")
        view = WorldView(store)
        result[name] = (view, WalkerPhysics(view), NavGraph(store, view.chunk_size))
    return result


def _spaced(xs, ys, rng, count, gap=4):
    """Índices bien separados entre sí (para que los personajes de la prueba no se empujen)."""
    taken, occupied = [], set()
    for i in rng.permutation(len(xs)):
        key = (xs[i] // gap, ys[i] // gap)
        if key in occupied:
            continue
        occupied |= {(key[0] + a, key[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
        taken.append(i)
        if len(taken) >= count:
            break
    return np.array(taken, np.int64)


def _walk(physics, graph, a, b, seconds=2.5):
    start, goal = graph.position(a), graph.position(b)
    w = Walkers.create(start)
    w.grounded[:] = True
    for _ in range(int(seconds * 60)):
        d = goal[:, [0, 2]] - w.pos[:, [0, 2]]
        dist = np.hypot(d[:, 0], d[:, 1])
        w.wish[:] = d / np.maximum(dist, 1e-9)[:, None] * np.minimum(1.0, dist / 0.5)[:, None]
        physics.step(w, 1 / 60)
    dist = np.hypot(*(w.pos[:, [0, 2]] - goal[:, [0, 2]]).T)
    return (dist < 0.3) & (np.abs(w.pos[:, 1] - goal[:, 1]) < 0.15)


@pytest.mark.parametrize("name", ["course", "levels", "terrain"])
def test_every_edge_a_person_fits_through_can_be_walked(worlds, name):
    _view, physics, g = worlds[name]
    a_all, d_all = np.nonzero(g.usable(0.25))
    tested = 0
    for r in range(8):
        pick = _spaced(g.x[a_all], g.y[a_all], np.random.default_rng(r), 200)
        a, di = a_all[pick], d_all[pick]
        ok = _walk(physics, g, a, g.target[a, di])
        bad = [(int(g.x[a[i]]), int(g.y[a[i]]), tuple(DIRS[di[i]])) for i in np.nonzero(~ok)[0]]
        assert not bad, bad
        tested += len(ok)
    assert tested > 50


@pytest.mark.parametrize("name", ["course", "levels", "terrain"])
def test_where_there_is_no_edge_nobody_gets_through(worlds, name):
    """Vecinos ortogonales sin arista, sin caída (pared o desnivel hacia arriba): no se llega."""
    _view, physics, g = worlds[name]
    pairs = []
    for di in range(4):
        dx, dy = DIRS[di]
        none = np.nonzero(g.target[:, di] < 0)[0]
        lo, hi = g.columns.ranges(g.x[none] + dx, g.y[none] + dy)
        for j in range(int((hi - lo).max(initial=0))):
            has = lo + j < hi
            idx = g.columns.order[np.minimum(lo + j, len(g.columns.order) - 1)]
            sel = has & (g.h[idx] >= g.h[none] - 0.3) & (np.abs(g.h[idx] - g.h[none]) < 2.5)
            pairs += list(zip(none[sel], idx[sel]))
    pairs = np.array(pairs)
    assert len(pairs) > 0
    pick = _spaced(g.x[pairs[:, 0]], g.y[pairs[:, 0]], np.random.default_rng(9), 200)
    assert not _walk(physics, g, pairs[pick, 0], pairs[pick, 1]).any()


def test_edges_land_on_the_neighbour_column(worlds):
    for _view, _physics, g in worlds.values():
        a, di = np.nonzero(g.target >= 0)
        b = g.target[a, di]
        assert np.array_equal(g.x[b], g.x[a] + DIRS[di, 0]) and np.array_equal(g.y[b], g.y[a] + DIRS[di, 1])


def test_bridge_stairs_and_diagonal_wall(worlds):
    _view, _p, levels = worlds["levels"]
    _cols, counts = np.unique(np.stack([levels.x, levels.y], 1), axis=0, return_counts=True)
    assert (counts > 1).sum() > 20                       # calle bajo el puente, dos plantas
    climbs = levels.usable(0.25) & (np.abs(levels.h[np.maximum(levels.target, 0)] - levels.h[:, None]) > 0.2)
    assert climbs.sum() > 5                              # la escalera interior
    view, _p, course = worlds["course"]
    # Celdas del muro diagonal (medias celdas macizas): sin nodo en su centro
    ys, xs = np.nonzero(np.isin(C.fill(view.cells(*np.meshgrid(np.arange(view.width), np.arange(view.depth)), 0)),
                                [C.FILLS.index(f) for f in ("half_nw", "half_ne", "half_se", "half_sw")]))
    assert len(xs) > 0
    have = set(zip(course.x[course.h < 1.0].tolist(), course.y[course.h < 1.0].tolist()))
    assert not have & set(zip(xs.tolist(), ys.tolist()))


def test_bigger_bodies_use_a_subset_of_the_edges(worlds):
    _v, _p, g = worlds["course"]
    small, person, big = g.usable(0.2), g.usable(0.25), g.usable(0.6)   # los centros quedan a 0.5 m o más
    assert (person <= small).all() and (big <= person).all()
    assert small.sum() >= person.sum() > big.sum()


def test_worlds_without_nav(tmp_path, shared_tileset):
    build_course(tmp_path / "c.tmx", shared_tileset)
    assert compile_tmx(tmp_path / "c.tmx", tmp_path / "c.cworld", nav=False).ok
    store = SQLiteWorldStore(tmp_path / "c.cworld")
    assert not store.has_nav()
    with pytest.raises(ValueError):
        NavGraph(store, 32)
