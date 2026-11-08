"""Lugares, costes, campos de flujo y rutinas casa ↔ plaza."""
import numpy as np
import pytest

from engine.compiler import compile_tmx
from engine.entities import BRAIN_ROUTINE, ActorTable, RoutineBrains, load_types
from engine.entities.world_sim import ActorSim
from engine.nav import CostProfile, FieldCache, NavGraph, edge_costs, load_places
from engine.world.materials import Materials
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.terrain import build_terrain


@pytest.fixture(scope="module")
def town(tmp_path_factory, shared_tileset):
    out = tmp_path_factory.mktemp("town")
    info = build_terrain(out / "t.tmx", shared_tileset, 128, 128, houses=6, seed=4)
    report = compile_tmx(out / "t.tmx", out / "t.cworld", chunk_size=16)
    assert report.ok, report.summary()
    store = SQLiteWorldStore(out / "t.cworld")
    view = WorldView(store)
    graph = NavGraph(store, view.chunk_size)
    places = load_places(store)
    materials = Materials.from_json(store.meta()["materials"])
    return info, store, view, graph, places, materials


def test_places_one_per_house_and_a_plaza(town):
    info, _store, _view, _graph, places, _m = town
    kinds = [p.kind for p in places]
    assert kinds.count("casa") == len(info.plots) and kinds.count("plaza") == 1
    plaza = next(p for p in places if p.kind == "plaza")
    assert plaza.x1 - plaza.x0 == 6 and plaza.y1 - plaza.y0 == 6


def test_going_up_costs_more_than_going_down(town):
    _i, _s, _v, g, _p, materials = town
    costs = edge_costs(g, CostProfile("persona", materials.costs.copy()))
    a, d = np.nonzero(np.isfinite(costs))
    b = g.target[a, d]
    rise = g.h[b] - g.h[a]
    same_material = g.material[a] == g.material[b]
    up = same_material & (rise > 0.3) & (d < 4)
    assert up.any()
    # La arista de vuelta (b -> a, dirección opuesta) es más barata
    back = np.array([costs[b[i], (d[i] + 2) % 4] for i in np.nonzero(up)[0]])
    assert (back < costs[a[up], d[up]]).all()


def _road_share(graph, cache, places, profile, materials):
    road = [materials.code("camino"), materials.code("adoquin")]
    plaza = next(p for p in places if p.kind == "plaza")
    field = cache.field(plaza.id, profile)
    shares, lengths = [], []
    for house in (p for p in places if p.kind == "casa"):
        path = field.path(int(cache.goal_nodes(house)[0]))
        assert path[-1] in set(field.goals.tolist())             # siempre se llega
        assert np.all(np.diff(field.dist[path]) < 0)              # el coste baja en cada paso
        shares.append(np.isin(graph.material[path], road).mean())
        lengths.append(len(path))
    return np.mean(shares), np.mean(lengths)


def test_people_prefer_roads_and_dogs_less(town):
    _i, _s, _v, g, places, materials = town
    cache = FieldCache(g, places)
    types = load_types()
    person = CostProfile.for_type(types["persona"], materials)
    dog = CostProfile.for_type(types["perro"], materials)
    flat = CostProfile("todo igual", np.ones(16), uphill=0.0, downhill=0.0)
    p_share, p_len = _road_share(g, cache, places, person, materials)
    d_share, _ = _road_share(g, cache, places, dog, materials)
    f_share, f_len = _road_share(g, cache, places, flat, materials)
    assert p_share > f_share + 0.25                    # con costes, mucho más camino
    assert p_len >= f_len                              # a cambio de algún rodeo
    assert d_share < p_share                           # al perro le da más igual


def test_routines_go_to_the_plaza_and_back_without_getting_stuck(town):
    _i, _store, view, g, places, materials = town
    types = load_types()
    table = ActorTable(types)
    rng = np.random.default_rng(5)
    routines = RoutineBrains(g, places, types, materials, rng)
    sim = ActorSim(view, table, rng, routines=routines)
    houses = [p for p in places if p.kind == "casa"]
    for i in range(16):
        home = houses[i % len(houses)]
        x, y, z = g.position(int(rng.choice(routines.cache.goal_nodes(home))))
        row = table.row(table.spawn(["persona", "niño", "perro", "guardia"][i % 4], [x, y, z], brain=BRAIN_ROUTINE))
        table.home[row] = table.target_place[row] = home.id
        table.routine_state[row] = 1
        table.routine_timer[row] = rng.uniform(0, 3)
    rows = table.rows()
    best = np.full(len(rows), np.inf)
    since = np.zeros(len(rows))
    target = table.target_place[rows].copy()
    worst = 0
    for _ in range(150):
        sim.step(1.0)
        for i, r in enumerate(rows):
            if table.routine_state[r] != 0 or table.target_place[r] != target[i] or table.last_node[r] < 0:
                best[i], since[i], target[i] = np.inf, 0, table.target_place[r]
                continue
            field = routines.cache.field(int(target[i]), routines.profiles[table.type_names[table.type[r]]])
            cur = field.dist[table.last_node[r]]
            best[i], since[i] = (cur, 0) if cur < best[i] - 1.0 else (best[i], since[i] + 1)
            worst = max(worst, since[i])
    assert worst < 20, "alguien pasó 20 s sin acercarse a su destino"
    assert routines.arrived >= 16                       # todos llegan al menos una vez
    went_to_plaza = (table.target_place[rows] != table.home[rows]) | (table.routine_state[rows] == 0)
    assert went_to_plaza.any()


def _reference_dijkstra(graph, costs, goals):
    """Dijkstra de libro (heapq), para comparar: coste desde cada nodo hasta el destino."""
    import heapq
    n = len(graph)
    incoming = [[] for _ in range(n)]
    for a, d in zip(*np.nonzero(np.isfinite(costs))):
        incoming[int(graph.target[a, d])].append((int(a), float(costs[a, d])))
    dist = [np.inf] * n
    heap = []
    for g in goals:
        dist[int(g)] = 0.0
        heap.append((0.0, int(g)))
    heapq.heapify(heap)
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist[u]:
            continue
        for v, c in incoming[u]:
            if du + c < dist[v]:
                dist[v] = du + c
                heapq.heappush(heap, (du + c, v))
    return np.array(dist)


def test_numpy_shortest_paths_match_dijkstra(town):
    from engine.nav.shortest import ReverseGraph, shortest_to_goals
    _i, _s, _v, g, places, materials = town
    cache = FieldCache(g, places)
    for profile in (CostProfile.for_type(load_types()["persona"], materials),
                    CostProfile.for_type(load_types()["perro"], materials)):
        costs = edge_costs(g, profile)
        rev = ReverseGraph.from_targets(g.target, costs)
        for place in places[:3]:
            goals = cache.goal_nodes(place)
            ours, ref = shortest_to_goals(rev, goals), _reference_dijkstra(g, costs, goals)
            assert np.array_equal(np.isfinite(ours), np.isfinite(ref))
            fin = np.isfinite(ref)
            assert np.allclose(ours[fin], ref[fin], rtol=0, atol=1e-9)


def test_fields_computed_across_frames_match_and_respect_the_budget(town):
    import time
    _i, _s, _v, g, places, materials = town
    profile = CostProfile.for_type(load_types()["persona"], materials)
    at_once = FieldCache(g, places).field(places[0].id, profile)
    cache = FieldCache(g, places)
    assert cache.request(places[0].id, profile) is None          # encargado, aún no está
    calls = 0
    while cache.request(places[0].id, profile) is None:
        t = time.perf_counter()
        cache.advance(0.002)
        assert time.perf_counter() - t < 0.05                     # un trozo pequeño cada vez
        calls += 1
    later = cache.request(places[0].id, profile)
    assert calls >= 1
    assert np.array_equal(later.next, at_once.next) and np.allclose(later.dist, at_once.dist)


def _routine_people(town, count=8, seed=7):
    _i, _store, view, g, places, materials = town
    types = load_types()
    table = ActorTable(types)
    rng = np.random.default_rng(seed)
    routines = RoutineBrains(g, places, types, materials, rng)
    sim = ActorSim(view, table, rng, routines=routines)
    houses = [p for p in places if p.kind == "casa"]
    rows = []
    for i in range(count):
        home = houses[i % len(houses)]
        x, y, z = g.position(int(routines.cache.goal_nodes(home)[0]))
        row = table.row(table.spawn("persona", [x, y, z], brain=BRAIN_ROUTINE))
        table.home[row] = home.id
        table.target_place[row] = routines.plazas[0]
        table.routine_state[row] = 0
        rows.append(row)
    return table, sim, np.array(rows)


def _first_arrivals(table, sim, rows, focus_at, seconds=400):
    first = np.full(len(rows), np.nan)
    worst_in = 0.0
    for sec in range(seconds):
        sim.step(1.0, focus_at(sec))
        y = table.pos[rows, 1]
        body = sim.physics.support(table.pos[rows, 0], table.pos[rows, 2], y + 1.7)
        worst_in = max(worst_in, float((body - y).max()))
        first[np.isnan(first) & (table.routine_state[rows] == 1)] = sec + 1
        if np.isfinite(first).all():
            break
    return first, worst_in


def test_far_npcs_keep_travelling_without_physics(town):
    near_t, _ = _first_arrivals(*_routine_people(town), lambda s: None)
    table, sim, rows = _routine_people(town)
    far_t, worst_in = _first_arrivals(table, sim, rows, lambda s: (9999.0, 9999.0))
    assert np.isfinite(far_t).all() and np.isfinite(near_t).all()
    assert worst_in < 0.06                                   # siempre sobre el suelo, nunca dentro
    assert np.abs(far_t - near_t).max() <= 0.35 * near_t.max() + 10      # tiempos parecidos
    assert sim.tier_counts[2] == len(rows)                   # de verdad estaban lejos


def test_far_to_near_continues_with_physics(town):
    table, sim, rows = _routine_people(town, seed=8)
    first, worst_in = _first_arrivals(table, sim, rows, lambda s: (9999.0, 9999.0) if s < 30 else None)
    assert np.isfinite(first).all() and worst_in < 0.06


def test_generated_roads_follow_the_terrain_and_connect_every_door(town):
    from tools.pieces.terrain import _straight_roads
    info = town[0]
    H = info.heights
    level = (H[:-1, :-1] + H[1:, :-1] + H[:-1, 1:] + H[1:, 1:]) / 4.0
    road = info.road | info.plaza
    # Conexas: todas las puertas en una sola red (8 vecinos)
    start = next(iter(road))
    seen, todo = {start}, [start]
    while todo:
        x, y = todo.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                c = (x + dx, y + dy)
                if c in road and c not in seen:
                    seen.add(c)
                    todo.append(c)
    assert all(tuple(d) in seen for d in info.doors)

    def climb(cells):
        cells = list(cells)
        diffs = []
        for x, y in cells:
            for dx, dy in ((1, 0), (0, 1)):
                if (x + dx, y + dy) in cells:
                    diffs.append(abs(level[y + dy, x + dx] - level[y, x]))
        return np.mean(diffs)
    straight = _straight_roads(info.doors, H.shape[1] - 1, H.shape[0] - 1)
    assert climb(info.road) < climb(straight)


@pytest.mark.parametrize("far", [False, True], ids=["con física", "lejos, sin física"])
def test_waiting_npcs_spread_inside_their_place(town, far):
    """Al llegar, cada uno reserva un sitio libre dentro del lugar (el más lejano de la entrada) y
    espera allí: nadie se queda apiñado en la puerta y no hay dos en el mismo sitio."""
    _i, _store, view, g, places, materials = town
    types = load_types()
    table = ActorTable(types)
    rng = np.random.default_rng(11)
    routines = RoutineBrains(g, places, types, materials, rng)
    sim = ActorSim(view, table, rng, routines=routines)
    houses = [p for p in places if p.kind == "casa"][:3]
    plaza = routines.plazas[0]
    rows = []
    for i in range(18):                               # 6 por casa, todos vienen de la plaza
        home = houses[i % len(houses)]
        x, y, z = g.position(int(rng.choice(routines.cache.goal_nodes(routines.places[plaza]))))
        row = table.row(table.spawn("persona", [x, y, z], brain=BRAIN_ROUTINE))
        table.home[row], table.target_place[row] = home.id, home.id
        rows.append(row)
    rows = np.array(rows)
    focus = (9999.0, 9999.0) if far else None
    inside_ok, seen_waiting = True, 0
    for _ in range(240):
        sim.step(1.0, focus)
        w = rows[(table.routine_state[rows] == 1) & (table.target_place[rows] != plaza)]
        seen_waiting = max(seen_waiting, len(w))
        cx = np.floor(table.pos[w, 0] / g.cell_w).astype(int)
        cy = np.floor(table.pos[w, 2] / g.cell_w).astype(int)
        inside_ok &= all(routines.places[int(table.target_place[r])].contains(x, y) for r, x, y in zip(w, cx, cy))
        spots = table.spot_node[w][table.spot_node[w] >= 0]
        assert len(spots) == len(set(spots.tolist()))
    assert seen_waiting >= 6 and inside_ok
