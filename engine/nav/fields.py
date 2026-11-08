"""Campos de flujo: hacia un destino, el coste restante desde cada nodo y por dónde seguir.

Un campo sirve a todos los personajes del mismo tipo que van al mismo sitio. Las distancias se
calculan con ``shortest`` (solo numpy). La caché puede calcularlos de una vez (``field``) o
repartidos entre frames con un presupuesto de tiempo (``request`` + ``advance``): al empezar una
partida se piden muchos a la vez y calcularlos de golpe pararía el juego.
"""

import time
from collections import OrderedDict

import numpy as np

from .costs import CostProfile, edge_costs
from .graph import NavGraph
from .shortest import ReverseGraph, ShortestPaths

ROUNDS_PER_SLICE = 16


class FlowField:
    def __init__(self, graph: NavGraph, costs: np.ndarray, goals: np.ndarray, dist: np.ndarray):
        n = len(graph)
        self.goals = np.unique(goals)
        total = costs + np.where(graph.target >= 0, dist[np.maximum(graph.target, 0)], np.inf)
        best = np.argmin(total, axis=1)
        nxt = np.where(np.isfinite(total[np.arange(n), best]), graph.target[np.arange(n), best], -1)
        nxt[self.goals] = -1
        # Compacto: ≈ 9 bytes por nodo y campo
        self.dist = dist.astype(np.float32)
        self.next = nxt.astype(np.int32)
        self.direction = np.where(nxt >= 0, best, -1).astype(np.int8)

    def reachable(self, nodes) -> np.ndarray:
        return np.isfinite(self.dist[nodes])

    def path(self, node: int, limit: int = 100000):
        out = [node]
        while self.next[out[-1]] >= 0 and len(out) < limit:
            out.append(int(self.next[out[-1]]))
        return out


class FieldCache:
    """Campos por (lugar, perfil), calculados al pedirlos y guardados (los menos usados se tiran)."""

    def __init__(self, graph: NavGraph, places, cell_h: float = 3.0, capacity: int = 128):
        self.graph, self.cell_h, self.capacity = graph, cell_h, capacity
        self.places = {p.id: p for p in places}
        self.costs, self.reverse = {}, {}
        self.fields = OrderedDict()
        self.pending = OrderedDict()            # clave -> (ShortestPaths, perfil, destinos)

    def goal_nodes(self, place) -> np.ndarray:
        g = self.graph
        inside = (g.x >= place.x0) & (g.x < place.x1) & (g.y >= place.y0) & (g.y < place.y1)
        level = np.floor((g.h + 0.05) / self.cell_h)
        same = inside & (level == place.level)
        return np.nonzero(same if same.any() else inside)[0]

    def _prepare(self, profile: CostProfile):
        if profile.key not in self.costs:
            self.costs[profile.key] = edge_costs(self.graph, profile)
            self.reverse[profile.key] = ReverseGraph.from_targets(self.graph.target, self.costs[profile.key])

    def _store(self, key, profile, goals, dist) -> FlowField:
        f = FlowField(self.graph, self.costs[profile.key], goals, dist)
        self.fields[key] = f
        while len(self.fields) > self.capacity:
            self.fields.popitem(last=False)
        return f

    def field(self, place_id: int, profile: CostProfile) -> FlowField:
        """El campo, calculándolo ahora si hace falta."""
        key = (place_id, profile.key)
        if key in self.fields:
            self.fields.move_to_end(key)
            return self.fields[key]
        self._prepare(profile)
        goals = self.goal_nodes(self.places[place_id])
        sp = self.pending.pop(key, (None,))[0] or ShortestPaths(self.reverse[profile.key], goals)
        sp.run()
        return self._store(key, profile, goals, sp.dist)

    def request(self, place_id: int, profile: CostProfile):
        """El campo si ya está; si no, lo encarga (``advance`` lo irá calculando) y devuelve None."""
        key = (place_id, profile.key)
        if key in self.fields:
            self.fields.move_to_end(key)
            return self.fields[key]
        if key not in self.pending:
            self._prepare(profile)
            goals = self.goal_nodes(self.places[place_id])
            self.pending[key] = (ShortestPaths(self.reverse[profile.key], goals), profile, goals)
        return None

    def advance(self, budget_s: float) -> int:
        """Avanza los campos encargados durante como mucho ``budget_s`` segundos. Devuelve cuántos
        ha terminado."""
        finished = 0
        deadline = time.perf_counter() + budget_s
        while self.pending and time.perf_counter() < deadline:
            key, (sp, profile, goals) = next(iter(self.pending.items()))
            if sp.run(ROUNDS_PER_SLICE):
                del self.pending[key]
                self._store(key, profile, goals, sp.dist)
                finished += 1
        return finished
