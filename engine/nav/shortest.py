"""Caminos mínimos hasta un conjunto de destinos, solo con numpy (sin scipy: el juego se
distribuye sin dependencias pesadas).

Delta-stepping vectorizado: los nodos que han mejorado su coste quedan pendientes; en cada
vuelta se propagan a la vez los pendientes cuyo coste está a menos de ``delta`` del mínimo (a los
nodos que llegan a ellos por una arista), y los que mejoran pasan a pendientes. Con costes
positivos el resultado es el mismo que el de Dijkstra. Propagar todo el frente a la vez
(Bellman-Ford) rehacía trabajo: un nodo recibía primero un coste caro por un camino de pocos
saltos y luego lo mejoraba una y otra vez (≈ 12 veces cada arista); con delta del orden del coste
de una arista, cada nodo se propaga casi una sola vez.

Para poder repartirlo entre varios frames, ``ShortestPaths.run(vueltas)`` avanza un número
limitado de vueltas y dice si ha terminado.
"""

import numpy as np


class ReverseGraph:
    """Aristas agrupadas por nodo de LLEGADA (para medir hacia atrás desde los destinos)."""

    def __init__(self, n: int, src: np.ndarray, dst: np.ndarray, cost: np.ndarray):
        order = np.argsort(dst, kind="stable")
        self.n = n
        self.src = src[order].astype(np.int64)
        self.cost = cost[order].astype(np.float64)
        counts = np.bincount(dst, minlength=n)
        self.ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    @classmethod
    def from_targets(cls, target: np.ndarray, costs: np.ndarray) -> "ReverseGraph":
        """Desde la tabla de aristas del grafo (n, 8) y sus costes (inf = no hay paso)."""
        a, d = np.nonzero(np.isfinite(costs) & (target >= 0))
        return cls(len(target), a, target[a, d], costs[a, d])


class ShortestPaths:
    def __init__(self, graph: ReverseGraph, goals, delta: float = 4.0):
        self.g = graph
        self.delta = delta
        self.dist = np.full(graph.n, np.inf)
        goals = np.unique(np.asarray(goals, np.int64))
        self.dist[goals] = 0.0
        self.pending = goals
        self.rounds = 0
        self.edges_processed = 0

    @property
    def done(self) -> bool:
        return self.pending.size == 0

    def run(self, max_rounds: int = None) -> bool:
        """Avanza hasta terminar (o ``max_rounds`` vueltas). Devuelve si ha terminado."""
        g = self.g
        k = 0
        while self.pending.size and (max_rounds is None or k < max_rounds):
            d = self.dist[self.pending]
            near = d <= d.min() + self.delta
            f, rest = self.pending[near], self.pending[~near]
            start, count = g.ptr[f], g.ptr[f + 1] - g.ptr[f]
            total = int(count.sum())
            self.edges_processed += total
            changed = f[:0]
            if total:
                # Aristas que llegan a los nodos propagados: posiciones start .. start + count
                owner = np.repeat(np.arange(f.size), count)
                idx = np.repeat(start - np.cumsum(count) + count, count) + np.arange(total)
                src, cand = g.src[idx], self.dist[f][owner] + g.cost[idx]
                better = cand < self.dist[src]
                src, cand = src[better], cand[better]
                if src.size:
                    # El mejor candidato de cada nodo: ordenar por (nodo, coste), el primero
                    order = np.lexsort((cand, src))
                    src, cand = src[order], cand[order]
                    first = np.ones(src.size, bool)
                    first[1:] = src[1:] != src[:-1]
                    changed = src[first]
                    self.dist[changed] = cand[first]
            self.pending = np.union1d(rest, changed)
            k += 1
            self.rounds += 1
        return self.done


def shortest_to_goals(graph: ReverseGraph, goals) -> np.ndarray:
    sp = ShortestPaths(graph, goals)
    sp.run()
    return sp.dist
