"""Rutinas: los NPC van de un lugar a otro siguiendo campos de flujo (brain = 2).

Rutina de prueba: cada NPC tiene su casa; va a la plaza, se queda un rato, vuelve a casa,
descansa y repite. En camino, mira dos nodos por delante si el camino sigue recto (así no anda
en zigzag de centro en centro) y uno si gira.

Lejos de la cámara (``update_far``), sin física: los que van de camino avanzan por su campo a su
velocidad de paseo, saltando de nodo en nodo y guardando lo que sobra del último tramo (su
posición queda entre dos nodos, sobre el grafo); los que esperan descuentan su espera. Al
volver cerca siguen con física desde donde estén.
"""

import numpy as np

from engine.nav import CostProfile, FieldCache, NavGraph

from .table import ActorTable

BRAIN_ROUTINE = 2
TRAVELLING, WAITING = 0, 1
WAIT_AT = {"plaza": (10.0, 25.0), "casa": (8.0, 30.0)}
NODE_REACH = 1.0            # m: altura máxima entre los pies y el nodo de su celda
FIELD_BUDGET = 0.006        # s por actualización para calcular campos encargados
ARRIVE_COST = 6.0           # coste restante con el que se da por llegado (un lugar lleno de gente
                            # no deja entrar hasta el centro: llegar al borde es haber llegado)


class RoutineBrains:
    def __init__(self, graph: NavGraph, places, types, materials, rng=None, cell_h: float = 3.0):
        self.graph = graph
        self.places = {p.id: p for p in places}
        self.plazas = [p.id for p in places if p.kind == "plaza"]
        self.cache = FieldCache(graph, places, cell_h)
        self.profiles = {name: CostProfile.for_type(t, materials) for name, t in types.items()}
        self.rng = rng or np.random.default_rng()
        self.arrived = 0                     # llegadas (para medir)
        self.physics = None                  # la pone ActorSim: para apoyar en el suelo a los lejanos

    def next_place(self, table: ActorTable, rows):
        """Destino siguiente: de casa a la plaza (si hay) y de la plaza a casa."""
        at_home = table.target_place[rows] == table.home[rows]
        plaza = self.rng.choice(self.plazas, len(rows)) if self.plazas else table.home[rows]
        return np.where(at_home | (table.target_place[rows] < 0), plaza, table.home[rows])

    def _arrive(self, table: ActorTable, rows, place: int):
        lo, hi = WAIT_AT.get(self.places[place].kind, (5.0, 15.0))
        table.routine_state[rows] = WAITING
        table.routine_timer[rows] = self.rng.uniform(lo, hi, len(rows))
        table.travel_progress[rows] = 0.0
        self.arrived += int(len(rows))

    def _tick_waiting(self, table: ActorTable, rows, dt: float):
        waiting = rows[table.routine_state[rows] == WAITING]
        table.routine_timer[waiting] -= dt
        done = waiting[table.routine_timer[waiting] <= 0]
        if done.size:
            table.target_place[done] = self.next_place(table, done)
            table.routine_state[done] = TRAVELLING
            table.travel_progress[done] = 0.0

    def update_far(self, table: ActorTable, dt: float, rows):
        """Lejos de la cámara, sin física: avanzar por el campo según el tiempo."""
        rows = rows[table.brain[rows] == BRAIN_ROUTINE]
        if rows.size == 0:
            return
        g = self.graph
        self.cache.advance(FIELD_BUDGET)
        self._tick_waiting(table, rows, dt)
        moving = rows[table.routine_state[rows] == TRAVELLING]
        if moving.size == 0:
            return
        unknown = table.last_node[moving] < 0
        if unknown.any():
            p = table.pos[moving[unknown]]
            table.last_node[moving[unknown]] = g.node_at(p[:, 0], p[:, 2], p[:, 1], NODE_REACH)
        type_names = np.array(table.type_names)[table.type[moving]]
        for (place, type_name) in {(int(p), str(t)) for p, t in zip(table.target_place[moving], type_names)}:
            r = moving[(table.target_place[moving] == place) & (type_names == type_name)]
            r = r[table.last_node[r] >= 0]
            if r.size == 0 or place not in self.places:
                continue
            field = self.cache.request(place, self.profiles[type_name])
            if field is None:
                continue
            node = table.last_node[r].copy()
            budget = table.travel_progress[r] + table.walk_speed[r] * dt
            # Saltar de nodo en nodo mientras alcance para el tramo entero
            for _ in range(64):
                nxt = field.next[node]
                go = (nxt >= 0) & (field.dist[node] > ARRIVE_COST)
                if not go.any():
                    break
                step_len = g.cell_w * np.hypot(g.x[np.maximum(nxt, 0)] - g.x[node], g.y[np.maximum(nxt, 0)] - g.y[node])
                jump = go & (budget >= step_len)
                if not jump.any():
                    break
                budget[jump] -= step_len[jump]
                node[jump] = nxt[jump]
            arrived = field.dist[node] <= ARRIVE_COST
            table.last_node[r] = node
            if arrived.any():
                self._arrive(table, r[arrived], place)
            # Posición: entre el nodo y el siguiente, según lo andado del tramo
            nxt = np.where(arrived, -1, field.next[node])
            a = g.position(node)
            b = g.position(np.maximum(nxt, 0))
            seg = np.maximum(np.hypot(*(b[:, [0, 2]] - a[:, [0, 2]]).T), 1e-6)
            t = np.where(nxt >= 0, np.clip(budget / seg, 0.0, 1.0), 0.0)
            pos = a + (b - a) * t[:, None]
            if self.physics is not None:
                # Entre dos nodos el suelo no es una recta (una casilla partida tiene dos planos):
                # apoyarlo en el suelo de verdad, para que al volver cerca no empiece enterrado
                ground = self.physics.support(pos[:, 0], pos[:, 2], pos[:, 1] + 0.6, search=2)
                pos[:, 1] = np.where(ground > -1e8, ground, pos[:, 1])
            table.pos[r] = pos
            table.vel[r] = 0.0
            table.grounded[r] = True
            table.support[r] = pos[:, 1]
            table.travel_progress[r] = np.where(arrived | (nxt < 0), 0.0, budget)

    def update(self, table: ActorTable, dt: float, rows=None):
        rows = table.rows() if rows is None else rows
        rows = rows[table.brain[rows] == BRAIN_ROUTINE]
        if rows.size == 0:
            return
        g = self.graph
        table.wish[rows] = 0.0
        table.travel_progress[rows] = 0.0          # con física, el avance lo lleva la posición
        self.cache.advance(FIELD_BUDGET)
        self._tick_waiting(table, rows, dt)
        moving = rows[table.routine_state[rows] == TRAVELLING]
        if moving.size == 0:
            return
        pos = table.pos[moving]
        node = g.node_at(pos[:, 0], pos[:, 2], pos[:, 1], NODE_REACH)
        node = np.where(node >= 0, node, table.last_node[moving])       # celda sin nodo: el último
        table.last_node[moving] = node
        type_names = np.array(table.type_names)[table.type[moving]]
        for (place, type_name) in {(int(p), str(t)) for p, t in zip(table.target_place[moving], type_names)}:
            sel = (table.target_place[moving] == place) & (type_names == type_name)
            r, n = moving[sel], node[sel]
            if place < 0 or place not in self.places:
                continue
            field = self.cache.request(place, self.profiles[type_name])
            if field is None:                  # aún calculándose: esperan un momento
                continue
            known = n >= 0
            arrived = known & (field.dist[np.maximum(n, 0)] <= ARRIVE_COST)
            if arrived.any():
                self._arrive(table, r[arrived], place)
            go = known & ~arrived & (field.next[np.maximum(n, 0)] >= 0)
            if not go.any():
                continue
            r, n = r[go], n[go]
            n1 = field.next[n]
            n2 = np.where(n1 >= 0, field.next[np.maximum(n1, 0)], -1)
            straight = (n2 >= 0) & (field.direction[n] == field.direction[np.maximum(n1, 0)])
            aim = g.position(np.where(straight, n2, n1))
            d = aim[:, [0, 2]] - table.pos[r][:, [0, 2]]
            dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
            frac = table.walk_speed[r] / table.max_speed[r]
            table.wish[r] = d / dist[:, None] * frac[:, None]
