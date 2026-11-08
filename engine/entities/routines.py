"""Rutinas: los NPC van de un lugar a otro siguiendo campos de flujo (brain = 2).

Rutina de prueba: cada NPC tiene su casa; va a la plaza, se queda un rato, vuelve a casa,
descansa y repite. En camino, mira dos nodos por delante si el camino sigue recto (así no anda
en zigzag de centro en centro) y uno si gira.

Al llegar: se llega al ENTRAR en el lugar (a uno de sus nodos). Entonces elige un sitio libre
dentro (un nodo que no haya reservado otro), el más lejano de por donde entra, para que el
lugar se llene desde el fondo y la puerta quede libre, y va hasta él antes de esperar. Si no
queda sitio, o lleva un rato sin poder avanzar junto al lugar (lleno de gente), espera donde
está. Al irse, libera su sitio.

Lejos de la cámara (``update_far``), sin física: los que van de camino avanzan por su campo a su
velocidad de paseo, saltando de nodo en nodo y guardando lo que sobra del último tramo (su
posición queda entre dos nodos, sobre el grafo, apoyada en el suelo); al llegar se les pone en
su sitio. Los que esperan descuentan su espera. Al volver cerca siguen con física.
"""

import numpy as np

from engine.nav import CostProfile, FieldCache, NavGraph

from .table import ActorTable

BRAIN_ROUTINE = 2
TRAVELLING, WAITING, SETTLING = 0, 1, 2
WAIT_AT = {"plaza": (10.0, 25.0), "casa": (8.0, 30.0)}
NODE_REACH = 1.0            # m: altura máxima entre los pies y el nodo de su celda
FIELD_BUDGET = 0.006        # s por actualización para calcular campos encargados
ARRIVE_COST = 6.0           # junto al lugar (coste restante), y ...
CROWD_STALL = 3.0           # ... tantos segundos sin avanzar: el lugar está lleno, espera ahí
SETTLE_REACH = 0.35         # m: ya está en su sitio
SETTLE_STALL = 4.0          # s sin avanzar hacia su sitio: se queda donde está
SLOW = 0.3                  # m/s: por debajo, "sin avanzar"


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

    # --- Destinos, llegadas y sitios ---

    def next_place(self, table: ActorTable, rows):
        """Destino siguiente: de casa a la plaza (si hay) y de la plaza a casa."""
        at_home = table.target_place[rows] == table.home[rows]
        plaza = self.rng.choice(self.plazas, len(rows)) if self.plazas else table.home[rows]
        return np.where(at_home | (table.target_place[rows] < 0), plaza, table.home[rows])

    def _arrive(self, table: ActorTable, rows, place: int):
        """Empieza la espera en el lugar."""
        lo, hi = WAIT_AT.get(self.places[place].kind, (5.0, 15.0))
        table.routine_state[rows] = WAITING
        table.routine_timer[rows] = self.rng.uniform(lo, hi, len(rows))
        table.travel_progress[rows] = 0.0
        self.arrived += int(len(rows))

    def _assign_spots(self, table: ActorTable, rows, place: int, field) -> np.ndarray:
        """Reserva a cada uno un sitio libre del lugar: el más lejano de donde está (el lugar se
        llena desde el fondo). -1 si no queda."""
        all_rows = table.rows()
        mine = all_rows[(table.target_place[all_rows] == place)
                        & np.isin(table.routine_state[all_rows], (WAITING, SETTLING))]
        taken = set(table.spot_node[mine][table.spot_node[mine] >= 0].tolist())
        free = np.array([n for n in field.goals.tolist() if n not in taken], np.int64)
        out = np.full(len(rows), -1, np.int64)
        g = self.graph
        for i, r in enumerate(rows):
            if free.size == 0:
                break
            p = table.pos[r]
            d = np.hypot((g.x[free] + 0.5) * g.cell_w - p[0], (g.y[free] + 0.5) * g.cell_w - p[2])
            k = int(np.argmax(d))
            out[i] = free[k]
            free = np.delete(free, k)
        table.spot_node[rows] = out
        return out

    def _tick_waiting(self, table: ActorTable, rows, dt: float):
        waiting = rows[table.routine_state[rows] == WAITING]
        table.routine_timer[waiting] -= dt
        done = waiting[table.routine_timer[waiting] <= 0]
        if done.size:
            table.target_place[done] = self.next_place(table, done)
            table.routine_state[done] = TRAVELLING
            table.routine_timer[done] = 0.0            # en camino: segundos sin avanzar
            table.travel_progress[done] = 0.0
            table.spot_node[done] = -1                 # libera su sitio

    # --- Lejos de la cámara: sin física ---

    def update_far(self, table: ActorTable, dt: float, rows):
        """Lejos de la cámara, sin física: avanzar por el campo según el tiempo."""
        rows = rows[table.brain[rows] == BRAIN_ROUTINE]
        if rows.size == 0:
            return
        g = self.graph
        self.cache.advance(FIELD_BUDGET)
        self._tick_waiting(table, rows, dt)
        # Los que iban a su sitio: ponerlos en él
        settling = rows[table.routine_state[rows] == SETTLING]
        for place in np.unique(table.target_place[settling]):
            r = settling[table.target_place[settling] == place]
            self._place_on_spot(table, r)
            self._arrive(table, r, int(place))
        moving = rows[table.routine_state[rows] == TRAVELLING]
        if moving.size == 0:
            return
        unknown = table.last_node[moving] < 0
        if unknown.any():
            # Sin física no hay nada que lo devuelva al suelo: si no hay nodo cerca de sus pies,
            # el de su columna más próximo en altura (si no, se quedaría quieto para siempre)
            p = table.pos[moving[unknown]]
            near = g.node_at(p[:, 0], p[:, 2], p[:, 1], NODE_REACH)
            anywhere = g.node_at(p[:, 0], p[:, 2], p[:, 1], np.inf)
            table.last_node[moving[unknown]] = np.where(near >= 0, near, anywhere)
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
                go = nxt >= 0
                if not go.any():
                    break
                step_len = g.cell_w * np.hypot(g.x[np.maximum(nxt, 0)] - g.x[node], g.y[np.maximum(nxt, 0)] - g.y[node])
                jump = go & (budget >= step_len)
                if not jump.any():
                    break
                budget[jump] -= step_len[jump]
                node[jump] = nxt[jump]
            table.last_node[r] = node
            arrived = field.dist[node] == 0
            if arrived.any():
                a = r[arrived]
                spots = self._assign_spots(table, a, place, field)
                table.last_node[a] = np.where(spots >= 0, spots, table.last_node[a])
                self._place_on_spot(table, a)
                self._arrive(table, a, place)
            still = r[~arrived]
            n, b = node[~arrived], budget[~arrived]
            # Posición: entre el nodo y el siguiente, según lo andado del tramo
            nxt = field.next[n]
            a_pos = g.position(n)
            b_pos = g.position(np.maximum(nxt, 0))
            seg = np.maximum(np.hypot(*(b_pos[:, [0, 2]] - a_pos[:, [0, 2]]).T), 1e-6)
            t = np.where(nxt >= 0, np.clip(b / seg, 0.0, 1.0), 0.0)
            self._set_ground_pos(table, still, a_pos + (b_pos - a_pos) * t[:, None])
            table.travel_progress[still] = np.where(nxt < 0, 0.0, b)

    def _place_on_spot(self, table: ActorTable, rows):
        has = table.spot_node[rows] >= 0
        if has.any():
            r = rows[has]
            self._set_ground_pos(table, r, self.graph.position(table.spot_node[r]))
            table.last_node[r] = table.spot_node[r]

    def _set_ground_pos(self, table: ActorTable, rows, pos):
        pos = np.array(pos, float)
        if self.physics is not None and len(rows):
            # Entre dos nodos el suelo no es una recta (una casilla partida tiene dos planos):
            # apoyarlo en el suelo de verdad, para que al volver cerca no empiece enterrado
            ground = self.physics.support(pos[:, 0], pos[:, 2], pos[:, 1] + 0.6, search=2)
            pos[:, 1] = np.where(ground > -1e8, ground, pos[:, 1])
        table.pos[rows] = pos
        table.vel[rows] = 0.0
        table.grounded[rows] = True
        table.support[rows] = pos[:, 1]

    # --- Cerca: con física ---

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
        speed = np.hypot(table.vel[rows, 0], table.vel[rows, 2])
        slow = rows[speed < SLOW]
        moving_on = rows[speed >= SLOW]
        active = np.isin(table.routine_state[rows], (TRAVELLING, SETTLING))
        # Segundos sin avanzar (en camino o yendo a su sitio)
        table.routine_timer[np.intersect1d(slow, rows[active])] += dt
        table.routine_timer[np.intersect1d(moving_on, rows[active])] = 0.0
        self._settle(table, rows[table.routine_state[rows] == SETTLING])
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
            d = field.dist[np.maximum(n, 0)]
            inside = known & (d == 0)
            crowded = known & ~inside & (d <= ARRIVE_COST) & (table.routine_timer[r] >= CROWD_STALL)
            if inside.any():
                a = r[inside]
                spots = self._assign_spots(table, a, place, field)
                with_spot = a[spots >= 0]
                table.routine_state[with_spot] = SETTLING
                table.routine_timer[with_spot] = 0.0
                if (spots < 0).any():
                    self._arrive(table, a[spots < 0], place)          # sin sitio: espera ahí
            if crowded.any():
                table.spot_node[r[crowded]] = -1
                self._arrive(table, r[crowded], place)                 # lleno: espera junto al lugar
            go = known & ~inside & ~crowded & (field.next[np.maximum(n, 0)] >= 0)
            if not go.any():
                continue
            r, n = r[go], n[go]
            n1 = field.next[n]
            n2 = np.where(n1 >= 0, field.next[np.maximum(n1, 0)], -1)
            straight = (n2 >= 0) & (field.direction[n] == field.direction[np.maximum(n1, 0)])
            self._steer(table, r, g.position(np.where(straight, n2, n1)))
        self._settle(table, moving[table.routine_state[moving] == SETTLING])

    def _settle(self, table: ActorTable, rows):
        """Yendo a su sitio dentro del lugar: en línea recta; al llegar (o si no puede), espera."""
        if rows.size == 0:
            return
        spot = self.graph.position(np.maximum(table.spot_node[rows], 0))
        d = np.hypot(*(spot[:, [0, 2]] - table.pos[rows][:, [0, 2]]).T)
        done = (d <= SETTLE_REACH) | (table.spot_node[rows] < 0) | (table.routine_timer[rows] >= SETTLE_STALL)
        for place in np.unique(table.target_place[rows[done]]):
            r = rows[done][table.target_place[rows[done]] == place]
            table.wish[r] = 0.0
            self._arrive(table, r, int(place))
        go = rows[~done]
        if go.size:
            self._steer(table, go, spot[~done], slow_near=True)

    def _steer(self, table: ActorTable, rows, aim, slow_near: bool = False):
        d = aim[:, [0, 2]] - table.pos[rows][:, [0, 2]]
        dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        frac = table.walk_speed[rows] / table.max_speed[rows]
        if slow_near:                                   # frenar al acercarse a su sitio
            frac = frac * np.minimum(1.0, dist / 0.8)
        table.wish[rows] = d / dist[:, None] * frac[:, None]
