"""Sucesos en el juego: quién entra, sale, está dentro o usa cada uno, y sus acciones.

Cada suceso es un rectángulo de celdas en una planta, con su disparo (enter, exit, inside cada
segundo, use al pulsar la tecla de usar), quién lo dispara (player, all o un tipo de personaje),
cuántas veces puede ocurrir, una condición y una lista de acciones (ver ``script``). Las acciones
se ejecutan en orden; ``wait`` deja la lista en pausa y sigue en las actualizaciones siguientes.

Para no recorrer a todos los personajes, se buscan de forma vectorizada los que están en una
celda con suceso o estaban en alguno (para saber quién sale).
"""

import contextlib
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from .script import Action, Condition, parse_actions, parse_condition

INSIDE_EVERY = 1.0          # s: cada cuánto se dispara un suceso "inside"
MESSAGE_SECONDS = 4.0


@dataclass
class Event:
    id: int
    name: str
    level: int
    x0: int
    y0: int
    x1: int
    y1: int
    trigger: str
    who: str
    times: int
    condition: Condition
    actions: List[Action]
    enabled: bool = True
    count: int = 0


@dataclass
class Running:
    """Una lista de acciones en marcha (por quien disparó el suceso)."""
    event: Event
    actor: Optional[int]          # identificador del personaje
    pc: int = 0
    wait: float = 0.0


@dataclass
class Message:
    text: str
    time_left: float = MESSAGE_SECONDS


@dataclass
class EventSystem:
    events: List[Event]
    markers: Dict[str, tuple]                     # nombre -> (planta, x, z) en metros
    cell_w: float = 2.0
    cell_h: float = 3.0
    variables: Dict[str, object] = field(default_factory=dict)
    handlers: Dict[str, Callable] = field(default_factory=dict)
    messages: deque = field(default_factory=lambda: deque(maxlen=6))
    log: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.by_name: Dict[str, List[Event]] = {}
        self.cells: Dict[tuple, List[Event]] = {}
        for e in self.events:
            self.by_name.setdefault(e.name, []).append(e)
            for y in range(e.y0, e.y1):
                for x in range(e.x0, e.x1):
                    self.cells.setdefault((e.level, x, y), []).append(e)
        self._keys = np.array(sorted(self._key(*k) for k in self.cells), np.int64)
        self.inside: Dict[int, Set[int]] = {}          # fila del personaje -> sucesos en los que está
        self.inside_timer: Dict[tuple, float] = {}
        self.running: List[Running] = []
        self.fired = 0

    @staticmethod
    def _key(level, x, y):
        return (np.int64(level) + 512) << 42 | (np.int64(y) + (1 << 20)) << 21 | (np.int64(x) + (1 << 20))

    @classmethod
    def from_store(cls, store, cell_w: float = 2.0, cell_h: float = 3.0) -> "EventSystem":
        events = [Event(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], int(r[9]),
                        parse_condition(r[10]), parse_actions(r[11]), bool(r[12]))
                  for r in store.events()]
        markers = {name: (level, x, z) for name, level, x, z in store.markers()}
        return cls(events, markers, cell_w, cell_h)

    def on(self, name: str, function: Callable):
        """Registra una función de Python para la acción ``call nombre``: recibe (sistema,
        identificador del personaje, *argumentos)."""
        self.handlers[name] = function

    # --- Cada actualización ---

    def update(self, table, dt: float, physics=None, player: Optional[int] = None, use: bool = False):
        """Detecta disparos y avanza las acciones. ``use``: el jugador ha pulsado usar."""
        self._detect(table, dt, player, use)
        self._run(table, dt, physics)
        for m in list(self.messages):
            m.time_left -= dt
        while self.messages and self.messages[0].time_left <= 0:
            self.messages.popleft()

    def _who_matches(self, event: Event, table, row: int, player_row: Optional[int]) -> bool:
        if event.who == "player":
            return row == player_row
        if event.who == "all":
            return True
        return table.type_names[table.type[row]] == event.who

    def _detect(self, table, dt, player, use):
        rows = table.rows()
        if rows.size == 0 or not self.events:
            return
        pos = table.pos[rows]
        level = np.floor((pos[:, 1] + 0.05) / self.cell_h).astype(np.int64)
        cx = np.floor(pos[:, 0] / self.cell_w).astype(np.int64)
        cy = np.floor(pos[:, 2] / self.cell_w).astype(np.int64)
        in_cell = np.isin(self._key(level, cx, cy), self._keys)
        was_inside = np.array([bool(self.inside.get(int(r))) for r in rows], bool) if self.inside else \
            np.zeros(len(rows), bool)
        player_row = table.row(player) if player is not None else None
        for i in np.nonzero(in_cell | was_inside)[0]:
            row = int(rows[i])
            here = self.cells.get((int(level[i]), int(cx[i]), int(cy[i])), [])
            now = {e.id for e in here if self._who_matches(e, table, row, player_row)}
            before = self.inside.get(row, set())
            events = {e.id: e for e in here}
            for eid in now - before:
                e = events[eid]
                if e.trigger == "enter":
                    self.fire(e, table, row)
                if e.trigger == "inside":
                    self.inside_timer[(row, eid)] = 0.0
                    self.fire(e, table, row)
            for eid in before - now:
                self.inside_timer.pop((row, eid), None)
                for e in self._event_by_id(eid):
                    if e.trigger == "exit":
                        self.fire(e, table, row)
            for eid in now & before:
                e = events[eid]
                if e.trigger == "inside":
                    t = self.inside_timer.get((row, eid), 0.0) + dt
                    if t >= INSIDE_EVERY:
                        t -= INSIDE_EVERY
                        self.fire(e, table, row)
                    self.inside_timer[(row, eid)] = t
            if use and row == player_row:
                for e in here:
                    if e.trigger == "use" and e.id in now:
                        self.fire(e, table, row)
            if now:
                self.inside[row] = now
            else:
                self.inside.pop(row, None)

    def _event_by_id(self, eid):
        return [e for e in self.events if e.id == eid]

    def fire(self, event: Event, table, row: Optional[int]):
        """Dispara el suceso si está activo, no ha agotado sus veces y se cumple su condición."""
        if not event.enabled or (event.times and event.count >= event.times):
            return False
        if not event.condition(self.variables):
            return False
        event.count += 1
        self.fired += 1
        handle = table.handle(row) if row is not None else None
        self.running.append(Running(event, handle))
        return True

    # --- Acciones ---

    def _run(self, table, dt, physics):
        still = []
        for job in self.running:
            job.wait -= dt
            while job.wait <= 0 and job.pc < len(job.event.actions):
                action = job.event.actions[job.pc]
                job.pc += 1
                job.wait = self._do(action, job, table, physics)
            if job.pc < len(job.event.actions) or job.wait > 0:
                still.append(job)
        self.running = still

    def _format(self, text: str) -> str:
        class Vars(dict):
            def __missing__(self, key):
                return 0
        try:
            return text.format_map(Vars({k: (int(v) if isinstance(v, float) and v.is_integer() else v)
                                         for k, v in self.variables.items()}))
        except (ValueError, IndexError):
            return text

    def _do(self, action: Action, job: Running, table, physics) -> float:
        """Ejecuta una acción. Devuelve los segundos que hay que esperar antes de la siguiente."""
        name, args = action.name, action.args
        if name == "message":
            self.messages.append(Message(self._format(args[0])))
        elif name == "teleport":
            self.teleport(table, job.actor, args[0], physics)
        elif name == "set":
            value = args[1] if len(args) > 1 else 1
            with contextlib.suppress(TypeError, ValueError):
                value = float(value)                       # si no es un número, se queda el texto
            self.variables[args[0]] = value
        elif name == "clear":
            self.variables.pop(args[0], None)
        elif name == "add":
            current = self.variables.get(args[0], 0)
            try:
                current = float(current)
            except (TypeError, ValueError):
                current = 0.0
            self.variables[args[0]] = current + float(args[1])
        elif name == "wait":
            return float(args[0])
        elif name in ("enable", "disable"):
            for e in self.by_name.get(args[0], []):
                e.enabled = name == "enable"
        elif name == "call":
            fn = self.handlers.get(args[0])
            if fn is None:
                self.log.append(f"call {args[0]}: no hay ninguna función registrada con ese nombre")
            else:
                fn(self, job.actor, *args[1:])
        return 0.0

    def teleport(self, table, actor: Optional[int], marker: str, physics=None):
        if actor is None or not table.valid(actor) or marker not in self.markers:
            return
        level, x, z = self.markers[marker]
        y = level * self.cell_h
        if physics is not None:
            ground = physics.support(np.array([x]), np.array([z]), np.array([(level + 0.99) * self.cell_h]),
                                     search=3)[0]
            if ground > -1e8:
                y = float(ground)
        row = table.row(actor)
        table.pos[row] = (x, y, z)
        table.vel[row] = 0.0
        table.grounded[row] = True
        table.support[row] = y
