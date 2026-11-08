"""Tabla de personajes: una fila por personaje, arrays de numpy, huecos estables.

Cada personaje ocupa una fila que no se mueve mientras vive; una máscara marca las vivas.
Se le referencia con un identificador (handle) que combina la fila y un contador de
generación: al borrar un personaje la generación de su fila aumenta, así que un handle
antiguo deja de ser válido aunque la fila se reutilice para otro.

Los sistemas trabajan con arrays (``rows()`` son las filas vivas); el código de juego puede
usar ``Actor``, una vista cómoda sobre una fila.
"""

from typing import Dict, Iterator, List, Optional

import numpy as np

from engine.physics import Walkers

from .types import ActorType

GEN_SHIFT = 32


class ActorTable:
    def __init__(self, types: Dict[str, ActorType], capacity: int = 64):
        self.types = types
        self.type_names: List[str] = list(types)
        self.type_index = {name: i for i, name in enumerate(self.type_names)}
        self._alloc(capacity)

    # --- Almacenamiento ---

    def _alloc(self, capacity: int):
        old = getattr(self, "capacity", 0)
        def grow(arr, shape_tail=(), fill=0, dtype=float):
            new = np.full((capacity,) + shape_tail, fill, dtype=dtype)
            if arr is not None:
                new[:old] = arr
            return new
        g = lambda name: getattr(self, name, None)
        self.alive = grow(g("alive"), fill=False, dtype=bool)
        self.generation = grow(g("generation"), fill=0, dtype=np.int64)
        self.type = grow(g("type"), fill=0, dtype=np.int32)
        self.pos = grow(g("pos"), (3,))
        self.vel = grow(g("vel"), (3,))
        self.wish = grow(g("wish"), (2,))
        self.jump = grow(g("jump"), fill=False, dtype=bool)
        self.grounded = grow(g("grounded"), fill=False, dtype=bool)
        self.support = grow(g("support"), fill=-1e9)
        for name in ("radius", "height", "step", "mass", "max_speed", "walk_speed",
                     "grip_ground", "grip_air", "jump_speed"):
            setattr(self, name, grow(g(name)))
        self.tint = grow(g("tint"), (3,), fill=1.0)
        self.facing = grow(g("facing"), fill=1, dtype=np.int32)
        self.anim_time = grow(g("anim_time"))
        # Comportamiento (cerebro de paseo): estado, temporizador, rumbo
        self.brain = grow(g("brain"), fill=0, dtype=np.int8)        # 0 ninguno (jugador), 1 paseo
        self.brain_state = grow(g("brain_state"), fill=0, dtype=np.int8)
        self.brain_timer = grow(g("brain_timer"))
        self.heading = grow(g("heading"))
        # Rutina (brain = 2): casa, lugar de destino, estado (0 en camino, 1 esperando), espera
        self.home = grow(g("home"), fill=-1, dtype=np.int32)
        self.target_place = grow(g("target_place"), fill=-1, dtype=np.int32)
        self.routine_state = grow(g("routine_state"), fill=0, dtype=np.int8)
        self.routine_timer = grow(g("routine_timer"))
        self.last_node = grow(g("last_node"), fill=-1, dtype=np.int64)
        self.travel_progress = grow(g("travel_progress"))      # m recorridos del tramo actual (lejos)
        self.capacity = capacity

    # --- Altas y bajas ---

    def spawn(self, type_name: str, pos, brain: int = 1) -> int:
        free = np.nonzero(~self.alive)[0]
        if free.size == 0:
            self._alloc(self.capacity * 2)
            free = np.nonzero(~self.alive)[0]
        row = int(free[0])
        t = self.types[type_name]
        self.alive[row] = True
        self.type[row] = self.type_index[type_name]
        self.pos[row] = pos
        self.vel[row] = 0
        self.wish[row] = 0
        self.jump[row] = False
        self.grounded[row] = True
        self.support[row] = pos[1]
        for name in ("radius", "height", "step", "mass", "max_speed", "walk_speed",
                     "grip_ground", "grip_air", "jump_speed"):
            getattr(self, name)[row] = getattr(t, name)
        self.tint[row] = t.tint
        self.facing[row] = 1
        self.anim_time[row] = 0
        self.brain[row] = brain
        self.brain_state[row] = 0
        self.brain_timer[row] = 0
        self.heading[row] = 0
        self.home[row] = self.target_place[row] = -1
        self.routine_state[row] = 0
        self.routine_timer[row] = 0
        self.last_node[row] = -1
        self.travel_progress[row] = 0.0
        return self.handle(row)

    def despawn(self, handle: int):
        row = self.row(handle)
        if row is None:
            raise KeyError("Identificador no válido (ya borrado)")
        self.alive[row] = False
        self.generation[row] += 1

    # --- Identificadores ---

    def handle(self, row: int) -> int:
        return (int(self.generation[row]) << GEN_SHIFT) | int(row)

    def row(self, handle: int) -> Optional[int]:
        row = handle & ((1 << GEN_SHIFT) - 1)
        if row >= self.capacity or not self.alive[row] or self.generation[row] != handle >> GEN_SHIFT:
            return None
        return row

    def valid(self, handle: int) -> bool:
        return self.row(handle) is not None

    def rows(self) -> np.ndarray:
        return np.nonzero(self.alive)[0]

    def __len__(self):
        return int(self.alive.sum())

    def __iter__(self) -> Iterator["Actor"]:
        for row in self.rows():
            yield Actor(self, self.handle(row))

    def get(self, handle: int) -> "Actor":
        if not self.valid(handle):
            raise KeyError("Identificador no válido (ya borrado)")
        return Actor(self, handle)

    # --- Puente con la física ---

    def walkers(self, rows: np.ndarray) -> Walkers:
        """Copia de las filas indicadas como Walkers (lo que consume la física)."""
        return Walkers(self.pos[rows].copy(), self.radius[rows].copy(), self.height[rows].copy(),
                       self.step[rows].copy(), self.max_speed[rows].copy(), self.grip_ground[rows].copy(),
                       self.grip_air[rows].copy(), self.jump_speed[rows].copy(),
                       vel=self.vel[rows].copy(), grounded=self.grounded[rows].copy(),
                       wish=self.wish[rows].copy(), jump=self.jump[rows].copy(),
                       support=self.support[rows].copy(), mass=self.mass[rows].copy())

    def store(self, rows: np.ndarray, w: Walkers):
        """Devuelve a la tabla el estado que cambia la física."""
        self.pos[rows], self.vel[rows] = w.pos, w.vel
        self.grounded[rows], self.support[rows], self.jump[rows] = w.grounded, w.support, w.jump


class Actor:
    """Vista cómoda sobre una fila de la tabla (para el código de juego)."""

    def __init__(self, table: ActorTable, handle: int):
        self.table, self.handle = table, handle

    @property
    def _row(self) -> int:
        row = self.table.row(self.handle)
        if row is None:
            raise KeyError("Este personaje ya no existe")
        return row

    @property
    def type_name(self) -> str:
        return self.table.type_names[self.table.type[self._row]]

    @property
    def pos(self) -> np.ndarray:
        return self.table.pos[self._row]

    @pos.setter
    def pos(self, value):
        self.table.pos[self._row] = value

    @property
    def wish(self) -> np.ndarray:
        return self.table.wish[self._row]

    @wish.setter
    def wish(self, value):
        self.table.wish[self._row] = value

    def request_jump(self):
        self.table.jump[self._row] = True

    @property
    def grounded(self) -> bool:
        return bool(self.table.grounded[self._row])

    @property
    def mass(self) -> float:
        return float(self.table.mass[self._row])

    @mass.setter
    def mass(self, value):
        self.table.mass[self._row] = value
