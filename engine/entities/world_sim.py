"""Simulación de los personajes de un mundo: comportamiento + física a paso fijo, por zonas.

Niveles de detalle alrededor de un punto de interés (el jugador o la cámara):
    cercana     (< near)        física y comportamiento completos, cada paso (60 Hz)
    intermedia  (near … far)    física cada dos pasos, con paso doble (30 Hz); comportamiento
    lejana      (> far)         sin física; los que siguen una rutina avanzan por su camino
                                según el tiempo (dos veces por segundo), y los demás, quietos
Sin punto de interés, todos en la cercana.
"""

import numpy as np

from engine.physics import WalkerPhysics
from engine.world import cell as C

from .brains import WanderBrains
from .table import ActorTable

PHYSICS_HZ = 60
BRAIN_EVERY = 6                 # el comportamiento decide 10 veces por segundo


NEAR, FAR = 60.0, 160.0       # metros
FAR_EVERY = 30                # pasos entre avances de la zona lejana (dos veces por segundo)


class ActorSim:
    def __init__(self, world, table: ActorTable, rng=None, near: float = NEAR, far: float = FAR,
                 routines=None):
        self.physics = WalkerPhysics(world)
        self.table = table
        self.brains = WanderBrains(self.physics, rng)
        self.routines = routines              # RoutineBrains, si el mundo tiene navegación y lugares
        if routines is not None:
            routines.physics = self.physics
        self.accumulator = 0.0
        self.steps = 0
        self.near, self.far = near, far
        self.tier_counts = (0, 0, 0)

    def spawn_points(self, count: int, rng, levels=None, x_range=None, z_range=None, tries=50):
        """Posiciones al azar sobre suelo pisable (opcionalmente en ciertas plantas)."""
        w = self.physics.world
        cw, ch = self.physics.p.cell_w, self.physics.p.cell_h
        x_lo, x_hi = x_range or (w.x0 * cw + 0.5, (w.x0 + w.width) * cw - 0.5)
        z_lo, z_hi = z_range or (w.y0 * cw + 0.5, (w.y0 + w.depth) * cw - 0.5)
        out = []
        for _ in range(tries):
            n = count - len(out)
            if n <= 0:
                break
            x, z = rng.uniform(x_lo, x_hi, n * 3), rng.uniform(z_lo, z_hi, n * 3)
            lv = rng.choice(levels if levels is not None else [w.level_max], n * 3)
            y = self.physics.support(x, z, (lv + 0.99) * ch, search=lv.max() - w.level_min + 3)
            ok = (y > -1e8) & (np.floor((y + 0.05) / ch) == lv)
            # Que no esté enterrada: lo que hay por debajo de una columna es macizo, y una
            # "superficie" ahí queda bajo el terreno. El cuerpo debe estar libre: nada macizo donde
            # están los pies, ningún suelo dentro del cuerpo y sitio bajo el techo
            yy = np.where(ok, y, 0.0)
            feet_cells = w.cells(np.floor(x / cw).astype(int), np.floor(z / cw).astype(int),
                                 np.floor((yy + 0.1) / ch).astype(int))
            u, v = x / cw - np.floor(x / cw), z / cw - np.floor(z / cw)
            ok &= ~C.fill_contains(C.fill(feet_cells), u, v)
            ok &= self.physics.support(x, z, yy + 1.8) <= yy + 0.01
            ok &= self.physics.ceiling(x, z, yy) >= yy + 1.8
            out.extend(np.stack([x[ok], y[ok], z[ok]], axis=1)[:n].tolist())
        return np.array(out[:count])

    def tiers(self, rows, focus):
        """(cercanos, intermedios) entre ``rows`` según la distancia en planta a ``focus`` (X, Z)."""
        if focus is None:
            return rows, rows[:0]
        d = np.hypot(self.table.pos[rows, 0] - focus[0], self.table.pos[rows, 2] - focus[1])
        return rows[d < self.near], rows[(d >= self.near) & (d < self.far)]

    def _physics(self, rows, dt):
        if rows.size:
            w = self.table.walkers(rows)
            self.physics.step(w, dt)
            self.table.store(rows, w)

    def step(self, dt: float, focus=None, max_steps: int = None):
        """Avanza ``dt`` segundos de reloj a paso fijo, por zonas alrededor de ``focus`` (X, Z en
        metros, o None para simular a todos por completo). Devuelve los pasos dados.

        ``max_steps`` limita los pasos por llamada: si un frame tarda mucho, el juego va más lento
        un momento en lugar de dar cada vez más pasos por frame (lo que lo haría aún más lento)."""
        self.accumulator += dt
        step = 1.0 / PHYSICS_HZ
        if max_steps is not None and self.accumulator > max_steps * step:
            self.dropped_time = getattr(self, "dropped_time", 0.0) + self.accumulator - max_steps * step
            self.accumulator = max_steps * step
        done = 0
        while self.accumulator >= step:
            rows = self.table.rows()
            near, mid = self.tiers(rows, focus)
            self.tier_counts = (near.size, mid.size, len(self.table) - near.size - mid.size)
            if self.routines is not None and focus is not None and self.steps % FAR_EVERY == 0:
                far = np.setdiff1d(rows, np.concatenate([near, mid]), assume_unique=True)
                if far.size:
                    self.routines.update_far(self.table, step * FAR_EVERY, far)
            if self.steps % BRAIN_EVERY == 0:
                active = np.concatenate([near, mid])
                self.brains.update(self.table, step * BRAIN_EVERY, active)
                if self.routines is not None:
                    self.routines.update(self.table, step * BRAIN_EVERY, active)
            self._physics(near, step)
            if self.steps % 2 == 0:
                self._physics(mid, 2 * step)
            self.accumulator -= step
            self.steps += 1
            done += 1
        return done
