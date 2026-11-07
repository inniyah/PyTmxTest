"""Simulación de los personajes de un mundo: comportamiento + física a paso fijo."""

import numpy as np

from engine.physics import WalkerPhysics

from .brains import WanderBrains
from .table import ActorTable

PHYSICS_HZ = 60
BRAIN_EVERY = 6                 # el comportamiento decide 10 veces por segundo


class ActorSim:
    def __init__(self, world, table: ActorTable, rng=None):
        self.physics = WalkerPhysics(world)
        self.table = table
        self.brains = WanderBrains(self.physics, rng)
        self.accumulator = 0.0
        self.steps = 0

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
            out.extend(np.stack([x[ok], y[ok], z[ok]], axis=1)[:n].tolist())
        return np.array(out[:count])

    def step(self, dt: float):
        """Avanza la simulación ``dt`` segundos de reloj (a paso fijo). Devuelve los pasos dados."""
        self.accumulator += dt
        step = 1.0 / PHYSICS_HZ
        done = 0
        while self.accumulator >= step:
            if self.steps % BRAIN_EVERY == 0:
                self.brains.update(self.table, step * BRAIN_EVERY)
            rows = self.table.rows()
            if rows.size:
                w = self.table.walkers(rows)
                self.physics.step(w, step)
                self.table.store(rows, w)
            self.accumulator -= step
            self.steps += 1
            done += 1
        return done
