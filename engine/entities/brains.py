"""Comportamientos de los NPC, vectorizados para todos a la vez.

Paseo (brain = 1): alternan entre andar en un rumbo unos segundos y quedarse quietos. Miran
delante de sus pies (una consulta al mundo para todos): si hay una caída o algo más alto que
un escalón, cambian de rumbo antes de llegar; si aun así avanzan mucho menos de lo que
quieren (atascados contra algo o contra otros), también.
"""

import numpy as np

from .table import ActorTable

BRAIN_NONE, BRAIN_WANDER = 0, 1
IDLE, WALKING = 0, 1

WALK_SECONDS = (2.0, 6.0)
IDLE_SECONDS = (0.5, 2.5)
IDLE_CHANCE = 0.35
LOOK_AHEAD = 0.7          # metros por delante del borde del cilindro (caídas)
LOOK_AHEAD_HIGH = 0.3     # más cerca para lo que sube: así una ladera de 21° no parece un muro
DROP_LIMIT = 0.5          # una caída mayor que esto, delante: no seguir
STUCK_FRACTION = 0.3      # avanzar menos de esta fracción de lo deseado: atascado


class WanderBrains:
    def __init__(self, physics, rng=None):
        self.physics = physics
        self.rng = rng or np.random.default_rng()

    def update(self, table: ActorTable, dt: float, rows=None):
        rows = table.rows() if rows is None else rows
        rows = rows[table.brain[rows] == BRAIN_WANDER]
        if rows.size == 0:
            return
        rng = self.rng
        table.brain_timer[rows] -= dt
        state = table.brain_state[rows]

        # Fin del tiempo en el estado: elegir lo siguiente
        due = table.brain_timer[rows] <= 0
        go_idle = due & (state == WALKING) & (rng.random(rows.size) < IDLE_CHANCE)
        go_walk = due & ~go_idle
        state = np.where(go_idle, IDLE, np.where(go_walk, WALKING, state))
        new_heading = rng.uniform(0, 2 * np.pi, rows.size)
        table.heading[rows] = np.where(go_walk, new_heading, table.heading[rows])
        table.brain_timer[rows] = np.where(go_idle, rng.uniform(*IDLE_SECONDS, rows.size),
                                  np.where(go_walk, rng.uniform(*WALK_SECONDS, rows.size),
                                           table.brain_timer[rows]))

        # Mirar delante: caídas, escalones demasiado altos o atascos -> girar
        walking = state == WALKING
        if walking.any():
            r = rows[walking]
            h = table.heading[r]
            dx, dz = np.cos(h), np.sin(h)
            reach = table.radius[r] + LOOK_AHEAD
            x, y, z = table.pos[r, 0], table.pos[r, 1], table.pos[r, 2]
            ahead = self.physics.support(x + dx * reach, z + dz * reach, y + table.step[r] + 0.01)
            near = table.radius[r] + LOOK_AHEAD_HIGH
            blocked_high = self.physics.support(x + dx * near, z + dz * near, y + table.height[r]) \
                > y + table.step[r] + 1e-3
            drop = ahead < y - DROP_LIMIT
            speed = np.hypot(table.vel[r, 0], table.vel[r, 2])
            stuck = table.grounded[r] & (speed < STUCK_FRACTION * table.walk_speed[r]) \
                & (table.brain_timer[r] < WALK_SECONDS[0] - 0.3)
            turn = drop | blocked_high | stuck
            if turn.any():
                # Darse la vuelta más o menos (entre 90° y 270°) y seguir un rato
                table.heading[r[turn]] = h[turn] + rng.uniform(0.5 * np.pi, 1.5 * np.pi, int(turn.sum()))
                table.brain_timer[r[turn]] = rng.uniform(*WALK_SECONDS, int(turn.sum()))

        table.brain_state[rows] = state
        h = table.heading[rows]
        frac = table.walk_speed[rows] / table.max_speed[rows]
        table.wish[rows, 0] = np.where(state == WALKING, np.cos(h) * frac, 0.0)
        table.wish[rows, 1] = np.where(state == WALKING, np.sin(h) * frac, 0.0)
