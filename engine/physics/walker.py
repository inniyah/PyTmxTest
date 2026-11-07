"""Física del modo "andar" para cilindros verticales, vectorizada para muchos a la vez.

Coordenadas en metros: X este, Y arriba, Z sur; la posición es el centro de la base (los
pies). Celda: (floor(X / ancho), floor(Z / ancho), floor(Y / alto de planta)).

Por paso de física (paso fijo):
  1. Velocidad horizontal hacia la objetivo con agarre (suelo o aire); gravedad; salto.
     Contactos entre cilindros: los que se solapan (en planta y en altura) se apartan, cada uno
     en proporción a la masa del otro, y pierden la velocidad con la que se acercaban. El
     apartamiento se suma al movimiento antes de las paredes: las paredes siempre ganan.
  2. Avance horizontal y separación de las paredes: el cilindro se empuja fuera del segmento
     con más penetración (paredes de borde, diagonales finas, lados de lo macizo) y se quita
     la componente de la velocidad que entra en él, así que resbala en cualquier ángulo.
     Un obstáculo cuya parte de arriba no supere los pies + el escalón no cuenta como pared.
  3. Validez en el centro y en el borde del cilindro: si un suelo de destino sube más que el
     escalón o un techo no deja caber el cuerpo, se prueba a moverse en un solo eje (resbalar);
     si nada vale, se queda (y si su sitio ya no fuera válido, se le saca por el camino más corto).
  4. Vertical: en el suelo se sigue la superficie (subiendo escalones y pegándose al bajar);
     en el aire actúa la gravedad hasta tocar una superficie. Al caer junto a un canto, si el
     cilindro queda solapado con un suelo más alto que un escalón, se le saca en horizontal.
El escalón es el mismo en el suelo y en el aire.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from engine.world import cell as C

NEG_INF = -1e9


def _fill_edges():
    """Segmentos (u0, v0, u1, v1) de cada relleno, en coordenadas de celda; NaN = sin segmento."""
    sq = [(0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)]
    table = {
        "solid": sq,
        "half_nw": [(0, 0, 1, 0), (1, 0, 0, 1), (0, 1, 0, 0)],
        "half_ne": [(0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0)],
        "half_se": [(1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 1, 0)],
        "half_sw": [(0, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)],
        "thin_slash": [(0, 1, 1, 0)],
        "thin_backslash": [(0, 0, 1, 1)],
    }
    out = np.full((len(C.FILLS), 4, 4), np.nan)
    for name, segs in table.items():
        out[C.FILL[name], :len(segs)] = segs
    return out


FILL_EDGES = _fill_edges()
SEGMENTS_PER_CELL = 6          # 4 del relleno + pared norte + pared oeste
CONTACT_PASSES = 2             # pasadas de separación entre cilindros por paso

# Puntos del cilindro en los que se comprueba si choca con suelos y techos: el centro y 8 en
# el borde, a 0.98 r (tocar un borde exactamente es válido: 5 mm de tolerancia con r = 25 cm).
# Entre dos puntos del borde, un canto recto puede entrar como mucho r·(1 − 0.98·cos 22.5°)
# (2.4 cm). Para apoyarse solo cuenta el centro.
_ANGLES = np.radians(np.arange(8) * 45.0)
FOOTPRINT = np.concatenate([[[0.0, 0.0]], 0.98 * np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1)])


@dataclass
class WalkerParams:
    cell_w: float = 2.0
    cell_h: float = 3.0
    gravity: float = 20.0          # m/s² (algo más que la real: saltos más "de videojuego")
    snap_down: float = 0.35        # bajar sin despegar hasta esta distancia
    search_levels: int = 4         # plantas hacia abajo en las que se busca suelo
    push_iterations: int = 4


@dataclass
class Walkers:
    """Cilindros que andan: una fila por entidad (arrays de numpy)."""
    pos: np.ndarray                                  # (n, 3) X, Y, Z en metros
    radius: np.ndarray                               # (n,)
    height: np.ndarray                               # (n,) altura actual del cuerpo
    step: np.ndarray                                 # (n,) escalón máximo
    max_speed: np.ndarray                            # (n,) m/s
    grip_ground: np.ndarray                          # (n,) 1/s
    grip_air: np.ndarray                             # (n,) 1/s
    jump_speed: np.ndarray                           # (n,) m/s
    vel: np.ndarray = None                           # (n, 3)
    grounded: np.ndarray = None                      # (n,) bool
    wish: np.ndarray = None                          # (n, 2) dirección deseada en X, Z (-1..1)
    jump: np.ndarray = None                          # (n,) bool, petición de salto
    support: np.ndarray = None                       # (n,) altura del suelo bajo los pies
    mass: np.ndarray = None                          # (n,) kg; inf = inamovible

    def __post_init__(self):
        n = len(self.pos)
        self.pos = np.asarray(self.pos, float)
        self.mass = np.full(n, 70.0) if self.mass is None else np.asarray(self.mass, float)
        self.vel = np.zeros((n, 3)) if self.vel is None else self.vel
        self.grounded = np.zeros(n, bool) if self.grounded is None else self.grounded
        self.wish = np.zeros((n, 2)) if self.wish is None else self.wish
        self.jump = np.zeros(n, bool) if self.jump is None else self.jump
        self.support = np.full(n, NEG_INF) if self.support is None else self.support

    @classmethod
    def create(cls, positions, radius=0.25, height=1.7, step=0.3, max_speed=4.0,
               grip_ground=14.0, grip_air=2.0, jump_speed=6.5):
        n = len(positions)
        full = lambda v: np.full(n, float(v))
        return cls(np.asarray(positions, float), full(radius), full(height), full(step),
                   full(max_speed), full(grip_ground), full(grip_air), full(jump_speed))


class WalkerPhysics:
    """Paso de física para un conjunto de ``Walkers``.

    Cada paso hace solo tres consultas al mundo, sea cual sea el número de cilindros:
    los segmentos de pared, la validez de tres movimientos posibles (completo, solo en X,
    solo en Z) con su suelo y su techo, y nada más: el apoyo final sale de la misma consulta.
    """

    def __init__(self, world, params: Optional[WalkerParams] = None):
        """``world``: algo con ``cells(cx, cy, nivel)`` vectorizado (WorldView)."""
        self.world = world
        meta = getattr(world, "meta", {})
        self.p = params or WalkerParams(cell_w=float(meta.get("cell_width_m", 2.0)),
                                        cell_h=float(meta.get("cell_height_m", 3.0)))
        self._steps = 0

    # --- Columna de celdas bajo un punto ----------------------------------------------

    def _column(self, x, z, base_level, count):
        """Celdas de ``count`` plantas desde ``base_level`` bajo cada punto: una sola consulta.

        Devuelve (niveles (m, K), celdas (m, K), u (m, 1), v (m, 1)).
        """
        cw = self.p.cell_w
        cx, cy = np.floor(x / cw).astype(np.int64), np.floor(z / cw).astype(np.int64)
        u, v = (x / cw - cx)[:, None], (z / cw - cy)[:, None]
        levels = base_level[:, None] + np.arange(count)[None, :]
        cells = self.world.cells(np.repeat(cx[:, None], count, 1), np.repeat(cy[:, None], count, 1), levels)
        return levels, cells, u, v

    def _surfaces(self, levels, cells, u, v):
        """Alturas (m, 2K) de las superficies pisables: suelos y parte de arriba de lo macizo."""
        ch = self.p.cell_h
        floor_y = (levels + C.surface_at(cells, u, v)) * ch
        solid_y = np.where(C.fill_contains(C.fill(cells), u, v), (levels + 1) * ch, np.nan)
        return np.concatenate([floor_y, solid_y], axis=1)

    @staticmethod
    def _highest_below(surfaces, limit):
        ok = np.isfinite(surfaces) & (surfaces <= limit[:, None] + 1e-6)
        return np.where(ok, surfaces, NEG_INF).max(axis=1)

    def _ceiling_above(self, levels, cells, u, v, stand, reach=None):
        """Primer techo por encima de ``stand``: un techo bajo, o la base de lo que haya en una
        planta superior cuya parte de arriba quede fuera de alcance (lo que se puede pisar con un
        escalón, hasta ``stand + reach``, es suelo y no techo)."""
        ch = self.p.cell_h
        reach = np.zeros_like(stand) if reach is None else reach
        low = C.ceiling_at(cells, u, v)
        low_y = np.where(low < 1.0, (levels + low) * ch, np.inf)
        floor_top = (levels + C.surface_at(cells, u, v)) * ch
        solid_top = np.where(C.fill_contains(C.fill(cells), u, v), (levels + 1) * ch, np.nan)
        top = np.fmax(floor_top, solid_top)                        # NaN si la celda está vacía ahí
        overhead = np.isfinite(top) & (top > (stand + reach)[:, None] + 1e-6)
        base_y = np.where(overhead, levels * ch, np.inf)
        cand = np.concatenate([low_y, base_y], axis=1)
        cand = np.where(cand > stand[:, None] + 1e-3, cand, np.inf)
        return cand.min(axis=1)

    def _levels_for(self, w: Walkers):
        ch, p = self.p.cell_h, self.p
        base = np.floor((w.pos[:, 1] - p.snap_down) / ch).astype(np.int64) - 1
        count = int(np.ceil((w.height.max() + p.snap_down) / ch)) + 3
        return base, count

    # --- Consultas públicas (herramientas y tests) -------------------------------------

    def support(self, x, z, y_limit, search=None):
        """Altura de la superficie más alta bajo (x, z) que no supere ``y_limit``."""
        x, z, y_limit = (np.atleast_1d(np.asarray(a, float)) for a in (x, z, y_limit))
        search = search or self.p.search_levels
        top = np.floor(y_limit / self.p.cell_h).astype(np.int64)
        levels, cells, u, v = self._column(x, z, top - search, search + 1)
        return self._highest_below(self._surfaces(levels, cells, u, v), y_limit)

    def ceiling(self, x, z, y_feet):
        x, z, y_feet = (np.atleast_1d(np.asarray(a, float)) for a in (x, z, y_feet))
        base = np.floor((y_feet + 0.05) / self.p.cell_h).astype(np.int64)
        levels, cells, u, v = self._column(x, z, base, 3)
        return self._ceiling_above(levels, cells, u, v, y_feet)

    # --- Paredes ----------------------------------------------------------------------

    def _segments(self, w: Walkers, x, z, margin):
        """Segmentos (n, S, 4) en metros que pueden tocar a cada cilindro: una sola consulta.

        No cuentan los obstáculos cuya parte de arriba no supera los pies + ``margin`` (el
        escalón en el suelo, casi nada en el aire): a esos se sube."""
        cw, ch = self.p.cell_w, self.p.cell_h
        n = len(x)
        c0x = np.floor((x - w.radius) / cw).astype(np.int64)
        c0y = np.floor((z - w.radius) / cw).astype(np.int64)
        y = w.pos[:, 1]
        lo = np.floor((y + margin) / ch).astype(np.int64)
        hi = np.floor((y + w.height - 1e-3) / ch).astype(np.int64)
        di = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        dj = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        dl = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        cx, cy = c0x[:, None] + di, c0y[:, None] + dj                     # (n, 8)
        level = lo[:, None] + dl
        cells = self.world.cells(cx, cy, level)
        segs = np.concatenate([FILL_EDGES[C.fill(cells)],                  # (n, 8, 4, 4)
                               np.full((n, 8, 2, 4), np.nan)], axis=2)    # (n, 8, 6, 4)
        segs[..., 4, :] = np.where(C.wall_n(cells)[..., None], [0, 0, 1, 0], np.nan)
        segs[..., 5, :] = np.where(C.wall_w(cells)[..., None], [0, 0, 0, 1], np.nan)
        segs[..., [0, 2]] = (segs[..., [0, 2]] + cx[..., None, None]) * cw
        segs[..., [1, 3]] = (segs[..., [1, 3]] + cy[..., None, None]) * cw
        segs[~(level <= hi[:, None])] = np.nan
        return segs.reshape(n, -1, 4)

    def _push_out(self, w: Walkers, x, z, vel, margin):
        segs = self._segments(w, x, z, margin)
        ax, az, bx, bz = (segs[..., k] for k in range(4))
        dx, dz = bx - ax, bz - az
        length2 = np.maximum(dx * dx + dz * dz, 1e-12)
        rows_all = np.arange(len(x))
        for _ in range(self.p.push_iterations):
            t = np.clip(((x[:, None] - ax) * dx + (z[:, None] - az) * dz) / length2, 0, 1)
            ox, oz = x[:, None] - (ax + t * dx), z[:, None] - (az + t * dz)
            dist = np.sqrt(ox * ox + oz * oz)
            depth = np.where(np.isfinite(dist), w.radius[:, None] - dist, -np.inf)
            k = np.argmax(depth, axis=1)
            d = depth[rows_all, k]
            hit = d > 1e-9
            if not hit.any():
                break
            rows = np.nonzero(hit)[0]
            nd = np.maximum(dist[rows, k[rows]], 1e-9)
            nx, nz = ox[rows, k[rows]] / nd, oz[rows, k[rows]] / nd
            x[rows] += nx * d[rows]
            z[rows] += nz * d[rows]
            into = np.minimum(vel[rows, 0] * nx + vel[rows, 2] * nz, 0)     # quitar lo que entra
            vel[rows, 0] -= into * nx
            vel[rows, 2] -= into * nz
        return x, z

    def _check_moves(self, w: Walkers, x, z, y, margin, base, count, rows):
        """¿Puede cada cilindro (filas ``rows`` de ``w``) estar en (x, z)? Una sola consulta.

        Devuelve "valid", y las superficies y el techo bajo su centro ("surf", "ceil")."""
        P = len(FOOTPRINT)
        r = w.radius[rows]
        px = (x[:, None] + FOOTPRINT[None, :, 0] * r[:, None]).ravel()
        pz = (z[:, None] + FOOTPRINT[None, :, 1] * r[:, None]).ravel()
        levels, cells, u, v = self._column(px, pz, np.repeat(base, P), count)
        surf = self._surfaces(levels, cells, u, v)                         # (m·P, 2K)
        center = np.arange(len(x)) * P
        stand = np.maximum(y, self._highest_below(surf[center], y + margin))
        yP, hP = np.repeat(y, P), np.repeat(w.height[rows], P)
        mP, sP = np.repeat(margin, P), np.repeat(stand, P)
        # Solo los suelos: de lo macizo se encargan sus segmentos (que ya mantienen el cilindro a un
        # radio; los puntos del borde caerían justo sobre la arista y lo darían por choque)
        floors = surf[:, :levels.shape[1]]
        bottom = levels * self.p.cell_h
        crosses = (np.isfinite(floors) & (floors > (yP + mP)[:, None] + 1e-6)
                   & (bottom < (yP + hP)[:, None] - 1e-6)).any(axis=1)
        ceil = self._ceiling_above(levels, cells, u, v, sP, mP)
        no_room = ceil < sP + hP
        valid = ~(crosses | no_room).reshape(len(x), P).any(axis=1)
        return {"valid": valid, "surf": surf[center], "ceil": ceil[center]}

    def _escape(self, w: Walkers, x, z, y, margin, base, count, rows):
        """Saca por el camino más corto a los cilindros ``rows`` cuya posición no es válida."""
        here = self._check_moves(w, x[rows], z[rows], y[rows], margin[rows], base[rows], count, rows)
        stuck = rows[~here["valid"]]
        if stuck.size == 0:
            return x, z
        dirs = FOOTPRINT[1:] / 0.98
        steps = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25])                  # en radios
        k = len(dirs) * len(steps)
        offs = (steps[:, None, None] * dirs[None]).reshape(-1, 2)            # (k, 2), de cerca a lejos
        idx = np.repeat(stuck, k)
        cx = x[idx] + np.tile(offs[:, 0], stuck.size) * w.radius[idx]
        cz = z[idx] + np.tile(offs[:, 1], stuck.size) * w.radius[idx]
        res = self._check_moves(w, cx, cz, y[idx], margin[idx], base[idx], count, idx)
        ok = res["valid"].reshape(stuck.size, k)
        first = np.argmax(ok, axis=1)
        found = ok[np.arange(stuck.size), first]
        pick = np.arange(stuck.size) * k + first
        x, z = x.copy(), z.copy()
        x[stuck[found]] = cx[pick[found]]
        z[stuck[found]] = cz[pick[found]]
        return x, z

    # --- Contactos entre cilindros ------------------------------------------------------

    @staticmethod
    def _neighbour_pairs(x, z, cell):
        """Pares (i, j), i < j, en casillas vecinas de una rejilla de lado ``cell``. Vectorizado."""
        n = len(x)
        if n < 2:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        cx = np.floor(x / cell).astype(np.int64)
        cz = np.floor(z / cell).astype(np.int64)
        big = np.int64(1 << 31)
        key = cx * big + cz
        order = np.argsort(key, kind="stable")
        skey = key[order]
        ii, jj = [], []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                target = (cx + dx) * big + (cz + dz)
                lo = np.searchsorted(skey, target, "left")
                hi = np.searchsorted(skey, target, "right")
                counts = hi - lo
                total = int(counts.sum())
                if total == 0:
                    continue
                i = np.repeat(np.arange(n), counts)
                starts = np.repeat(lo - (np.cumsum(counts) - counts), counts)
                j = order[starts + np.arange(total)]
                keep = i < j
                ii.append(i[keep])
                jj.append(j[keep])
        if not ii:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(ii), np.concatenate(jj)

    def contacts(self, w: Walkers, x=None, z=None, relax: float = 1.0):
        """Apartamiento (n, 2) en X, Z de los cilindros que se solapan en (x, z) (por defecto,
        donde están); ajusta ``w.vel``."""
        n = len(w.pos)
        sep = np.zeros((n, 2))
        if n < 2:
            return sep
        y = w.pos[:, 1]
        x = w.pos[:, 0] if x is None else x
        z = w.pos[:, 2] if z is None else z
        i, j = self._neighbour_pairs(x, z, 2.0 * float(w.radius.max()) + 1e-6)
        if i.size == 0:
            return sep
        dx, dz = x[i] - x[j], z[i] - z[j]
        dist = np.sqrt(dx * dx + dz * dz)
        overlap = w.radius[i] + w.radius[j] - dist
        vertical = (y[i] < y[j] + w.height[j]) & (y[j] < y[i] + w.height[i])
        hit = (overlap > 0) & vertical
        if not hit.any():
            return sep
        i, j, dx, dz, dist, overlap = i[hit], j[hit], dx[hit], dz[hit], dist[hit], overlap[hit]
        same = dist < 1e-9                                   # encima uno de otro: separar en X
        nx = np.where(same, 1.0, dx / np.maximum(dist, 1e-9))
        nz = np.where(same, 0.0, dz / np.maximum(dist, 1e-9))
        inv_i = np.where(np.isfinite(w.mass[i]), 1.0 / w.mass[i], 0.0)
        inv_j = np.where(np.isfinite(w.mass[j]), 1.0 / w.mass[j], 0.0)
        total = inv_i + inv_j
        share_i = np.where(total > 0, inv_i / np.maximum(total, 1e-12), 0.0)
        share_j = np.where(total > 0, inv_j / np.maximum(total, 1e-12), 0.0)
        push = overlap * relax
        np.add.at(sep, i, np.stack([nx * push * share_i, nz * push * share_i], axis=1))
        np.add.at(sep, j, np.stack([-nx * push * share_j, -nz * push * share_j], axis=1))
        # Velocidad: quitar la parte con la que se acercan, repartida igual que el empuje
        rel = (w.vel[i, 0] - w.vel[j, 0]) * nx + (w.vel[i, 2] - w.vel[j, 2]) * nz
        closing = np.minimum(rel, 0.0)
        dv_i = -closing * share_i
        dv_j = closing * share_j
        np.add.at(w.vel, (i, 0), dv_i * nx)
        np.add.at(w.vel, (i, 2), dv_i * nz)
        np.add.at(w.vel, (j, 0), dv_j * nx)
        np.add.at(w.vel, (j, 2), dv_j * nz)
        return sep

    # --- Paso de física ---------------------------------------------------------------

    def step(self, w: Walkers, dt: float):
        p = self.p
        n = len(w.pos)
        # 1. Velocidad horizontal hacia la objetivo (agarre de suelo o de aire); salto
        wish = w.wish.copy()
        norm = np.linalg.norm(wish, axis=1)
        wish[norm > 1] /= norm[norm > 1, None]
        grip = np.where(w.grounded, w.grip_ground, w.grip_air)
        w.vel[:, [0, 2]] += (wish * w.max_speed[:, None] - w.vel[:, [0, 2]]) * (1 - np.exp(-grip * dt))[:, None]
        jumping = w.jump & w.grounded
        w.vel[jumping, 1] = w.jump_speed[jumping]
        w.grounded = w.grounded & ~jumping
        w.jump[:] = False

        # 2. Avance horizontal y separación de las paredes
        x0, z0, y = w.pos[:, 0].copy(), w.pos[:, 2].copy(), w.pos[:, 1].copy()
        # Lo que no supera los pies + un escalón no es pared, en el suelo y en el aire (en el aire
        # funciona como ayuda de borde: rozar un saliente bajo te sube a él)
        margin = w.step.copy()
        # Contactos sobre las posiciones tras avanzar (donde de verdad estarían), en dos pasadas
        # para las aglomeraciones; después las paredes, que siempre ganan
        x, z = x0 + w.vel[:, 0] * dt, z0 + w.vel[:, 2] * dt
        for _ in range(CONTACT_PASSES):
            sep = self.contacts(w, x, z)
            x, z = x + sep[:, 0], z + sep[:, 1]
        x, z = self._push_out(w, x, z, w.vel, margin)

        # 3. Validez del movimiento en el centro y en el borde del cilindro. No vale si en algún
        #    punto hay un suelo cuyo volumen (macizo de la base de su celda a su superficie) se
        #    cruza con el cuerpo por encima de los pies + el margen (escalón en el suelo, casi nada
        #    en el aire), o un techo que no deja caber el cuerpo. Para apoyarse cuenta el centro.
        #    Primero se prueba el movimiento completo; solo a quien no le vale se le prueba
        #    moverse en un único eje (resbalar contra el borde).
        base, count = self._levels_for(w)
        full = self._check_moves(w, x, z, y, margin, base, count, np.arange(n))
        choice = np.where(full["valid"], 0, 3)
        surf_here, ceil_here = full["surf"], full["ceil"]
        retry = np.nonzero(~full["valid"])[0]
        if retry.size:
            m = retry.size
            idx = np.concatenate([retry, retry])
            ax = np.concatenate([x[retry], x0[retry]])               # solo X, solo Z
            az = np.concatenate([z0[retry], z[retry]])
            alt = self._check_moves(w, ax, az, y[idx], margin[idx], base[idx], count, idx)
            ok_x, ok_z = alt["valid"][:m], alt["valid"][m:]
            choice[retry] = np.where(ok_x, 1, np.where(ok_z, 2, 3))
            for k, ok in ((0, ok_x), (m, ok_z)):
                sel = retry[ok & (choice[retry] == (1 if k == 0 else 2))]
                rows = k + np.nonzero(ok & (choice[retry] == (1 if k == 0 else 2)))[0]
                surf_here[sel], ceil_here[sel] = alt["surf"][rows], alt["ceil"][rows]
        x = np.where((choice == 0) | (choice == 1), x, x0)
        z = np.where((choice == 0) | (choice == 2), z, z0)
        w.vel[(choice == 2) | (choice == 3), 0] = 0
        w.vel[(choice == 1) | (choice == 3), 2] = 0
        w.pos[:, 0], w.pos[:, 2] = x, z
        stay = choice == 3
        # Red de seguridad periódica (4 veces por segundo): si algún caso imprevisto dejara a un
        # cilindro dentro de un suelo, se le saca por el camino más corto. Las posiciones a las que
        # se mueve ya se han comprobado, y las caídas se revisan en cada paso (más abajo).
        self._steps += 1
        if self._steps % max(1, int(round(0.25 / dt))) == 0:
            x, z = self._escape(w, x, z, y, margin, base, count, np.arange(n))
            w.pos[:, 0], w.pos[:, 2] = x, z
        if stay.any():                                             # quedarse quieto: su sitio actual
            lv, cl, uu, vv = self._column(x[stay], z[stay], base[stay], count)
            surf_here[stay] = self._surfaces(lv, cl, uu, vv)
            st = np.maximum(y[stay], self._highest_below(surf_here[stay], y[stay] + margin[stay]))
            ceil_here[stay] = self._ceiling_above(lv, cl, uu, vv, st, margin[stay])

        # 4. Vertical: seguir el suelo (subir escalones, pegarse al bajar) o caer
        sup = self._highest_below(surf_here, y + w.step)
        on_ground = w.grounded & (sup >= y - p.snap_down)
        new_y = np.where(on_ground, sup, y)
        w.vel[on_ground, 1] = 0
        falling = ~on_ground
        w.vel[falling, 1] -= p.gravity * dt
        air_y = y + w.vel[:, 1] * dt
        bump = falling & (air_y + w.height > ceil_here) & (w.vel[:, 1] > 0)
        air_y = np.where(bump, ceil_here - w.height, air_y)
        w.vel[bump, 1] = 0
        # Aterrizar: al descender se aceptan superficies hasta un escalón por encima de los pies,
        # igual que al andar (en el aire se puede avanzar hasta un punto cuyo suelo quede unos cm
        # por encima de los pies: sin esto, al saltar subiendo una rampa se la atravesaría).
        descending = air_y <= y
        land_sup = self._highest_below(surf_here, np.maximum(y, air_y) + np.where(descending, w.step, 0.0))
        landed = falling & descending & (air_y <= land_sup)
        # Subiendo, unos pies que quedaran por debajo de una superficie (a menos de un escalón) se
        # suben a ella
        inside = self._highest_below(surf_here, y + w.step)
        air_y = np.where(falling & ~descending, np.maximum(air_y, inside), air_y)
        new_y = np.where(falling, np.where(landed, land_sup, air_y), new_y)
        w.vel[landed, 1] = 0
        w.pos[:, 1] = new_y
        w.grounded = on_ground | landed
        # Al caer junto a un canto (el lateral de una rampa) el cilindro puede quedar solapado con
        # el volumen de un suelo que ahora le queda por encima de los pies: se le saca en
        # horizontal por el camino más corto, como si resbalara por el canto.
        moved_down = falling & ~landed
        if moved_down.any():
            rows = np.nonzero(moved_down)[0]
            base2, count2 = self._levels_for(w)
            # Al descender, lo que queda a menos de un escalón por encima de los pies es algo sobre lo
            # que se aterrizaría (una pendiente bajo el borde delantero), no un canto del que salir
            x2, z2 = self._escape(w, w.pos[:, 0].copy(), w.pos[:, 2].copy(), new_y, w.step,
                                  base2, count2, rows)
            w.pos[:, 0], w.pos[:, 2] = x2, z2
        w.support = self._highest_below(surf_here, new_y + w.step)
