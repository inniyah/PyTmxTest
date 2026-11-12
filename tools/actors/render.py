"""Render de un fotograma del personaje con la misma proyección que las piezas del mundo.

La cámara del motor es ortográfica: un punto (X, Y, Z) en metros (X este, Y arriba, Z sur)
cae en pantalla en ``x = X · sx`` y ``y_arriba = Y · sy − Z · sz``, y todos los puntos de la
recta de dirección ``(0, sz, sy)`` caen en el mismo píxel. Por cada píxel del lienzo se lanza
un rayo en esa dirección desde el lado de la cámara (arriba y al sur) y se avanza por la
función de distancia del cuerpo hasta tocarlo (*sphere tracing*). Así, cada píxel del sprite
tiene su punto 3D exacto, su normal y su parte del cuerpo, y encaja con los tiles en tamaño,
perspectiva y profundidad.

El lienzo tiene los pies del personaje en ``anchor`` (píxel donde cae el punto (0, 0, 0)).
"""

from dataclasses import dataclass

import numpy as np

from tools.pieces.render import DEFAULT_PROJECTION, Projection

from .body import SIDES, Proportions, build, eye_points

LIGHT = np.array([-0.45, 1.0, 0.65]) / np.linalg.norm([-0.45, 1.0, 0.65])   # la de las piezas
HIT_EPS = 0.0015          # m: distancia a la que el rayo ya toca la superficie
MAX_STEPS = 160


@dataclass
class Frame:
    """Mapas de un fotograma (alto × ancho del lienzo)."""
    hit: np.ndarray        # bool: el rayo toca el cuerpo
    group: np.ndarray      # int: índice en ``names`` (-1 si no toca)
    point: np.ndarray      # (h, w, 3) punto en metros respecto a los pies
    normal: np.ndarray     # (h, w, 3) normal en ejes del mundo
    near: np.ndarray       # cercanía a la cámara (m a lo largo del rayo): mayor = más cerca
    eyes: list             # (x, y, cerca) de cada ojo visible, en píxeles del lienzo
    names: tuple = ()      # nombre de cada parte (grupo)


def facing_axes(angle_deg: float):
    """Ejes del mundo de los ejes locales x (lado izquierdo) y z (delante) del personaje que
    mira hacia ``angle_deg`` (0 = este, 90 = sur, sentido horario en pantalla)."""
    a = np.radians(angle_deg)
    forward = np.array([np.cos(a), 0.0, np.sin(a)])
    left = np.array([np.sin(a), 0.0, -np.cos(a)])
    return left, forward


def fall_direction(angle_deg: float):
    """Hacia dónde cae (x, z locales) quien mira hacia ``angle_deg``: hacia atrás, pero llevado
    hacia el este o el oeste. En esta cámara, alguien tendido de norte a sur se ve casi igual
    que de pie; tendido de este a oeste, se lee bien. Mirando al norte o al sur, cae hacia su
    derecha."""
    left, forward = facing_axes(angle_deg)
    back = -forward
    if abs(back[0]) > 0.3:
        side = np.sign(back[0])
    else:
        side = -left[0] if abs(left[0]) > 1e-6 else 1.0          # su derecha
        side = np.sign(side)
    world = np.array([side, 0.0, 0.3 * back[2]])
    world /= np.linalg.norm(world)
    return float(world @ left), float(world @ forward)


def view_dir(proj: Projection) -> np.ndarray:
    """Dirección hacia la cámara, normalizada."""
    d = np.array([0.0, proj.sz, proj.sy])
    return d / np.linalg.norm(d)


def render_frame(pose, angle_deg: float, height_m: float, model_height: float,
                 size=(96, 160), anchor=(48, 124), proj: Projection = DEFAULT_PROJECTION,
                 eyes: bool = True, prop: Proportions = Proportions(), equipment=()) -> Frame:  # noqa: B008
    left, forward = facing_axes(angle_deg)
    parts, low, xform = build(pose, prop, equipment, fall_direction(angle_deg))
    names = tuple(name for name, _ in parts)
    m = height_m / model_height                     # metros por unidad del modelo
    up = np.array([0.0, 1.0, 0.0])

    def to_local(P):
        x = P @ left / m
        z = P @ forward / m
        return np.stack([x, P[..., 1] / m + low, z], -1)

    def field(P):
        p = to_local(P)
        ds = np.stack([fn(p) for _, fn in parts], -1) * m
        return ds.min(-1), ds.argmin(-1)

    w, h = size
    ax, ay = anchor
    dn = view_dir(proj)
    ii, jj = np.meshgrid(np.arange(w) + 0.5 - ax, ay - (np.arange(h) + 0.5))
    k = proj.sy ** 2 + proj.sz ** 2
    base = np.stack([ii / proj.sx, jj * proj.sy / k, -jj * proj.sz / k], -1).reshape(-1, 3)

    centre = np.array([0.0, height_m * 0.4, -height_m * 0.2])     # cubre también al caído
    radius = height_m * 1.05
    t_c = (centre - base) @ dn
    t = t_c + radius
    t_end = t_c - radius
    hit = np.zeros(len(base), bool)
    active = np.ones(len(base), bool)
    for _ in range(MAX_STEPS):
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        d, _ = field(base[idx] + t[idx, None] * dn)
        touched = d < HIT_EPS
        hit[idx[touched]] = True
        t[idx] -= np.where(touched, 0.0, np.maximum(d * 0.9, HIT_EPS * 0.5))
        active[idx[touched | (t[idx] < t_end[idx])]] = False

    P = base + t[:, None] * dn
    group = np.full(len(base), -1)
    normal = np.zeros_like(P)
    if hit.any():
        Ph = P[hit]
        _, group[hit] = field(Ph)
        e = 0.003
        for a in range(3):
            dv = np.zeros(3)
            dv[a] = e
            normal[hit, a] = field(Ph + dv)[0] - field(Ph - dv)[0]
        normal[hit] /= np.maximum(np.linalg.norm(normal[hit], axis=-1, keepdims=True), 1e-9)

    def screen(point):
        return (ax + point[0] * proj.sx, ay - (point[1] * proj.sy - point[2] * proj.sz))

    frame = Frame(hit.reshape(h, w), group.reshape(h, w), P.reshape(h, w, 3),
                  normal.reshape(h, w, 3), t.reshape(h, w), [], names)
    if eyes:
        head = names.index("cabeza")
        for _side, p, n in eye_points(pose, prop):
            p, n = xform(p), xform(n, direction=True)
            world = left * p[0] * m + up * (p[1] - low) * m + forward * p[2] * m
            nw = left * n[0] + up * n[1] + forward * n[2]
            facing = float(nw @ dn)
            if facing < 0.15:
                continue
            x, y = screen(world)
            px, py = int(np.floor(x)), int(np.floor(y))
            if not (0 <= px < w and 0 <= py < h) or frame.group[py, px] != head:
                continue
            frame.eyes.append((px, py, facing > 0.45))
    return frame


def far_side(group: str, angle_deg: float, proj: Projection = DEFAULT_PROJECTION) -> bool:
    """La parte está en el lado del cuerpo que no ve la cámara (se pinta en sombra)."""
    side = SIDES.get(group, 0)
    if side == 0:
        return False
    left, _ = facing_axes(angle_deg)
    return float(side * left @ view_dir(proj)) < -0.2
