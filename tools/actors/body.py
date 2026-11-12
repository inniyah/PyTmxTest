"""Cuerpo del personaje: partes descritas con funciones de distancia (elipsoides y cápsulas).

Unidades del modelo ("u"): con las proporciones de partida, el maniquí mide unos 74 u de pies
a cabeza; al renderizar se escala a la altura del personaje en metros. Ejes locales: x al lado
izquierdo del personaje, y arriba, z hacia delante (hacia donde mira).

Proporciones (``Proportions``): tamaño de la cabeza (escalada desde la barbilla, con orejas,
ojos y casco), largo de las piernas y de los brazos. Equipo opcional: casco, mochila y fusil.

Una pose es un diccionario (ver ``pose``): ángulos en radianes de muslos, rodillas, brazos y
codos (muslo y brazo positivos = hacia delante), rebote del cuerpo en u, inclinación del tronco
hacia delante desde la cadera (``lean``), caída de todo el cuerpo desde los pies (``tilt``; la dirección la elige
quien renderiza), y si los brazos sujetan el fusil apuntando (``aim``). ``throw`` sustituye el brazo
izquierdo por los ángulos (brazo, codo) dados, para lanzar.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

HIP = 28.0               # altura (u) del pivote de la cadera
CHIN = 52.0              # punto desde el que se escala la cabeza
HEAD_TOP = 74.0          # lo alto de la cabeza con escala 1
THIGH, SHIN = 13.0, 12.0
UPPER_ARM, FOREARM, HAND = 11.5, 9.5, 2.6

# Lado de cada grupo (-1 derecho, +1 izquierdo): para oscurecer el lado lejano
SIDES = {"oreja_d": -1, "oreja_i": 1, "brazo_d": -1, "brazo_i": 1, "mano_d": -1, "mano_i": 1,
         "pierna_d": -1, "pierna_i": 1, "bota_d": -1, "bota_i": 1}
# Material por defecto de una parte que la descripción no nombra
FALLBACK = {"mano_d": "brazo_d", "mano_i": "brazo_i", "bota_d": "pierna_d", "bota_i": "pierna_i"}
EQUIPMENT = ("casco", "mochila", "fusil")


@dataclass(frozen=True)
class Proportions:
    head: float = 1.0
    legs: float = 1.0
    arms: float = 1.0

    @classmethod
    def from_desc(cls, desc: dict) -> "Proportions":
        p = desc.get("proporciones", {})
        return cls(float(p.get("cabeza", 1.0)), float(p.get("piernas", 1.0)), float(p.get("brazos", 1.0)))


def pose(**kw) -> dict:
    """Pose con valores por defecto (de pie, brazos caídos)."""
    base = {"thigh_r": 0.0, "knee_r": 0.0, "thigh_l": 0.0, "knee_l": 0.0,
            "arm_r": np.radians(2), "arm_l": np.radians(2), "elbow_r": np.radians(6),
            "elbow_l": np.radians(6), "bob": 0.0, "lean": 0.0, "tilt": 0.0, "aim": False,
            "throw": None}
    unknown = set(kw) - set(base)
    if unknown:
        raise ValueError(f"Claves de pose desconocidas: {sorted(unknown)}")
    return {**base, **kw}


# --- Funciones de distancia ---------------------------------------------------------------

def sd_ell(p, c, r):
    q = (p - c) / r
    return (np.linalg.norm(q, axis=-1) - 1.0) * min(r)


def sd_cap(p, a, b, ra, rb):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba, pa = b - a, p - a
    t = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    d = np.linalg.norm(pa - t[..., None] * ba, axis=-1)
    return d - (ra + (rb - ra) * t)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def rotx(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    v = np.asarray(v, float)
    return np.stack([v[..., 0], v[..., 1] * c - v[..., 2] * s, v[..., 1] * s + v[..., 2] * c], -1)


def two_bone(shoulder, target, l1, l2, pole):
    """Codo de un brazo de dos segmentos que llega a ``target`` (o lo más cerca posible), doblado
    hacia ``pole``."""
    d_vec = target - shoulder
    d = min(np.linalg.norm(d_vec), l1 + l2 - 1e-3)
    u = d_vec / np.linalg.norm(d_vec)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = np.sqrt(max(l1 * l1 - a * a, 0.0))
    side = pole - (pole @ u) * u
    side /= max(np.linalg.norm(side), 1e-9)
    return shoulder + u * a + side * h, shoulder + u * d


# Fusil apuntando (u, coordenadas del cuerpo): culata junto al hombro derecho, cañón al frente
RIFLE_STOCK = np.array([-6.0, 45.0, -1.0])
RIFLE_MUZZLE = np.array([-2.0, 41.5, 34.0])
AIM_HAND_R = 0.29          # posición de cada mano a lo largo del fusil (0 culata, 1 boca)
AIM_HAND_L = 0.56


def _rifle_point(t):
    return RIFLE_STOCK + t * (RIFLE_MUZZLE - RIFLE_STOCK)


Part = Tuple[str, Callable[[np.ndarray], np.ndarray]]


def build(ps: dict, prop: Proportions = Proportions(), equipment=(), fall=(0.0, -1.0)):  # noqa: B008
    """(partes, altura en u del punto más bajo, transformación): las partes del cuerpo en la pose,
    y cómo llevar un punto o una dirección del cuerpo de pie (ojos) a la pose, si cae. ``fall``:
    dirección horizontal local (x, z) hacia la que cae con ``tilt`` (por defecto, de espaldas)."""
    lean, tilt = ps["lean"], ps["tilt"]
    piv = np.array([0, HIP, 0])
    cl, sl = np.cos(lean), np.sin(lean)
    hs = prop.head
    chin = np.array([0, CHIN, 0])
    l_arm, l_fore = UPPER_ARM * prop.arms, FOREARM * prop.arms
    l_thigh, l_shin = THIGH * prop.legs, SHIN * prop.legs

    def upper(fn):
        """Inclina el tren superior hacia delante, girando alrededor de la cadera."""
        def g(p):
            q = p - piv
            y = q[..., 1] * cl + q[..., 2] * sl
            z = -q[..., 1] * sl + q[..., 2] * cl
            return fn(np.stack([q[..., 0], y, z], -1) + piv)
        return g

    def headspace(fn):
        """La cabeza y lo que va en ella, escaladas desde la barbilla."""
        return upper(lambda p: fn((p - chin) / hs + chin) * hs)

    def head(p):
        d = sd_ell(p, np.array([0, 64.0, 0]), np.array([10.2, 10.0, 9.8]))
        return smin(d, sd_ell(p, np.array([0, 57.0, 1.8]), np.array([7.4, 6.0, 7.0])), 3.0)

    def helmet(p):
        dome = sd_ell(p, np.array([0, 65.2, -0.6]), np.array([11.6, 10.2, 11.4]))
        return np.maximum(dome, 61.0 - p[..., 1])                     # cortado por debajo

    def torso(p):
        d = sd_cap(p, [0, 49, 0], [0, 55, 0.3], 3.0, 2.8)                                   # cuello
        d = smin(d, sd_ell(p, np.array([0, 44, 0.3]), np.array([9.6, 6.0, 5.4])), 2.5)       # pecho
        d = smin(d, sd_cap(p, [0, 33, 0], [0, 40, 0.2], 6.6, 7.2), 3.0)                       # cintura
        return smin(d, sd_ell(p, np.array([0, 28.5, -0.2]), np.array([8.8, 5.2, 5.4])), 3.0)  # cadera

    def arm_from_angles(side, swing, elbow):
        sh = np.array([side * 11.2, 46.8, 0.0])
        down = np.array([side * 0.06, -1.0, 0.0])
        down /= np.linalg.norm(down)
        el = sh + rotx(down, -swing) * l_arm
        f = rotx(down, -(swing + elbow))
        return sh, el, el + f * l_fore, el + f * (l_fore + HAND)

    def arm_to(side, hand_target, pole):
        sh = np.array([side * 11.2, 46.8, 0.0])
        to = hand_target - sh
        wrist_target = hand_target - to / np.linalg.norm(to) * HAND
        el, wr = two_bone(sh, wrist_target, l_arm, l_fore, pole)
        return sh, el, wr, wr + (wr - el) / np.linalg.norm(wr - el) * HAND

    def arm_parts(name, hand_name, joints):
        sh, el, wr, hand = joints

        def arm(p):
            return smin(sd_cap(p, sh, el, 3.4, 2.9), sd_cap(p, el, wr, 2.9, 2.5), 1.0)
        return [(name, upper(arm)), (hand_name, upper(lambda p: sd_ell(p, hand, np.array([2.9, 3.3, 2.9]))))]

    aiming = ps["aim"] and "fusil" in equipment
    if aiming:
        right = arm_to(-1, _rifle_point(AIM_HAND_R) + np.array([0, -1.2, 0]), np.array([-1.0, -1.0, -0.3]))
    else:
        right = arm_from_angles(-1, ps["arm_r"], ps["elbow_r"])
    if ps["throw"] is not None:
        left = arm_from_angles(1, *ps["throw"])
    elif aiming:
        left = arm_to(1, _rifle_point(AIM_HAND_L) + np.array([0, -1.2, 0]), np.array([1.0, -1.0, 0.0]))
    else:
        left = arm_from_angles(1, ps["arm_l"], ps["elbow_l"])

    def leg(side, thigh, knee):
        hp = np.array([side * 4.6, HIP, 0.0])
        down = np.array([0, -1.0, 0])
        kn = hp + rotx(down, -thigh) * l_thigh
        an = kn + rotx(down, -(thigh - knee)) * l_shin
        fd = rotx(np.array([0, 0, 1.0]), (thigh - knee) * 0.7)       # el pie acompaña a la pierna
        toe = an + fd * 4.8 + np.array([0, -0.6, 0])
        heel = an + np.array([0, -0.6, -0.8])
        boot_top = an + (kn - an) * 0.35

        def thigh_shin(p):
            return smin(sd_cap(p, hp, kn, 4.6, 3.6), sd_cap(p, kn, an, 3.6, 2.7), 1.2)

        def boot(p):
            return smin(sd_cap(p, boot_top, an, 3.0, 2.9), sd_cap(p, heel, toe, 2.8, 2.5), 1.2)
        return thigh_shin, boot, min(heel[1], toe[1]) - 2.8

    leg_r, boot_r, low_r = leg(-1, ps["thigh_r"], ps["knee_r"])
    leg_l, boot_l, low_l = leg(1, ps["thigh_l"], ps["knee_l"])
    parts: List[Part] = [
        ("cabeza", headspace(head)),
        ("oreja_d", headspace(lambda p: sd_ell(p, np.array([-10.0, 59.5, -0.3]), np.array([1.6, 2.6, 1.8])))),
        ("oreja_i", headspace(lambda p: sd_ell(p, np.array([10.0, 59.5, -0.3]), np.array([1.6, 2.6, 1.8])))),
        ("torso", upper(torso)),
        *arm_parts("brazo_d", "mano_d", right),
        *arm_parts("brazo_i", "mano_i", left),
        ("pierna_d", leg_r), ("bota_d", boot_r),
        ("pierna_i", leg_l), ("bota_i", boot_l),
    ]
    if "casco" in equipment:
        parts.append(("casco", headspace(helmet)))
    if "mochila" in equipment:
        parts.append(("mochila", upper(lambda p: sd_ell(p, np.array([0, 41.0, -7.2]), np.array([7.6, 8.4, 3.8])))))
    if "fusil" in equipment:
        if aiming:
            stock, muzzle = RIFLE_STOCK, RIFLE_MUZZLE
        else:                                  # en la mano derecha, siguiendo el antebrazo
            hand = right[3]
            along = (right[3] - right[1]) / np.linalg.norm(right[3] - right[1])
            stock = hand - along * 9.0
            muzzle = stock + along * np.linalg.norm(RIFLE_MUZZLE - RIFLE_STOCK)
        axis = (muzzle - stock) / np.linalg.norm(muzzle - stock)
        mag_top = stock + (muzzle - stock) * 0.42

        def rifle(p, stock=stock, muzzle=muzzle, mag_top=mag_top):
            d = sd_cap(p, stock, muzzle, 1.0, 0.8)
            d = np.minimum(d, sd_cap(p, stock, stock + axis * 9.0, 1.9, 1.5))      # culata
            return np.minimum(d, sd_cap(p, mag_top, mag_top + np.array([0, -4.5, 1.0]), 1.1, 1.0))
        parts.append(("fusil", upper(rifle)))

    low = min(low_r, low_l) - ps["bob"]
    if not tilt:
        return parts, low, lambda v, direction=False: np.asarray(v, float)
    return _tilted(parts, tilt, low, fall)


def rotation_towards(fall, angle):
    """Giro que lleva la vertical hacia la dirección horizontal ``fall`` (x, z locales)."""
    f = np.array([fall[0], 0.0, fall[1]])
    f /= np.linalg.norm(f)
    k = np.cross([0.0, 1.0, 0.0], f)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


SPIN_UNTIL = np.radians(60)     # al caer, gira para quedar de espaldas antes de esta inclinación


def _tilted(parts, tilt, low, fall):
    """Todo el cuerpo cae hacia ``fall`` girando desde los pies, y a la vez gira sobre sí mismo
    hasta dar la espalda a ``fall``: acaba tendido de espaldas. El punto más bajo se busca
    girando puntos del interior del cuerpo de pie. Devuelve también la transformación de un
    punto (o una dirección) del cuerpo de pie al cuerpo caído."""
    psi = np.arctan2(-fall[0], -fall[1]) * min(1.0, tilt / SPIN_UNTIL)
    c, s_ = np.cos(psi), np.sin(psi)
    spin = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
    R = rotation_towards(fall, tilt) @ spin
    pivot = np.array([0.0, low, 0.0])

    def wrap(fn):
        return lambda p: fn((p - pivot) @ R + pivot)          # (p - pivot) @ R = Rᵀ (p - pivot)

    def xform(v, direction=False):
        v = np.asarray(v, float)
        return v @ R.T if direction else (v - pivot) @ R.T + pivot
    g = np.mgrid[-30:30:1.5, low:low + 90:1.5, -30:40:1.5].reshape(3, -1).T
    inside = np.min(np.stack([fn(g) for _, fn in parts], -1), -1) < 0
    lowest = float(xform(g[inside])[:, 1].min()) if inside.any() else low
    return [(name, wrap(fn)) for name, fn in parts], lowest, xform


def eye_points(ps: dict, prop: Proportions = Proportions()):  # noqa: B008
    """Ojos: (lado, posición en u, normal) de cada ojo, con la inclinación del tronco."""
    eyes = []
    chin = np.array([0, CHIN, 0])
    for side in (-1, 1):
        p = (np.array([side * 4.4, 58.6, 6.5]) - chin) * prop.head + chin
        p = rotx(p - np.array([0, HIP, 0]), ps["lean"]) + np.array([0, HIP, 0])
        n = np.array([side * 0.45, 0, 0.9])
        eyes.append((side, p, rotx(n / np.linalg.norm(n), ps["lean"])))
    return eyes


def standing_height(prop: Proportions = Proportions()) -> float:  # noqa: B008
    """Altura (u) de pie, de la planta del pie a lo alto de la cabeza."""
    _, low, _ = build(pose(), prop)
    return CHIN + (HEAD_TOP - CHIN) * prop.head - low
