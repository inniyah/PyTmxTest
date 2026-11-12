"""Poses de las animaciones, por nombre.

Cada animación es una lista de poses (ver ``body.pose``). ``quieto`` y ``agachado`` son una sola
pose. Las de andar empiezan con un paso, para que se note enseguida. Las que no se repiten
(``granada``, ``morir``) se quedan en su último fotograma.
"""

import numpy as np

from .body import pose

r = np.radians


def _stride(right_forward: bool, **kw):
    a, b = r(22), r(-20)
    legs = dict(thigh_r=a, knee_r=r(6), thigh_l=b, knee_l=r(28)) if right_forward else \
        dict(thigh_r=b, knee_r=r(28), thigh_l=a, knee_l=r(6))
    arms = dict(arm_r=r(-24), arm_l=r(24), elbow_r=r(18), elbow_l=r(30)) if right_forward else \
        dict(arm_r=r(24), arm_l=r(-24), elbow_r=r(30), elbow_l=r(18))
    return pose(**legs, **arms, **kw)


IDLE = pose()

# Andar: apoyo con la derecha delante, cruce, apoyo con la izquierda delante, cruce.
# En los cruces el cuerpo sube un poco y los brazos se balancean menos.
WALK = [
    _stride(True),
    pose(thigh_r=r(-4), knee_r=r(4), thigh_l=r(12), knee_l=r(38), arm_r=r(-6), arm_l=r(8),
         elbow_r=r(12), elbow_l=r(14), bob=1.0),
    _stride(False),
    pose(thigh_r=r(12), knee_r=r(38), thigh_l=r(-4), knee_l=r(4), arm_r=r(8), arm_l=r(-6),
         elbow_r=r(14), elbow_l=r(12), bob=1.0),
]

# Correr: apoyo, vuelo, apoyo con la otra, vuelo; el tronco inclinado hacia delante
_LEAN = r(12)
RUN = [
    pose(thigh_r=r(36), knee_r=r(18), thigh_l=r(-30), knee_l=r(62), arm_r=r(-38), arm_l=r(42),
         elbow_r=r(95), elbow_l=r(85), lean=_LEAN),
    pose(thigh_r=r(-4), knee_r=r(28), thigh_l=r(30), knee_l=r(95), arm_r=r(-8), arm_l=r(14),
         elbow_r=r(90), elbow_l=r(90), bob=1.5, lean=_LEAN),
    pose(thigh_r=r(-30), knee_r=r(62), thigh_l=r(36), knee_l=r(18), arm_r=r(42), arm_l=r(-38),
         elbow_r=r(85), elbow_l=r(95), lean=_LEAN),
    pose(thigh_r=r(30), knee_r=r(95), thigh_l=r(-4), knee_l=r(28), arm_r=r(14), arm_l=r(-8),
         elbow_r=r(90), elbow_l=r(90), bob=1.5, lean=_LEAN),
]

# --- Con el fusil apuntando (los brazos lo sujetan; solo cuentan las piernas) ---------------

AIM_IDLE = pose(thigh_r=r(6), thigh_l=r(-6), knee_l=r(4), aim=True)
AIM_WALK = [{**p, "aim": True} for p in WALK]

# Agachado: rodilla derecha en el suelo, pierna izquierda doblada delante, tronco un poco
# adelantado
CROUCH = pose(thigh_r=r(-10), knee_r=r(120), thigh_l=r(80), knee_l=r(95), lean=r(8), aim=True)
# Andar agachado: piernas muy dobladas, pasos cortos
CROUCH_WALK = [
    pose(thigh_r=r(70), knee_r=r(85), thigh_l=r(30), knee_l=r(105), lean=r(14), aim=True),
    pose(thigh_r=r(48), knee_r=r(100), thigh_l=r(56), knee_l=r(110), lean=r(14), bob=1.0, aim=True),
    pose(thigh_r=r(30), knee_r=r(105), thigh_l=r(70), knee_l=r(85), lean=r(14), aim=True),
    pose(thigh_r=r(56), knee_r=r(110), thigh_l=r(48), knee_l=r(100), lean=r(14), bob=1.0, aim=True),
]

# Granada: el brazo izquierdo atrás y arriba, el lanzamiento por encima de la cabeza y el final
GRENADE = [
    pose(thigh_r=r(10), thigh_l=r(-14), knee_l=r(10), lean=r(-6), aim=True, throw=(r(-150), r(-50))),
    pose(thigh_r=r(14), thigh_l=r(-10), knee_l=r(8), lean=r(4), aim=True, throw=(r(160), r(10))),
    pose(thigh_r=r(18), knee_r=r(10), thigh_l=r(-8), knee_l=r(6), lean=r(10), aim=True, throw=(r(75), r(15))),
]

# Morir: el impacto lo echa atrás, se le doblan las rodillas, cae de espaldas y queda tendido
DIE = [
    pose(thigh_r=r(10), knee_r=r(20), thigh_l=r(4), knee_l=r(14), arm_r=r(30), arm_l=r(-40),
         elbow_r=r(40), elbow_l=r(30), lean=r(-14), tilt=r(8)),
    pose(thigh_r=r(30), knee_r=r(50), thigh_l=r(16), knee_l=r(40), arm_r=r(-60), arm_l=r(-80),
         elbow_r=r(30), elbow_l=r(20), lean=r(-20), tilt=r(35)),
    pose(thigh_r=r(30), knee_r=r(40), thigh_l=r(20), knee_l=r(30), arm_r=r(-50), arm_l=r(-60),
         elbow_r=r(20), elbow_l=r(10), lean=r(-10), tilt=r(68)),
    pose(thigh_r=r(14), knee_r=r(16), thigh_l=r(6), knee_l=r(10), arm_r=r(-8), arm_l=r(-20),
         elbow_r=r(10), elbow_l=r(20), tilt=r(88)),
]

POSES = {
    "quieto": [IDLE], "andar": WALK, "correr": RUN,
    "agachado": [CROUCH], "andar_agachado": CROUCH_WALK, "granada": GRENADE, "morir": DIE,
}
# Con fusil, estas animaciones llevan los brazos apuntando
ARMED = {"quieto": [AIM_IDLE], "andar": AIM_WALK}

# Fotogramas por segundo, velocidad (m/s) a la que el ciclo se ve natural y si se repite
TIMING = {
    "andar": (8.0, 1.4, True), "correr": (10.0, 4.0, True), "andar_agachado": (7.0, 0.9, True),
    "granada": (10.0, None, False), "morir": (8.0, None, False),
}


def animation(name: str, armed: bool):
    """Poses de una animación; con fusil, la versión armada si la hay."""
    if armed and name in ARMED:
        return ARMED[name]
    return POSES[name]
