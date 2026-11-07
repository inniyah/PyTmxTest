"""Mallas 3D mínimas de una pieza, generadas a partir de su CellSpec.

Ejes (los del .obj): X hacia el este, Y hacia arriba, Z hacia el sur (hacia la cámara).
Unidades: metros. La celda ocupa x ∈ [0, W], y ∈ [0, H], z ∈ [0, D], con el origen en
la esquina noroeste del suelo.

Cada parte de la celda es un prisma vertical sobre una región de la planta:
    suelo         de 0 hasta el plano del suelo (macizo debajo, como acordamos)
    techo bajo    de su altura hasta el techo de la celda (macizo encima)
    relleno       de 0 a 1 sobre su región (diagonales finas: una banda estrecha)
    pared fina    de 0 a 1 sobre una banda junto a su borde, dentro de la celda
"""

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

from engine.world import cell as C
from engine.world.cellspec import CellSpec

THIN = 0.08          # grosor visual de una pared fina, en fracción de celda

# Regiones de la planta en (u, v), en sentido antihorario visto desde arriba
REGIONS = {
    "all": [(0, 0), (0, 1), (1, 1), (1, 0)],
    "nw": [(0, 0), (0, 1), (1, 0)],
    "se": [(1, 0), (0, 1), (1, 1)],
    "ne": [(0, 0), (1, 1), (1, 0)],
    "sw": [(0, 0), (0, 1), (1, 1)],
}
FILL_REGIONS = {"solid": "all", "half_nw": "nw", "half_se": "se", "half_ne": "ne", "half_sw": "sw"}
WALL_BANDS = {
    "N": [(0, 0), (0, THIN), (1, THIN), (1, 0)],
    "S": [(0, 1 - THIN), (0, 1), (1, 1), (1, 1 - THIN)],
    "W": [(0, 0), (0, 1), (THIN, 1), (THIN, 0)],
    "E": [(1 - THIN, 0), (1 - THIN, 1), (1, 1), (1, 0)],
}
DIAGONAL_BANDS = {
    "thin_slash": [(0, 1), (THIN, 1), (1, THIN), (1, 0), (1 - THIN, 0), (0, 1 - THIN)],
    "thin_backslash": [(0, 0), (0, THIN), (1 - THIN, 1), (1, 1), (1, 1 - THIN), (THIN, 0)],
}

# Materiales: un color plano por tipo de parte, para que se distingan en las pruebas
MATERIALS = {
    "floor": (0.62, 0.74, 0.55),
    "ramp": (0.86, 0.66, 0.38),
    "wall": (0.55, 0.66, 0.84),
    "fill": (0.62, 0.62, 0.66),
    "ceiling": (0.70, 0.56, 0.80),
}


@dataclass
class Face:
    points: np.ndarray        # (n, 3) en metros, en orden con la normal hacia fuera
    material: str


@dataclass
class Mesh:
    faces: List[Face]

    def bounds(self):
        pts = np.concatenate([f.points for f in self.faces]) if self.faces else np.zeros((1, 3))
        return pts.min(axis=0), pts.max(axis=0)


@dataclass(frozen=True)
class CellSize:
    width: float = 2.0       # X (y Z): metros por celda
    height: float = 3.0      # Y: metros por planta


DEFAULT_CELL = CellSize()

Height = Callable[[float, float], float]


def _const(h: float) -> Height:
    return lambda u, v: h


def _plane(code: int) -> Height:
    return lambda u, v: float(C.floor_height(code, u, v))


def _prism(region: Sequence[Tuple[float, float]], bottom: Height, top: Height,
           material: str, size: CellSize) -> List[Face]:
    """Prisma vertical entre dos alturas (en plantas) sobre una región de la planta."""
    def p(u, v, h):
        return (u * size.width, h * size.height, v * size.width)

    faces = []
    ring = list(region)
    # Tapa superior (normal hacia arriba): la región ya va antihoraria vista desde arriba
    faces.append(Face(np.array([p(u, v, top(u, v)) for u, v in ring], float), material))
    flat_floor = all(abs(top(u, v) - bottom(u, v)) < 1e-9 for u, v in ring)
    if flat_floor:
        return faces                 # una lámina: solo la cara de arriba
    faces.append(Face(np.array([p(u, v, bottom(u, v)) for u, v in reversed(ring)], float), material))
    for (u0, v0), (u1, v1) in zip(ring, ring[1:] + ring[:1]):
        quad = [p(u0, v0, bottom(u0, v0)), p(u1, v1, bottom(u1, v1)),
                p(u1, v1, top(u1, v1)), p(u0, v0, top(u0, v0))]
        quad = np.array(quad, float)
        if np.linalg.norm(np.cross(quad[1] - quad[0], quad[3] - quad[0])) < 1e-9 and \
           np.linalg.norm(np.cross(quad[2] - quad[1], quad[2] - quad[3])) < 1e-9:
            continue                 # lado degenerado (altura cero en ese borde)
        faces.append(Face(quad, material))
    return faces


def _orient_outwards(faces: List[Face]) -> List[Face]:
    """Gira las caras laterales cuya normal apunte hacia dentro del prisma."""
    if len(faces) <= 1:
        return faces
    center = np.concatenate([f.points for f in faces]).mean(axis=0)
    out = []
    for f in faces:
        n = np.cross(f.points[1] - f.points[0], f.points[2] - f.points[0])
        if np.dot(n, f.points.mean(axis=0) - center) < 0:
            f = Face(f.points[::-1].copy(), f.material)
        out.append(f)
    return out


def mesh_for(spec: CellSpec, size: CellSize = DEFAULT_CELL) -> Mesh:
    faces: List[Face] = []
    zero, one = _const(0.0), _const(1.0)

    for part, name in spec.floors.items():
        code = C.FLOOR[name]
        if code == C.FLOOR_NONE:
            continue
        material = "ramp" if name.startswith("ramp") else "floor"
        faces += _orient_outwards(_prism(REGIONS[part], zero, _plane(code), material, size))

    for part, h in spec.ceilings.items():
        faces += _orient_outwards(_prism(REGIONS[part], _const(h), one, "ceiling", size))

    if spec.fill in FILL_REGIONS:
        faces += _orient_outwards(_prism(REGIONS[FILL_REGIONS[spec.fill]], zero, one, "fill", size))
    elif spec.fill in DIAGONAL_BANDS:
        faces += _orient_outwards(_prism(DIAGONAL_BANDS[spec.fill], zero, one, "wall", size))

    for side in "NESW":
        if side in spec.walls:
            faces += _orient_outwards(_prism(WALL_BANDS[side], zero, one, "wall", size))
    return Mesh(faces)
