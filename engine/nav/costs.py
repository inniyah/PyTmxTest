"""Coste de andar por cada arista del grafo, según el tipo de personaje.

    coste = longitud (m) × material × pendiente

- Material: la media de los costes de material de los dos nodos (``materials.json``, con los
  ajustes de cada tipo: a un perro el camino y la hierba le dan igual).
- Pendiente: 1 + subida · desnivel/longitud (si sube) + bajada · desnivel/longitud (si baja).
  Subir cansa mucho más que bajar: así se rodean las colinas cuando el rodeo compensa.
- Las aristas por las que no cabe su cuerpo (radio) no cuentan.
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .build import DIRS
from .graph import NavGraph

UPHILL, DOWNHILL = 4.0, 1.0


@dataclass
class CostProfile:
    name: str
    material_costs: np.ndarray                 # (16,) por código de material
    radius: float = 0.25
    uphill: float = UPHILL
    downhill: float = DOWNHILL
    overrides: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def for_type(cls, actor_type, materials) -> "CostProfile":
        """Perfil de un tipo de personaje: sus ajustes de material (``material_costs`` en
        types.json, por nombre) sobre los costes generales del mundo."""
        costs = materials.costs.copy()
        overrides = dict(getattr(actor_type, "material_costs", None) or {})
        for name, value in overrides.items():
            costs[materials.code(name)] = float(value)
        return cls(actor_type.name, costs, float(actor_type.radius),
                   float(getattr(actor_type, "uphill", UPHILL) or UPHILL),
                   float(getattr(actor_type, "downhill", DOWNHILL) or DOWNHILL), overrides)

    @property
    def key(self):
        return (self.name, round(self.radius, 3))


def edge_costs(graph: NavGraph, profile: CostProfile) -> np.ndarray:
    """(n, 8): coste de cada arista para este perfil (inf si no existe o no cabe)."""
    usable = graph.usable(profile.radius)
    b = np.maximum(graph.target, 0)
    length = graph.cell_w * np.hypot(DIRS[:, 0], DIRS[:, 1])[None, :]
    rise = graph.h[b] - graph.h[:, None]
    slope = 1.0 + profile.uphill * np.maximum(rise, 0) / length + profile.downhill * np.maximum(-rise, 0) / length
    mat = 0.5 * (profile.material_costs[graph.material][:, None] + profile.material_costs[graph.material[b]])
    return np.where(usable, length * mat * slope, np.inf)
