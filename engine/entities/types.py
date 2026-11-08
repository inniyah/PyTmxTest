"""Tipos de personaje, definidos en datos (assets/actors/types.json)."""

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_TYPES = Path(__file__).resolve().parents[2] / "assets" / "actors" / "types.json"


@dataclass(frozen=True)
class ActorType:
    name: str
    radius: float
    height: float
    step: float
    mass: float
    max_speed: float
    walk_speed: float
    grip_ground: float
    grip_air: float
    jump_speed: float
    sheet: str
    tint: Tuple[float, float, float]
    # Navegación (opcional): ajustes del coste de los materiales por nombre, y de la pendiente
    material_costs: Optional[Dict[str, float]] = None
    uphill: Optional[float] = None
    downhill: Optional[float] = None


def load_types(path=DEFAULT_TYPES) -> Dict[str, ActorType]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    base = data.get("defecto", {})
    names = {f.name for f in fields(ActorType)} - {"name"}
    optional = {"material_costs", "uphill", "downhill"}
    types = {}
    for name, values in data.items():
        if name.startswith("_") or name == "defecto":
            continue
        merged = {**base, **values}
        unknown = set(merged) - names
        if unknown:
            raise ValueError(f"Tipo '{name}': campos desconocidos {sorted(unknown)}")
        missing = names - optional - set(merged)
        if missing:
            raise ValueError(f"Tipo '{name}': faltan {sorted(missing)}")
        merged["tint"] = tuple(float(c) for c in merged["tint"])
        types[name] = ActorType(name=name, **merged)
    return types
