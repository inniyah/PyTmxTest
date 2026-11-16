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
    # Ropa (opcional): para cada parte recoloreable de su hoja (zapatos, pantalones…), los colores
    # entre los que elige al aparecer ("#rrggbb"); sin paleta, la de la hoja
    ropa: Optional[Dict[str, list]] = None
    # Radio envolvente (m): la distancia mínima a paredes y macizos, para que nada de su dibujo (un
    # brazo) llegue al otro lado. Sin él: el alcance lateral de su hoja (generada), a su altura; o
    # el sólido más BOUND_MARGIN. El sólido (radius) es para los choques entre personajes
    bound_radius: Optional[float] = None


def load_types(path=DEFAULT_TYPES) -> Dict[str, ActorType]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    base = data.get("defecto", {})
    names = {f.name for f in fields(ActorType)} - {"name"}
    optional = {"material_costs", "uphill", "downhill", "ropa", "bound_radius"}
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
        if merged.get("bound_radius") is None:
            merged["bound_radius"] = sheet_bound(Path(path).resolve().parents[1], merged)
        merged["bound_radius"] = max(float(merged["bound_radius"]), float(merged["radius"]))
        types[name] = ActorType(name=name, **merged)
    return types


BOUND_MARGIN = 0.08          # m: sin alcance en su hoja, el envolvente es el sólido más esto


def sheet_bound(assets: Path, t: dict) -> float:
    """Radio envolvente sacado de la hoja: su alcance lateral (``reach`` en su .json, a la altura
    del modelo), escalado a la altura del tipo; si no lo tiene, el sólido más BOUND_MARGIN."""
    meta = (assets / t["sheet"]).with_suffix(".json")
    try:
        layout = json.loads(meta.read_text(encoding="utf-8"))
        return float(layout["reach"]) * float(t["height"]) / float(layout["height"])
    except (OSError, KeyError, ValueError):
        return float(t["radius"]) + BOUND_MARGIN
