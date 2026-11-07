"""Especificación legible de (parte de) una celda.

Cada tile del tileset aporta una parte de la celda en la que se pinta: una pared en
un borde, un suelo en una mitad, un techo bajo… El compilador combina las partes de
todos los tiles de una celda con ``merge`` y obtiene su valor de 32 bits con
``encode``. Las paredes sur y este no caben en la celda (son el norte y el oeste de
sus vecinas): ``encode`` las ignora y el compilador las traslada.

En Tiled, una ``CellSpec`` se guarda como propiedades de texto del tile:

    walls        "N", "NW", "NESW"…
    fill         uno de cell.FILLS
    floor        uno de cell.FLOOR_NAMES ("flat 0", "ramp NE 0.25"…)
    floor_part   all, nw, se, ne, sw          (por defecto all)
    ceiling      0.75, 0.5 o 0.25
    ceiling_part all, nw, se, ne, sw          (por defecto all)
    flags        "ladder"…
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

from . import cell as C

WALL_SIDES = "NESW"
FLAG_NAMES = {"ladder": C.FLAG_LADDER, "active": C.FLAG_ACTIVE}


class CellConflict(ValueError):
    """Dos partes de una celda piden cosas incompatibles."""


@dataclass(frozen=True)
class CellSpec:
    walls: FrozenSet[str] = frozenset()
    fill: Optional[str] = None
    floors: Dict[str, str] = field(default_factory=dict)       # parte -> nombre de suelo
    ceilings: Dict[str, float] = field(default_factory=dict)   # parte -> altura
    flags: FrozenSet[str] = frozenset()

    def __post_init__(self):
        if not set(self.walls) <= set(WALL_SIDES):
            raise ValueError(f"Paredes no válidas: {sorted(self.walls)}")
        if self.fill is not None and self.fill not in C.FILL:
            raise ValueError(f"Relleno desconocido: {self.fill}")
        for part, name in self.floors.items():
            if part not in C.PARTS or name not in C.FLOOR:
                raise ValueError(f"Suelo no válido: {part}: {name}")
        for part, h in self.ceilings.items():
            if part not in C.PARTS or h not in C.CEILING or h == 1.0:
                raise ValueError(f"Techo no válido: {part}: {h}")
        if not set(self.flags) <= set(FLAG_NAMES):
            raise ValueError(f"Flags desconocidos: {sorted(self.flags)}")
        self.split()            # comprueba que las partes usan una sola diagonal

    # --- Diagonal y mitades ---

    def split(self) -> Optional[int]:
        """La diagonal que exigen las partes de la celda, o None si ninguna exige una."""
        needed = set()
        for part in list(self.floors) + list(self.ceilings):
            s = C.PARTS[part][0]
            if s is not None:
                needed.add(s)
        if self.fill in C.FILL_HALF:
            needed.add(C.FILL_HALF[self.fill][0])
        if self.fill in C.FILL_THIN_SPLIT:
            needed.add(C.FILL_THIN_SPLIT[self.fill])
        if len(needed) > 1:
            raise CellConflict("La celda mezcla las diagonales / y \\")
        return needed.pop() if needed else None

    @staticmethod
    def _per_half(parts: dict, what: str) -> Dict[int, object]:
        halves: Dict[int, object] = {}
        for part, value in parts.items():
            for half in C.PARTS[part][1]:
                if half in halves and halves[half] != value:
                    raise CellConflict(f"Dos {what} distintos en la misma mitad")
                halves[half] = value
        return halves

    # --- Combinar ---

    def merge(self, other: "CellSpec") -> "CellSpec":
        if self.fill and other.fill and self.fill != other.fill:
            raise CellConflict(f"Rellenos incompatibles: {self.fill} y {other.fill}")
        floors = dict(self.floors)
        for part, name in other.floors.items():
            floors[part] = name if part not in floors or floors[part] == name else _conflict(
                f"Dos suelos en la parte {part}: {floors[part]} y {name}")
        ceilings = dict(self.ceilings)
        for part, h in other.ceilings.items():
            ceilings[part] = h if part not in ceilings or ceilings[part] == h else _conflict(
                f"Dos techos en la parte {part}")
        merged = CellSpec(walls=self.walls | other.walls, fill=self.fill or other.fill,
                          floors=floors, ceilings=ceilings, flags=self.flags | other.flags)
        merged._per_half(merged.floors, "suelos")        # solapes entre "all" y mitades
        merged._per_half(merged.ceilings, "techos")
        return merged

    # --- Validar y codificar ---

    def problems(self) -> List[str]:
        """Incoherencias que no impiden codificar pero que el compilador debe avisar."""
        out = []
        floors = self._per_half(self.floors, "suelos")
        ceilings = self._per_half(self.ceilings, "techos")
        for half in (1, 2):
            if half in floors:
                top = C.floor_range(C.FLOOR[floors[half]])[1]
                ceil = ceilings.get(half, 1.0)
                if top > ceil + 1e-9:
                    out.append(f"Mitad {half}: el suelo llega a {top:g} y el techo está a {ceil:g}")
        return out

    def encode(self) -> int:
        """Valor de 32 bits de la celda (sin las paredes sur y este, que van a las vecinas)."""
        split = self.split()
        floors = self._per_half(self.floors, "suelos")
        ceilings = self._per_half(self.ceilings, "techos")
        value = 0
        value |= C.WALL_N if "N" in self.walls else 0
        value |= C.WALL_W if "W" in self.walls else 0
        value |= C.FILL[self.fill or "empty"] << C.FILL_SHIFT
        value |= C.SPLIT_BIT if split == C.SPLIT_BACKSLASH else 0
        value |= C.FLOOR[floors.get(1, "none")] << C.FLOOR1_SHIFT
        value |= C.FLOOR[floors.get(2, "none")] << C.FLOOR2_SHIFT
        value |= C.CEILING[ceilings.get(1, 1.0)] << C.CEIL1_SHIFT
        value |= C.CEILING[ceilings.get(2, 1.0)] << C.CEIL2_SHIFT
        for name in self.flags:
            value |= FLAG_NAMES[name]
        return value

    # --- Propiedades de Tiled ---

    def to_properties(self) -> Dict[str, str]:
        props = {}
        if self.walls:
            props["walls"] = "".join(s for s in WALL_SIDES if s in self.walls)
        if self.fill:
            props["fill"] = self.fill
        if len(self.floors) > 1 or len(self.ceilings) > 1:
            raise ValueError("Un tile aporta como mucho un suelo y un techo")
        for part, name in self.floors.items():
            props["floor"] = name
            if part != "all":
                props["floor_part"] = part
        for part, h in self.ceilings.items():
            props["ceiling"] = f"{h:g}"
            if part != "all":
                props["ceiling_part"] = part
        if self.flags:
            props["flags"] = " ".join(sorted(self.flags))
        return props

    @classmethod
    def from_properties(cls, props: Dict[str, str]) -> "CellSpec":
        floors = {props.get("floor_part", "all"): props["floor"]} if "floor" in props else {}
        ceilings = ({props.get("ceiling_part", "all"): float(props["ceiling"])}
                    if "ceiling" in props else {})
        return cls(walls=frozenset(props.get("walls", "")), fill=props.get("fill"),
                   floors=floors, ceilings=ceilings,
                   flags=frozenset(props.get("flags", "").split()))


def _conflict(message):
    raise CellConflict(message)
