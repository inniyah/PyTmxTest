"""Materiales del suelo (assets/materials.json): ordenados de peor a mejor.

En una celda con varios tiles que declaran material cuenta el peor, que es el de código menor
(código = posición en la lista + 1; 0 = sin material). Los costes de andar se usan en la
navegación; los colores, solo para depurar.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from . import cell as C

DEFAULT_MATERIALS = Path(__file__).resolve().parents[2] / "assets" / "materials.json"
MAX_MATERIALS = C.MATERIAL_MASK            # 15 (el 0 es "sin material")


@dataclass
class Materials:
    names: List[str]                       # índice = código - 1, de peor a mejor
    costs: np.ndarray                      # (16,) por código; [0] = sin material
    colors: np.ndarray                     # (16, 3) en 0..1

    def code(self, name: str) -> int:
        try:
            return self.names.index(name) + 1
        except ValueError:
            raise KeyError(f"Material desconocido '{name}' (hay: {', '.join(self.names)})") from None

    def name(self, code: int) -> str:
        return self.names[code - 1] if code > 0 else "sin material"

    def to_json(self) -> str:
        return json.dumps({"materiales": [
            {"name": n, "cost": float(self.costs[i + 1]), "color": _hex(self.colors[i + 1])}
            for i, n in enumerate(self.names)],
            "sin_material": {"cost": float(self.costs[0]), "color": _hex(self.colors[0])}}, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Materials":
        data = json.loads(text)
        items = data["materiales"]
        if len(items) > MAX_MATERIALS:
            raise ValueError(f"Como mucho {MAX_MATERIALS} materiales (hay {len(items)})")
        names = [m["name"] for m in items]
        if len(set(names)) != len(names):
            raise ValueError("Materiales repetidos")
        costs = np.ones(MAX_MATERIALS + 1)
        colors = np.ones((MAX_MATERIALS + 1, 3))
        none = data.get("sin_material", {})
        costs[0], colors[0] = float(none.get("cost", 1.0)), _rgb(none.get("color", "#ffffff"))
        for i, m in enumerate(items, start=1):
            costs[i], colors[i] = float(m["cost"]), _rgb(m.get("color", "#ffffff"))
        return cls(names, costs, colors)


def _rgb(text: str) -> np.ndarray:
    t = text.lstrip("#")
    return np.array([int(t[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def load_materials(path=DEFAULT_MATERIALS) -> Materials:
    return Materials.from_json(Path(path).read_text(encoding="utf-8"))


def material_of(values) -> np.ndarray:
    return (np.asarray(values, np.uint32) >> C.MATERIAL_SHIFT) & C.MATERIAL_MASK
