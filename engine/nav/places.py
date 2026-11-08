"""Lugares del mundo compilado (tabla ``places``): casas, plaza, taberna…"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Place:
    id: int
    name: str
    kind: str
    level: int
    x0: int
    y0: int
    x1: int
    y1: int

    def contains(self, cx, cy) -> bool:
        return self.x0 <= cx < self.x1 and self.y0 <= cy < self.y1


def load_places(store) -> List[Place]:
    return [Place(*row) for row in store.places()]
