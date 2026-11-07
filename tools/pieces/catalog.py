"""Catálogo de piezas del tileset de prueba.

Cada pieza es una parte de celda (CellSpec) con un nombre de archivo, una categoría
(para la galería) y la clase de la capa de Tiled en la que se pinta.
"""

from dataclasses import dataclass
from typing import List

from engine.world import cell as C
from engine.world.cellspec import CellSpec


@dataclass(frozen=True)
class Piece:
    name: str          # también nombre del .obj
    category: str
    layer: str         # floor, object u overhead
    spec: CellSpec


def _h(x: float) -> str:
    return f"{int(round(x * 100)):03d}"          # 0.25 -> "025"


def build_catalog() -> List[Piece]:
    pieces: List[Piece] = []

    # Suelos de celda entera: planos y las 24 rampas
    for name in C.FLOOR_NAMES[1:]:
        kind, *rest = name.split()
        tag = f"flat_{_h(float(rest[0]))}" if kind == "flat" else f"ramp_{rest[0]}_{_h(float(rest[1]))}"
        category = "suelos" if kind == "flat" else ("rampas rectas" if len(rest[0]) == 1 else "rampas diagonales")
        pieces.append(Piece(f"floor_{tag}", category, "floor", CellSpec(floors={"all": name})))

    # Medios suelos planos, en las cuatro mitades
    for part in ("nw", "se", "ne", "sw"):
        for h in (0.0, 0.25, 0.5, 0.75):
            pieces.append(Piece(f"floor_{part}_flat_{_h(h)}", "medios suelos", "floor",
                                CellSpec(floors={part: f"flat {h:g}"})))

    # Medias rampas diagonales: las mitades paralelas a la rampa (bordes de escaleras diagonales)
    for direction, parts in (("NE", ("nw", "se")), ("SW", ("nw", "se")),
                             ("NW", ("ne", "sw")), ("SE", ("ne", "sw"))):
        for part in parts:
            for base in (0.0, 0.25, 0.5, 0.75):
                pieces.append(Piece(f"floor_{part}_ramp_{direction}_{_h(base)}", "medias rampas", "floor",
                                    CellSpec(floors={part: f"ramp {direction} {base:g}"})))

    # Paredes finas en bordes y esquinas
    for walls in ("N", "E", "S", "W", "NE", "ES", "SW", "WN"):
        pieces.append(Piece(f"wall_{walls}", "paredes", "object", CellSpec(walls=frozenset(walls))))

    # Rellenos: macizo, medias macizas, diagonales finas
    for fill in C.FILLS[1:]:
        pieces.append(Piece(f"fill_{fill}", "rellenos", "object", CellSpec(fill=fill)))

    # Techos bajos, de celda entera y por mitades
    for part in ("all", "nw", "se", "ne", "sw"):
        for h in (0.75, 0.5, 0.25):
            prefix = "ceiling" if part == "all" else f"ceiling_{part}"
            pieces.append(Piece(f"{prefix}_{_h(h)}", "techos", "overhead",
                                CellSpec(ceilings={part: h})))

    names = [p.name for p in pieces]
    assert len(names) == len(set(names)), "nombres de pieza repetidos"
    return pieces
