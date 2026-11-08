"""Catálogo de piezas del tileset de prueba.

Cada pieza es una parte de celda (CellSpec) con un nombre de archivo, una categoría
(para la galería) y la clase de la capa de Tiled en la que se pinta.
"""

from dataclasses import dataclass
from typing import Callable, List

from engine.world import cell as C
from engine.world.cellspec import CellSpec


@dataclass(frozen=True)
class Piece:
    name: str          # también nombre del .obj
    category: str
    layer: str         # floor, object u overhead
    spec: CellSpec
    material: str = None   # material del suelo (los marcadores solo llevan esto, sin geometría)
    mesh: Callable = None      # malla propia (tamaño de celda -> Mesh); si no, la de su spec
    finish: Callable = None    # retoque de sus capas (ver surfaces): metal, rugosidad, brillo, pulso


def _h(x: float) -> str:
    return f"{int(round(x * 100)):03d}"          # 0.25 -> "025"


def build_catalog() -> List[Piece]:
    from engine.world.materials import load_materials
    pieces: List[Piece] = []

    # Suelos de celda entera: planos, rampas rectas (escaleras), diagonales y pendientes suaves
    for name in C.FLOOR_NAMES[1:]:
        kind, *rest = name.split()
        if kind == "flat":
            tag, category = f"flat_{_h(float(rest[0]))}", "suelos"
        else:
            tag = f"{kind}_{rest[0]}_{_h(float(rest[1]))}"
            category = ("pendientes suaves" if kind == "slope" else
                        "rampas rectas" if len(rest[0]) == 1 else "rampas diagonales")
        pieces.append(Piece(f"floor_{tag}", category, "floor", CellSpec(floors={"all": name})))

    # Medias celdas, en las cuatro mitades: planos, diagonales y pendientes suaves (para partir
    # cualquier casilla de un terreno por cualquiera de las dos diagonales)
    for part in ("nw", "se", "ne", "sw"):
        for h in (0.0, 0.25, 0.5, 0.75):
            pieces.append(Piece(f"floor_{part}_flat_{_h(h)}", "medios suelos", "floor",
                                CellSpec(floors={part: f"flat {h:g}"})))
    for part in ("nw", "se", "ne", "sw"):
        for kind, dirs, category in (("ramp", ("NE", "SE", "SW", "NW"), "medias rampas"),
                                     ("slope", ("N", "S", "E", "W"), "medias pendientes")):
            for direction in dirs:
                for base in (0.0, 0.25, 0.5, 0.75):
                    pieces.append(Piece(f"floor_{part}_{kind}_{direction}_{_h(base)}", category, "floor",
                                        CellSpec(floors={part: f"{kind} {direction} {base:g}"})))

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
    # Marcadores de material: sin geometría ni imagen visible; se pintan encima del suelo, en una
    # capa aparte, y en cada celda cuenta el peor material de todos sus tiles
    for name in load_materials().names:
        pieces.append(Piece(f"material_{name}", "materiales", "floor", CellSpec(), material=name))

    # Superficies: metal, pulido y brillo propio (capa de superficie)
    from . import surfaces
    from .mesh import mesh_for, pipe_mesh, recolored
    flat = CellSpec(floors={"all": "flat 0"})
    pieces += [
        Piece("pipe_copper", "superficies", "object", CellSpec(),
              mesh=lambda size: pipe_mesh("copper", size), finish=surfaces.constant(metal=1.0, rough=0.3)),
        Piece("pipe_steel", "superficies", "object", CellSpec(),
              mesh=lambda size: pipe_mesh("steel", size), finish=surfaces.constant(metal=1.0, rough=0.18)),
        Piece("floor_polished", "superficies", "floor", flat,
              mesh=lambda size: recolored(mesh_for(flat, size), "polished"), finish=surfaces.constant(rough=0.12)),
        Piece("floor_glow", "superficies", "floor", flat,
              mesh=lambda size: recolored(mesh_for(flat, size), "glowfloor"), finish=surfaces.glow_lines()),
    ]
    return pieces
