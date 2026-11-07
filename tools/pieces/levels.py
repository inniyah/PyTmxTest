"""Mapa de dos niveles: suelos con aire debajo (puente, edificio de dos plantas, balcón).

Zonas (coordenadas de celda):
    puente     fila y = 8: rampa de subida (x 2-3), tablero en la planta 1 (x 4-9) con pilares
               macizos en x 4 y x 9, rampa de bajada (x 10-11). Por debajo, en la planta 0, se
               pasa entre los pilares (x 5-8).
    edificio   x 14-21, y 3-9, dos plantas con paredes finas. Puerta al sur en x = 17 (planta 0).
               Escalera interior junto a la pared este (x = 20, y 7-8), que se toma desde (20, 9)
               y llega a la planta 1 en (20, 6) por un hueco del forjado con barandilla.
               Agujero en el suelo de la planta 1 en (16, 5).
    balcón     x 15-16, y = 10, en la planta 1, sobre la calle, con salida por la pared sur.
"""

from pathlib import Path
from typing import List

import tmx_manager as tm

from .catalog import Piece, build_catalog
from .course import CourseBuilder
from .render import DEFAULT_PROJECTION, Projection

BRIDGE_ROW = 8
BUILDING = (14, 3, 21, 9)          # x0, y0, x1, y1
DOOR_X = 17
STAIRS = [(20, 8), (20, 7)]        # rampa N 0 y rampa N 0.5 (se sube hacia el norte)
HOLE = (16, 5)
BALCONY = [(15, 10), (16, 10)]


def build_levels(tmx_path: Path, tsx_path: Path, proj: Projection = DEFAULT_PROJECTION,
                 pieces: List[Piece] = None) -> tm.TiledMap:
    b = CourseBuilder(26, 18, pieces or build_catalog())
    W, H = b.width, b.height
    y = BRIDGE_ROW
    x0, y0, x1, y1 = BUILDING

    # --- Planta 0: suelo general, salvo donde hay rampas o pilares
    ramps0 = {(2, y): "floor_ramp_E_000", (3, y): "floor_ramp_E_050",
              (10, y): "floor_ramp_W_050", (11, y): "floor_ramp_W_000"}
    piers = {(4, y), (9, y)}
    ramps0.update({STAIRS[0]: "floor_ramp_N_000", STAIRS[1]: "floor_ramp_N_050"})
    for yy in range(H):
        for xx in range(W):
            if (xx, yy) in ramps0:
                b.put(0, xx, yy, ramps0[(xx, yy)])
            elif (xx, yy) in piers:
                b.put(0, xx, yy, "fill_solid")
            else:
                b.put(0, xx, yy, "floor_flat_000")

    # --- Puente: tablero en la planta 1 (sobre los pilares y sobre el paso)
    for xx in range(4, 10):
        b.put(1, xx, y, "floor_flat_000")

    # --- Edificio: paredes en las dos plantas, puerta abajo, salida al balcón arriba
    for level in (0, 1):
        for xx in range(x0, x1 + 1):
            b.put(level, xx, y0, "wall_N")
            door = (level == 0 and xx == DOOR_X) or (level == 1 and (xx, y1 + 1) in BALCONY)
            if not door:
                b.put(level, xx, y1, "wall_S")
        for yy in range(y0, y1 + 1):
            b.put(level, x0, yy, "wall_W")
            b.put(level, x1, yy, "wall_E")

    # Forjado de la planta 1, con el hueco de la escalera y un agujero
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            if (xx, yy) not in STAIRS and (xx, yy) != HOLE:
                b.put(1, xx, yy, "floor_flat_000")
    for sx, sy in STAIRS:                                  # barandilla del hueco de la escalera
        b.put(1, sx, sy, "wall_W")
    b.put(1, STAIRS[0][0], STAIRS[0][1] + 1, "wall_N")     # y por el sur

    # --- Balcón: suelo de la planta 1 fuera del edificio, con aire debajo
    for xx, yy in BALCONY:
        b.put(1, xx, yy, "floor_flat_000")

    m = b.to_map(tsx_path, tmx_path, proj)
    tmx_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(tmx_path, encoding="csv")
    return m
