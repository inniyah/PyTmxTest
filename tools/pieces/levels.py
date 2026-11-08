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

Superficies: suelo pulido en el edificio (planta 0), dos filas de suelo con líneas que brillan
y laten en la calle (x 1-9, y 16-17), y tubos metálicos de cobre (13, 11), (22, 11) y de acero
(22, 4), (22, 8).

Sucesos de prueba: una placa en la calle (3, 12) que lleva al balcón; un mensaje al entrar en el
edificio (una vez); tres palancas en la calle (x 1, 6, 11; y = 15) que se usan con E y cuentan;
y un paso al este (x 23-24, y 14-15) que, con las tres palancas, lleva a la planta 1 del
edificio y, si no, dice cuántas faltan.
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

    # Luces: calle, bajo el puente, las dos plantas del edificio y el balcón
    for x, y in ((3.0, 12.0), (12.5, 12.5), (24.0, 13.0), (10.0, 3.0)):
        b.light(0, x, y, "#ffcf8a", radius=7.0, intensity=1.6)
    b.light(0, 6.5, 8.5, "#8fbfff", radius=5.0, intensity=1.3, height=1.8)
    b.light(0, 15.5, 5.0, "#ffe2b8", radius=5.5, intensity=1.4)
    b.light(0, 19.0, 8.0, "#ffd0a0", radius=5.0, intensity=1.2)
    b.light(1, 17.5, 6.0, "#ffb070", radius=6.0, intensity=1.5)
    b.light(1, 16.0, 10.5, "#ff9a6a", radius=4.0, intensity=1.2, height=1.6)

    # --- Superficies: suelo pulido en el edificio, líneas que brillan en la calle y tubos ---
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            if (xx, yy) not in STAIRS:
                b.put(0, xx, yy, "floor_polished", replace=True)
    for yy in (16, 17):
        for xx in range(1, 10):
            b.put(0, xx, yy, "floor_glow", replace=True)
    for (xx, yy), name in (((13, 11), "pipe_copper"), ((22, 11), "pipe_copper"),
                           ((22, 4), "pipe_steel"), ((22, 8), "pipe_steel")):
        b.put(0, xx, yy, name)
    b.light(0, 5.5, 15.0, "#9fe8ff", radius=6.0, intensity=1.2)

    # --- Sucesos de prueba ---
    b.marker(1, 15.9, 10.5, "balcony")
    b.marker(1, 17.5, 4.5, "upstairs")
    b.event(0, 3, 12, 4, 13, "plate", when="enter",
            do='message "Una placa mágica te lleva al balcón"\nteleport balcony')
    b.event(0, x0 + 1, y1, x1, y1 + 1, "welcome", when="enter", times=1,
            do='message "Bienvenido al edificio"')
    for i, lx in enumerate((1, 6, 11), start=1):
        b.event(0, lx, 15, lx + 1, 16, f"lever_{i}", when="use", times=1,
                do='add levers 1\nmessage "Palanca {levers} de 3"')
    b.event(0, 23, 14, 25, 16, "gate", when="enter", **{"if": "levers >= 3"},
            do='message "Las tres palancas abren el paso"\nwait 1\nteleport upstairs')
    b.event(0, 23, 14, 25, 16, "gate_closed", when="enter", **{"if": "levers < 3"},
            do='message "El paso está cerrado: faltan palancas ({levers} de 3)"')
    m = b.to_map(tsx_path, tmx_path, proj)
    tmx_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(tmx_path, encoding="csv")
    return m
