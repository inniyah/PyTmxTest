"""Formato de celda de 32 bits (ver docs/diseño, "Formato de celda y obstáculos").

    bits 0      pared en el borde norte
    bit  1      pared en el borde oeste          (sur y este son N/O de la vecina)
    bits 2-4    relleno: vacío, macizo, 4 medias diagonales macizas, 2 diagonales finas
    bit  5      diagonal que divide la celda: 0 = "/" (SO-NE), 1 = "\\" (NO-SE)
    bits 6-11   suelo de la mitad 1
    bits 12-17  suelo de la mitad 2
    bits 18-19  techo bajo de la mitad 1
    bits 20-21  techo bajo de la mitad 2
    bits 22-31  comportamiento: material (22-25) y flags (26-31)

Coordenadas dentro de la celda: u de 0 (oeste) a 1 (este), v de 0 (norte) a 1 (sur).
Alturas en plantas: 0 = suelo de la celda, 1 = su techo.

Mitades: con "/" la mitad 1 es la NO (u + v < 1) y la 2 la SE; con "\\" la mitad 1
es la NE (u > v) y la 2 la SO. Una celda vacía, sin paredes ni suelo, vale 0.

Todo funciona con escalares y con arrays de numpy.
"""

from typing import Dict, Tuple

import numpy as np

# --- Campos ---------------------------------------------------------------

WALL_N = 1 << 0
WALL_W = 1 << 1
FILL_SHIFT, FILL_MASK = 2, 0b111 << 2
SPLIT_BIT = 1 << 5
FLOOR1_SHIFT, FLOOR2_SHIFT, FLOOR_MASK = 6, 12, 0b111111
CEIL1_SHIFT, CEIL2_SHIFT, CEIL_MASK = 18, 20, 0b11
MATERIAL_SHIFT, MATERIAL_MASK = 22, 0b1111
FLAGS_SHIFT, FLAGS_MASK = 26, 0b111111
GEOMETRY_MASK = (1 << 22) - 1

# Flags de comportamiento (bits 26-31)
FLAG_ACTIVE = 1 << 26       # celda activa: tiene una entrada en la tabla de eventos
FLAG_LADDER = 1 << 27       # escalera de mano

# Diagonales
SPLIT_SLASH = 0             # "/"  de la esquina SO a la NE
SPLIT_BACKSLASH = 1         # "\\" de la esquina NO a la SE

# --- Relleno -------------------------------------------------------------

FILLS = ("empty", "solid", "half_nw", "half_ne", "half_se", "half_sw",
         "thin_slash", "thin_backslash")
FILL = {name: i for i, name in enumerate(FILLS)}

# Valor de una celda maciza: lo que responden las consultas fuera del mapa o por debajo
# del tramo guardado de una columna
CELL_SOLID = FILL["solid"] << FILL_SHIFT

# Medias macizas: qué diagonal implican y qué mitad ocupan (1 o 2)
FILL_HALF = {"half_nw": (SPLIT_SLASH, 1), "half_se": (SPLIT_SLASH, 2),
             "half_ne": (SPLIT_BACKSLASH, 1), "half_sw": (SPLIT_BACKSLASH, 2)}
FILL_THIN_SPLIT = {"thin_slash": SPLIT_SLASH, "thin_backslash": SPLIT_BACKSLASH}

# --- Suelos: 45 piezas, cada una un plano  h = a + b·u + c·v ---------------
#
# Rampas rectas: suben ½ planta a lo ancho de la celda, base 0 o ½ (escaleras, 37°).
# Rampas diagonales: suben ½ planta de esquina a esquina, base 0, ¼, ½ o ¾
# (con base ¾ la esquina alta llega a 1¼: sobresale de la celda, y se permite).
# Pendientes suaves ("slope"): suben ¼ de planta a lo ancho de la celda (21°), base 0, ¼, ½
# o ¾. Con ellas y las diagonales, cualquier casilla de un terreno de alturas en las esquinas
# (tipo Populous: vecinas a un cuarto de planta como mucho) se representa exactamente.
# Los códigos 0-28 no cambian; las pendientes suaves son los 29-44.

_AXIS_DIRS = {"N": (0.5, 0.0, -0.5), "S": (0.0, 0.0, 0.5),
              "E": (0.0, 0.5, 0.0), "W": (0.5, -0.5, 0.0)}
_DIAG_DIRS = {"NE": (0.25, 0.25, -0.25), "SE": (0.0, 0.25, 0.25),
              "SW": (0.25, -0.25, 0.25), "NW": (0.5, -0.25, -0.25)}
_SLOPE_DIRS = {"N": (0.25, 0.0, -0.25), "S": (0.0, 0.0, 0.25),
               "E": (0.0, 0.25, 0.0), "W": (0.25, -0.25, 0.0)}


def _build_floor_table():
    names = ["none"]
    coefs = [(np.nan, np.nan, np.nan)]
    for h in (0.0, 0.25, 0.5, 0.75):
        names.append(f"flat {h:g}")
        coefs.append((h, 0.0, 0.0))
    for d, (a, b, c) in _AXIS_DIRS.items():
        for base in (0.0, 0.5):
            names.append(f"ramp {d} {base:g}")
            coefs.append((a + base, b, c))
    for d, (a, b, c) in _DIAG_DIRS.items():
        for base in (0.0, 0.25, 0.5, 0.75):
            names.append(f"ramp {d} {base:g}")
            coefs.append((a + base, b, c))
    for d, (a, b, c) in _SLOPE_DIRS.items():
        for base in (0.0, 0.25, 0.5, 0.75):
            names.append(f"slope {d} {base:g}")
            coefs.append((a + base, b, c))
    return names, np.array(coefs, dtype=np.float64)


FLOOR_NAMES, FLOOR_COEFS = _build_floor_table()
# Altura máxima de cada pieza de suelo en la celda (NaN sin suelo): mayor que 1 = sobresale
FLOOR_MAX = np.array([np.nan] + [max(a + b * u + c * v for u in (0, 1) for v in (0, 1))
                                  for a, b, c in FLOOR_COEFS[1:]])
FLOOR = {name: i for i, name in enumerate(FLOOR_NAMES)}
FLOOR_NONE = FLOOR["none"]
assert len(FLOOR_NAMES) == 45 and FLOOR_NONE == 0 and len(FLOOR_NAMES) <= FLOOR_MASK + 1

# --- Techos bajos ---------------------------------------------------------

CEILING_HEIGHTS = (1.0, 0.75, 0.5, 0.25)        # código 0 = sin techo bajo
CEILING = {h: i for i, h in enumerate(CEILING_HEIGHTS)}

# --- Partes de la celda ---------------------------------------------------

# parte -> (diagonal que implica o None, mitades que cubre)
PARTS: Dict[str, Tuple] = {
    "all": (None, (1, 2)),
    "nw": (SPLIT_SLASH, (1,)), "se": (SPLIT_SLASH, (2,)),
    "ne": (SPLIT_BACKSLASH, (1,)), "sw": (SPLIT_BACKSLASH, (2,)),
}


# --- Lectura de campos ----------------------------------------------------

def _arr(cells):
    return np.asarray(cells, dtype=np.uint32)


def wall_n(cells):
    return (_arr(cells) & WALL_N) != 0


def wall_w(cells):
    return (_arr(cells) & WALL_W) != 0


def fill(cells):
    return (_arr(cells) >> FILL_SHIFT) & 0b111


def split(cells):
    return (_arr(cells) & SPLIT_BIT) != 0


def floor_code(cells, half):
    shift = FLOOR1_SHIFT if half == 1 else FLOOR2_SHIFT
    return (_arr(cells) >> shift) & FLOOR_MASK


def ceiling_code(cells, half):
    shift = CEIL1_SHIFT if half == 1 else CEIL2_SHIFT
    return (_arr(cells) >> shift) & CEIL_MASK


# --- Geometría dentro de la celda ------------------------------------------

def half_at(split_bits, u, v):
    """1 o 2: la mitad de la celda que contiene el punto (u, v)."""
    split_bits = np.asarray(split_bits, dtype=bool)
    in_first = np.where(split_bits, np.asarray(u) > np.asarray(v),
                        np.asarray(u) + np.asarray(v) < 1.0)
    return np.where(in_first, 1, 2)


def floor_height(codes, u, v):
    """Altura del suelo (en plantas) en (u, v) para cada pieza; NaN si no hay suelo."""
    a, b, c = np.moveaxis(FLOOR_COEFS[np.asarray(codes)], -1, 0)   # sirve con cualquier forma
    return a + b * np.asarray(u) + c * np.asarray(v)


def surface_at(cells, u, v):
    """Altura del suelo de la celda en el punto (u, v), según la mitad que le toca."""
    cells = _arr(cells)
    half = half_at(split(cells), u, v)
    code = np.where(half == 1, floor_code(cells, 1), floor_code(cells, 2))
    return floor_height(code, u, v)


def ceiling_at(cells, u, v):
    """Altura del techo de la celda en (u, v): 1 sin techo bajo, si no ¾, ½ o ¼."""
    cells = _arr(cells)
    half = half_at(split(cells), u, v)
    code = np.where(half == 1, ceiling_code(cells, 1), ceiling_code(cells, 2))
    return np.asarray(CEILING_HEIGHTS)[code]


def fill_contains(fill_codes, u, v):
    """¿Está (u, v) dentro de la parte maciza del relleno? (Las diagonales finas no ocupan.)"""
    f = np.asarray(fill_codes)
    u, v = np.asarray(u), np.asarray(v)
    return ((f == FILL["solid"])
            | ((f == FILL["half_nw"]) & (u + v < 1)) | ((f == FILL["half_se"]) & (u + v > 1))
            | ((f == FILL["half_ne"]) & (u > v)) | ((f == FILL["half_sw"]) & (u < v)))


def floor_range(code) -> Tuple[float, float]:
    """(mínimo, máximo) de una pieza de suelo sobre la celda entera."""
    a, b, c = FLOOR_COEFS[code]
    corners = [a + b * u + c * v for u in (0, 1) for v in (0, 1)]
    return min(corners), max(corners)
