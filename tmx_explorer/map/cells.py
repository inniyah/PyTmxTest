"""Formato de celda de 16 bits: forma de colisión + flags de comportamiento.

    bits 0-3    forma            EMPTY=0, BLOCK=1, resto reservado (ver SHAPES)
    bits 4-5    variante         orientación (N, E, S, O) o parámetro propio de la forma
    bits 6-7    altura superior  1, 3/4, 1/2, 1/4 del nivel (0 = altura completa)
    bits 8-9    altura inferior  0, 1/4, 1/2, 3/4
    bits 10-15  flags            libres (escalera de mano, agua, daño...)

``EMPTY`` vale 0 y ``BLOCK`` con altura completa vale 1, igual que el antiguo
"sólido", así que el camino rápido (vacío / bloque) es una sola comparación.

Hoy solo están implementadas EMPTY y BLOCK. El resto de formas tienen el código
reservado; mientras no se implementen, cualquier geometría distinta de EMPTY se
trata como sólida (lo prudente: mejor chocar que atravesar).
"""

from typing import Dict, Optional

import numpy as np

# --- Campos ---
SHAPE_MASK = 0x000F
VARIANT_SHIFT, VARIANT_MASK = 4, 0x0030
TOP_SHIFT, TOP_MASK = 6, 0x00C0
BASE_SHIFT, BASE_MASK = 8, 0x0300
GEOMETRY_MASK = 0x03FF
FLAGS_SHIFT, FLAGS_MASK = 10, 0xFC00

# --- Formas (códigos reservados; solo EMPTY y BLOCK implementadas) ---
SHAPES = {
    "empty": 0, "block": 1, "half": 2, "diagonal": 3, "corner": 4, "inner_corner": 5,
    "ramp": 6, "ramp_outer": 7, "ramp_inner": 8, "plate": 9, "thin_wall": 10, "pillar": 11,
}
EMPTY = SHAPES["empty"]
BLOCK = SHAPES["block"]
IMPLEMENTED_SHAPES = {EMPTY, BLOCK}

VARIANTS = {"north": 0, "east": 1, "south": 2, "west": 3}
TOP_HEIGHTS = {1.0: 0, 0.75: 1, 0.5: 2, 0.25: 3}
BASE_HEIGHTS = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3}


def encode(shape: int = BLOCK, variant: int = 0, top: float = 1.0, base: float = 0.0,
           flags: int = 0) -> int:
    if top not in TOP_HEIGHTS or base not in BASE_HEIGHTS:
        raise ValueError(f"Alturas no válidas: top={top}, base={base}")
    return ((shape & SHAPE_MASK)
            | (variant & 3) << VARIANT_SHIFT
            | TOP_HEIGHTS[top] << TOP_SHIFT
            | BASE_HEIGHTS[base] << BASE_SHIFT
            | (flags & 0x3F) << FLAGS_SHIFT)


def shape_of(cells):
    return np.asarray(cells) & SHAPE_MASK


def is_solid(cells):
    """True donde la celda tiene geometría (vale para escalares y arrays)."""
    return (np.asarray(cells) & GEOMETRY_MASK) != EMPTY


# --- Desde las propiedades de los tiles en Tiled ---

def _prop(tile, name, default=None):
    prop = tile.properties.get(name)
    return default if prop is None else prop.value


def cell_value_for_tile(tile) -> Optional[int]:
    """Valor de celda definido por las propiedades de un tile, o None si no define nada.

    ``solid = true`` equivale a ``shape = block``. Propiedades admitidas:
    ``shape`` (nombre de SHAPES), ``facing`` (north/east/south/west), ``top``, ``base``.
    """
    shape_name = _prop(tile, "shape")
    if shape_name is None:
        solid = _prop(tile, "solid")
        if solid is None:
            return None
        solid = solid if isinstance(solid, bool) else str(solid).lower() == "true"
        return encode(BLOCK) if solid else EMPTY

    shape = SHAPES.get(str(shape_name).lower())
    if shape is None:
        raise ValueError(f"Tile {tile.id}: forma desconocida '{shape_name}'")
    variant = VARIANTS.get(str(_prop(tile, "facing", "north")).lower(), 0)
    return encode(shape, variant, float(_prop(tile, "top", 1.0)), float(_prop(tile, "base", 0.0)))


def build_gid_table(tilesets, max_gid: int) -> np.ndarray:
    """Tabla GID -> valor de celda (uint16), para convertir mapas enteros con numpy."""
    table = np.zeros(max_gid + 1, dtype=np.uint16)
    for tileset in tilesets:
        for tile_id, tile in tileset.tiles.items():
            gid = tileset.firstgid + tile_id
            value = cell_value_for_tile(tile)
            if value is not None and gid <= max_gid:
                table[gid] = value
    return table


def describe(value: int) -> Dict[str, object]:
    """Desglose legible de un valor de celda (para depuración)."""
    names = {v: k for k, v in SHAPES.items()}
    tops = {v: k for k, v in TOP_HEIGHTS.items()}
    bases = {v: k for k, v in BASE_HEIGHTS.items()}
    return {
        "shape": names.get(value & SHAPE_MASK, f"reservada({value & SHAPE_MASK})"),
        "variant": (value & VARIANT_MASK) >> VARIANT_SHIFT,
        "top": tops[(value & TOP_MASK) >> TOP_SHIFT],
        "base": bases[(value & BASE_MASK) >> BASE_SHIFT],
        "flags": (value & FLAGS_MASK) >> FLAGS_SHIFT,
    }
