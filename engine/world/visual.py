"""Hueco visual compilado: 32 bits por hueco, 4 huecos por celda (de abajo arriba).

    bits 0-17   región del atlas (1..262143; 0 = hueco vacío)
    bits 18-19  rol: 0 FLOOR, 1 OBJECT, 2 OVERHEAD, 3 ROOF
    bit  20     volteado en horizontal
    bits 21-24  opacidad: alfa = código / 15 (15 = opaco)
    bits 25-31  reservados
"""

import numpy as np

REGION_MASK = (1 << 18) - 1
ROLE_SHIFT = 18
FLIP_H = 1 << 20
OPACITY_SHIFT = 21
SLOTS = 4

ROLE_FLOOR, ROLE_OBJECT, ROLE_OVERHEAD, ROLE_ROOF = range(4)


def pack(region, role, flip_h, opacity):
    """Vectorizado: región (int), rol (0-3), volteo (bool), opacidad (0.0-1.0)."""
    region = np.asarray(region, dtype=np.uint32)
    code = np.clip(np.rint(np.asarray(opacity, dtype=float) * 15), 0, 15).astype(np.uint32)
    value = (region & REGION_MASK) | (np.asarray(role, np.uint32) << ROLE_SHIFT) \
        | np.where(np.asarray(flip_h, bool), FLIP_H, 0).astype(np.uint32) | (code << OPACITY_SHIFT)
    return np.where(region == 0, 0, value).astype(np.uint32)


def region(values):
    return np.asarray(values, np.uint32) & REGION_MASK


def role(values):
    return (np.asarray(values, np.uint32) >> ROLE_SHIFT) & 0b11


def flipped(values):
    return (np.asarray(values, np.uint32) & FLIP_H) != 0


def opacity(values):
    return ((np.asarray(values, np.uint32) >> OPACITY_SHIFT) & 0b1111) / 15.0
