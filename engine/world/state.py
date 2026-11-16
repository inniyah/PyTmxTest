"""Estado de las celdas que cambian durante la partida: 16 bits por celda, solo en las que lo
tienen (en un mapa grande, un porcentaje pequeño: puertas, trampas, palancas, celdas dañadas…).

Lo que significa cada bit lo decide quien lo usa (una puerta: abierta o cerrada; una trampa:
armada; daño acumulado…). Una celda sin entrada tiene estado 0. Leer muchas celdas a la vez es
vectorizado: claves ordenadas y búsqueda binaria.
"""

from typing import Dict, Iterable, Tuple

import numpy as np


def _key(x, y, z):
    return ((np.asarray(z, np.int64) + 512) << 42) | ((np.asarray(y, np.int64) + (1 << 20)) << 21) \
        | (np.asarray(x, np.int64) + (1 << 20))


class CellState:
    def __init__(self, rows: Iterable[Tuple[int, int, int, int]] = ()):
        """``rows``: (x, y, nivel, estado)."""
        self._values: Dict[int, int] = {}
        self._sorted = None
        for x, y, z, value in rows:
            self.set(x, y, z, value)

    def __len__(self) -> int:
        return len(self._values)

    def set(self, x: int, y: int, z: int, value: int):
        if not 0 <= int(value) <= 0xFFFF:
            raise ValueError(f"El estado de una celda es de 16 bits: {value}")
        k = int(_key(x, y, z))
        if value:
            self._values[k] = int(value)
        else:
            self._values.pop(k, None)               # estado 0: sin entrada
        self._sorted = None

    def get(self, x, y, z) -> np.ndarray:
        """Estado (uint16) de una o muchas celdas."""
        keys = _key(x, y, z)
        out = np.zeros(np.shape(keys), np.uint16)
        if not self._values:
            return out
        if self._sorted is None:
            k = np.fromiter(self._values.keys(), np.int64, len(self._values))
            order = np.argsort(k)
            self._sorted = (k[order], np.fromiter(self._values.values(), np.uint16, len(self._values))[order])
        sk, sv = self._sorted
        pos = np.clip(np.searchsorted(sk, keys), 0, len(sk) - 1)
        hit = sk[pos] == keys
        out[hit] = sv[pos[hit]]
        return out

    def rows(self):
        """(x, y, nivel, estado) de cada celda con estado."""
        out = []
        for k, v in self._values.items():
            z = (k >> 42) - 512
            y = ((k >> 21) & ((1 << 21) - 1)) - (1 << 20)
            x = (k & ((1 << 21) - 1)) - (1 << 20)
            out.append((int(x), int(y), int(z), v))
        return out
