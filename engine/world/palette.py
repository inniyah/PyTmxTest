"""Paleta de celdas del mundo: cada celda guarda un índice de 16 bits a una descripción.

Índices reservados: 0 = aire, 0xFFFF = macizo (lo que hay bajo el terreno, bloques rellenos);
ninguno de los dos necesita entrada. Las descripciones van del 1 al 0xFFFE (65 534): cada
combinación distinta de paredes, suelos, techos, material y flags que aparece en el mundo, una
sola vez. Cada entrada guarda su valor de 32 bits de siempre (ver ``cell``: el resto del motor
sigue leyendo las celdas así, con una consulta a ``lut``) y un diccionario ``props`` abierto a
propiedades nuevas sin cambiar el formato.

La paleta puede crecer durante la partida (una pared que se rompe es otra combinación).
"""

import json
from typing import Dict, Iterable, Optional

import numpy as np

from . import cell as C

AIR = 0
SOLID = 0xFFFF
MAX_ENTRIES = 0xFFFE           # descripciones posibles (1 … 0xFFFE)


class PaletteFull(ValueError):
    pass


class Palette:
    def __init__(self, entries: Iterable = ()):
        """``entries``: (id, valor de 32 bits, props) en el orden de sus identificadores."""
        self.lut = np.zeros(65536, np.uint32)        # índice -> valor de 32 bits
        self.lut[SOLID] = C.CELL_SOLID
        self.props: Dict[int, dict] = {}
        self._ids: Dict[int, int] = {0: AIR, int(C.CELL_SOLID): SOLID}
        self.next_id = 1
        for pid, value, props in entries:
            if pid != self.next_id:
                raise ValueError(f"Paleta con huecos: se esperaba el {self.next_id} y llega el {pid}")
            self.add(int(value), props)

    def __len__(self) -> int:
        """Descripciones guardadas (sin contar aire ni macizo)."""
        return self.next_id - 1

    def add(self, value: int, props: Optional[dict] = None) -> int:
        """Índice de una combinación (de 32 bits, y sus propiedades nuevas si las hay); si no
        está, se añade."""
        value = int(value)
        if not props and value in self._ids:
            return self._ids[value]
        if self.next_id > MAX_ENTRIES:
            raise PaletteFull(f"La paleta no admite más de {MAX_ENTRIES} combinaciones de celda distintas")
        pid = self.next_id
        self.next_id += 1
        self.lut[pid] = value
        if props:
            self.props[pid] = dict(props)
        else:
            self._ids[value] = pid
        return pid

    def ids_for(self, values: np.ndarray) -> np.ndarray:
        """Índices (uint16) de un array de valores de 32 bits, añadiendo los que falten."""
        values = np.asarray(values, np.uint32)
        uniq, inverse = np.unique(values, return_inverse=True)
        ids = np.array([self.add(int(v)) for v in uniq], np.uint16)
        return ids[inverse].reshape(values.shape)

    def rows(self):
        """(id, valor, props en JSON) de cada descripción, para guardarla."""
        return [(pid, int(self.lut[pid]), json.dumps(self.props[pid]) if pid in self.props else None)
                for pid in range(1, self.next_id)]

    @classmethod
    def from_rows(cls, rows) -> "Palette":
        return cls((pid, value, json.loads(props) if props else None) for pid, value, props in rows)
