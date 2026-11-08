"""Tiempo de GPU por tramo de dibujo, con consultas de OpenGL (GL_TIME_ELAPSED).

El resultado de una consulta no está listo hasta unos frames después; para no parar el dibujo
esperándolo, las consultas de los últimos frames se guardan en un anillo y cada una se lee
cuando ya está disponible. Los tiempos se suavizan (media móvil) y se dan en milisegundos.

    timer = GpuTimer()
    with timer("mundo"):
        ...dibujar...
    timer.frame_done()             # al terminar cada frame
    timer.ms                       # {"mundo": 3.2, ...}
"""

from contextlib import contextmanager

import numpy as np
from OpenGL.GL import (
    GL_QUERY_RESULT, GL_QUERY_RESULT_AVAILABLE, GL_TIME_ELAPSED, glBeginQuery, glDeleteQueries, glEndQuery,
    glGenQueries, glGetQueryObjectiv, glGetQueryObjectuiv,
)

SMOOTHING = 0.1


class GpuTimer:
    def __init__(self, frames_in_flight: int = 4):
        self.ring = [[] for _ in range(frames_in_flight)]
        self.index = 0
        self.free = []
        self.ms = {}
        self.enabled = True
        self._all = []

    def _query(self) -> int:
        if self.free:
            return self.free.pop()
        q = int(np.atleast_1d(glGenQueries(1))[0])
        self._all.append(q)
        return q

    @contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        q = self._query()
        glBeginQuery(GL_TIME_ELAPSED, q)
        try:
            yield
        finally:
            glEndQuery(GL_TIME_ELAPSED)
            self.ring[self.index].append((name, q))

    def frame_done(self):
        """Cierra el frame y recoge los resultados del más antiguo del anillo."""
        self.index = (self.index + 1) % len(self.ring)
        oldest = self.ring[self.index]
        if oldest:
            totals = {}
            ready = True
            for _name, q in oldest:
                available = np.zeros(1, np.int32)
                glGetQueryObjectiv(q, GL_QUERY_RESULT_AVAILABLE, available)
                ready &= bool(available[0])
            if ready:                                   # si no, se descarta (sin esperar a la GPU)
                for name, q in oldest:
                    ns = np.zeros(1, np.uint32)            # nanosegundos: 32 bits llegan a 4 s
                    glGetQueryObjectuiv(q, GL_QUERY_RESULT, ns)
                    totals[name] = totals.get(name, 0.0) + float(ns[0]) / 1e6
                for name in set(totals) | set(self.ms):
                    self.ms[name] = (1 - SMOOTHING) * self.ms.get(name, totals.get(name, 0.0)) \
                        + SMOOTHING * totals.get(name, 0.0)
            self.free.extend(q for _name, q in oldest)
        self.ring[self.index] = []

    def delete(self):
        if self._all:
            glDeleteQueries(len(self._all), self._all)
        self._all, self.free = [], []
