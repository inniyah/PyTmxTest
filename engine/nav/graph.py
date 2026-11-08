"""Grafo de navegación cargado desde un mundo compilado (tabla ``nav``)."""

import io

import numpy as np

from .build import DIRS, _Columns


class NavGraph:
    """Nodos de todo el mundo como arrays planos, con sus aristas como índices globales."""

    def __init__(self, store, chunk_size: int, cell_w: float = 2.0):
        self.cell_w = cell_w
        xs, ys, hs, mats, kts, cls, ecls = [], [], [], [], [], [], []
        n = chunk_size
        for cx, cy in store.chunks():
            raw = store.read_nav(cx, cy)
            if raw is None:
                continue
            d = np.load(io.BytesIO(raw))
            col = d["col"].astype(np.int64)
            xs.append(cx * n + col % n)
            ys.append(cy * n + col // n)
            hs.append(d["h"].astype(np.float64))
            mats.append(d["material"])
            kts.append(d["target"].astype(np.int64))
            cls.append(d["clear"].astype(np.float64) / 100)
            ecls.append(d["edge_clear"].astype(np.float64) / 100)
        if not xs:
            raise ValueError("El mundo no tiene grafo de navegación: vuelve a compilarlo")
        self.x, self.y, self.h = np.concatenate(xs), np.concatenate(ys), np.concatenate(hs)
        self.material = np.concatenate(mats)
        self.clear = np.concatenate(cls)                    # holgura del nodo (m)
        self.edge_clear = np.concatenate(ecls)              # (n, 8) holgura de cada arista (m)
        k_target = np.concatenate(kts)
        self.columns = _Columns(self.x, self.y)
        # Índice global del nodo k de la columna (x, y): inicio de la columna + k
        self.target = np.full(k_target.shape, -1, np.int64)
        for di, (dx, dy) in enumerate(DIRS):
            has = k_target[:, di] >= 0
            lo, hi = self.columns.ranges(self.x[has] + dx, self.y[has] + dy)
            k = k_target[has, di]
            valid = lo + k < hi
            idx = np.full(has.sum(), -1, np.int64)
            idx[valid] = self.columns.order[lo[valid] + k[valid]]
            self.target[np.nonzero(has)[0], di] = idx

    def __len__(self):
        return len(self.x)

    @property
    def edges(self) -> int:
        return int((self.target >= 0).sum())

    def usable(self, radius: float) -> np.ndarray:
        """(n, 8): aristas por las que cabe un cuerpo de ese radio."""
        return (self.target >= 0) & (self.edge_clear >= radius - 0.005)

    def node_at(self, px, pz, py, reach: float = 0.5):
        """Nodo de la celda bajo (px, pz) cuya altura está más cerca de ``py`` (dentro de
        ``reach``), o -1."""
        px, pz, py = (np.atleast_1d(np.asarray(a, float)) for a in (px, pz, py))
        cx = np.floor(px / self.cell_w).astype(np.int64)
        cy = np.floor(pz / self.cell_w).astype(np.int64)
        lo, hi = self.columns.ranges(cx, cy)
        out = np.full(len(px), -1, np.int64)
        best = np.full(len(px), np.inf)
        for j in range(int((hi - lo).max(initial=0))):
            has = lo + j < hi
            idx = self.columns.order[np.minimum(lo + j, len(self.columns.order) - 1)]
            dist = np.abs(self.h[idx] - py)
            better = has & (dist < best) & (dist <= reach)
            out[better], best[better] = idx[better], dist[better]
        return out

    def position(self, node):
        """Centro (x, y, z) en metros del nodo."""
        node = np.asarray(node)
        return np.stack([(self.x[node] + 0.5) * self.cell_w, self.h[node], (self.y[node] + 0.5) * self.cell_w], -1)
