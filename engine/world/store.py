"""Almacenamiento del mundo compilado.

El motor solo habla con la interfaz ``WorldStore``. La implementación actual es un único
archivo SQLite (``SQLiteWorldStore``); si algún día hiciera falta otra (un binario propio),
basta con escribirla sin tocar el resto del motor.

Esquema:
    meta          (key, value)                        valores JSON
    chunks        (cx, cy, col_start, ..., vis_index)  un chunk por fila, cada parte un BLOB
    atlas_regions (id, page, x, y, w, h, border)       región 1.. de cada tile visual
    atlas_pages   (page, kind, png)                    kind: color, albedo, normal, position
    lights        (id, x, y, z, r, g, b, radius, intensity)   luces puntuales, en metros
"""

import json
import sqlite3
import zlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

from .chunk import PARTS, ChunkData

FORMAT_VERSION = 2          # 2: suelos de 6 bits por mitad (pendientes suaves)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS chunks (cx INTEGER, cy INTEGER,
    {", ".join(p + " BLOB" for p in PARTS)}, PRIMARY KEY (cx, cy));
CREATE TABLE IF NOT EXISTS atlas_regions (id INTEGER PRIMARY KEY, page INTEGER,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER, border INTEGER);
CREATE TABLE IF NOT EXISTS atlas_pages (page INTEGER, kind TEXT, png BLOB, PRIMARY KEY (page, kind));
CREATE TABLE IF NOT EXISTS nav (cx INTEGER, cy INTEGER, data BLOB, PRIMARY KEY (cx, cy));
CREATE TABLE IF NOT EXISTS places (id INTEGER PRIMARY KEY, name TEXT, kind TEXT, level INTEGER,
                                   x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER);
CREATE TABLE IF NOT EXISTS lights (id INTEGER PRIMARY KEY, x REAL, y REAL, z REAL,
    r REAL, g REAL, b REAL, radius REAL, intensity REAL);
"""


class WorldStore:
    """Interfaz mínima del almacenamiento del mundo."""

    def meta(self) -> dict: ...
    def read_chunk(self, cx: int, cy: int, parts: Sequence[str] = PARTS) -> Optional[ChunkData]: ...
    def write_chunk(self, chunk: ChunkData): ...
    def chunks(self) -> Iterable[Tuple[int, int]]: ...


class SQLiteWorldStore(WorldStore):
    def __init__(self, path, mode: str = "r"):
        """``mode``: "r" solo lectura (el juego), "w" crea o reescribe (el compilador), "a" abre uno
        existente para añadirle datos (por ejemplo, el grafo de navegación)."""
        self.path = Path(path)
        if mode == "w":
            if self.path.exists():
                self.path.unlink()
            self.db = sqlite3.connect(self.path)
            self.db.executescript(_SCHEMA)
        elif mode == "r":
            if not self.path.exists():
                raise FileNotFoundError(self.path)
            self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        elif mode == "a":
            if not self.path.exists():
                raise FileNotFoundError(self.path)
            self.db = sqlite3.connect(self.path)
            self.db.executescript(_SCHEMA)
        else:
            raise ValueError(mode)
        self._meta: Optional[dict] = None

    # --- Parámetros ---

    def meta(self) -> dict:
        if self._meta is None:
            self._meta = {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM meta")}
        return self._meta

    def set_meta(self, values: Dict[str, object]):
        self.db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                            [(k, json.dumps(v, ensure_ascii=False)) for k, v in values.items()])
        self._meta = None

    @property
    def _compressed(self) -> bool:
        return self.meta().get("compression") == "zlib"

    # --- Chunks ---

    def read_chunk(self, cx: int, cy: int, parts: Sequence[str] = PARTS) -> Optional[ChunkData]:
        row = self.db.execute(f"SELECT {', '.join(parts)} FROM chunks WHERE cx = ? AND cy = ?",
                              (cx, cy)).fetchone()
        if row is None:
            return None
        blobs = {p: (zlib.decompress(b) if self._compressed and b is not None else b)
                 for p, b in zip(parts, row)}
        return ChunkData.from_blobs(cx, cy, self.meta()["chunk_size"], blobs)

    def write_chunk(self, chunk: ChunkData):
        blobs = chunk.to_blobs()
        values = [blobs.get(p) for p in PARTS]
        if self._compressed:
            values = [zlib.compress(v, 6) if v is not None else None for v in values]
        self.db.execute(f"INSERT OR REPLACE INTO chunks VALUES (?, ?, {', '.join('?' * len(PARTS))})",
                        (chunk.cx, chunk.cy, *values))

    def chunks(self) -> Iterator[Tuple[int, int]]:
        yield from self.db.execute("SELECT cx, cy FROM chunks ORDER BY cy, cx")

    # --- Atlas ---

    def write_atlas(self, regions: Sequence[Tuple[int, int, int, int, int, int, int]],
                    pages: Dict[Tuple[int, str], bytes]):
        self.db.executemany("INSERT INTO atlas_regions VALUES (?, ?, ?, ?, ?, ?, ?)", regions)
        self.db.executemany("INSERT INTO atlas_pages VALUES (?, ?, ?)",
                            [(page, kind, png) for (page, kind), png in pages.items()])

    def atlas_regions(self):
        """Filas (id, page, x, y, w, h, border) ordenadas por id."""
        return self.db.execute("SELECT * FROM atlas_regions ORDER BY id").fetchall()

    def atlas_page(self, page: int, kind: str = "color") -> Optional[bytes]:
        row = self.db.execute("SELECT png FROM atlas_pages WHERE page = ? AND kind = ?",
                              (page, kind)).fetchone()
        return row[0] if row else None

    def write_lights(self, rows):
        """Filas (x, y, z, r, g, b, radio, intensidad), en metros y color lineal 0-1."""
        self.db.executemany("INSERT INTO lights (x, y, z, r, g, b, radius, intensity) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def lights(self):
        try:
            return self.db.execute("SELECT x, y, z, r, g, b, radius, intensity FROM lights").fetchall()
        except sqlite3.OperationalError:                  # mundos compilados antes de las luces
            return []

    def atlas_kinds(self):
        return [k for (k,) in self.db.execute("SELECT DISTINCT kind FROM atlas_pages ORDER BY kind")]

    # --- Ciclo de vida ---

    # --- Lugares (rectángulos de celdas [x0, x1) × [y0, y1) en una planta) ---

    def write_places(self, rows):
        self.db.execute("DELETE FROM places")
        self.db.executemany("INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def places(self):
        try:
            return self.db.execute("SELECT id, name, kind, level, x0, y0, x1, y1 FROM places ORDER BY id").fetchall()
        except sqlite3.OperationalError:            # mundo compilado antes de los lugares
            return []

    # --- Navegación (opcional: mundos sin ella siguen funcionando) ---

    def write_nav(self, cx: int, cy: int, data: bytes):
        self.db.execute("INSERT OR REPLACE INTO nav VALUES (?, ?, ?)", (cx, cy, data))

    def read_nav(self, cx: int, cy: int) -> Optional[bytes]:
        try:
            row = self.db.execute("SELECT data FROM nav WHERE cx = ? AND cy = ?", (cx, cy)).fetchone()
        except sqlite3.OperationalError:            # mundo compilado antes de la navegación
            return None
        return row[0] if row else None

    def has_nav(self) -> bool:
        try:
            return self.db.execute("SELECT 1 FROM nav LIMIT 1").fetchone() is not None
        except sqlite3.OperationalError:
            return False

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.commit()
        self.close()
