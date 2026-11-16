"""Celdas de 16 bits con paleta del mundo, y estado disperso de las celdas que lo tienen."""
import numpy as np
import pytest

from engine.compiler import compile_tmx
from engine.world import cell as C
from engine.world.palette import AIR, MAX_ENTRIES, SOLID, Palette, PaletteFull
from engine.world.state import CellState
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.levels import build_levels


def test_reserved_ids_and_deduplication():
    p = Palette()
    assert p.lut[AIR] == 0 and p.lut[SOLID] == C.CELL_SOLID and len(p) == 0
    assert p.add(0) == AIR and p.add(int(C.CELL_SOLID)) == SOLID          # sin entrada
    a = p.add(1234)
    assert a == 1 and p.add(1234) == a and len(p) == 1
    b = p.add(1234, {"friction": 0.1})                                     # propiedades nuevas: otra entrada
    assert b == 2 and p.props[b] == {"friction": 0.1}
    ids = p.ids_for(np.array([[0, 1234], [99, int(C.CELL_SOLID)]], np.uint32))
    assert ids.dtype == np.uint16 and ids.tolist() == [[0, 1], [3, SOLID]]
    back = Palette.from_rows(p.rows())
    assert np.array_equal(back.lut, p.lut) and back.props == p.props and back.add(99) == 3


def test_palette_full():
    p = Palette()
    p.ids_for(np.arange(1, MAX_ENTRIES + 1, dtype=np.uint32) + 0x100000)
    assert len(p) == MAX_ENTRIES
    with pytest.raises(PaletteFull):
        p.add(7)


@pytest.fixture(scope="module")
def levels(tmp_path_factory, shared_tileset):
    d = tmp_path_factory.mktemp("pal")
    build_levels(d / "l.tmx", shared_tileset)
    report = compile_tmx(d / "l.tmx", d / "l.cworld", nav=False)
    assert report.ok and 0 < report.stats["paleta"] < 100
    return d / "l.cworld"


def test_world_cells_are_16_bit_ids(levels):
    v = WorldView(SQLiteWorldStore(levels))
    ys, xs = np.mgrid[0:18, 0:26]
    ids = v.cell_ids(xs, ys, 0)
    assert ids.dtype == np.uint16 and v.id_pool.dtype == np.uint16
    assert np.array_equal(v.cells(xs, ys, 0), v.palette.lut[ids])
    assert (v.cell_ids(xs, ys, -1) == SOLID).all()                         # bajo las columnas, macizo
    assert C.wall_n(v.cells(17, 3, 0)) and C.fill(v.cells(3, 14, -1)) == C.FILLS.index("solid")


def test_set_cell_grows_the_palette(levels):
    v = WorldView(SQLiteWorldStore(levels))
    before, n = int(v.cells(5, 14, 0)), len(v.palette)
    pid = v.set_cell(5, 14, 0, before | C.FLAG_ACTIVE)
    assert pid == n + 1 and int(v.cells(5, 14, 0)) == before | C.FLAG_ACTIVE
    assert v.chunk(0, 0).obstacles[v.chunk(0, 0).offset[14 * 32 + 5]] == before | C.FLAG_ACTIVE
    with pytest.raises(ValueError):
        v.set_cell(5, 14, 9, 1)                                            # fuera del tramo guardado


def test_cell_state_is_sparse_and_vectorised(tmp_path, levels):
    s = CellState()
    s.set(5, 14, 0, 3)
    s.set(-7, 2, 1, 0xFFFF)
    s.set(6, 14, 0, 9)
    s.set(6, 14, 0, 0)                                                     # 0: sin entrada
    assert len(s) == 2
    got = s.get(np.array([5, 6, -7, 0]), np.array([14, 14, 2, 0]), np.array([0, 0, 1, 0]))
    assert got.dtype == np.uint16 and got.tolist() == [3, 0, 0xFFFF, 0]
    with pytest.raises(ValueError):
        s.set(0, 0, 0, 0x10000)
    assert sorted(CellState(s.rows()).rows()) == sorted(s.rows())
    store = SQLiteWorldStore(levels, "a")
    store.write_cell_state(s)
    store.db.commit()
    assert WorldView(SQLiteWorldStore(levels)).state.get(-7, 2, 1) == 0xFFFF


def test_old_worlds_ask_to_be_recompiled(tmp_path):
    store = SQLiteWorldStore(tmp_path / "old.cworld", "w")
    store.set_meta({"format_version": 2, "chunk_size": 32})
    store.db.commit()
    with pytest.raises(ValueError, match="vuelve a compilarlo"):
        WorldView(SQLiteWorldStore(tmp_path / "old.cworld"))
