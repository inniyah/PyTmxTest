"""Formato de celda de 32 bits y CellSpec."""
import itertools

import numpy as np
import pytest

from engine.world import CellConflict, CellSpec
from engine.world import cell as C

EDGE = np.linspace(0, 1, 9)


def h(name, u, v):
    return C.floor_height(C.FLOOR[name], u, v)


def test_field_layout():
    assert C.GEOMETRY_MASK == 0xFFFFF
    assert len(C.FILLS) == 8                       # los 3 bits de relleno, completos
    assert len(C.FLOOR_NAMES) == 29 and C.FLOOR_NONE == 0
    assert CellSpec().encode() == 0                # celda vacía = 0


@pytest.mark.parametrize("d", ["N", "S", "E", "W"])
def test_axis_ramps_chain_across_two_cells(d):
    # Una planta en dos celdas: la de abajo (base 0) y la siguiente en la dirección de subida (base 0.5)
    lo, hi = f"ramp {d} 0", f"ramp {d} 0.5"
    exit_edge = {"N": (EDGE, 0), "S": (EDGE, 1), "E": (1, EDGE), "W": (0, EDGE)}[d]
    entry_edge = {"N": (EDGE, 1), "S": (EDGE, 0), "E": (0, EDGE), "W": (1, EDGE)}[d]
    assert np.allclose(h(lo, *exit_edge), h(hi, *entry_edge))
    assert np.isclose(h(hi, *exit_edge).max(), 1.0)
    assert np.isclose(C.floor_range(C.FLOOR[lo])[0], 0.0)


@pytest.mark.parametrize("d", ["NE", "SE", "SW", "NW"])
def test_diagonal_ramps_are_continuous_with_their_neighbours(d):
    du = 1 if "E" in d else -1                     # la vecina en X hacia donde sube empieza 1/4 más arriba
    dv = -1 if "N" in d else 1
    for base in (0.0, 0.25, 0.5):
        a, b = f"ramp {d} {base:g}", f"ramp {d} {base + 0.25:g}"
        ue_a, ue_b = (1, 0) if du == 1 else (0, 1)
        assert np.allclose(h(a, ue_a, EDGE), h(b, ue_b, EDGE))
        ve_a, ve_b = (0, 1) if dv == -1 else (1, 0)
        assert np.allclose(h(a, EDGE, ve_a), h(b, EDGE, ve_b))
    lo, hi = C.floor_range(C.FLOOR[f"ramp {d} 0"])
    assert (lo, hi) == (0.0, 0.5)
    assert C.floor_range(C.FLOOR[f"ramp {d} 0.75"])[1] == 1.25        # sobresale 1/4


def test_halves():
    u, v = np.array([0.2, 0.8]), np.array([0.2, 0.8])
    assert list(C.half_at(C.SPLIT_SLASH, u, v)) == [1, 2]              # NO, SE
    assert list(C.half_at(C.SPLIT_BACKSLASH, [0.8, 0.2], [0.2, 0.8])) == [1, 2]   # NE, SO


def test_surface_and_ceiling_follow_the_half():
    v = CellSpec(floors={"nw": "flat 0", "se": "ramp N 0.5"}, ceilings={"nw": 0.5}).encode()
    assert C.surface_at(v, 0.1, 0.1) == 0.0
    assert np.isclose(C.surface_at(v, 0.9, 0.9), 0.5 + 0.5 * 0.1)
    assert C.ceiling_at(v, 0.1, 0.1) == 0.5 and C.ceiling_at(v, 0.9, 0.9) == 1.0
    assert np.isnan(C.surface_at(CellSpec().encode(), 0.5, 0.5))


def test_fill_regions():
    f = C.FILL
    assert C.fill_contains(f["solid"], 0.5, 0.5)
    assert C.fill_contains(f["half_nw"], 0.1, 0.1) and not C.fill_contains(f["half_nw"], 0.9, 0.9)
    assert C.fill_contains(f["half_ne"], 0.9, 0.1) and not C.fill_contains(f["half_ne"], 0.1, 0.9)
    assert not C.fill_contains(f["thin_slash"], 0.5, 0.5)                 # las finas no ocupan


def test_walls_only_north_and_west_are_encoded():
    v = CellSpec(walls=frozenset("NESW")).encode()
    assert C.wall_n(v) and C.wall_w(v) and v == C.WALL_N | C.WALL_W


def test_merge_combines_and_detects_conflicts():
    cliff = CellSpec(fill="half_nw").merge(CellSpec(floors={"se": "flat 0"}))
    v = cliff.encode()
    assert C.FILLS[int(C.fill(v))] == "half_nw" and not C.split(v)
    assert C.FLOOR_NAMES[int(C.floor_code(v, 2))] == "flat 0"
    walls = CellSpec(walls=frozenset("N")).merge(CellSpec(walls=frozenset("W")))
    assert walls.walls == frozenset("NW")
    with pytest.raises(CellConflict):
        CellSpec(floors={"nw": "flat 0"}).merge(CellSpec(floors={"ne": "flat 0"}))   # / y \
    with pytest.raises(CellConflict):
        CellSpec(floors={"all": "flat 0"}).merge(CellSpec(floors={"se": "flat 0.5"}))
    with pytest.raises(CellConflict):
        CellSpec(fill="solid").merge(CellSpec(fill="half_ne"))


def test_problems_floor_above_ceiling():
    assert CellSpec(floors={"all": "flat 0.5"}, ceilings={"all": 0.75}).problems() == []
    assert CellSpec(floors={"all": "ramp N 0.5"}, ceilings={"all": 0.75}).problems()


@pytest.mark.parametrize("spec", [
    CellSpec(walls=frozenset("ES")),
    CellSpec(fill="thin_backslash"),
    CellSpec(floors={"sw": "ramp NW 0.25"}),
    CellSpec(ceilings={"ne": 0.25}, flags=frozenset({"ladder"})),
])
def test_properties_round_trip(spec):
    assert CellSpec.from_properties(spec.to_properties()) == spec


def test_every_encoding_decodes_back():
    for fill, floor, ceiling in itertools.product(C.FILLS, C.FLOOR_NAMES[::4], (1.0, 0.5)):
        if fill in C.FILL_HALF or fill in C.FILL_THIN_SPLIT:
            continue
        spec = CellSpec(fill=None if fill == "empty" else fill, floors={"all": floor},
                        ceilings={} if ceiling == 1.0 else {"all": ceiling})
        v = spec.encode()
        assert C.FILLS[int(C.fill(v))] == fill
        assert C.FLOOR_NAMES[int(C.floor_code(v, 1))] == floor == C.FLOOR_NAMES[int(C.floor_code(v, 2))]
        assert C.CEILING_HEIGHTS[int(C.ceiling_code(v, 1))] == ceiling
