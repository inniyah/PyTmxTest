"""Piezas de prueba: mallas, .obj, render, tileset y galería."""
import numpy as np
import pytest

import tmx_manager as tm
from engine.world import CellSpec
from engine import png16
from tools.pieces.build import build_gallery, build_tileset
from tools.pieces.catalog import build_catalog
from tools.pieces.mesh import MATERIALS, mesh_for
from tools.pieces.objio import read_obj, write_mtl, write_obj
from tools.pieces.render import Projection, render

PROJ = Projection()
CATALOG = build_catalog()


def faces_of(spec):
    return [(f.points, MATERIALS[f.material]) for f in mesh_for(spec).faces]


def test_catalog():
    assert len(CATALOG) == 106
    assert {p.layer for p in CATALOG} == {"floor", "object", "overhead"}


@pytest.mark.parametrize("piece", CATALOG, ids=lambda p: p.name)
def test_meshes_are_closed_and_point_outwards(piece):
    mesh = mesh_for(piece.spec)
    lo, hi = mesh.bounds()
    assert (lo >= -1e-9).all() and hi[0] <= 2 + 1e-9 and hi[2] <= 2 + 1e-9 and hi[1] <= 3.75 + 1e-9
    pts = np.concatenate([f.points for f in mesh.faces])
    # Volumen con signo (teorema de la divergencia): positivo y coherente si las caras miran hacia fuera
    volume = 0.0
    for f in mesh.faces:
        for k in range(1, len(f.points) - 1):
            a, b, c = f.points[0], f.points[k], f.points[k + 1]
            volume += np.dot(a, np.cross(b, c)) / 6
    is_plate = len(mesh.faces) == 1 or (piece.spec.floors and "flat 0" in piece.spec.floors.values()
                                        and len(mesh.faces) == 1)
    assert volume > 0 or is_plate, f"volumen {volume}"
    assert len(pts) > 0


def test_block_silhouette_is_exact():
    rgba, nrm, pos = render(faces_of(CellSpec(fill="solid")), PROJ)
    rows, cols = np.nonzero(rgba[..., 3])
    assert (rows.max() - rows.min() + 1, cols.max() - cols.min() + 1) == (PROJ.row_h + PROJ.level_h, PROJ.tile_w)
    assert rows.max() == rgba.shape[0] - 1                     # anclado al borde inferior
    assert tuple(nrm[rows.min() + 5, 64, :3]) == (128, 255, 128)  # cara de arriba: normal +Y
    assert tuple(nrm[-5, 64, :3]) == (128, 128, 255)              # cara sur: normal +Z


def test_position_map_gives_world_coordinates():
    _, _, pos = render(faces_of(CellSpec(fill="solid")), PROJ)
    (xr, yr, zr) = PROJ.position_range
    decode = lambda j, i: [pos[j, i, a] / 65535 * (r[1] - r[0]) + r[0] for a, r in enumerate((xr, yr, zr))]
    H = pos.shape[0]
    assert np.allclose(decode(H - 1, 0), (0, 0, 2), atol=0.03)                 # esquina SO del suelo
    assert np.allclose(decode(H - PROJ.row_h - PROJ.level_h, 0), (0, 3, 0), atol=0.03)  # NO del techo
    assert pos[0, 0, 3] == 0                                    # fuera de la pieza: transparente


def test_obj_round_trip(tmp_path):
    write_mtl(tmp_path / "pieces.mtl")
    spec = CellSpec(floors={"all": "ramp NE 0.25"}, walls=frozenset("N"))
    write_obj(tmp_path / "p.obj", mesh_for(spec), "p")
    a = render(faces_of(spec), PROJ)[0]
    b = render(read_obj(tmp_path / "p.obj"), PROJ)[0]
    assert np.array_equal(a, b)


def test_png16_round_trip(tmp_path):
    data = np.random.default_rng(0).integers(0, 65536, (7, 5, 4)).astype(np.uint16)
    png16.write_rgba16(tmp_path / "x.png", data)
    assert np.array_equal(png16.read_rgba16(tmp_path / "x.png"), data)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("pieces")
    tsx = build_tileset(out / "assets")
    build_gallery(out / "maps" / "gallery.tmx", tsx)
    return out, tsx


def test_tileset_properties_reproduce_each_spec(built):
    _, tsx = built
    m = tm.TiledMap.load(built[0] / "maps" / "gallery.tmx")
    tileset = m.tilesets[0]
    assert tileset.tilecount == len(CATALOG)
    for piece in CATALOG:
        tile = next(t for t in tileset.tiles.values() if t.properties["piece"].value == piece.name)
        props = {k: p.value for k, p in tile.properties.items() if k != "piece"}
        assert CellSpec.from_properties(props) == piece.spec
        assert tile.type == piece.layer


def test_three_maps_share_the_layout(built):
    from PIL import Image
    out, _ = built
    c = np.asarray(Image.open(out / "assets" / "pieces.png"))
    n = np.asarray(Image.open(out / "assets" / "pieces.n.png"))
    p = png16.read_rgba16(out / "assets" / "pieces.p.png")
    a = np.asarray(Image.open(out / "assets" / "pieces.a.png"))
    assert c.shape[:2] == n.shape[:2] == p.shape[:2] == a.shape[:2]
    for other in (n, p, a):
        assert np.array_equal(c[..., 3] > 0, other[..., 3] > 0)
    # El albedo no tiene sombreado: cada cara tiene el color del material, sin luz
    assert (a[..., :3].astype(int) >= c[..., :3].astype(int) - 1)[c[..., 3] > 0].mean() > 0.95


def test_gallery_loads_in_the_engine_world(built):
    from tmx_explorer.map.world import World
    w = World(tm.TiledMap.load(built[0] / "maps" / "gallery.tmx"))
    assert w.level_height == PROJ.level_h and w.H == 1
    assert {l.role.name for l in w.layers} == {"FLOOR", "OBJECT", "OVERHEAD"}
    assert (w.tiles[0, :, :, 0] != 0).all()                       # todo el suelo cubierto


@pytest.mark.parametrize("piece", CATALOG, ids=lambda p: p.name)
def test_every_face_has_a_valid_outward_normal(piece):
    """Una cara con vértices alineados al principio daba normal indefinida y no se dibujaba."""
    from tools.pieces.mesh import newell_normal
    mesh = mesh_for(piece.spec)
    center = np.concatenate([f.points for f in mesh.faces]).mean(axis=0)
    for f in mesh.faces:
        assert len(f.points) >= 3
        n = newell_normal(f.points)
        assert np.isfinite(n).all() and np.linalg.norm(n) > 1e-9
        single_prism = len(piece.spec.walls) <= 1                 # las esquinas son dos prismas
        if len(mesh.faces) > 1 and single_prism:                 # hacia fuera del prisma
            assert np.dot(n, f.points.mean(axis=0) - center) > 0
