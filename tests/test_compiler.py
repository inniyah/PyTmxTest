"""Compilador TMX -> mundo compilado, WorldStore y WorldView."""
import io

import numpy as np
import pytest
from PIL import Image

import tmx_manager as tm
from engine.compiler import compile_tmx
from engine.world import CellSpec
from engine.world import cell as C
from engine.world import visual as V
from engine.world.chunk import ChunkData
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.build import build_gallery

FLIP_H = 0x80000000


# --- La galería, comprobada celda a celda de forma independiente ---------------------

@pytest.fixture(scope="module")
def gallery(tmp_path_factory, shared_tileset):
    out = tmp_path_factory.mktemp("gal")
    tsx = shared_tileset
    tmx = out / "maps" / "gallery.tmx"
    build_gallery(tmx, tsx)
    report = compile_tmx(tmx, out / "gallery.cworld", chunk_size=16)
    return tmx, out / "gallery.cworld", report


def expected_values(tmx_path):
    """Valor esperado de cada celda (una sola planta): bucle directo sobre las capas."""
    m = tm.TiledMap.load(tmx_path)
    ts = m.tilesets[0]
    keys = {"walls", "fill", "floor", "floor_part", "ceiling", "ceiling_part", "flags"}
    specs = {ts.firstgid + i: CellSpec.from_properties({k: str(p.value) for k, p in t.properties.items()
                                                        if k in keys})
             for i, t in ts.tiles.items()}
    layers = [l for l in m.iter_layers() if isinstance(l, tm.TileLayer)]
    values = np.zeros((m.height, m.width), np.int64)
    south, east = np.zeros_like(values, bool), np.zeros_like(values, bool)
    for y in range(m.height):
        for x in range(m.width):
            spec = CellSpec()
            for layer in layers:
                g = layer.get_tile_gid(x, y)
                if g:
                    part = specs[g & 0x0FFFFFFF]
                    spec = spec.merge(part.mirrored() if g & FLIP_H else part)
            values[y, x] = spec.encode()
            south[y, x], east[y, x] = "S" in spec.walls, "E" in spec.walls
    for y in range(m.height):
        for x in range(m.width):
            if south[y, x] and y + 1 < m.height:
                values[y + 1, x] |= C.WALL_N
            if east[y, x] and x + 1 < m.width:
                values[y, x + 1] |= C.WALL_W
    return m, layers, values


def test_gallery_compiles_cleanly(gallery):
    _, _, report = gallery
    assert report.ok and not report.warnings, report.summary()
    m = tm.TiledMap.load(gallery[0])
    expected = -(-m.width // 16) * -(-m.height // 16)        # chunks de 16 que cubren el mapa
    assert report.stats["chunks"] == expected


def test_every_cell_matches_an_independent_computation(gallery):
    tmx, world_path, _ = gallery
    m, layers, expected = expected_values(tmx)
    view = WorldView(SQLiteWorldStore(world_path))
    ys, xs = np.mgrid[0:m.height, 0:m.width]
    got = view.cells(xs, ys, 0)
    assert np.array_equal(got, expected.astype(np.uint32))
    assert (C.wall_n(got).sum(), C.wall_w(got).sum()) == (C.wall_n(expected).sum(), C.wall_w(expected).sum())
    # huecos visuales: tantos como tiles, con el rol de su capa
    slots = view.visual(xs, ys, 0)
    role_of_layer = {"floor": V.ROLE_FLOOR, "object": V.ROLE_OBJECT, "overhead": V.ROLE_OVERHEAD}
    for y in range(m.height):
        for x in range(m.width):
            tiles = [(l, l.get_tile_gid(x, y)) for l in layers if l.get_tile_gid(x, y)]
            used = [s for s in slots[y, x] if V.region(s)]
            assert len(used) == len(tiles)
            assert [int(V.role(s)) for s in used] == [role_of_layer[l.klass] for l, _ in tiles]


def test_atlas_regions_reproduce_the_tile_images(gallery):
    tmx, world_path, _ = gallery
    store = SQLiteWorldStore(world_path)
    page = np.asarray(Image.open(io.BytesIO(store.atlas_page(0, "color"))))
    m = tm.TiledMap.load(tmx)
    ts = m.tilesets[0]
    sheet = np.asarray(Image.open((tmx.parent / ts.source).parent / ts.image.source).convert("RGBA"))
    view = WorldView(store)
    ys, xs = np.mgrid[0:m.height, 0:m.width]
    slots = view.visual(xs, ys, 0)
    regions = {r[0]: r for r in store.atlas_regions()}
    role_of_layer = {"floor": V.ROLE_FLOOR, "object": V.ROLE_OBJECT, "overhead": V.ROLE_OVERHEAD}
    checked = set()
    for layer in (l for l in m.iter_layers() if isinstance(l, tm.TileLayer)):
        role = role_of_layer[layer.klass]
        for y in range(m.height):
            for x in range(m.width):
                g = layer.get_tile_gid(x, y) & 0x0FFFFFFF
                if not g:
                    continue
                rid = next(int(V.region(s)) for s in slots[y, x] if V.role(s) == role)
                _, _, rx, ry, w, h, _ = regions[rid]
                col, row = (g - 1) % ts.columns, (g - 1) // ts.columns
                tile = sheet[row * ts.tileheight:(row + 1) * ts.tileheight,
                             col * ts.tilewidth:(col + 1) * ts.tilewidth]
                assert np.array_equal(page[ry:ry + h, rx:rx + w], tile)
                checked.add(g)
    assert len(checked) == ts.tilecount == len(regions)          # las 106 regiones comprobadas
    assert set(store.atlas_kinds()) == {"color", "albedo", "normal", "position"}


# --- Mapas sintéticos ----------------------------------------------------------------

def make_map(levels, width=6, height=5, specs=None, infinite_offset=None):
    """levels: {nivel: [capa, ...]}; cada capa {(x, y): nombre de pieza o (nombre, volteada)}."""
    specs = specs or {}
    names = list(specs)
    m = tm.create_empty_map(width, height, 128, 92)
    ts = tm.Tileset(firstgid=1, name="t", tilewidth=128, tileheight=259, tilecount=len(names), columns=0)
    for i, name in enumerate(names):
        ts.tiles[i] = tm.Tile(id=i, properties={k: tm.Property(k, "string", v)
                                                for k, v in specs[name].to_properties().items()})
    m.tilesets.append(ts)
    gid = {n: i + 1 for i, n in enumerate(names)}
    for level, layers in levels.items():
        group = tm.LayerGroup(name=f"P{level}", properties={"Z": tm.Property("Z", "int", level)})
        for k, cells in enumerate(layers):
            layer = tm.create_layer(f"l{level}_{k}", width, height)
            for (x, y), piece in cells.items():
                name, flip = piece if isinstance(piece, tuple) else (piece, False)
                layer.set_tile_gid(x, y, gid[name] | (FLIP_H if flip else 0))
            group.layers.append(layer)
        m.layers.append(group)
    return m


SPECS = {
    "floor": CellSpec(floors={"all": "flat 0"}),
    "floor_nw": CellSpec(floors={"nw": "flat 0"}),
    "floor_ne": CellSpec(floors={"ne": "flat 0.5"}),
    "wall_S": CellSpec(walls=frozenset("S")),
    "wall_E": CellSpec(walls=frozenset("E")),
    "ramp_E": CellSpec(floors={"se": "ramp NE 0.25"}, walls=frozenset("E")),
    "block": CellSpec(fill="solid"),
}


def compile_map(tmp_path, m, **kw):
    path = tmp_path / "m.tmx"
    m.save(path)
    report = compile_tmx(path, tmp_path / "m.cworld", **kw)
    return report, (WorldView(SQLiteWorldStore(tmp_path / "m.cworld")) if report.ok else None)


def test_conflicts_are_reported_with_coordinates(tmp_path):
    m = make_map({0: [{(2, 3): "floor_nw"}, {(2, 3): "floor_ne"}]}, specs=SPECS)
    report, view = compile_map(tmp_path, m)
    assert not report.ok and view is None
    assert "diagonales" in report.errors[0] and "x=2, y=3, planta=0" in report.errors[0]


def test_south_and_east_walls_move_to_the_neighbour(tmp_path):
    m = make_map({0: [{(1, 1): "wall_S", (4, 2): "wall_E", (2, 4): "wall_S", (5, 0): "wall_E"}]},
                 specs=SPECS)
    report, view = compile_map(tmp_path, m)
    assert report.ok
    assert C.wall_n(view.cells(1, 2, 0)) and not C.wall_n(view.cells(1, 1, 0))
    assert C.wall_w(view.cells(5, 2, 0)) and not C.wall_w(view.cells(4, 2, 0))
    # En el borde del mapa caen fuera: fuera ya es macizo
    assert view.cells(2, 5, 0) == C.CELL_SOLID and view.cells(6, 0, 0) == C.CELL_SOLID


def test_horizontal_flip_mirrors_the_cell(tmp_path):
    m = make_map({0: [{(1, 1): "ramp_E", (3, 1): ("ramp_E", True)}]}, specs=SPECS)
    report, view = compile_map(tmp_path, m)
    plain, flipped = int(view.cells(1, 1, 0)), int(view.cells(3, 1, 0))
    assert C.FLOOR_NAMES[int(C.floor_code(plain, 2))] == "ramp NE 0.25"
    assert C.FLOOR_NAMES[int(C.floor_code(flipped, 2))] == "ramp NW 0.25"   # mitad SE -> SO
    assert C.split(flipped) and not C.split(plain)
    assert C.wall_w(view.cells(2, 1, 0)) and C.wall_w(view.cells(3, 1, 0))   # E -> pared en su O
    assert report.warnings == ["GIDs sin imagen (no se dibujan): [6]"]


def test_columns_with_negative_levels(tmp_path):
    m = make_map({-1: [{(1, 1): "floor", (2, 2): "block"}], 0: [{(1, 1): "floor"}],
                  2: [{(1, 1): "block"}]}, specs=SPECS)
    report, view = compile_map(tmp_path, m)
    assert report.ok and (view.level_min, view.level_max) == (-1, 2)
    assert view.column(1, 1) == (-1, 4)                  # de -1 a 2: incluye el aire del nivel 1
    assert view.cells(1, 1, -2) == C.CELL_SOLID          # por debajo del tramo: macizo
    assert view.cells(1, 1, 1) == 0                      # aire dentro del tramo
    assert view.cells(1, 1, 2) == C.CELL_SOLID and view.cells(1, 1, 3) == 0   # bloque; encima, aire
    assert view.column(0, 0) == (-1, 0)                  # columna vacía: suelo macizo bajo el nivel mínimo
    assert view.cells(0, 0, -1) == 0 and view.cells(0, 0, -2) == C.CELL_SOLID


def test_infinite_map_with_negative_coordinates(tmp_path):
    m = make_map({0: [{}]}, specs=SPECS)
    m.infinite = True
    layer = m.layers[0].layers[0]
    layer.data.tiles = tm.array.array("I")
    layer.data.chunks = [tm.Chunk(-8, -4, 8, 8, tm.array.array("I", [1] * 64))]
    layer.width = layer.height = 8
    report, view = compile_map(tmp_path, m, chunk_size=4)
    assert report.ok and (view.x0, view.y0) == (-8, -4)
    assert report.stats["chunks"] == 4
    floor = SPECS["floor"].encode()
    assert (view.cells(np.arange(-8, 0), -4, 0) == floor).all()
    assert (view.cells(np.arange(-8, 0), 3, 0) == floor).all()
    assert view.cells(0, 0, 0) == C.CELL_SOLID          # fuera del mapa


def test_compression_does_not_change_the_world(tmp_path):
    m = make_map({0: [{(x, y): "floor" for x in range(6) for y in range(5)}, {(2, 2): "wall_S"}]}, specs=SPECS)
    m.save(tmp_path / "m.tmx")
    compile_tmx(tmp_path / "m.tmx", tmp_path / "a.cworld", compression="zlib")
    compile_tmx(tmp_path / "m.tmx", tmp_path / "b.cworld", compression="none")
    ys, xs = np.mgrid[0:5, 0:6]
    a = WorldView(SQLiteWorldStore(tmp_path / "a.cworld")).cells(xs, ys, 0)
    b = WorldView(SQLiteWorldStore(tmp_path / "b.cworld")).cells(xs, ys, 0)
    assert np.array_equal(a, b)


def test_palette_index_width_adapts(tmp_path):
    n = 300                                               # más de 256 valores distintos
    chunk = ChunkData(0, 0, 32, np.zeros(1024, np.int16), np.r_[np.ones(n, np.uint8), np.zeros(1024 - n, np.uint8)],
                      np.arange(n, dtype=np.uint32) * 7, np.zeros((n, 4), np.uint32))
    blobs = chunk.to_blobs()
    assert len(blobs["obst_index"]) == 2 * n and len(blobs["vis_index"]) == n
    back = ChunkData.from_blobs(0, 0, 32, blobs)
    assert np.array_equal(back.obstacles, chunk.obstacles)


def test_cli(tmp_path, capsys):
    from engine.compiler.__main__ import main as compile_main
    from engine.world.__main__ import main as world_main
    m = make_map({0: [{(1, 1): "floor"}]}, specs=SPECS)
    m.save(tmp_path / "m.tmx")
    assert compile_main([str(tmp_path / "m.tmx")]) == 0
    assert world_main(["cell", str(tmp_path / "m.cworld"), "1", "1", "0"]) == 0
    assert "flat 0" in capsys.readouterr().out


def test_evict_and_reload_give_the_same_answers(gallery):
    _, world_path, _ = gallery
    view = WorldView(SQLiteWorldStore(world_path))
    rng = np.random.default_rng(3)
    xs, ys, zs = rng.integers(-3, 34, 5000), rng.integers(-3, 40, 5000), rng.integers(-1, 2, 5000)
    before = view.cells(xs, ys, zs), view.visual(xs, ys, zs)
    assert len(view.loaded) == 6
    view.evict(keep=[(0, 0)])
    assert view.loaded == [(0, 0)]
    after = view.cells(xs, ys, zs), view.visual(xs, ys, zs)        # recarga lo que falta
    assert all(np.array_equal(a, b) for a, b in zip(before, after)) and len(view.loaded) == 6


def test_obstacles_only_view(gallery):
    _, world_path, _ = gallery
    full, light = WorldView(SQLiteWorldStore(world_path)), WorldView(SQLiteWorldStore(world_path), with_visual=False)
    ys, xs = np.mgrid[0:37, 0:31]
    assert np.array_equal(full.cells(xs, ys, 0), light.cells(xs, ys, 0))
    with pytest.raises(ValueError):
        light.visual(0, 0, 0)
