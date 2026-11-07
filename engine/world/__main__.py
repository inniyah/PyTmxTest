"""Herramientas del mundo compilado.

    python -m engine.world info  MUNDO.cworld          parámetros, informe de compilación
    python -m engine.world stats MUNDO.cworld          chunks, celdas, paletas
    python -m engine.world cell  MUNDO.cworld X Y Z    contenido de una celda
"""

import argparse
import json
import sys

import numpy as np

from . import cell as C
from . import visual as V
from .store import SQLiteWorldStore
from .view import WorldView


def describe_cell(value: int) -> dict:
    split = int(C.split(value))
    halves = ("NO", "SE") if split == C.SPLIT_SLASH else ("NE", "SO")
    return {
        "valor": f"0x{value:08x}",
        "paredes": "".join(s for s, f in (("N", C.wall_n), ("O", C.wall_w)) if f(value)) or "-",
        "relleno": C.FILLS[int(C.fill(value))],
        "diagonal": "/" if split == C.SPLIT_SLASH else "\\",
        f"suelo {halves[0]}": C.FLOOR_NAMES[int(C.floor_code(value, 1))],
        f"suelo {halves[1]}": C.FLOOR_NAMES[int(C.floor_code(value, 2))],
        f"techo {halves[0]}": C.CEILING_HEIGHTS[int(C.ceiling_code(value, 1))],
        f"techo {halves[1]}": C.CEILING_HEIGHTS[int(C.ceiling_code(value, 2))],
        "flags": f"0x{(value >> C.FLAGS_SHIFT) & C.FLAGS_MASK:02x}",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="engine.world")
    parser.add_argument("command", choices=["info", "stats", "cell"])
    parser.add_argument("world")
    parser.add_argument("coords", nargs="*", type=int)
    args = parser.parse_args(argv)
    store = SQLiteWorldStore(args.world)
    meta = store.meta()

    if args.command == "info":
        for k, v in meta.items():
            print(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}")
    elif args.command == "stats":
        view = WorldView(store)
        cells, obst_pal, vis_pal = 0, [], []
        for cx, cy in store.chunks():
            ch = view.chunk(cx, cy)
            cells += ch.cell_count
            obst_pal.append(len(np.unique(ch.obstacles)))
            vis_pal.append(len(np.unique(ch.visual, axis=0)))
        print(f"chunks: {len(obst_pal)}  celdas guardadas: {cells}")
        print(f"paleta de obstáculos por chunk: máx {max(obst_pal, default=0)}, media {np.mean(obst_pal or [0]):.0f}")
        print(f"paleta visual por chunk: máx {max(vis_pal, default=0)}, media {np.mean(vis_pal or [0]):.0f}")
        print(f"regiones de atlas: {len(store.atlas_regions())}  mapas: {store.atlas_kinds()}")
    elif args.command == "cell":
        if len(args.coords) != 3:
            parser.error("cell necesita X Y Z")
        view = WorldView(store)
        x, y, z = args.coords
        value = int(view.cells(x, y, z))
        for k, v in describe_cell(value).items():
            print(f"{k:>10}: {v}")
        slots = view.visual(x, y, z)
        for k, s in enumerate(slots):
            if V.region(s):
                print(f"   hueco {k}: región {int(V.region(s))}, rol {int(V.role(s))}, "
                      f"volteado {bool(V.flipped(s))}, opacidad {float(V.opacity(s)):.2f}")
        print(f"   columna: {view.column(x, y)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
