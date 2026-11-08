"""python -m tools.pieces [--out DIR] [--gallery MAPA.tmx]"""

import argparse
import time
from pathlib import Path

from .build import build_gallery, build_tileset
from .catalog import build_catalog
from .course import build_course
from .levels import build_levels
from .terrain import build_terrain


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tools.pieces",
                                     description="Genera piezas .obj, el tileset de prueba y la galería")
    parser.add_argument("--out", type=Path, default=Path("assets/pieces"), help="carpeta del tileset")
    parser.add_argument("--gallery", type=Path, default=Path("maps/gallery/gallery.tmx"),
                        help="mapa galería a generar")
    parser.add_argument("--course", type=Path, default=Path("maps/course/course.tmx"),
                        help="mapa de recorrido a generar")
    parser.add_argument("--levels", type=Path, default=Path("maps/levels/levels.tmx"),
                        help="mapa de dos niveles a generar")
    parser.add_argument("--terrain", type=Path, default=Path("maps/terrain/terrain.tmx"),
                        help="mapa de terreno a generar")
    parser.add_argument("--terrain-size", type=int, default=256,
                        help="lado del mapa de terreno en celdas (256 = 512 m; 512 = 1 km)")
    parser.add_argument("--houses", type=int, default=None, help="casas en el terreno")
    parser.add_argument("--seed", type=int, default=1, help="semilla del terreno")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    pieces = build_catalog()
    tsx = build_tileset(args.out, pieces=pieces)
    build_gallery(args.gallery, tsx, pieces=pieces)
    build_course(args.course, tsx, pieces=pieces)
    build_levels(args.levels, tsx, pieces=pieces)
    n = args.terrain_size
    build_terrain(args.terrain, tsx, n, n, houses=args.houses or max(4, n * n // 2800), seed=args.seed,
                  pieces=pieces)
    print(f"{len(pieces)} piezas -> {args.out}/, {args.gallery}, {args.course}, {args.levels} y "
          f"{args.terrain} ({n} × {n}) ({time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    main()
