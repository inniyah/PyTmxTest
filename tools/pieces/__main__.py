"""python -m tools.pieces [--out DIR] [--gallery MAPA.tmx]"""

import argparse
import time
from pathlib import Path

from .build import build_gallery, build_tileset
from .catalog import build_catalog
from .course import build_course


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tools.pieces",
                                     description="Genera piezas .obj, el tileset de prueba y la galería")
    parser.add_argument("--out", type=Path, default=Path("assets/pieces"), help="carpeta del tileset")
    parser.add_argument("--gallery", type=Path, default=Path("maps/gallery/gallery.tmx"),
                        help="mapa galería a generar")
    parser.add_argument("--course", type=Path, default=Path("maps/course/course.tmx"),
                        help="mapa de recorrido a generar")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    pieces = build_catalog()
    tsx = build_tileset(args.out, pieces=pieces)
    build_gallery(args.gallery, tsx, pieces=pieces)
    build_course(args.course, tsx, pieces=pieces)
    print(f"{len(pieces)} piezas -> {args.out}/, {args.gallery} y {args.course} "
          f"({time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    main()
