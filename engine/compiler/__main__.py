"""python -m engine.compiler MAPA.tmx [-o SALIDA.cworld] [--chunk-size 32] [--force]"""

import argparse
import sys
from pathlib import Path

from .compile import compile_tmx


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="engine.compiler", description="Compila un mapa de Tiled")
    parser.add_argument("map", type=Path)
    parser.add_argument("-o", "--out", type=Path, help="por defecto, junto al mapa con extensión .cworld")
    parser.add_argument("--chunk-size", type=int, default=32, help="potencia de dos (por defecto 32)")
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--force", action="store_true", help="escribir aunque haya errores")
    args = parser.parse_args(argv)

    out = args.out or args.map.with_suffix(".cworld")
    report = compile_tmx(args.map, out, args.chunk_size,
                         compression="none" if args.no_compression else "zlib", force=args.force)
    print(report.summary())
    if report.ok or args.force:
        print(f"-> {out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
