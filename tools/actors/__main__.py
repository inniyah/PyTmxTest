"""Genera las hojas de sprites de los personajes descritos en assets/actors/models.

    python -m tools.actors                 # todos
    python -m tools.actors maniqui         # solo ese

Cada descripción ``models/NOMBRE.json`` produce ``assets/actors/sheets/NOMBRE.*`` (ver
``tools.actors.sheet``).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from .sheet import DIRECTIONS, animation_poses, build_sheet, write_sheet

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "actors"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.actors", description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="personajes (por defecto, todos)")
    parser.add_argument("--models", type=Path, default=ASSETS / "models")
    parser.add_argument("--out", type=Path, default=ASSETS / "sheets")
    args = parser.parse_args(argv)
    paths = [args.models / f"{n}.json" for n in args.names] or sorted(args.models.glob("*.json"))
    for path in paths:
        if not path.exists():
            print(f"No existe {path}", file=sys.stderr)
            return 1
        desc = json.loads(path.read_text(encoding="utf-8"))
        total = DIRECTIONS * sum(len(p) for _, p in animation_poses(desc))
        done = [0]

        def progress(path=path, total=total, done=done):
            done[0] += 1
            print(f"\r{path.stem}: {done[0]}/{total} fotogramas", end="", flush=True)

        t0 = time.perf_counter()
        layers, layout = build_sheet(desc, progress=progress)
        write_sheet(layers, layout, args.out, path.stem)
        w, h = layout["frame"]
        print(f"\r{path.stem}: {total} fotogramas de {w}×{h} -> {args.out / path.stem}.* "
              f"({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
