"""Vista previa de las animaciones de una hoja: un GIF por animación, con las 8 direcciones.

    python -m tools.actors.preview assets/actors/sheets/soldado.png [--out DIR] [--scale 2]

Lee la hoja y su .json (sirve igual con hojas generadas o dibujadas por un grafista).
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

BACKGROUND = (104, 124, 84, 255)


def frames_of(sheet: Image.Image, layout: dict, name: str, scale: int):
    w, h = layout["frame"]
    anim = layout["animations"][name]
    rows = layout.get("direction_rows", list(range(layout["rows"])))
    cols = list(anim["cols"])
    if anim.get("loop", True) is False:
        cols += [cols[-1]] * 4                 # se queda en el último fotograma un momento
    out = []
    for c in cols:
        canvas = Image.new("RGBA", (w * 4, h * 2), BACKGROUND)
        for d, row in enumerate(rows):
            cell = sheet.crop((c * w, row * h, (c + 1) * w, (row + 1) * h))
            canvas.alpha_composite(cell, ((d % 4) * w, (d // 4) * h))
        out.append(canvas.convert("RGB").resize((canvas.width * scale, canvas.height * scale), Image.NEAREST))
    return out, int(1000 / anim.get("fps", 8))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.actors.preview", description=__doc__.split("\n\n")[0])
    parser.add_argument("sheet", type=Path)
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args(argv)
    layout = json.loads(args.sheet.with_suffix(".json").read_text(encoding="utf-8"))
    sheet = Image.open(args.sheet).convert("RGBA")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in layout["animations"]:
        frames, ms = frames_of(sheet, layout, name, args.scale)
        path = args.out / f"{args.sheet.stem}_{name}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
