"""Hoja de sprites de un personaje: 8 direcciones × las animaciones que pida su descripción.

Archivos que genera (``nombre`` = el de la descripción):

    nombre.png     color en el estilo de la hoja original: dos tonos por material y contorno
                   (es lo que se ve sin iluminación, y la referencia para el grafista)
    nombre.a.png   albedo: el color de cada material sin sombrear, con el contorno y los ojos
    nombre.n.png   normal por píxel en ejes del mundo (X este, Y arriba, Z sur): (n·0.5+0.5)·255
    nombre.p.png   posición por píxel en metros respecto a los pies, 16 bits, escalada con
                   ``position_range`` del .json
    nombre.s.png   capa de superficie (R metal, G rugosidad), solo si algún material es
                   metálico o pulido
    nombre.json    disposición: tamaño del fotograma, pies, escala, direcciones y animaciones

Proporciones, equipo (casco, mochila, fusil) y animaciones, de la descripción.
Filas: las 8 direcciones en el orden del motor (0 = este y en sentido horario en pantalla:
este, sureste, sur, suroeste, oeste, noroeste, norte, noreste). Columnas: ``quieto`` y
después cada animación, una tras otra.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from engine import png16
from tools.pieces.render import DEFAULT_PROJECTION, Projection

from .body import EQUIPMENT, FALLBACK, Proportions, standing_height
from .poses import DEFAULT_SMOOTHING, POSES, TIMING, animation, inbetween
from .render import LIGHT, Frame, far_side, render_frame, view_dir

DIRECTIONS = 8
DEFAULT_PITCH = 30.0           # grados: inclinación de la cámara de los personajes (el mundo, ~46)
CANVAS = (288, 176)            # lienzo de trabajo, antes de recortar (cabe alguien tendido)
ANCHOR = (144, 132)            # pies en el lienzo de trabajo
TOON_THRESHOLD = 0.6           # por encima, el tono claro del material
TOON_FAR = 0.9                 # en las partes del lado lejano, casi todo en sombra
WHITE = (255, 255, 255)


def camera_projection(pitch_deg: float, base: Projection = DEFAULT_PROJECTION) -> Projection:
    """Proyección para dibujar al personaje visto desde ``pitch_deg`` grados sobre el suelo: la
    misma escala que el mundo (un metro mide lo mismo en pantalla, en horizontal y en vertical) y
    una dirección de visión (0, sz, sy) con esa inclinación. Con la del mundo (unos 46°) encaja
    exactamente con los tiles; con menos, se le ve más de frente. La profundidad por píxel sigue
    siendo exacta: el mapa de posición guarda el punto 3D de cada píxel."""
    sy = base.level_h / base.cell.height
    sz = sy * np.tan(np.radians(pitch_deg))
    return Projection(base.tile_w, sz * base.cell.width, base.level_h, base.cell)


def pitch_of(desc: dict, pitch=None) -> float:
    """Inclinación: la que se pida, la de la descripción (``camara.inclinacion``) o la de siempre."""
    if pitch is not None:
        return float(pitch)
    return float(desc.get("camara", {}).get("inclinacion", DEFAULT_PITCH))


def material_of(parts: dict, group: str) -> str:
    """Material de una parte: el que diga la descripción, o el de la parte de la que sale."""
    if group in parts:
        return parts[group]
    if group in FALLBACK and FALLBACK[group] in parts:
        return parts[FALLBACK[group]]
    raise ValueError(f"La descripción no dice el material de '{group}'")


def _hex(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _neighbours4(mask):
    out = np.zeros_like(mask)
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def shade(frame: Frame, angle: float, desc: dict, model_m: float, proj: Projection = DEFAULT_PROJECTION):
    """(color, albedo) uint8 (h, w, 4), normal (h, w, 3), punto (h, w, 3) y alfa de un fotograma."""
    h, w = frame.hit.shape
    mats = {name: (_hex(m["luz"]), _hex(m["sombra"])) for name, m in desc["materiales"].items()}
    surf_of = {name: (int(round(float(m.get("metal", 0.0)) * 255)),
                      int(round(float(m.get("rugosidad", 1.0)) * 255))) for name, m in desc["materiales"].items()}
    parts = desc["partes"]
    outline = _hex(desc["contorno"])
    eye = _hex(desc.get("ojos", desc["contorno"]))
    dn = view_dir(proj)
    inten = 0.7 * (frame.normal @ dn) + 0.3 * (frame.normal @ LIGHT)

    color = np.zeros((h, w, 3), np.uint8)
    albedo = np.zeros((h, w, 3), np.uint8)
    # Parte de ropa recoloreable de cada píxel (1, 2…: el orden de "ropa"; 0, ninguna). Va en el
    # alfa del albedo como 255 - parte, así que en un editor sigue viéndose opaco
    ropa = list(desc.get("ropa", []))
    slot = np.zeros((h, w), np.uint8)
    surface = np.zeros((h, w, 4), np.uint8)
    surface[..., 1] = 255                                        # mate, sin metal ni brillo
    for gi, group in enumerate(frame.names):
        sel = frame.hit & (frame.group == gi)
        if not sel.any():
            continue
        material = material_of(parts, group)
        light, dark = mats[material]
        surface[sel, 0], surface[sel, 1] = surf_of[material]
        thr = TOON_FAR if far_side(group, angle, proj) else TOON_THRESHOLD
        color[sel] = np.where((inten[sel] > thr)[:, None], light, dark)
        albedo[sel] = light
        if material in ropa:
            slot[sel] = ropa.index(material) + 1

    # Contorno interior: donde una parte queda detrás de otra (salto de profundidad entre
    # grupos distintos), en el píxel de la que queda detrás; sin píxeles sueltos
    inner = np.zeros((h, w), bool)
    step = 2.0 * model_m
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb_hit = np.roll(frame.hit, (-dy, -dx), (0, 1))
        nb_group = np.roll(frame.group, (-dy, -dx), (0, 1))
        nb_near = np.roll(frame.near, (-dy, -dx), (0, 1))
        inner |= frame.hit & nb_hit & (nb_group != frame.group) & (nb_near - frame.near > step)
    count = sum(np.roll(inner, (dy, dx), (0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - inner
    inner &= count > 0
    color[inner] = outline
    albedo[inner] = outline
    slot[inner] = 0

    # Contorno exterior: un píxel alrededor de la silueta. No tiene geometría propia: toma la
    # normal media de sus vecinos y el punto del vecino más lejano (nunca tapa nada)
    outer = _neighbours4(frame.hit) & ~frame.hit
    color[outer] = outline
    albedo[outer] = outline
    slot[outer] = 0
    normal = frame.normal.copy()
    point = frame.point.copy()
    for y, x in zip(*np.where(outer)):
        ys, xs = slice(max(0, y - 1), y + 2), slice(max(0, x - 1), x + 2)
        nb = frame.hit[ys, xs]
        n = frame.normal[ys, xs][nb].mean(0)
        normal[y, x] = n / max(np.linalg.norm(n), 1e-9)
        far = np.argmin(np.where(nb, frame.near[ys, xs], np.inf))
        point[y, x] = frame.point[ys, xs].reshape(-1, 3)[far]

    # Ojos: el del lado hacia el que mira lleva el brillo blanco hacia fuera de la cara
    head = frame.hit & (frame.group == frame.names.index("cabeza"))
    head_x = np.where(head)[1].mean() if head.any() else w / 2
    for px, py, near in frame.eyes:
        if near:
            out_left = px < head_x
            pattern = [[eye, eye], [WHITE, eye] if out_left else [eye, WHITE], [eye, eye]]
            x0 = px if out_left else px - 1
        else:
            pattern, x0 = [[eye], [eye], [eye]], px
        for r, row in enumerate(pattern):
            for k, c in enumerate(row):
                yy, xx = py - 1 + r, x0 + k
                if 0 <= yy < h and 0 <= xx < w and head[yy, xx]:
                    color[yy, xx] = c
                    albedo[yy, xx] = c
                    slot[yy, xx] = 0

    alpha = frame.hit | outer
    surface[~frame.hit] = (0, 255, 0, 0)                         # el contorno, mate
    return color, albedo, normal, point, alpha, surface, slot


def equipment_of(desc: dict):
    eq = tuple(desc.get("equipo", ()))
    unknown = [e for e in eq if e not in EQUIPMENT]
    if unknown:
        raise ValueError(f"Equipo desconocido: {unknown} (hay {list(EQUIPMENT)})")
    return eq


def animation_poses(desc: dict):
    """[(nombre, [poses])]: ``quieto`` y las animaciones de la descripción, en orden."""
    names = ["quieto"] + [a for a in desc.get("animaciones", []) if a != "quieto"]
    unknown = [a for a in names if a not in POSES]
    if unknown:
        raise ValueError(f"Animaciones desconocidas: {unknown} (hay {sorted(POSES)})")
    armed = "fusil" in equipment_of(desc)
    factor = smoothing_of(desc)
    out = []
    for a in names:
        poses = animation(a, armed)
        if a in TIMING and TIMING[a][2]:                 # las que se repiten: con intermedias
            poses = inbetween(poses, factor)
        out.append((a, poses))
    return out


def smoothing_of(desc: dict) -> int:
    """Fotogramas por pose clave en las animaciones que se repiten (``suavizado``; 1 = sin
    intermedias)."""
    factor = int(desc.get("suavizado", DEFAULT_SMOOTHING))
    if factor < 1:
        raise ValueError(f"suavizado debe ser 1 o más, no {factor}")
    return factor


def build_sheet(desc: dict, proj: Projection = None, progress=None, pitch=None):
    """Renderiza todos los fotogramas y devuelve (capas, disposición) sin escribir nada.

    Sin ``proj``, la de la inclinación de la cámara (``pitch``, o la de la descripción).
    Cada animación se recorta a lo que ocupan sus fotogramas (en las 8 direcciones): andar no
    carga con el ancho de alguien tendido. La disposición describe cada columna (x, ancho, alto y
    pies); las filas de una columna van una debajo de otra."""
    pitch = pitch_of(desc, pitch) if proj is None else float(np.degrees(np.arctan2(proj.sz, proj.sy)))
    proj = camera_projection(pitch) if proj is None else proj
    height = float(desc["altura"])
    prop = Proportions.from_desc(desc)
    equipment = equipment_of(desc)
    model_h = standing_height(prop)
    model_m = height / model_h
    anims = animation_poses(desc)
    cols = sum(len(p) for _, p in anims)
    W, H = CANVAS
    layers = {k: np.zeros((DIRECTIONS * H, cols * W) + s, d) for k, s, d in (
        ("color", (3,), np.uint8), ("albedo", (3,), np.uint8), ("normal", (3,), float), ("slot", (), np.uint8),
        ("point", (3,), float), ("alpha", (), bool), ("surface", (4,), np.uint8))}
    layout_anims, col = {}, 0
    for name, poses in anims:
        entry = {"cols": list(range(col, col + len(poses)))}
        if name in TIMING:
            fps, speed, loop = TIMING[name]
            entry["fps"] = fps * (smoothing_of(desc) if loop else 1)      # el ciclo dura lo mismo
            if speed is not None:
                entry["speed"] = speed
            if not loop:
                entry["loop"] = False
        layout_anims[name] = entry
        for pose in poses:
            for row in range(DIRECTIONS):
                angle = row * 360.0 / DIRECTIONS
                frame = render_frame(pose, angle, height, model_h, CANVAS, ANCHOR, proj,
                                     eyes=desc.get("ojos") is not None, prop=prop, equipment=equipment)
                maps = shade(frame, angle, desc, model_m, proj)
                sl = (slice(row * H, (row + 1) * H), slice(col * W, (col + 1) * W))
                for key, value in zip(("color", "albedo", "normal", "point", "alpha", "surface", "slot"), maps):
                    layers[key][sl] = value
                if progress:
                    progress()
            col += 1

    # Recorte por animación: el menor fotograma que contiene los suyos (con un píxel de margen)
    boxes = []
    for name, _poses in anims:
        acols = layout_anims[name]["cols"]
        alpha = layers["alpha"].reshape(DIRECTIONS, H, cols, W)[:, :, acols].any(axis=(0, 2))
        ys, xs = np.where(alpha)
        if ys.size == 0:
            ys, xs = np.array([ANCHOR[1]]), np.array([ANCHOR[0]])
        boxes += [(max(0, ys.min() - 1), min(H, ys.max() + 2), max(0, xs.min() - 1), min(W, xs.max() + 2))] * len(acols)
    out_w = sum(x1 - x0 for _y0, _y1, x0, x1 in boxes)
    out_h = DIRECTIONS * max(y1 - y0 for y0, y1, _x0, _x1 in boxes)
    cropped = {k: np.zeros((out_h, out_w) + v.shape[2:], v.dtype) for k, v in layers.items()}
    columns, x = [], 0
    for c, (y0, y1, x0, x1) in enumerate(boxes):
        fw, fh = x1 - x0, y1 - y0
        for row in range(DIRECTIONS):
            for k, v in layers.items():
                cropped[k][row * fh:(row + 1) * fh, x:x + fw] = v[row * H + y0:row * H + y1, c * W + x0:c * W + x1]
        columns.append({"x": int(x), "w": int(fw), "h": int(fh), "anchor": [ANCHOR[0] - int(x0), ANCHOR[1] - int(y0)]})
        x += fw
    layers = cropped
    a = layers["alpha"]
    pts = layers["point"][a]
    lo = np.floor((pts.min(0) - 0.05) * 20) / 20
    hi = np.ceil((pts.max(0) + 0.05) * 20) / 20
    layout = {
        "_comentario": "Generado por tools.actors: no editar a mano. 'columns': x, ancho, alto y pies "
                       "(anchor) de cada columna; las filas de una columna, una debajo de otra. "
                       "'px_per_m' = píxeles por metro en horizontal; 'pitch' = inclinación de la "
                       "cámara; filas = direcciones desde el este en sentido horario.",
        "frame": [max(c["w"] for c in columns), max(c["h"] for c in columns)],
        "cols": cols,
        "rows": DIRECTIONS,
        "anchor": columns[0]["anchor"],
        "columns": columns,
        "ropa": list(desc.get("ropa", [])),
        "px_per_m": proj.sx,
        "pitch": round(pitch, 2),
        "projection": {"sx": proj.sx, "sy": proj.sy, "sz": proj.sz},
        "height": height,
        "directions": DIRECTIONS,
        "direction_rows": list(range(DIRECTIONS)),
        "animations": layout_anims,
        "position_range": [[float(lo[i]), float(hi[i])] for i in range(3)],
    }
    return layers, layout


def write_sheet(layers, layout, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    a = layers["alpha"]
    alpha8 = (a * 255).astype(np.uint8)[..., None]
    Image.fromarray(np.concatenate([layers["color"], alpha8], -1)).save(out / f"{name}.png")
    # Alfa del albedo: 0 fuera; 255 normal; 254, 253…: partes de ropa recoloreables (1, 2…)
    albedo_alpha = np.where(layers["alpha"], 255 - layers["slot"], 0).astype(np.uint8)[..., None]
    Image.fromarray(np.concatenate([layers["albedo"], albedo_alpha], -1)).save(out / f"{name}.a.png")
    nrm = np.clip((layers["normal"] * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)
    nrm[~a] = (128, 128, 255)
    Image.fromarray(np.concatenate([nrm, alpha8], -1)).save(out / f"{name}.n.png")
    lo, hi = np.array(layout["position_range"]).T
    pos = np.zeros(a.shape + (4,), np.uint16)
    pos[..., :3] = np.clip((layers["point"] - lo) / (hi - lo) * 65535 + 0.5, 0, 65535)
    pos[..., 3] = a * 65535
    pos[~a] = 0
    png16.write_rgba16(out / f"{name}.p.png", pos)
    surface_path = out / f"{name}.s.png"
    surf = layers["surface"]
    if (surf[a][:, 0] > 0).any() or (surf[a][:, 1] < 255).any():       # algo metálico o pulido
        Image.fromarray(surf).save(surface_path)
    elif surface_path.exists():
        surface_path.unlink()
    (out / f"{name}.json").write_text(json.dumps(layout, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
