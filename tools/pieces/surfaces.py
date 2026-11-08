"""Capas de superficie de las piezas de prueba: metal, rugosidad, brillo y pulso por píxel.

Cada función recibe las capas de una pieza ya dibujada (color, albedo, posición en metros,
máscara de lo cubierto y la capa de superficie, por defecto mate sin metal ni brillo) y las
retoca en su sitio. Canales de la capa de superficie (RGBA 8 bits): R metal, G rugosidad (0
pulido, 255 mate), B brillo propio (con el color del píxel), A pulso (0 fijo; si no, fase de un
latido lento: un degradado hace que la luz recorra la pieza).
"""

import numpy as np

DEFAULT_SURFACE = (0, 255, 0, 0)            # mate, sin metal ni brillo


def constant(metal: float = 0.0, rough: float = 1.0, glow: float = 0.0):
    def finish(maps):
        maps["surface"][maps["covered"]] = (round(metal * 255), round(rough * 255), round(glow * 255), 0)
    return finish


def glow_lines(spacing: float = 0.5, width: float = 0.06, line_color=(0.25, 0.95, 1.0), rough: float = 0.35):
    """Suelo oscuro algo pulido con una cuadrícula de líneas que brillan; el pulso avanza en
    diagonal (fase = (x + z) / 4 m), así que la luz recorre el suelo de una celda a la siguiente."""
    def finish(maps):
        pos, covered = maps["position"], maps["covered"]
        x, z = pos[..., 0], pos[..., 2]
        line = covered & ((np.mod(x, spacing) < width) | (np.mod(z, spacing) < width))
        surface = maps["surface"]
        surface[covered] = (0, round(rough * 255), 0, 0)
        phase = 1 + np.mod((x + z) / 4.0, 1.0) * 254              # 0 sería brillo fijo
        surface[line, 2] = 255
        surface[line, 3] = np.clip(np.rint(phase[line]), 1, 255)
        rgb = np.clip(np.array(line_color) * 255 + 0.5, 0, 255).astype(np.uint8)
        for key in ("color", "albedo"):
            maps[key][line, :3] = rgb
    return finish
