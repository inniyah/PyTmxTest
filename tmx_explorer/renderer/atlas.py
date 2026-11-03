"""Atlas de texturas: muchas imágenes pequeñas en pocas texturas grandes ("páginas").

Cada imagen se extruye ``border`` píxeles (repitiendo su borde) antes de colocarla,
para que el filtrado nunca muestree píxeles de la imagen vecina.

El empaquetado (``pack_shelves``) es Python puro y no necesita OpenGL; solo
``TextureAtlas.build`` sube las páginas a la GPU.
"""

import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from ..config import TEXTURE_BORDER


@dataclass(frozen=True)
class Placement:
    page: int
    x: int          # esquina superior izquierda de la celda (incluye el borde)
    y: int


@dataclass(frozen=True)
class AtlasRegion:
    """Zona de una página del atlas. UV en convención OpenGL: v1 es el borde superior."""
    page: int
    u0: float
    v0: float
    u1: float
    v1: float
    width: int      # tamaño del contenido en píxeles (sin borde)
    height: int

    @property
    def uv(self) -> Tuple[float, float, float, float]:
        return self.u0, self.v0, self.u1, self.v1


def _page_width(sizes: Sequence[Tuple[int, int]], max_size: int) -> int:
    """Ancho razonable: ~cuadrado para el área total, nunca menor que la celda más ancha."""
    widest = max(w for w, _ in sizes)
    area = sum(w * h for w, h in sizes)
    side = 1 << max(0, math.ceil(math.log2(math.sqrt(area * 1.1))))
    return min(max_size, max(widest, side))


def pack_shelves(sizes: Sequence[Tuple[int, int]], max_size: int
                 ) -> Tuple[List[Placement], List[Tuple[int, int]]]:
    """Coloca rectángulos (ancho, alto) en páginas de como mucho max_size x max_size.

    Algoritmo de estanterías: se ordena por altura y se llenan filas de izquierda a
    derecha. Devuelve una colocación por rectángulo (en el orden de entrada) y el
    tamaño final de cada página.
    """
    if not sizes:
        return [], []
    for w, h in sizes:
        if w > max_size or h > max_size:
            raise ValueError(f"Imagen de {w}x{h} no cabe en una página de {max_size}")

    page_w = _page_width(sizes, max_size)
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0]))
    placements: List[Placement] = [None] * len(sizes)
    pages: List[Tuple[int, int]] = []

    page = x = y = shelf_h = 0
    used_w = 0
    for i in order:
        w, h = sizes[i]
        if x + w > page_w:                      # siguiente estantería
            x, y, shelf_h = 0, y + shelf_h, 0
        if y + h > max_size:                    # siguiente página
            pages.append((used_w, y))
            page, x, y, shelf_h, used_w = page + 1, 0, 0, 0, 0
        placements[i] = Placement(page, x, y)
        x += w
        used_w = max(used_w, x)
        shelf_h = max(shelf_h, h)
    pages.append((used_w, y + shelf_h))
    return placements, pages


def extrude(image: Image.Image, border: int) -> np.ndarray:
    """RGBA como array (alto, ancho, 4) con el borde repetido ``border`` píxeles."""
    pixels = np.asarray(image.convert("RGBA"))
    if border:
        pixels = np.pad(pixels, ((border, border), (border, border), (0, 0)), mode="edge")
    return pixels


def layout_atlas(images: Dict[Hashable, Image.Image], max_size: int,
                 border: int = TEXTURE_BORDER
                 ) -> Tuple[List[np.ndarray], Dict[Hashable, AtlasRegion]]:
    """Compone las páginas en memoria (sin GPU). Devuelve (páginas RGBA, regiones)."""
    keys = list(images)
    cells = [extrude(images[k], border) for k in keys]
    placements, page_sizes = pack_shelves([(c.shape[1], c.shape[0]) for c in cells], max_size)

    pages = [np.zeros((h, w, 4), dtype=np.uint8) for w, h in page_sizes]
    regions: Dict[Hashable, AtlasRegion] = {}
    for key, cell, pl in zip(keys, cells, placements):
        ch, cw = cell.shape[:2]
        pages[pl.page][pl.y:pl.y + ch, pl.x:pl.x + cw] = cell
        page_h, page_w = pages[pl.page].shape[:2]
        w, h = cw - 2 * border, ch - 2 * border
        left, top = pl.x + border, pl.y + border
        # La página se sube volteada en vertical: la fila 0 de la imagen queda en v = 1
        regions[key] = AtlasRegion(
            page=pl.page,
            u0=left / page_w, u1=(left + w) / page_w,
            v0=1.0 - (top + h) / page_h, v1=1.0 - top / page_h,
            width=w, height=h)
    return pages, regions


class TextureAtlas:
    """Páginas subidas a la GPU más el diccionario clave -> AtlasRegion."""

    def __init__(self, textures: list, regions: Dict[Hashable, AtlasRegion]):
        self.textures = textures       # una Texture por página
        self.regions = regions

    @classmethod
    def build(cls, images: Dict[Hashable, Image.Image], max_size: int = None) -> "TextureAtlas":
        from .texture import Texture, max_texture_size
        from ..config import ATLAS_MAX_SIZE
        limit = min(max_size or ATLAS_MAX_SIZE, max_texture_size())
        pages, regions = layout_atlas(images, limit)
        textures = [Texture.from_array(page) for page in pages]
        return cls(textures, regions)

    def __getitem__(self, key) -> AtlasRegion:
        return self.regions[key]

    def texture_for(self, region: AtlasRegion):
        return self.textures[region.page]

    @property
    def page_count(self) -> int:
        return len(self.textures)
