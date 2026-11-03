"""Carga los tilesets del TMX en un atlas de texturas y crea tablas de consulta por GID.

Soporta tilesets de imagen única (rejilla con margin/spacing) y colecciones de
imágenes (un archivo por tile). Todos los tiles de todos los tilesets comparten
atlas; solo se reparte en varias páginas si no cabe en una textura.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from tmx_manager import TiledMap

log = logging.getLogger(__name__)


@dataclass
class TileLookup:
    """Tablas indexadas por GID (sin bits de flip). page == -1 significa "sin imagen"."""
    page: np.ndarray    # (max_gid + 1,) int32
    uv: np.ndarray      # (max_gid + 1, 4) float32: u0, v0, u1, v1 (v1 = arriba)
    size: np.ndarray    # (max_gid + 1, 2) float32: ancho, alto en píxeles

    @classmethod
    def from_regions(cls, regions: Dict[int, "AtlasRegion"]) -> "TileLookup":
        n = max(regions, default=0) + 1
        lookup = cls(page=np.full(n, -1, dtype=np.int32),
                     uv=np.zeros((n, 4), dtype=np.float32),
                     size=np.zeros((n, 2), dtype=np.float32))
        for gid, r in regions.items():
            lookup.page[gid] = r.page
            lookup.uv[gid] = r.uv
            lookup.size[gid] = (r.width, r.height)
        return lookup

    def get(self, gids: np.ndarray):
        """(page, uv, size) para un array de GIDs; los que se salen de la tabla dan page -1."""
        gids = np.asarray(gids)
        inside = gids < len(self.page)
        safe = np.where(inside, gids, 0)
        page = np.where(inside, self.page[safe], -1)
        return page, self.uv[safe], self.size[safe]


def load_tileset_images(tmx_map: TiledMap, tmx_path) -> Dict[int, Image.Image]:
    """GID -> imagen RGBA de cada tile. No necesita OpenGL."""
    tmx_dir = Path(tmx_path).parent
    images: Dict[int, Image.Image] = {}
    for tileset in tmx_map.tilesets:
        # Las rutas de un TSX externo son relativas al propio TSX
        base = tmx_dir / Path(tileset.source).parent if tileset.source else tmx_dir
        if tileset.image:
            images.update(_cut_image_tileset(tileset, base))
        elif tileset.tiles:
            images.update(_load_collection_tileset(tileset, base))
    return images


def _cut_image_tileset(tileset, base: Path) -> Dict[int, Image.Image]:
    image_path = base / tileset.image.source
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGBA")
    except OSError as e:
        log.warning("No se pudo cargar %s: %s", image_path, e)
        return {}
    if tileset.columns <= 0:
        log.warning("Tileset '%s' sin columnas; se ignora", tileset.name)
        return {}

    tw, th = tileset.tilewidth, tileset.tileheight
    tiles = {}
    for tile_id in range(tileset.tilecount):
        col, row = tile_id % tileset.columns, tile_id // tileset.columns
        x = tileset.margin + col * (tw + tileset.spacing)
        y = tileset.margin + row * (th + tileset.spacing)
        tiles[tileset.firstgid + tile_id] = image.crop((x, y, x + tw, y + th))
    log.info("Tileset '%s': %d tiles (%dx%d)", tileset.name, len(tiles),
             image.width, image.height)
    return tiles


def _load_collection_tileset(tileset, base: Path) -> Dict[int, Image.Image]:
    tiles = {}
    for tile_id, tile in tileset.tiles.items():
        if not tile.image:
            continue
        path = base / tile.image.source
        try:
            with Image.open(path) as img:
                tiles[tileset.firstgid + tile_id] = img.convert("RGBA")
        except OSError as e:
            log.warning("No se pudo cargar %s: %s", path, e)
    log.info("Colección '%s': %d tiles", tileset.name, len(tiles))
    return tiles


class TilesetRenderer:
    """Atlas de todos los tiles del mapa más sus tablas de consulta por GID."""

    def __init__(self, tmx_map: TiledMap, tmx_path: str, gl_renderer=None):
        from ..renderer.atlas import TextureAtlas
        images = load_tileset_images(tmx_map, tmx_path)
        self.atlas = TextureAtlas.build(images)
        self.lookup = TileLookup.from_regions(self.atlas.regions)
        log.info("Atlas de tiles: %d tiles en %d página(s): %s", len(images),
                 self.atlas.page_count,
                 ", ".join(f"{t.width}x{t.height}" for t in self.atlas.textures))

    @property
    def textures(self):
        """Una textura por página del atlas."""
        return self.atlas.textures

    def get_tile_size(self, gid: int) -> Optional[Tuple[int, int]]:
        region = self.atlas.regions.get(gid)
        return (region.width, region.height) if region else None
