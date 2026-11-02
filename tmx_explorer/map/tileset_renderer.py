"""Carga los tilesets del TMX y sube cada tile a la GPU como textura (GID -> Texture).

Soporta tilesets de imagen única (rejilla con margin/spacing) y colecciones de
imágenes (un archivo por tile).
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from PIL import Image

from tmx_manager import TiledMap

if TYPE_CHECKING:
    from ..renderer.opengl_renderer import OpenGLRenderer
    from ..renderer.texture import Texture

log = logging.getLogger(__name__)


class TilesetRenderer:
    def __init__(self, tmx_map: TiledMap, tmx_path: str, gl_renderer: "OpenGLRenderer"):
        self.tmx_map = tmx_map
        self.tmx_dir = Path(tmx_path).parent
        self.gl_renderer = gl_renderer
        self.tile_size_cache: Dict[int, Tuple[int, int]] = {}
        self.tile_texture_cache: Dict[int, "Texture"] = {}
        self._load_tilesets()

    def _load_tilesets(self):
        for tileset in self.tmx_map.tilesets:
            # Las rutas de un TSX externo son relativas al propio TSX
            base = self.tmx_dir
            if tileset.source:
                base = self.tmx_dir / Path(tileset.source).parent

            if tileset.image:
                self._load_image_tileset(tileset, base)
            elif tileset.tiles:
                self._load_collection_tileset(tileset, base)

    def _load_image_tileset(self, tileset, base: Path):
        image_path = base / tileset.image.source
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGBA")
        except OSError as e:
            log.warning("No se pudo cargar %s: %s", image_path, e)
            return
        if tileset.columns <= 0:
            log.warning("Tileset '%s' sin columnas; se ignora", tileset.name)
            return

        tw, th = tileset.tilewidth, tileset.tileheight
        for tile_id in range(tileset.tilecount):
            col, row = tile_id % tileset.columns, tile_id // tileset.columns
            x = tileset.margin + col * (tw + tileset.spacing)
            y = tileset.margin + row * (th + tileset.spacing)
            self._register(tileset.firstgid + tile_id, image.crop((x, y, x + tw, y + th)))
        log.info("Tileset '%s': %d tiles (%dx%d)", tileset.name, tileset.tilecount,
                 image.width, image.height)

    def _load_collection_tileset(self, tileset, base: Path):
        loaded = 0
        for tile_id, tile in tileset.tiles.items():
            if not tile.image:
                continue
            path = base / tile.image.source
            try:
                with Image.open(path) as img:
                    self._register(tileset.firstgid + tile_id, img.convert("RGBA"))
                loaded += 1
            except OSError as e:
                log.warning("No se pudo cargar %s: %s", path, e)
        log.info("Colección '%s': %d tiles", tileset.name, loaded)

    def _register(self, gid: int, image: Image.Image):
        self.tile_size_cache[gid] = image.size
        self.tile_texture_cache[gid] = self.gl_renderer.preload_texture(gid, image)

    def get_tile_texture(self, gid: int) -> Optional["Texture"]:
        return self.tile_texture_cache.get(gid) if gid else None

    def get_tile_size(self, gid: int) -> Optional[Tuple[int, int]]:
        """Tamaño en píxeles (ancho, alto) de la imagen del tile."""
        return self.tile_size_cache.get(gid) if gid else None

    # Nombre antiguo, se mantiene por compatibilidad
    get_tile_surface = get_tile_size
