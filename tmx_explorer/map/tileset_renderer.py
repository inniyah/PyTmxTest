"""
Tileset loading and tile texture management
"""

import pygame
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

from tmx_manager import TiledMap

if TYPE_CHECKING:
    from ..renderer.opengl_renderer import OpenGLRenderer
    from ..renderer.texture import Texture


class TilesetRenderer:
    """Loads tilesets and manages tile textures with bleeding prevention"""

    def __init__(self, tmx_map: TiledMap, tmx_path: str, gl_renderer: 'OpenGLRenderer'):
        self.tmx_map = tmx_map
        self.tmx_path = Path(tmx_path).parent
        self.gl_renderer = gl_renderer
        
        self.tileset_surfaces: Dict[int, dict] = {}
        self.tile_surface_cache: Dict[int, pygame.Surface] = {}
        self.tile_texture_cache: Dict[int, 'Texture'] = {}
        
        self._load_tilesets()

    def _load_tilesets(self):
        """Load all tilesets from the map"""
        print("\n=== Loading Tilesets ===")
        
        for tileset in self.tmx_map.tilesets:
            tileset_base = self.tmx_path
            if tileset.source:
                tileset_base = self.tmx_path / Path(tileset.source).parent

            if tileset.image:
                self._load_image_tileset(tileset, tileset_base)
            elif tileset.tiles:
                self._load_collection_tileset(tileset, tileset_base)

    def _load_image_tileset(self, tileset, tileset_base: Path):
        """Load a tileset based on a single image"""
        image_path = tileset_base / tileset.image.source
        try:
            surface = pygame.image.load(str(image_path))
            self.tileset_surfaces[tileset.firstgid] = {
                'surface': surface,
                'tileset': tileset
            }
            print(f"Loaded tileset: {tileset.name}")
            self._preload_tileset_tiles(tileset, surface)
        except pygame.error as e:
            print(f"Warning: {e}")

    def _load_collection_tileset(self, tileset, tileset_base: Path):
        """Load an image collection tileset"""
        print(f"Loading image collection: {tileset.name} ({len(tileset.tiles)} tiles)")
        
        for tile_id, tile in tileset.tiles.items():
            if tile.image:
                image_path = tileset_base / tile.image.source
                try:
                    surface = pygame.image.load(str(image_path))
                    gid = tileset.firstgid + tile_id
                    self.tile_surface_cache[gid] = surface
                    self.tile_texture_cache[gid] = self.gl_renderer.preload_texture(gid, surface)
                except pygame.error as e:
                    print(f"  Warning: {e}")

    def _preload_tileset_tiles(self, tileset, tileset_surface: pygame.Surface):
        """Pre-load ALL tiles with 1px border to prevent bleeding"""
        if tileset.columns <= 0:
            return

        tiles_loaded = 0
        border = 1
        
        for tile_id in range(tileset.tilecount):
            gid = tileset.firstgid + tile_id
            
            # Calculate tile position in tileset
            col = tile_id % tileset.columns
            row = tile_id // tileset.columns
            
            tile_x = col * tileset.tilewidth + tileset.margin + col * tileset.spacing
            tile_y = row * tileset.tileheight + tileset.margin + row * tileset.spacing

            # Create surface with 2px extra (1px border on each side)
            tile_surface = pygame.Surface(
                (tileset.tilewidth + border * 2, tileset.tileheight + border * 2),
                pygame.SRCALPHA
            )

            # Blit main tile in center
            tile_surface.blit(
                tileset_surface, (border, border),
                pygame.Rect(tile_x, tile_y, tileset.tilewidth, tileset.tileheight)
            )

            # Extrude edges to prevent bleeding
            self._extrude_tile_edges(
                tile_surface, tileset_surface, 
                tile_x, tile_y, 
                tileset.tilewidth, tileset.tileheight, 
                border
            )

            self.tile_surface_cache[gid] = tile_surface
            self.tile_texture_cache[gid] = self.gl_renderer.preload_texture(gid, tile_surface)
            tiles_loaded += 1

        print(f"  Pre-loaded {tiles_loaded} tiles (with 1px border)")

    def _extrude_tile_edges(self, dest: pygame.Surface, src: pygame.Surface,
                           tx: int, ty: int, tw: int, th: int, border: int):
        """Copy edge pixels to border area to prevent texture bleeding"""
        # Top edge
        dest.blit(src, (border, 0), pygame.Rect(tx, ty, tw, 1))
        # Bottom edge
        dest.blit(src, (border, th + border), pygame.Rect(tx, ty + th - 1, tw, 1))
        # Left edge
        dest.blit(src, (0, border), pygame.Rect(tx, ty, 1, th))
        # Right edge
        dest.blit(src, (tw + border, border), pygame.Rect(tx + tw - 1, ty, 1, th))
        
        # Corners
        dest.blit(src, (0, 0), pygame.Rect(tx, ty, 1, 1))
        dest.blit(src, (tw + border, 0), pygame.Rect(tx + tw - 1, ty, 1, 1))
        dest.blit(src, (0, th + border), pygame.Rect(tx, ty + th - 1, 1, 1))
        dest.blit(src, (tw + border, th + border), pygame.Rect(tx + tw - 1, ty + th - 1, 1, 1))

    def get_tile_texture(self, gid: int) -> Optional['Texture']:
        """Get pre-loaded texture for tile GID"""
        if gid == 0:
            return None
        return self.tile_texture_cache.get(gid)

    def get_tile_surface(self, gid: int) -> Optional[pygame.Surface]:
        """Get tile surface for GID"""
        if gid == 0:
            return None
        return self.tile_surface_cache.get(gid)
