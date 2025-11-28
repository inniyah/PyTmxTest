"""
Tileset loading and tile texture management (GLFW version - uses PIL)
"""

from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

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
        
        # Cache: gid -> (width, height) of original tile (without border)
        self.tile_size_cache: Dict[int, Tuple[int, int]] = {}
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
            image = Image.open(str(image_path)).convert('RGBA')
            print(f"Loaded tileset: {tileset.name} ({image.width}x{image.height})")
            self._preload_tileset_tiles(tileset, image)
        except Exception as e:
            print(f"Warning: Could not load {image_path}: {e}")

    def _load_collection_tileset(self, tileset, tileset_base: Path):
        """Load an image collection tileset"""
        print(f"Loading image collection: {tileset.name} ({len(tileset.tiles)} tiles)")
        
        for tile_id, tile in tileset.tiles.items():
            if tile.image:
                image_path = tileset_base / tile.image.source
                try:
                    tile_img = Image.open(str(image_path)).convert('RGBA')
                    gid = tileset.firstgid + tile_id
                    
                    self.tile_size_cache[gid] = (tile_img.width, tile_img.height)
                    self.tile_texture_cache[gid] = self.gl_renderer.preload_texture(gid, tile_img)
                except Exception as e:
                    print(f"  Warning: {e}")

    def _preload_tileset_tiles(self, tileset, tileset_image: Image.Image):
        """Pre-load ALL tiles from tileset image"""
        if tileset.columns <= 0:
            return

        tiles_loaded = 0
        tw = tileset.tilewidth
        th = tileset.tileheight
        
        for tile_id in range(tileset.tilecount):
            gid = tileset.firstgid + tile_id
            
            # Calculate tile position
            col = tile_id % tileset.columns
            row = tile_id // tileset.columns
            
            tile_x = col * tw + tileset.margin + col * tileset.spacing
            tile_y = row * th + tileset.margin + row * tileset.spacing
            
            # Crop tile from tileset
            tile_img = tileset_image.crop((tile_x, tile_y, tile_x + tw, tile_y + th))
            
            self.tile_size_cache[gid] = (tw, th)
            self.tile_texture_cache[gid] = self.gl_renderer.preload_texture(gid, tile_img)
            tiles_loaded += 1

        print(f"  Pre-loaded {tiles_loaded} tiles")

    def get_tile_texture(self, gid: int) -> Optional['Texture']:
        """Get pre-loaded texture for tile GID"""
        if gid == 0:
            return None
        return self.tile_texture_cache.get(gid)

    def get_tile_surface(self, gid: int) -> Optional[Tuple[int, int]]:
        """Get tile size (width, height) for GID"""
        if gid == 0:
            return None
        return self.tile_size_cache.get(gid)
