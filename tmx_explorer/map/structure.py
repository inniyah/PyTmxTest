"""
3D structure representation of TMX maps
"""

import numpy as np
from typing import List, Tuple, Dict
from tmx_manager import TiledMap, TileLayer, LayerGroup


class Map3DStructure:
    """
    3D representation of a TMX map.
    
    Dimensions:
    - W: Map width in tiles
    - D: Map depth (Y axis) in tiles  
    - H: Height levels (Z axis from layer properties)
    - N: Number of tile layers
    
    The 4D array `mapa[z, y, x, n]` stores tile GIDs for each position.
    """

    def __init__(self, tmx_map: TiledMap):
        self.map_width = tmx_map.width
        self.map_height = tmx_map.height
        self.tile_width = tmx_map.tilewidth
        self.tile_height = tmx_map.tileheight
        
        self.W = self.map_width
        self.D = self.map_height

        # Extract layer info: (layer, level, name)
        self.layer_info: List[Tuple[TileLayer, int, str]] = []
        self._extract_all_layers(tmx_map)

        # Determine height levels from layer properties
        self.levels: Dict[int, bool] = {}
        for _, level, _ in self.layer_info:
            self.levels[level] = True

        self.min_level = min(self.levels.keys()) if self.levels else 0
        self.max_level = max(self.levels.keys()) if self.levels else 0
        self.H = self.max_level - self.min_level + 1
        self.level_offset = -self.min_level
        self.N = len(self.layer_info)

        # 4D array: [height, depth, width, layer]
        self.mapa = np.zeros((self.H, self.D, self.W, self.N), dtype=np.uint16)
        self.layer_names: List[str] = []
        self.layer_levels: List[int] = []

        print(f"\n=== 3D Map Structure ===")
        print(f"Dimensions: W={self.W}, D={self.D}, H={self.H}, N={self.N}")

        self._load_layers()

    def _extract_all_layers(self, tmx_map: TiledMap):
        """Recursively extract all tile layers from map"""
        
        def process_layer(layer, layer_name: str = ""):
            if isinstance(layer, TileLayer):
                level = self._get_layer_level(layer)
                full_name = layer_name or layer.name
                self.layer_info.append((layer, level, full_name))
                
            elif isinstance(layer, LayerGroup):
                for sublayer in layer.layers:
                    subname = f"{layer_name}/{sublayer.name}" if layer_name else sublayer.name
                    process_layer(sublayer, subname)

        for layer in tmx_map.layers:
            process_layer(layer)

    def _get_layer_level(self, layer: TileLayer) -> int:
        """Extract Z level from layer properties"""
        level = 0
        for prop_name in ('Z', 'z', 'level'):
            if prop_name in layer.properties:
                level = layer.properties[prop_name].value
                break
        
        if isinstance(level, str):
            level = int(level, 0)
        elif isinstance(level, float):
            level = int(level)
            
        return level

    def _load_layers(self):
        """Load tile data into 4D array"""
        for layer_idx, (layer, level, layer_name) in enumerate(self.layer_info):
            z = level + self.level_offset
            if not (0 <= z < self.H):
                continue

            for y in range(min(layer.height, self.D)):
                for x in range(min(layer.width, self.W)):
                    gid = layer.get_tile_gid(x, y)
                    if gid > 0:
                        self.mapa[z, y, x, layer_idx] = gid

            self.layer_names.append(layer_name)
            self.layer_levels.append(level)

    def get_level_value(self, z: int) -> int:
        """Convert internal Z index to original level value"""
        return z - self.level_offset

    def get_tile(self, x: int, y: int, z: int, layer: int) -> int:
        """Get tile GID at position"""
        if (0 <= x < self.W and 0 <= y < self.D and 
            0 <= z < self.H and 0 <= layer < self.N):
            return self.mapa[z, y, x, layer]
        return 0

    def set_tile(self, x: int, y: int, z: int, layer: int, gid: int):
        """Set tile GID at position"""
        if (0 <= x < self.W and 0 <= y < self.D and 
            0 <= z < self.H and 0 <= layer < self.N):
            self.mapa[z, y, x, layer] = gid
