#!/usr/bin/env python3

"""
TMX Map Explorer with 3D NumPy Structure
Loads a TMX map and converts it into a 4D numpy array: [z, y, x, layer]

The structure handles:
- W, D: Map width and depth from TMX dimensions
- H: Heights from layer 'level' property (automatic offset calculation)
- N: Number of actual layers (each TMX layer gets its own index)

Controls:
- Arrow keys or WASD: Move camera
- Mouse wheel or +/-: Zoom in/out
- Shift +/-: Adjust level height offset (for isometric stacking)
- Page Up/Down: Change maximum visible height level
- Space: Reset camera
- L: Toggle layer visibility menu
- 0-9: Toggle individual layer visibility (when menu is open)
- G: Toggle grid
- I: Toggle info panel
- ESC or Q: Quit

Usage:
    ./tmx_explorer.py <map.tmx>
"""

import sys
import pygame
import numpy as np
from pathlib import Path
from tmx_manager import TiledMap, TileLayer, ObjectGroup, LayerGroup

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)


class Camera:
    """Camera for scrolling and zooming"""
    def __init__(self, width, height):
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.width = width
        self.height = height
        
    def apply(self, x, y):
        """Apply camera transformation to coordinates"""
        return int((x - self.x) * self.zoom), int((y - self.y) * self.zoom)
    
    def apply_rect(self, rect):
        """Apply camera transformation to a rectangle"""
        x, y = self.apply(rect.x, rect.y)
        w = int(rect.width * self.zoom)
        h = int(rect.height * self.zoom)
        return pygame.Rect(x, y, w, h)
    
    def move(self, dx, dy):
        """Move camera"""
        self.x += dx / self.zoom
        self.y += dy / self.zoom
    
    def set_zoom(self, zoom):
        """Set zoom level"""
        # Keep center point
        center_x = self.x + self.width / (2 * self.zoom)
        center_y = self.y + self.height / (2 * self.zoom)
        
        self.zoom = max(0.1, min(5.0, zoom))
        
        # Adjust position to keep center
        self.x = center_x - self.width / (2 * self.zoom)
        self.y = center_y - self.height / (2 * self.zoom)
    
    def reset(self, map_width, map_height, tile_width, tile_height):
        """Reset camera to show entire map"""
        self.x = 0
        self.y = 0
        
        # Calculate zoom to fit map
        zoom_x = self.width / (map_width * tile_width)
        zoom_y = self.height / (map_height * tile_height)
        self.zoom = min(zoom_x, zoom_y, 1.0)


class TilesetRenderer:
    """Handles tileset image loading and rendering"""
    def __init__(self, tmx_map, tmx_path):
        self.tmx_map = tmx_map
        self.tmx_path = Path(tmx_path).parent
        self.tileset_surfaces = {}
        self.tile_cache = {}
        self.tileset_base_paths = {}
        self.load_tilesets()
    
    def load_tilesets(self):
        """Load all tileset images"""
        for tileset in self.tmx_map.tilesets:
            if tileset.source:
                tileset_base = self.tmx_path / Path(tileset.source).parent
            else:
                tileset_base = self.tmx_path
            
            self.tileset_base_paths[tileset.firstgid] = tileset_base
            
            if tileset.image:
                image_path = tileset_base / tileset.image.source
                try:
                    surface = pygame.image.load(str(image_path))
                    self.tileset_surfaces[tileset.firstgid] = {
                        'surface': surface,
                        'tileset': tileset
                    }
                    print(f"Loaded tileset: {tileset.name} from {image_path}")
                except pygame.error as e:
                    print(f"Warning: Could not load tileset image {image_path}: {e}")
                    surface = pygame.Surface((tileset.tilewidth, tileset.tileheight))
                    surface.fill(GRAY)
                    self.tileset_surfaces[tileset.firstgid] = {
                        'surface': surface,
                        'tileset': tileset
                    }
            elif tileset.tiles:
                print(f"Loading image collection tileset: {tileset.name} ({len(tileset.tiles)} tiles)")
                for tile_id, tile in tileset.tiles.items():
                    if tile.image:
                        image_path = tileset_base / tile.image.source
                        try:
                            surface = pygame.image.load(str(image_path))
                            gid = tileset.firstgid + tile_id
                            self.tile_cache[gid] = surface
                        except pygame.error as e:
                            print(f"  Warning: Could not load tile image {image_path}: {e}")
                            surface = pygame.Surface((tileset.tilewidth, tileset.tileheight))
                            surface.fill(RED)
                            gid = tileset.firstgid + tile_id
                            self.tile_cache[gid] = surface
                
                self.tileset_surfaces[tileset.firstgid] = {
                    'surface': None,
                    'tileset': tileset
                }
    
    def get_tile_surface(self, gid):
        """Get surface for a specific tile GID"""
        if gid == 0:
            return None
        
        if gid in self.tile_cache:
            return self.tile_cache[gid]
        
        tileset_data = None
        for firstgid in sorted(self.tileset_surfaces.keys(), reverse=True):
            if gid >= firstgid:
                tileset_data = self.tileset_surfaces[firstgid]
                break
        
        if not tileset_data:
            return None
        
        tileset = tileset_data['tileset']
        tileset_surface = tileset_data['surface']
        
        if tileset_surface is None:
            return None
        
        local_id = gid - tileset.firstgid
        
        if tileset.columns > 0:
            tile_x = (local_id % tileset.columns) * tileset.tilewidth
            tile_y = (local_id // tileset.columns) * tileset.tileheight
            
            tile_x += tileset.margin + (local_id % tileset.columns) * tileset.spacing
            tile_y += tileset.margin + (local_id // tileset.columns) * tileset.spacing
            
            tile_surface = pygame.Surface((tileset.tilewidth, tileset.tileheight), pygame.SRCALPHA)
            tile_surface.blit(tileset_surface, (0, 0), 
                            pygame.Rect(tile_x, tile_y, tileset.tilewidth, tileset.tileheight))
            
            self.tile_cache[gid] = tile_surface
            return tile_surface
        
        return None


class Map3DStructure:
    """Handles the 3D numpy array structure of the map"""
    def __init__(self, tmx_map):
        self.tmx_map = tmx_map
        
        # Store map dimensions (so we don't need tmx_map later)
        self.map_width = tmx_map.width   # Map width in tiles
        self.map_height = tmx_map.height # Map height in tiles
        self.tile_width = tmx_map.tilewidth   # Tile width in pixels
        self.tile_height = tmx_map.tileheight # Tile height in pixels
        
        self.W = self.map_width   # Width
        self.D = self.map_height  # Depth
        
        # First pass: extract all layers and their levels
        self.layer_info = []  # List of (layer, level, layer_name)
        self._extract_all_layers()
        
        # Calculate dimensions
        self.levels = {}
        for _, level, _ in self.layer_info:
            self.levels[level] = True
        
        self.min_level = min(self.levels.keys()) if self.levels else 0
        self.max_level = max(self.levels.keys()) if self.levels else 0
        self.H = self.max_level - self.min_level + 1  # Height (number of levels)
        self.level_offset = -self.min_level  # Offset to make array start at 0
        
        self.N = len(self.layer_info)  # Number of layers (each layer gets its own index)
        
        # Create the 4D array: [z, y, x, layer_index]
        self.mapa = np.zeros((self.H, self.D, self.W, self.N), dtype=np.uint16)
        
        # Create mapping from layer index to layer info
        self.layer_names = []
        self.layer_levels = []
        
        print(f"\n=== 3D Map Structure ===")
        print(f"Dimensions: W={self.W}, D={self.D}, H={self.H}, N={self.N}")
        print(f"Tile size: {self.tile_width}x{self.tile_height}px")
        print(f"Level range: {self.min_level} to {self.max_level} (offset: {self.level_offset})")
        print(f"Array shape: {self.mapa.shape}")
        print(f"Memory usage: {self.mapa.nbytes / 1024 / 1024:.2f} MB")
        
        # Load data into array
        self._load_layers()
        
        # After loading, we don't need tmx_map anymore
        self.tmx_map = None
        print("\nTMX map unloaded - now using numpy array only")
    
    def _extract_all_layers(self):
        """Extract all tile layers from the TMX structure"""
        def process_layer(layer, layer_name=""):
            if isinstance(layer, TileLayer):
                # Get level for this layer
                level = 0
                if 'level' in layer.properties:
                    level = layer.properties['level'].value
                
                full_name = layer_name or layer.name
                self.layer_info.append((layer, level, full_name))
            
            elif isinstance(layer, LayerGroup):
                for sublayer in layer.layers:
                    subname = f"{layer_name}/{sublayer.name}" if layer_name else sublayer.name
                    process_layer(sublayer, subname)
        
        for layer in self.tmx_map.layers:
            process_layer(layer)
    
    def _load_layers(self):
        """Load all tile layers into the numpy array"""
        print(f"\nLoading {len(self.layer_info)} layers:")
        
        for layer_idx, (layer, level, layer_name) in enumerate(self.layer_info):
            # Calculate z index with offset
            z = level + self.level_offset
            
            if not (0 <= z < self.H):
                print(f"Warning: Layer '{layer_name}' level {level} out of bounds")
                continue
            
            # Copy tile data into array at this layer's index
            tiles_copied = 0
            for y in range(min(layer.height, self.D)):
                for x in range(min(layer.width, self.W)):
                    gid = layer.get_tile_gid(x, y)
                    if gid > 0:
                        self.mapa[z, y, x, layer_idx] = gid
                        tiles_copied += 1
            
            # Store layer metadata
            self.layer_names.append(layer_name)
            self.layer_levels.append(level)
            
            # Get layer type for display
            layer_type = "?"
            if 'tipo' in layer.properties:
                layer_type = layer.properties['tipo'].value
            
            print(f"  [{layer_idx}] '{layer_name}' -> z={z} (level={level}), tipo={layer_type}, tiles={tiles_copied}")
        
        # Statistics per level
        print(f"\nStatistics per level:")
        for z in range(self.H):
            level_tiles = np.count_nonzero(self.mapa[z, :, :, :])
            if level_tiles > 0:
                # Show which layers contribute to this level
                contributing_layers = []
                for n in range(self.N):
                    count = np.count_nonzero(self.mapa[z, :, :, n])
                    if count > 0:
                        contributing_layers.append(f"{self.layer_names[n]}({count})")
                
                layers_str = ", ".join(contributing_layers[:3])  # Show first 3
                if len(contributing_layers) > 3:
                    layers_str += f" + {len(contributing_layers)-3} more"
                
                print(f"  z={z} (level={self.get_level_value(z)}): {level_tiles:,} tiles from {len(contributing_layers)} layers")
                print(f"    Layers: {layers_str}")
        
        # Overall statistics
        non_zero = np.count_nonzero(self.mapa)
        total = self.mapa.size
        print(f"\nTotal non-zero tiles: {non_zero:,} / {total:,} ({non_zero/total*100:.2f}%)")
    
    def get_slice(self, z):
        """Get a 2D slice at height z"""
        if 0 <= z < self.H:
            return self.mapa[z, :, :, :]
        return None
    
    def get_level_value(self, z):
        """Convert z index to actual level value"""
        return z - self.level_offset


class TMXExplorer:
    """Main TMX explorer application"""
    def __init__(self, tmx_path):
        pygame.init()
        
        # Load TMX map
        print(f"Loading TMX map: {tmx_path}")
        self.tmx_map = TiledMap.load(tmx_path)
        self.tmx_path = tmx_path
        
        print(f"Map size: {self.tmx_map.width}x{self.tmx_map.height}")
        print(f"Tile size: {self.tmx_map.tilewidth}x{self.tmx_map.tileheight}")
        
        # Set up display
        self.screen_width = 1280
        self.screen_height = 720
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), 
                                              pygame.RESIZABLE)
        pygame.display.set_caption(f"TMX Explorer - {Path(tmx_path).name}")
        
        # Tileset renderer (needs tmx_map)
        self.tileset_renderer = TilesetRenderer(self.tmx_map, tmx_path)
        
        # 3D structure (converts TMX to numpy)
        self.map_3d = Map3DStructure(self.tmx_map)
        
        # We no longer need the tmx_map object after loading into numpy
        # All necessary data is now in map_3d
        self.tmx_map = None
        
        # Camera (needs map_3d dimensions)
        self.camera = Camera(self.screen_width, self.screen_height)
        self.camera.reset(self.map_3d.map_width, self.map_3d.map_height,
                         self.map_3d.tile_width, self.map_3d.tile_height)
        
        # Current height level to display - start at the highest level to see everything
        self.current_z = self.map_3d.H - 1  # Start at max height
        print(f"\nStarting at height level: {self.current_z} (level={self.map_3d.get_level_value(self.current_z)})")
        print(f"Showing all levels: 0 to {self.current_z}")
        
        # Height offset between levels (for isometric rendering)
        # This should match the vertical offset between layers in TMX
        self.level_height_offset = 128  # Default, can be adjusted with +/- keys
        
        # Adjust camera to show all levels at start
        self._adjust_camera_for_all_levels()
        
        # UI state
        self.show_grid = False
        self.show_info = True
        self.show_layer_menu = False
        self.layer_visibility = [True] * self.map_3d.N  # Now N is number of actual layers
        
        # Mouse panning state
        self.panning = False
        self.pan_start_pos = (0, 0)
        self.pan_start_camera = (0, 0)
        
        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Clock
        self.clock = pygame.time.Clock()
        self.running = True
    
    def _adjust_camera_for_all_levels(self):
        """Adjust camera position and zoom to show all levels"""
        # Calculate the total vertical span needed to show all levels
        # The highest level is offset by: current_z * level_height_offset
        total_height_offset = self.current_z * self.level_height_offset
        
        # Calculate the map dimensions
        map_pixel_width = self.map_3d.map_width * self.map_3d.tile_width
        map_pixel_height = self.map_3d.map_height * self.map_3d.tile_height
        
        # Add the vertical offset to the height
        total_pixel_height = map_pixel_height + total_height_offset
        
        # Calculate zoom to fit everything
        zoom_x = self.screen_width / map_pixel_width
        zoom_y = self.screen_height / total_pixel_height
        target_zoom = min(zoom_x, zoom_y, 1.0) * 0.85  # 0.85 for some margin
        
        # Adjust camera position to center on the full stack
        # Center y should be adjusted to show from bottom (z=0) to top (current_z)
        center_x = map_pixel_width / 2
        center_y = (map_pixel_height - total_height_offset) / 2
        
        # Set new camera position
        self.camera.x = center_x - self.screen_width / (2 * target_zoom)
        self.camera.y = center_y - self.screen_height / (2 * target_zoom)
        self.camera.zoom = target_zoom
        
        if self.current_z > 0:
            print(f"Camera adjusted to show all levels 0-{self.current_z} (offset: {total_height_offset}px, zoom: {target_zoom:.2f})")
    
    def handle_events(self):
        """Handle input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.VIDEORESIZE:
                self.screen_width = event.w
                self.screen_height = event.h
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height),
                                                     pygame.RESIZABLE)
                self.camera.width = self.screen_width
                self.camera.height = self.screen_height
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.panning = True
                    self.pan_start_pos = event.pos
                    self.pan_start_camera = (self.camera.x, self.camera.y)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.panning = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.panning:
                    dx = event.pos[0] - self.pan_start_pos[0]
                    dy = event.pos[1] - self.pan_start_pos[1]
                    self.camera.x = self.pan_start_camera[0] - dx / self.camera.zoom
                    self.camera.y = self.pan_start_camera[1] - dy / self.camera.zoom
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.camera.reset(self.map_3d.map_width, self.map_3d.map_height,
                                    self.map_3d.tile_width, self.map_3d.tile_height)
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_l:
                    self.show_layer_menu = not self.show_layer_menu
                elif event.key == pygame.K_PAGEUP:
                    old_z = self.current_z
                    self.current_z = min(self.current_z + 1, self.map_3d.H - 1)
                    if old_z != self.current_z:
                        print(f"Height level: {self.current_z} (level={self.map_3d.get_level_value(self.current_z)})")
                        self._adjust_camera_for_all_levels()
                elif event.key == pygame.K_PAGEDOWN:
                    old_z = self.current_z
                    self.current_z = max(self.current_z - 1, 0)
                    if old_z != self.current_z:
                        print(f"Height level: {self.current_z} (level={self.map_3d.get_level_value(self.current_z)})")
                        self._adjust_camera_for_all_levels()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # With shift: adjust level offset
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        self.level_height_offset += 16
                        print(f"Level height offset: {self.level_height_offset}")
                        self._adjust_camera_for_all_levels()
                    else:
                        self.camera.set_zoom(self.camera.zoom * 1.2)
                elif event.key == pygame.K_MINUS:
                    # With shift: adjust level offset
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        self.level_height_offset = max(0, self.level_height_offset - 16)
                        print(f"Level height offset: {self.level_height_offset}")
                        self._adjust_camera_for_all_levels()
                    else:
                        self.camera.set_zoom(self.camera.zoom / 1.2)
                
                # Toggle individual layer visibility (0-9)
                if self.show_layer_menu and pygame.K_0 <= event.key <= pygame.K_9:
                    layer_idx = event.key - pygame.K_0
                    if layer_idx < len(self.layer_visibility):
                        self.layer_visibility[layer_idx] = not self.layer_visibility[layer_idx]
                        print(f"Layer {layer_idx} ({self.map_3d.layer_names[layer_idx]}): {'ON' if self.layer_visibility[layer_idx] else 'OFF'}")
            
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.camera.set_zoom(self.camera.zoom * 1.1)
                else:
                    self.camera.set_zoom(self.camera.zoom / 1.1)
        
        # Continuous key presses
        keys = pygame.key.get_pressed()
        base_speed = 10
        move_speed = base_speed / self.camera.zoom
        
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.camera.move(-move_speed, 0)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.camera.move(move_speed, 0)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.camera.move(0, -move_speed)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.camera.move(0, move_speed)
    
    def draw_tiles(self):
        """Draw tiles from the 3D structure using proper isometric depth ordering"""
        # Always render all levels from 0 to current_z (bottom to top)
        z_start = 0
        z_end = self.current_z + 1
        
        # Calculate visible range for culling (optimization)
        start_x = max(0, int(self.camera.x / self.map_3d.tile_width) - 3)
        start_y = max(0, int(self.camera.y / self.map_3d.tile_height) - 4)
        end_x = min(self.map_3d.W, int((self.camera.x + self.screen_width / self.camera.zoom) / 
                                       self.map_3d.tile_width) + 4)
        end_y = min(self.map_3d.D, int((self.camera.y + self.screen_height / self.camera.zoom) / 
                                       self.map_3d.tile_height) + 5)
        
        tiles_drawn = 0
        
        # Main rendering loop for isometric projection
        # Order: H (height/z) → N (layers) → diagonal sorting → W, D
        # This ensures: bottom-to-top heights, then layer order, then isometric depth
        
        for z in range(z_start, z_end):
            # Calculate vertical offset for this level (isometric stacking)
            level_y_offset = z * self.level_height_offset
            
            # Draw each layer at this height
            for n in range(self.map_3d.N):
                # Skip if layer is not visible
                if not self.layer_visibility[n]:
                    continue
                
                # Skip if this layer is not at this z level
                if self.map_3d.layer_levels[n] != self.map_3d.get_level_value(z):
                    continue
                
                # For this layer, draw in isometric diagonal order
                # Calculate the range of diagonals we need to draw
                min_diagonal = start_x + start_y
                max_diagonal = (end_x - 1) + (end_y - 1)
                
                # Draw tiles diagonal by diagonal (back to front)
                for diagonal in range(min_diagonal, max_diagonal + 1):
                    # For each diagonal, iterate through all (x, y) pairs where x + y = diagonal
                    for x in range(start_x, end_x):
                        y = diagonal - x
                        
                        # Check if y is within bounds
                        if y < start_y or y >= end_y:
                            continue
                        
                        # Get tile ID from array: mapa[z, y, x, n]
                        tile_id = self.map_3d.mapa[z, y, x, n]
                        
                        if tile_id != 0:  # 0 = empty
                            # Draw tile
                            self._draw_tile(tile_id, x, y, z, level_y_offset)
                            tiles_drawn += 1
        
        # Debug: show tiles drawn in first few frames
        if tiles_drawn == 0 and self.clock.get_fps() < 5:
            print(f"Warning: No tiles drawn at z={self.current_z}")
            print(f"  Visible range: x=[{start_x}, {end_x}), y=[{start_y}, {end_y})")
            print(f"  Array shape: {self.map_3d.mapa.shape}")
            print(f"  Visible layers: {sum(self.layer_visibility)}/{len(self.layer_visibility)}")
    
    def _draw_tile(self, tile_id, x, y, z, level_y_offset):
        """Draw a single tile at the given position"""
        tile_surface = self.tileset_renderer.get_tile_surface(tile_id)
        if not tile_surface:
            return
        
        tile_width = tile_surface.get_width()
        tile_height = tile_surface.get_height()
        
        # World position - tiles are anchored at bottom
        # For isometric maps, tiles are positioned on a grid but drawn larger
        world_x = x * self.map_3d.tile_width
        world_y = (y + 1) * self.map_3d.tile_height - tile_height - level_y_offset
        
        # Apply camera transformation
        screen_x, screen_y = self.camera.apply(world_x, world_y)
        screen_x = round(screen_x)
        screen_y = round(screen_y)
        
        # Scale tile if zoomed
        if self.camera.zoom != 1.0:
            scaled_w = round(tile_width * self.camera.zoom) + 1
            scaled_h = round(tile_height * self.camera.zoom) + 1
            tile_surface = pygame.transform.smoothscale(tile_surface, (scaled_w, scaled_h))
        
        # Blit to screen
        self.screen.blit(tile_surface, (screen_x, screen_y))
    
    def draw_grid(self):
        """Draw tile grid"""
        if not self.show_grid:
            return
        
        start_x = max(0, int(self.camera.x / self.map_3d.tile_width))
        start_y = max(0, int(self.camera.y / self.map_3d.tile_height))
        end_x = min(self.map_3d.map_width, int((self.camera.x + self.screen_width / self.camera.zoom) / 
                                           self.map_3d.tile_width) + 2)
        end_y = min(self.map_3d.map_height, int((self.camera.y + self.screen_height / self.camera.zoom) / 
                                            self.map_3d.tile_height) + 2)
        
        for x in range(start_x, end_x + 1):
            world_x = x * self.map_3d.tile_width
            screen_x, _ = self.camera.apply(world_x, 0)
            pygame.draw.line(self.screen, DARK_GRAY, 
                           (screen_x, 0), (screen_x, self.screen_height))
        
        for y in range(start_y, end_y + 1):
            world_y = y * self.map_3d.tile_height
            _, screen_y = self.camera.apply(0, world_y)
            pygame.draw.line(self.screen, DARK_GRAY, 
                           (0, screen_y), (self.screen_width, screen_y))
    
    def draw_ui(self):
        """Draw UI overlays"""
        if self.show_info:
            # Info panel
            level_value = self.map_3d.get_level_value(self.current_z)
            slice_data = self.map_3d.get_slice(self.current_z)
            tiles_in_slice = np.count_nonzero(slice_data) if slice_data is not None else 0
            
            # Count visible layers at current level
            visible_layers = sum(1 for i in range(self.map_3d.N) 
                               if self.layer_visibility[i] and 
                               self.map_3d.layer_levels[i] == level_value)
            
            info_lines = [
                f"Zoom: {self.camera.zoom:.2f}x",
                f"Position: ({int(self.camera.x)}, {int(self.camera.y)})",
                f"Height: {self.current_z}/{self.map_3d.H-1} (level={level_value})",
                f"Showing levels: 0 to {self.current_z}",
                f"Level offset: {self.level_height_offset}px",
                f"Visible layers: {visible_layers}/{self.map_3d.N}",
                f"Tiles in view: {tiles_in_slice:,}",
                f"FPS: {int(self.clock.get_fps())}",
            ]
            
            y = 10
            for line in info_lines:
                text = self.small_font.render(line, True, WHITE)
                text_rect = text.get_rect(topleft=(10, y))
                bg_rect = text_rect.inflate(8, 4)
                pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
                self.screen.blit(text, text_rect)
                y += 25
        
        # Controls help
        help_lines = [
            "Mouse: Click & Drag | Wheel: Zoom | PgUp/PgDn: Change Max Height",
            "Shift+/-: Level Offset | Space: Reset | G: Grid | I: Info | L: Layers (0-9) | ESC/Q: Quit"
        ]
        
        y = self.screen_height - 55
        for line in help_lines:
            text = self.small_font.render(line, True, WHITE)
            text_rect = text.get_rect(topleft=(10, y))
            bg_rect = text_rect.inflate(8, 4)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
            self.screen.blit(text, text_rect)
            y += 25
        
        # Layer menu
        if self.show_layer_menu:
            menu_x = self.screen_width - 350
            menu_y = 10
            menu_width = 340
            menu_height = min(len(self.map_3d.layer_names) * 25 + 50, self.screen_height - 20)
            
            pygame.draw.rect(self.screen, (0, 0, 0, 200), 
                           pygame.Rect(menu_x, menu_y, menu_width, menu_height))
            pygame.draw.rect(self.screen, WHITE, 
                           pygame.Rect(menu_x, menu_y, menu_width, menu_height), 2)
            
            title = self.font.render(f"Layers (0-9 to toggle) - Level {self.map_3d.get_level_value(self.current_z)}:", True, WHITE)
            self.screen.blit(title, (menu_x + 10, menu_y + 10))
            
            # Scrollable layer list
            y = menu_y + 40
            max_y = menu_y + menu_height - 10
            
            for i in range(self.map_3d.N):
                if y >= max_y:
                    # Show "..." if there are more layers
                    if i < self.map_3d.N - 1:
                        text = self.small_font.render(f"... and {self.map_3d.N - i} more", True, GRAY)
                        self.screen.blit(text, (menu_x + 10, y))
                    break
                
                visible = self.layer_visibility[i]
                color = WHITE if visible else GRAY
                
                # Show layer index, name, and level
                layer_name = self.map_3d.layer_names[i]
                layer_level = self.map_3d.layer_levels[i]
                
                # Truncate long names
                display_name = layer_name if len(layer_name) < 25 else layer_name[:22] + "..."
                
                # Show if layer is at current z level
                at_current = "●" if layer_level == self.map_3d.get_level_value(self.current_z) else " "
                
                key_hint = str(i) if i < 10 else ""
                text = self.small_font.render(f"{at_current} {key_hint} {display_name} (L{layer_level})", True, color)
                self.screen.blit(text, (menu_x + 10, y))
                
                # Checkbox
                checkbox_rect = pygame.Rect(menu_x + menu_width - 30, y + 2, 15, 15)
                pygame.draw.rect(self.screen, color, checkbox_rect, 1)
                if visible:
                    pygame.draw.line(self.screen, color, 
                                   (checkbox_rect.left + 3, checkbox_rect.centery),
                                   (checkbox_rect.centerx, checkbox_rect.bottom - 3), 2)
                    pygame.draw.line(self.screen, color,
                                   (checkbox_rect.centerx, checkbox_rect.bottom - 3),
                                   (checkbox_rect.right - 3, checkbox_rect.top + 3), 2)
                
                y += 25
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(BLACK)
        
        # Draw tiles from 3D structure
        self.draw_tiles()
        
        # Draw grid
        self.draw_grid()
        
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable TMX files in current directory:")
        tmx_files = list(Path('.').glob('*.tmx'))
        if tmx_files:
            for f in tmx_files:
                print(f"  - {f}")
        else:
            print("  (none found)")
        sys.exit(1)
    
    tmx_path = sys.argv[1]
    
    if not Path(tmx_path).exists():
        print(f"Error: File '{tmx_path}' not found")
        sys.exit(1)
    
    try:
        explorer = TMXExplorer(tmx_path)
        explorer.run()
    except Exception as e:
        print(f"Error loading TMX file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()