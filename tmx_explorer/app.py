"""
TMX Map Explorer - Main Application
"""

import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import pygame

from tmx_manager import TiledMap
from .camera import Camera
from .renderer.opengl_renderer import OpenGLRenderer
from .map.structure import Map3DStructure
from .map.tileset_renderer import TilesetRenderer
from .entities import EntityManager, Character

DARK_GRAY = (64, 64, 64)


class TMXExplorer:
    """Main TMX map explorer application"""

    def __init__(self, source_path: str):
        pygame.init()
        
        self.source_path = Path(source_path)
        self.screen_width = 1280
        self.screen_height = 720

        # Initialize renderer
        self.renderer = OpenGLRenderer(self.screen_width, self.screen_height)
        pygame.display.set_caption(f"TMX Explorer - {self.source_path.name}")

        # Load map
        print(f"\nLoading TMX: {source_path}")
        tmx_map = TiledMap.load(source_path)

        # Initialize components
        self.tileset_renderer = TilesetRenderer(tmx_map, source_path, self.renderer)
        self.map_3d = Map3DStructure(tmx_map)
        self.entity_manager = EntityManager()
        
        print(f"\nTotal textures cached: {len(self.renderer.texture_cache)}")

        # Initialize camera
        self.camera = Camera(self.screen_width, self.screen_height)
        self.camera.reset(
            self.map_3d.map_width, self.map_3d.map_height,
            self.map_3d.tile_width, self.map_3d.tile_height
        )

        # View state
        self.current_z = self.map_3d.H - 1
        self.level_height_offset = 128
        self._adjust_camera_for_all_levels()

        # UI state
        self.show_grid = False
        self.show_info = True
        self.show_profiling = False
        self.layer_visibility = [True] * self.map_3d.N

        # Mouse panning
        self.panning = False
        self.pan_start_pos = (0, 0)
        self.pan_start_camera = (0.0, 0.0)

        self.clock = pygame.time.Clock()
        self.running = True

        # Profiling
        self.frame_times = {k: [] for k in ['collect', 'render', 'grid', 'ui', 'total']}
        
        # Delta time tracking
        self.last_time = time.perf_counter()

        print("\n=== Ready! ===")
        print("Press 'P' to toggle profiling")

    def add_character(
        self,
        spritesheet_path: str,
        x: float = 0.0,
        y: float = 0.0,
        z: int = 0,
        speed: float = 100.0,
        is_player: bool = False
    ) -> Character:
        """
        Add a character to the scene.
        
        Args:
            spritesheet_path: Path to 4x4 spritesheet
            x, y: World position (bottom-center anchor)
            z: Height level
            speed: Movement speed in pixels/second
            is_player: If True, this character responds to player input
        
        Returns:
            The created Character instance
        """
        return self.entity_manager.create_character(
            spritesheet_path, x, y, z, speed,
            is_player=is_player
        )

    def _adjust_camera_for_all_levels(self):
        """Adjust camera to show all visible levels"""
        total_height_offset = self.current_z * self.level_height_offset
        map_pixel_width = self.map_3d.map_width * self.map_3d.tile_width
        map_pixel_height = self.map_3d.map_height * self.map_3d.tile_height
        total_pixel_height = map_pixel_height + total_height_offset

        zoom_x = self.screen_width / map_pixel_width
        zoom_y = self.screen_height / total_pixel_height
        target_zoom = min(zoom_x, zoom_y, 1.0) * 0.85

        center_x = map_pixel_width / 2
        center_y = (map_pixel_height - total_height_offset) / 2

        self.camera.x = center_x - self.screen_width / (2 * target_zoom)
        self.camera.y = center_y - self.screen_height / (2 * target_zoom)
        self.camera.zoom = target_zoom

    def handle_events(self):
        """Process all input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self._handle_resize(event.w, event.h)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_up(event)
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event)
            elif event.type == pygame.KEYDOWN:
                self._handle_key_down(event)
            elif event.type == pygame.KEYUP:
                self._handle_key_up(event)
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_mouse_wheel(event)

    def _handle_resize(self, width: int, height: int):
        self.screen_width = width
        self.screen_height = height
        self.renderer.resize(width, height)
        self.camera.width = width
        self.camera.height = height

    def _handle_mouse_down(self, event):
        if event.button == 1:
            self.panning = True
            self.pan_start_pos = event.pos
            self.pan_start_camera = (self.camera.x, self.camera.y)

    def _handle_mouse_up(self, event):
        if event.button == 1:
            self.panning = False

    def _handle_mouse_motion(self, event):
        if self.panning:
            dx = event.pos[0] - self.pan_start_pos[0]
            dy = event.pos[1] - self.pan_start_pos[1]
            self.camera.x = self.pan_start_camera[0] - dx / self.camera.zoom
            self.camera.y = self.pan_start_camera[1] - dy / self.camera.zoom

    def _handle_key_down(self, event):
        """Key pressed"""
        if event.key in (pygame.K_ESCAPE, pygame.K_q):
            self.running = False
        elif event.key == pygame.K_SPACE:
            self.camera.reset(
                self.map_3d.map_width, self.map_3d.map_height,
                self.map_3d.tile_width, self.map_3d.tile_height
            )
        elif event.key == pygame.K_g:
            self.show_grid = not self.show_grid
        elif event.key == pygame.K_i:
            self.show_info = not self.show_info
        elif event.key == pygame.K_p:
            self.show_profiling = not self.show_profiling
            print(f"Profiling: {'ON' if self.show_profiling else 'OFF'}")
        elif event.key == pygame.K_PAGEUP:
            self._change_height(1)
        elif event.key == pygame.K_PAGEDOWN:
            self._change_height(-1)
        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
            self._handle_zoom_key(event, 1.2)
        elif event.key == pygame.K_MINUS:
            self._handle_zoom_key(event, 1/1.2)
        # Movement
        elif event.key in (pygame.K_UP, pygame.K_w):
            self._update_player_velocity(dy=-1)
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self._update_player_velocity(dy=1)
        elif event.key in (pygame.K_LEFT, pygame.K_a):
            self._update_player_velocity(dx=-1)
        elif event.key in (pygame.K_RIGHT, pygame.K_d):
            self._update_player_velocity(dx=1)

    def _handle_key_up(self, event):
        """Key released"""
        if event.key in (pygame.K_UP, pygame.K_w):
            self._update_player_velocity(dy=0, reset_y=True)
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self._update_player_velocity(dy=0, reset_y=True)
        elif event.key in (pygame.K_LEFT, pygame.K_a):
            self._update_player_velocity(dx=0, reset_x=True)
        elif event.key in (pygame.K_RIGHT, pygame.K_d):
            self._update_player_velocity(dx=0, reset_x=True)

    def _update_player_velocity(self, dx=None, dy=None, reset_x=False, reset_y=False):
        """Update player velocity based on key events"""
        player = self.entity_manager.player
        if player is None:
            return
        
        # Get current direction from velocity
        current_dx = 1 if player.velocity_x > 0 else (-1 if player.velocity_x < 0 else 0)
        current_dy = 1 if player.velocity_y > 0 else (-1 if player.velocity_y < 0 else 0)
        
        if reset_x:
            current_dx = 0
        elif dx is not None:
            current_dx = dx
            
        if reset_y:
            current_dy = 0
        elif dy is not None:
            current_dy = dy
        
        player.move(current_dx, current_dy)

    def _change_height(self, delta: int):
        old_z = self.current_z
        self.current_z = max(0, min(self.map_3d.H - 1, self.current_z + delta))
        if old_z != self.current_z:
            self._adjust_camera_for_all_levels()

    def _handle_zoom_key(self, event, factor: float):
        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
            if factor > 1:
                self.level_height_offset += 16
            else:
                self.level_height_offset = max(0, self.level_height_offset - 16)
            self._adjust_camera_for_all_levels()
        else:
            self.camera.zoom_by(factor)

    def _handle_mouse_wheel(self, event):
        factor = 1.1 if event.y > 0 else 1/1.1
        self.camera.zoom_by(factor)

    def collect_visible_tiles_ordered(self) -> Dict:
        """Collect visible tiles with culling and depth ordering"""
        tile_batches = defaultdict(list)
        
        tile_w = self.map_3d.tile_width
        tile_h = self.map_3d.tile_height
        margin_tiles = 3

        # Calculate visible bounds with margin
        screen_left = self.camera.x - (margin_tiles * tile_w)
        screen_right = self.camera.x + (self.screen_width / self.camera.zoom) + (margin_tiles * tile_w)
        screen_top = self.camera.y - (margin_tiles * tile_h)
        screen_bottom = self.camera.y + (self.screen_height / self.camera.zoom) + (margin_tiles * tile_h)

        start_x = max(0, int(screen_left / tile_w))
        end_x = min(self.map_3d.W, int(screen_right / tile_w) + 1)
        start_y = max(0, int(screen_top / tile_h))
        end_y = min(self.map_3d.D, int(screen_bottom / tile_h) + 1)

        if start_x >= end_x or start_y >= end_y:
            return tile_batches

        base_offset = -1000.0

        # Collect tiles with depth: offset + y + z + n * 0.1
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                for z in range(0, self.current_z + 1):
                    level_y_offset = z * self.level_height_offset

                    for n in range(self.map_3d.N):
                        if not self.layer_visibility[n]:
                            continue
                        if self.map_3d.layer_levels[n] != self.map_3d.get_level_value(z):
                            continue

                        tile_id = self.map_3d.mapa[z, y, x, n]
                        if tile_id == 0:
                            continue

                        texture = self.tileset_renderer.get_tile_texture(tile_id)
                        surface = self.tileset_renderer.get_tile_surface(tile_id)
                        if not texture or not surface:
                            continue

                        tile_height = surface.get_height()
                        world_x = x * tile_w
                        world_y = (y + 1) * tile_h - tile_height - level_y_offset

                        depth = base_offset + y + z + (n * 0.1)
                        tile_batches[texture].append((
                            world_x, world_y,
                            surface.get_width(), surface.get_height(),
                            depth
                        ))

        return tile_batches

    def draw_grid(self):
        """Draw tile grid overlay"""
        if not self.show_grid:
            return

        tile_w = self.map_3d.tile_width
        tile_h = self.map_3d.tile_height
        
        start_x = max(0, int(self.camera.x / tile_w))
        start_y = max(0, int(self.camera.y / tile_h))
        end_x = min(self.map_3d.map_width, 
                   int((self.camera.x + self.screen_width / self.camera.zoom) / tile_w) + 2)
        end_y = min(self.map_3d.map_height,
                   int((self.camera.y + self.screen_height / self.camera.zoom) / tile_h) + 2)

        lines = []
        for x in range(start_x, end_x + 1):
            screen_x = (x * tile_w - self.camera.x) * self.camera.zoom
            lines.append((screen_x, 0, screen_x, self.screen_height))

        for y in range(start_y, end_y + 1):
            screen_y = (y * tile_h - self.camera.y) * self.camera.zoom
            lines.append((0, screen_y, self.screen_width, screen_y))

        self.renderer.draw_lines(lines, DARK_GRAY)

    def draw_ui(self):
        """Draw UI panels"""
        if self.show_info:
            level_value = self.map_3d.get_level_value(self.current_z)
            info_lines = [
                f"Zoom: {self.camera.zoom:.2f}x | FPS: {int(self.clock.get_fps())}",
                f"Height: {self.current_z}/{self.map_3d.H-1} (level={level_value})",
                f"Textures: {len(self.renderer.texture_cache)}",
                f"Z-Buffer: ON | Depth = offset + y + z + n*0.1",
            ]

            if self.show_profiling:
                for key in ['collect', 'render', 'grid', 'ui']:
                    if self.frame_times[key]:
                        avg = sum(self.frame_times[key][-60:]) / min(60, len(self.frame_times[key]))
                        info_lines.append(f"{key}: {avg*1000:.1f}ms")

            self.renderer.draw_ui_panel(info_lines, 10, 10)

        help_lines = ["Mouse: Drag | Wheel: Zoom | PgUp/PgDn: Height | G: Grid | I: Info | P: Profile | ESC: Quit"]
        self.renderer.draw_ui_panel(help_lines, 10, self.screen_height - 35)

    def draw(self):
        """Main draw method with profiling"""
        frame_start = time.perf_counter()

        self.renderer.begin_frame()
        self.renderer.set_camera(self.camera.x, self.camera.y, self.camera.zoom)

        t1 = time.perf_counter()
        tile_batches = self.collect_visible_tiles_ordered()
        t2 = time.perf_counter()

        self.renderer.draw_batched_tiles(tile_batches)
        
        # Draw characters
        self._draw_characters()
        
        t3 = time.perf_counter()

        self.draw_grid()
        t4 = time.perf_counter()

        self.draw_ui()
        t5 = time.perf_counter()

        self.renderer.end_frame()

        if self.show_profiling:
            self.frame_times['collect'].append(t2 - t1)
            self.frame_times['render'].append(t3 - t2)
            self.frame_times['grid'].append(t4 - t3)
            self.frame_times['ui'].append(t5 - t4)
            self.frame_times['total'].append(t5 - frame_start)

            for key in self.frame_times:
                if len(self.frame_times[key]) > 120:
                    self.frame_times[key].pop(0)

    def _draw_characters(self):
        """Draw all characters with proper depth sorting"""
        char_data = self.entity_manager.collect_render_data(
            self.renderer,
            self.level_height_offset
        )
        
        if not char_data:
            return
        
        # Group by texture and draw
        char_batches = defaultdict(list)
        for texture, x, y, w, h, depth in char_data:
            char_batches[texture].append((x, y, w, h, depth))
        
        self.renderer.draw_batched_tiles(char_batches)

    def run(self):
        """Main application loop"""
        while self.running:
            # Calculate delta time
            current_time = time.perf_counter()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # 1. Process events
            self.handle_events()
            
            # 2. Update entities
            self.entity_manager.update(dt)
            
            # 3. Render
            self.draw()
            
            self.clock.tick(60)
        
        pygame.quit()
