"""
TMX Map Explorer - Main Application (GLFW Version)
"""

import glfw
from OpenGL.GL import *
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

from tmx_manager import TiledMap
from .camera import Camera
from .renderer.opengl_renderer import OpenGLRenderer
from .map.structure import Map3DStructure
from .map.tileset_renderer import TilesetRenderer
from .entities import EntityManager, Character
from .gamepad import GamepadManager

DARK_GRAY = (64, 64, 64)


class TMXExplorer:
    """Main TMX map explorer application - GLFW version"""

    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
        self.screen_width = 1280
        self.screen_height = 720
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
        # Request OpenGL 3.3 Core
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.RESIZABLE, GL_TRUE)
        
        # Create window
        self.window = glfw.create_window(
            self.screen_width, self.screen_height,
            f"TMX Explorer - {self.source_path.name}",
            None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # VSync
        
        # Set callbacks
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)
        
        # Input state
        self.pressed_keys = set()
        self.mouse_pos = (0, 0)
        self.mouse_buttons = set()
        
        # Gamepad support
        self.gamepad = GamepadManager("gamecontrollerdb.txt")

        # Initialize renderer
        self.renderer = OpenGLRenderer(self.screen_width, self.screen_height)

        # Load map
        print(f"\nLoading TMX: {source_path}")
        tmx_map = TiledMap.load(source_path)

        # Initialize components
        self.tileset_renderer = TilesetRenderer(tmx_map, source_path, self.renderer)
        self.map_3d = Map3DStructure(tmx_map)
        
        # Entity manager con max_z del mapa
        self.entity_manager = EntityManager(
            tile_width=self.map_3d.tile_width,
            tile_height=self.map_3d.tile_height,
            collision_map=self.map_3d.collision,
            max_z=self.map_3d.H
        )
        
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
        self.show_collision_debug = False
        self.layer_visibility = [True] * self.map_3d.N

        # Mouse panning
        self.panning = False
        self.pan_start_pos = (0, 0)
        self.pan_start_camera = (0.0, 0.0)

        self.running = True

        # Profiling / FPS
        self.frame_times = {k: [] for k in ['collect', 'render', 'grid', 'ui', 'total']}
        self.last_time = time.perf_counter()
        self.fps_samples = []
        self.current_fps = 0.0
        self.frame_count = 0

        print("\n=== Ready! ===")
        print("WASD/Arrows: Move | Q/E: Player height down/up")
        print("Mouse drag: Pan | Wheel: Zoom | PgUp/PgDn: View height")
        print("G: Grid | I: Info | P: Profiling | C: Collision debug | ESC: Quit")

    def add_character(self, spritesheet_path: str, x: float = 0.0, y: float = 0.0,
                      z: float = 0, speed: float = 100.0, is_player: bool = False,
                      collision_width: float = None, collision_depth: float = None,
                      collision_height: float = None) -> Character:
        """Add a character to the scene"""
        return self.entity_manager.create_character(
            spritesheet_path, x, y, z, speed, is_player=is_player,
            collision_width=collision_width,
            collision_depth=collision_depth,
            collision_height=collision_height
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

    # === GLFW Callbacks ===
    
    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        if action == glfw.PRESS:
            self.pressed_keys.add(key)
            self._handle_key_action(key, mods)
        elif action == glfw.RELEASE:
            self.pressed_keys.discard(key)
    
    def _handle_key_action(self, key, mods):
        """Handle single-press key actions"""
        if key == glfw.KEY_ESCAPE:
            self.running = False
        elif key == glfw.KEY_SPACE:
            self.camera.reset(
                self.map_3d.map_width, self.map_3d.map_height,
                self.map_3d.tile_width, self.map_3d.tile_height
            )
        elif key == glfw.KEY_G:
            self.show_grid = not self.show_grid
        elif key == glfw.KEY_I:
            self.show_info = not self.show_info
        elif key == glfw.KEY_P:
            self.show_profiling = not self.show_profiling
            print(f"Profiling: {'ON' if self.show_profiling else 'OFF'}")
        elif key == glfw.KEY_C:
            self.show_collision_debug = not self.show_collision_debug
            print(f"Collision debug: {'ON' if self.show_collision_debug else 'OFF'}")
        elif key == glfw.KEY_PAGE_UP:
            self._change_height(1)
        elif key == glfw.KEY_PAGE_DOWN:
            self._change_height(-1)
        elif key == glfw.KEY_EQUAL:
            if mods & glfw.MOD_SHIFT:
                self.level_height_offset += 16
                self._adjust_camera_for_all_levels()
            else:
                self.camera.zoom_by(1.2)
        elif key == glfw.KEY_MINUS:
            if mods & glfw.MOD_SHIFT:
                self.level_height_offset = max(0, self.level_height_offset - 16)
                self._adjust_camera_for_all_levels()
            else:
                self.camera.zoom_by(1/1.2)

    def _mouse_button_callback(self, window, button, action, mods):
        """Handle mouse buttons"""
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.panning = True
                self.pan_start_pos = self.mouse_pos
                self.pan_start_camera = (self.camera.x, self.camera.y)
            elif action == glfw.RELEASE:
                self.panning = False

    def _cursor_pos_callback(self, window, xpos, ypos):
        """Handle mouse movement"""
        self.mouse_pos = (xpos, ypos)
        
        if self.panning:
            dx = xpos - self.pan_start_pos[0]
            dy = ypos - self.pan_start_pos[1]
            self.camera.x = self.pan_start_camera[0] - dx / self.camera.zoom
            self.camera.y = self.pan_start_camera[1] - dy / self.camera.zoom

    def _scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll"""
        factor = 1.1 if yoffset > 0 else 1/1.1
        self.camera.zoom_by(factor)

    def _resize_callback(self, window, width, height):
        """Handle window resize"""
        self.screen_width = width
        self.screen_height = height
        self.renderer.resize(width, height)
        self.camera.width = width
        self.camera.height = height

    def _change_height(self, delta: int):
        old_z = self.current_z
        self.current_z = max(0, min(self.map_3d.H - 1, self.current_z + delta))
        if old_z != self.current_z:
            self._adjust_camera_for_all_levels()

    def _update_player_movement(self):
        """Update player based on keyboard and gamepad input"""
        player = self.entity_manager.player
        if player is None:
            return
        
        dx, dy, dz = 0.0, 0.0, 0.0
        
        # Keyboard input
        if glfw.KEY_W in self.pressed_keys or glfw.KEY_UP in self.pressed_keys:
            dy = -1
        if glfw.KEY_S in self.pressed_keys or glfw.KEY_DOWN in self.pressed_keys:
            dy = 1
        if glfw.KEY_A in self.pressed_keys or glfw.KEY_LEFT in self.pressed_keys:
            dx = -1
        if glfw.KEY_D in self.pressed_keys or glfw.KEY_RIGHT in self.pressed_keys:
            dx = 1
        
        if glfw.KEY_Q in self.pressed_keys:
            dz = -1
        if glfw.KEY_E in self.pressed_keys:
            dz = 1
        
        # Gamepad input
        self.gamepad.update()
        if self.gamepad.is_connected():
            gp_dx, gp_dy = self.gamepad.get_movement()
            gp_dz = self.gamepad.get_height_change()
            
            if abs(gp_dx) > 0.1 or abs(gp_dy) > 0.1:
                dx, dy = gp_dx, gp_dy
            if abs(gp_dz) > 0.1:
                dz = gp_dz
            
            if self.gamepad.button_just_pressed('start'):
                self.show_info = not self.show_info
            if self.gamepad.button_just_pressed('back'):
                self.show_grid = not self.show_grid
        
        # Camera movement with shift
        if glfw.KEY_LEFT_SHIFT in self.pressed_keys or glfw.KEY_RIGHT_SHIFT in self.pressed_keys:
            camera_speed = 500
            dt = 1/60
            self.camera.x += dx * camera_speed * dt / self.camera.zoom
            self.camera.y += dy * camera_speed * dt / self.camera.zoom
            player.move(0, 0, 0)
        else:
            player.move(dx, dy, dz)

    def collect_visible_tiles_ordered(self) -> Dict:
        """Collect visible tiles with culling and depth ordering"""
        tile_batches = defaultdict(list)
        
        tile_w = self.map_3d.tile_width
        tile_h = self.map_3d.tile_height
        margin_tiles = 3

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

                        tile_height = surface[1]
                        world_x = x * tile_w
                        world_y = (y + 1) * tile_h - tile_height - level_y_offset

                        depth = base_offset + y + z + (n * 0.1)
                        tile_batches[texture].append((
                            world_x, world_y,
                            surface[0], surface[1],
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

    def draw_collision_debug(self):
        """Draw collision tiles as colored overlay with height offset"""
        if not self.show_collision_debug:
            return
        
        tile_w = self.map_3d.tile_width
        tile_h = self.map_3d.tile_height
        collision = self.map_3d.collision
        
        start_x = max(0, int(self.camera.x / tile_w) - 1)
        start_y = max(0, int(self.camera.y / tile_h) - 1)
        end_x = min(collision.W, int((self.camera.x + self.screen_width / self.camera.zoom) / tile_w) + 2)
        end_y = min(collision.D, int((self.camera.y + self.screen_height / self.camera.zoom) / tile_h) + 2)
        
        player_z = int(self.entity_manager.player.z) if self.entity_manager.player else 0
        
        # Dibujar cada nivel Z por separado con su offset
        for z in range(self.map_3d.H):
            level_value = self.map_3d.get_level_value(z)
            level_y_offset = level_value * self.level_height_offset
            
            # Transparencia según distancia al nivel del jugador
            z_distance = abs(z - player_z)
            if z_distance == 0:
                alpha = 120
            elif z_distance == 1:
                alpha = 60
            else:
                alpha = 30
            
            solid_rects = []
            empty_rects = []
            
            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    screen_x = (x * tile_w - self.camera.x) * self.camera.zoom
                    screen_y = (y * tile_h - level_y_offset - self.camera.y) * self.camera.zoom
                    screen_w = tile_w * self.camera.zoom
                    screen_h = tile_h * self.camera.zoom
                    
                    if collision.is_solid(x, y, z):
                        solid_rects.append((screen_x, screen_y, screen_w, screen_h))
                    else:
                        empty_rects.append((screen_x, screen_y, screen_w, screen_h))
            
            if empty_rects:
                self.renderer.draw_rects(empty_rects, (0, 200, 0, alpha))
            
            if solid_rects:
                self.renderer.draw_rects(solid_rects, (200, 0, 0, alpha))

    def draw_ui(self):
        """Draw UI info"""
        if not self.show_info:
            return
        
        level_value = self.map_3d.get_level_value(self.current_z)
        lines = [
            f"FPS: {int(self.current_fps)} | Zoom: {self.camera.zoom:.2f}x",
            f"View Height: {self.current_z}/{self.map_3d.H-1} (level={level_value})",
            f"Textures: {len(self.renderer.texture_cache)}",
        ]
        
        # Mostrar info del jugador si existe
        if self.entity_manager.player:
            p = self.entity_manager.player
            lines.append(f"Player: ({p.x:.0f}, {p.y:.0f}) Z={p.z:.2f}")
        
        if self.show_profiling:
            for key in ['collect', 'render', 'grid', 'ui']:
                if self.frame_times[key]:
                    avg = sum(self.frame_times[key][-60:]) / min(60, len(self.frame_times[key]))
                    lines.append(f"{key}: {avg*1000:.1f}ms")
        
        self.renderer.draw_text_lines(lines, 10, 10)

    def _draw_characters(self):
        """Draw all characters with proper depth sorting"""
        char_data = self.entity_manager.collect_render_data(
            self.renderer, self.level_height_offset
        )
        
        if not char_data:
            return
        
        char_batches = defaultdict(list)
        for texture, x, y, w, h, depth in char_data:
            char_batches[texture].append((x, y, w, h, depth))
        
        self.renderer.draw_batched_tiles(char_batches)

    def draw(self):
        """Main draw method"""
        frame_start = time.perf_counter()

        self.renderer.begin_frame()
        self.renderer.set_camera(self.camera.x, self.camera.y, self.camera.zoom)

        t1 = time.perf_counter()
        tile_batches = self.collect_visible_tiles_ordered()
        t2 = time.perf_counter()

        self.renderer.draw_batched_tiles(tile_batches)
        self._draw_characters()
        t3 = time.perf_counter()

        self.draw_collision_debug()
        self.draw_grid()
        t4 = time.perf_counter()

        self.draw_ui()
        t5 = time.perf_counter()

        if self.show_profiling:
            self.frame_times['collect'].append(t2 - t1)
            self.frame_times['render'].append(t3 - t2)
            self.frame_times['grid'].append(t4 - t3)
            self.frame_times['ui'].append(t5 - t4)
            self.frame_times['total'].append(t5 - frame_start)

            for key in self.frame_times:
                if len(self.frame_times[key]) > 120:
                    self.frame_times[key].pop(0)

    def run(self):
        """Main application loop"""
        while self.running and not glfw.window_should_close(self.window):
            current_time = time.perf_counter()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            self.frame_count += 1
            self.fps_samples.append(1.0 / dt if dt > 0 else 0)
            if len(self.fps_samples) > 60:
                self.fps_samples.pop(0)
            self.current_fps = sum(self.fps_samples) / len(self.fps_samples)
            
            glfw.poll_events()
            self._update_player_movement()
            self.entity_manager.update(dt)
            self.draw()
            glfw.swap_buffers(self.window)
        
        glfw.terminate()
