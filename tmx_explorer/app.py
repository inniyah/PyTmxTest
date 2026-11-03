"""Aplicación principal: ventana GLFW, entrada, bucle y dibujo de la escena."""

import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import glfw
import numpy as np
from OpenGL.GL import GL_TRUE

from tmx_manager import TiledMap

from . import config
from .camera import Camera
from .entities import Character, EntityManager
from .gamepad import GamepadManager
from .map.collision import FLAG_SOLID
from .map.structure import Map3DStructure
from .map.tileset_renderer import TilesetRenderer
from .map.visible_tiles import collect_visible_tiles
from .renderer.opengl_renderer import OpenGLRenderer

log = logging.getLogger(__name__)

GRID_COLOR = (64, 64, 64)
COLLISION_SOLID_COLOR = (200, 0, 0)
COLLISION_EMPTY_COLOR = (0, 200, 0)
COLLISION_ALPHA_BY_DISTANCE = (120, 60, 30)   # nivel del jugador, ±1, resto

MOVE_KEYS = {
    "up": (glfw.KEY_W, glfw.KEY_UP),
    "down": (glfw.KEY_S, glfw.KEY_DOWN),
    "left": (glfw.KEY_A, glfw.KEY_LEFT),
    "right": (glfw.KEY_D, glfw.KEY_RIGHT),
}
HEIGHT_DOWN_KEY = glfw.KEY_Q
HEIGHT_UP_KEY = glfw.KEY_E
SHIFT_KEYS = (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)

PROFILE_SECTIONS = ("collect", "render", "grid", "ui", "total")
PROFILE_HISTORY = 120
FPS_HISTORY = 60

CONTROLS_HELP = (
    "WASD/Flechas: mover (cámara si no hay jugador) | Shift+WASD: cámara | Q/E: altura jugador\n"
    "Arrastrar: pan | Rueda o +/-: zoom | Shift +/-: separación de niveles | PgUp/PgDn: nivel visible\n"
    "G: grid | I: info | P: profiling | C: debug colisión | Espacio: reset vista | ESC: salir"
)


class TMXExplorer:
    """Visor de mapas TMX con personajes."""

    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
        self.screen_width = config.WINDOW_WIDTH
        self.screen_height = config.WINDOW_HEIGHT

        self.window = self._create_window()

        # Entrada
        self.pressed_keys = set()
        self.mouse_pos = (0.0, 0.0)
        self.panning = False
        self.pan_start_pos = (0.0, 0.0)
        self.pan_start_camera = (0.0, 0.0)
        self.gamepad = GamepadManager()

        # Mundo
        self.renderer = OpenGLRenderer(self.screen_width, self.screen_height)
        log.info("Cargando TMX: %s", source_path)
        tmx_map = TiledMap.load(source_path)
        self.tileset_renderer = TilesetRenderer(tmx_map, source_path)
        self.map_3d = Map3DStructure(tmx_map)
        self.entity_manager = EntityManager(
            tile_width=self.map_3d.tile_width,
            tile_height=self.map_3d.tile_height,
            collision_map=self.map_3d.collision,
            max_z=self.map_3d.H,
        )

        # Vista
        self.camera = Camera(self.screen_width, self.screen_height)
        self._reset_camera()
        self.current_z = self.map_3d.H - 1          # nivel más alto que se dibuja
        self.level_height_offset = config.LEVEL_HEIGHT_OFFSET
        self._fit_camera_to_levels()

        # UI
        self.show_grid = False
        self.show_info = True
        self.show_profiling = False
        self.show_collision_debug = False
        self.layer_visibility = [True] * self.map_3d.N

        # Tiempo / profiling
        self.running = True
        self.last_time = time.perf_counter()
        self.frame_count = 0
        self.fps_samples = deque(maxlen=FPS_HISTORY)
        self.current_fps = 0.0
        self.frame_times = {k: deque(maxlen=PROFILE_HISTORY) for k in PROFILE_SECTIONS}

        print("\n" + CONTROLS_HELP + "\n")

    # --- Ventana ---

    def _create_window(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.RESIZABLE, GL_TRUE)

        window = glfw.create_window(self.screen_width, self.screen_height,
                                    f"{config.WINDOW_TITLE} - {self.source_path.name}",
                                    None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")

        glfw.make_context_current(window)
        glfw.swap_interval(1 if config.VSYNC else 0)
        glfw.set_key_callback(window, self._key_callback)
        glfw.set_mouse_button_callback(window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(window, self._cursor_pos_callback)
        glfw.set_scroll_callback(window, self._scroll_callback)
        glfw.set_framebuffer_size_callback(window, self._resize_callback)
        return window

    # --- API pública ---

    def add_character(self, spritesheet_path: str, x: float = 0.0, y: float = 0.0,
                      z: float = 0, speed: float = 100.0, is_player: bool = False,
                      collision_width: float = None, collision_depth: float = None,
                      collision_height: float = None) -> Character:
        return self.entity_manager.create_character(
            spritesheet_path, x, y, z, speed, is_player=is_player,
            collision_width=collision_width, collision_depth=collision_depth,
            collision_height=collision_height)

    def level_screen_offset(self, z: float) -> float:
        """Cuánto sube en pantalla (px de mundo) algo situado en el índice de nivel z.

        Tiles, personajes y overlays deben usar SIEMPRE esto para quedar alineados.
        """
        return z * self.level_height_offset

    # --- Cámara ---

    @property
    def map_pixel_size(self) -> Tuple[int, int]:
        return (self.map_3d.map_width * self.map_3d.tile_width,
                self.map_3d.map_height * self.map_3d.tile_height)

    def _reset_camera(self):
        self.camera.reset(self.map_3d.map_width, self.map_3d.map_height,
                          self.map_3d.tile_width, self.map_3d.tile_height)

    def _fit_camera_to_levels(self):
        """Encuadra el mapa completo incluyendo el desplazamiento de los niveles visibles."""
        map_w, map_h = self.map_pixel_size
        offset = self.level_screen_offset(self.current_z)
        zoom = min(self.screen_width / map_w, self.screen_height / (map_h + offset), 1.0)
        # Asignación directa: puede quedar por debajo de Camera.MIN_ZOOM en mapas grandes
        self.camera.zoom = zoom * config.INITIAL_FIT_FACTOR
        self.camera.center_on(map_w / 2, (map_h - offset) / 2)

    def _change_view_level(self, delta: int):
        new_z = max(0, min(self.map_3d.H - 1, self.current_z + delta))
        if new_z != self.current_z:
            self.current_z = new_z
            self._fit_camera_to_levels()

    def _change_level_spacing(self, delta: int):
        self.level_height_offset = max(0, self.level_height_offset + delta)
        self._fit_camera_to_levels()

    # --- Callbacks GLFW ---

    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.pressed_keys.add(key)
            self._handle_key_press(key, mods)
        elif action == glfw.RELEASE:
            self.pressed_keys.discard(key)

    def _handle_key_press(self, key, mods):
        shift = bool(mods & glfw.MOD_SHIFT)
        if key == glfw.KEY_ESCAPE:
            self.running = False
        elif key == glfw.KEY_SPACE:
            self._reset_camera()
        elif key == glfw.KEY_G:
            self.show_grid = not self.show_grid
        elif key == glfw.KEY_I:
            self.show_info = not self.show_info
        elif key == glfw.KEY_P:
            self.show_profiling = not self.show_profiling
        elif key == glfw.KEY_C:
            self.show_collision_debug = not self.show_collision_debug
        elif key == glfw.KEY_PAGE_UP:
            self._change_view_level(1)
        elif key == glfw.KEY_PAGE_DOWN:
            self._change_view_level(-1)
        elif key == glfw.KEY_EQUAL:
            if shift:
                self._change_level_spacing(config.LEVEL_HEIGHT_STEP)
            else:
                self.camera.zoom_by(config.KEY_ZOOM_FACTOR)
        elif key == glfw.KEY_MINUS:
            if shift:
                self._change_level_spacing(-config.LEVEL_HEIGHT_STEP)
            else:
                self.camera.zoom_by(1 / config.KEY_ZOOM_FACTOR)

    def _mouse_button_callback(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if action == glfw.PRESS:
            self.panning = True
            self.pan_start_pos = self.mouse_pos
            self.pan_start_camera = (self.camera.x, self.camera.y)
        elif action == glfw.RELEASE:
            self.panning = False

    def _cursor_pos_callback(self, window, xpos, ypos):
        self.mouse_pos = (xpos, ypos)
        if self.panning:
            self.camera.x = self.pan_start_camera[0] - (xpos - self.pan_start_pos[0]) / self.camera.zoom
            self.camera.y = self.pan_start_camera[1] - (ypos - self.pan_start_pos[1]) / self.camera.zoom

    def _scroll_callback(self, window, xoffset, yoffset):
        self.camera.zoom_by(config.WHEEL_ZOOM_FACTOR if yoffset > 0 else 1 / config.WHEEL_ZOOM_FACTOR)

    def _resize_callback(self, window, width, height):
        if width == 0 or height == 0:      # ventana minimizada
            return
        self.screen_width, self.screen_height = width, height
        self.renderer.resize(width, height)
        self.camera.resize(width, height)

    # --- Entrada por frame ---

    def _key_down(self, *keys) -> bool:
        return any(k in self.pressed_keys for k in keys)

    def _read_movement_input(self) -> Tuple[float, float, float]:
        """Dirección (dx, dy, dz) en -1..1 combinando teclado y mando."""
        dx = dy = dz = 0.0
        if self._key_down(*MOVE_KEYS["up"]):
            dy = -1
        if self._key_down(*MOVE_KEYS["down"]):
            dy = 1
        if self._key_down(*MOVE_KEYS["left"]):
            dx = -1
        if self._key_down(*MOVE_KEYS["right"]):
            dx = 1
        if self._key_down(HEIGHT_DOWN_KEY):
            dz = -1
        if self._key_down(HEIGHT_UP_KEY):
            dz = 1

        self.gamepad.update()
        if self.gamepad.is_connected():
            threshold = config.GAMEPAD_ACTIVE_THRESHOLD
            gp_dx, gp_dy = self.gamepad.get_movement()
            gp_dz = self.gamepad.get_height_change()
            if abs(gp_dx) > threshold or abs(gp_dy) > threshold:
                dx, dy = gp_dx, gp_dy
            if abs(gp_dz) > threshold:
                dz = gp_dz
            if self.gamepad.button_just_pressed("start"):
                self.show_info = not self.show_info
            if self.gamepad.button_just_pressed("back"):
                self.show_grid = not self.show_grid
        return dx, dy, dz

    def _update_input(self, dt: float):
        dx, dy, dz = self._read_movement_input()
        player = self.entity_manager.player

        if player is None or self._key_down(*SHIFT_KEYS):
            step = config.CAMERA_KEY_SPEED * dt
            self.camera.move(dx * step, dy * step)
            if player is not None:
                player.stop()
        else:
            player.move(dx, dy, dz)

    # --- Recolección de tiles ---

    def _visible_tile_range(self, margin: int = 0) -> Tuple[int, int, int, int]:
        """(x0, x1, y0, y1) de tiles visibles, con ``margin`` tiles extra, recortado al mapa."""
        tw, th = self.map_3d.tile_width, self.map_3d.tile_height
        left, top, right, bottom = self.camera.get_visible_bounds()
        x0 = max(0, int((left - margin * tw) / tw))
        x1 = min(self.map_3d.W, int((right + margin * tw) / tw) + 1)
        y0 = max(0, int((top - margin * th) / th))
        y1 = min(self.map_3d.D, int((bottom + margin * th) / th) + 1)
        return x0, x1, y0, y1

    def collect_visible_tiles(self) -> Dict[int, "np.ndarray"]:
        """``{página del atlas: vértices}`` de los tiles visibles hasta current_z."""
        x0, x1, y0, y1 = self._visible_tile_range(config.TILE_CULL_MARGIN)
        return collect_visible_tiles(self.map_3d, self.tileset_renderer.lookup,
                                     x0, x1, y0, y1, self.current_z,
                                     self.level_height_offset, self.layer_visibility)

    # --- Dibujo ---

    def _draw_tiles(self, tile_batches: Dict[int, "np.ndarray"]):
        pages = self.tileset_renderer.textures
        for page, vertices in tile_batches.items():
            self.renderer.draw_quads(pages[page], vertices)

    def _draw_characters(self):
        batches = self.entity_manager.collect_render_data(self.level_height_offset)
        for texture, vertices in batches.items():
            self.renderer.draw_quads(texture, vertices)

    def draw_grid(self):
        if not self.show_grid:
            return
        tw, th, zoom = self.map_3d.tile_width, self.map_3d.tile_height, self.camera.zoom
        left, top, right, bottom = self.camera.get_visible_bounds()
        x0, y0 = max(0, int(left / tw)), max(0, int(top / th))
        x1 = min(self.map_3d.map_width, int(right / tw) + 2)
        y1 = min(self.map_3d.map_height, int(bottom / th) + 2)

        lines = []
        for x in range(x0, x1 + 1):
            sx = (x * tw - self.camera.x) * zoom
            lines.append((sx, 0, sx, self.screen_height))
        for y in range(y0, y1 + 1):
            sy = (y * th - self.camera.y) * zoom
            lines.append((0, sy, self.screen_width, sy))
        self.renderer.draw_lines(lines, GRID_COLOR)

    def draw_collision_debug(self):
        """Celdas sólidas (rojo) y libres (verde) de cada nivel, desplazadas como los tiles."""
        if not self.show_collision_debug:
            return
        collision = self.map_3d.collision
        tw, th, zoom = self.map_3d.tile_width, self.map_3d.tile_height, self.camera.zoom
        x0, x1, y0, y1 = self._visible_tile_range(margin=1)

        player = self.entity_manager.player
        player_z = int(player.z) if player else 0
        if x0 >= x1 or y0 >= y1:
            return

        # Rejilla de esquinas en pantalla, común a todos los niveles salvo el desplazamiento
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        sx = (gx * tw - self.camera.x) * zoom
        cell = np.full(gx.shape, tw * zoom), np.full(gx.shape, th * zoom)

        for z in range(self.map_3d.H):
            distance = min(abs(z - player_z), len(COLLISION_ALPHA_BY_DISTANCE) - 1)
            alpha = COLLISION_ALPHA_BY_DISTANCE[distance]
            sy = (gy * th - self.level_screen_offset(z) - self.camera.y) * zoom
            rects = np.stack([sx, sy, *cell], axis=-1)
            solid = (collision.data[z, y0:y1, x0:x1] & FLAG_SOLID) != 0

            self.renderer.draw_rects(rects[~solid], (*COLLISION_EMPTY_COLOR, alpha))
            self.renderer.draw_rects(rects[solid], (*COLLISION_SOLID_COLOR, alpha))

    def draw_ui(self):
        if not self.show_info:
            return
        m = self.map_3d
        lines = [
            f"FPS: {int(self.current_fps)} | Zoom: {self.camera.zoom:.2f}x",
            f"View Height: {self.current_z}/{m.H - 1} (level={m.get_level_value(self.current_z)})",
            f"Draw calls: {self.renderer.draw_calls} | Atlas: "
            f"{len(self.tileset_renderer.textures)} página(s)",
        ]
        player = self.entity_manager.player
        if player:
            lines.append(f"Player: ({player.x:.0f}, {player.y:.0f}) Z={player.z:.2f}")
        if self.show_profiling:
            for key in PROFILE_SECTIONS[:-1]:
                samples = list(self.frame_times[key])[-60:]
                if samples:
                    lines.append(f"{key}: {sum(samples) / len(samples) * 1000:.1f}ms")
        self.renderer.draw_text_lines(lines, 10, 10)

    def draw(self):
        t0 = time.perf_counter()
        self.renderer.begin_frame()
        self.renderer.set_camera(self.camera.x, self.camera.y, self.camera.zoom)

        t1 = time.perf_counter()
        tile_batches = self.collect_visible_tiles()
        t2 = time.perf_counter()
        self._draw_tiles(tile_batches)
        self._draw_characters()
        t3 = time.perf_counter()
        self.draw_collision_debug()
        self.draw_grid()
        t4 = time.perf_counter()
        self.draw_ui()
        t5 = time.perf_counter()

        if self.show_profiling:
            for key, value in zip(PROFILE_SECTIONS, (t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0)):
                self.frame_times[key].append(value)

    # --- Bucle ---

    def _tick_clock(self) -> float:
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        self.frame_count += 1
        if dt > 0:
            self.fps_samples.append(1.0 / dt)
            self.current_fps = sum(self.fps_samples) / len(self.fps_samples)
        return dt

    def run(self):
        try:
            while self.running and not glfw.window_should_close(self.window):
                dt = self._tick_clock()
                glfw.poll_events()
                self._update_input(dt)
                self.entity_manager.update(dt)
                self.draw()
                glfw.swap_buffers(self.window)
        finally:
            glfw.terminate()
