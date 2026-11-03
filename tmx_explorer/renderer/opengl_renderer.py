"""Renderer OpenGL 3.3 core.

Dos espacios de coordenadas:
    mundo     tiles y personajes; la cámara (posición + zoom) va en la matriz de proyección
    pantalla  overlays y HUD, en píxeles con origen arriba a la izquierda
"""

import ctypes
import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_DYNAMIC_DRAW, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER, GL_LESS,
    GL_LINES, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_TRIANGLES, GL_VERSION,
    GL_VERTEX_SHADER, glBindBuffer, glBindVertexArray, glBlendFunc, glBufferData,
    glBufferSubData, glClear, glClearColor, glDepthFunc, glDisable, glDrawArrays,
    glEnable, glEnableVertexAttribArray, glGenBuffers, glGenVertexArrays, glGetString,
    glGetUniformLocation, glUniform1i, glUniformMatrix4fv, glUseProgram,
    glVertexAttribPointer, glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image, ImageDraw, ImageFont

from .. import config
from ..shaders.sources import (
    FRAGMENT_SHADER, SIMPLE_FRAGMENT_SHADER, SIMPLE_VERTEX_SHADER, VERTEX_SHADER,
)
from .sprite_batch import SpriteBatch, quad_vertices
from .texture import Texture

log = logging.getLogger(__name__)

Color = Sequence[int]   # (r, g, b) o (r, g, b, a) en 0..255

FLOAT_SIZE = 4
SIMPLE_FLOATS_PER_VERTEX = 6   # x, y, r, g, b, a
HUD_LINE_HEIGHT = 18
HUD_WIDTH = 350
HUD_PADDING = 5
HUD_BACKGROUND = (0, 0, 0, 180)
HUD_DEPTH = 9999.0


def ortho_matrix(left, right, bottom, top, near, far) -> np.ndarray:
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = 2.0 / (right - left)
    mat[1, 1] = 2.0 / (top - bottom)
    mat[2, 2] = -2.0 / (far - near)
    mat[3, 3] = 1.0
    mat[0, 3] = -(right + left) / (right - left)
    mat[1, 3] = -(top + bottom) / (top - bottom)
    mat[2, 3] = -(far + near) / (far - near)
    return mat


def _rgba(color: Color) -> Tuple[float, float, float, float]:
    r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    a = color[3] / 255.0 if len(color) > 3 else 1.0
    return r, g, b, a


def _load_hud_font():
    for candidate in config.HUD_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, config.HUD_FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


class OpenGLRenderer:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_zoom = 1.0
        self.draw_calls = 0          # draw calls del frame anterior (para el HUD)

        self._init_shaders()
        self._init_buffers()
        self._init_state()

        self._hud_font = _load_hud_font()
        self._hud_texture: Optional[Texture] = None
        self._hud_cache_key = ""
        self._hud_frame_counter = 0

        log.info("OpenGL: %s", glGetString(GL_VERSION).decode())

    # --- Inicialización ---

    def _init_shaders(self):
        self.shader_program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        self.simple_shader = compileProgram(
            compileShader(SIMPLE_VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(SIMPLE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        self.proj_loc = glGetUniformLocation(self.shader_program, "projection")
        self.tex_loc = glGetUniformLocation(self.shader_program, "texture0")
        self.simple_proj_loc = glGetUniformLocation(self.simple_shader, "projection")

    def _init_buffers(self):
        self.batch = SpriteBatch(max_sprites=config.MAX_SPRITES_PER_BATCH)

        self.simple_vao = glGenVertexArrays(1)
        self.simple_vbo = glGenBuffers(1)
        self._simple_vbo_size = 0
        glBindVertexArray(self.simple_vao)
        self._ensure_simple_vbo(1024 * 1024)
        stride = SIMPLE_FLOATS_PER_VERTEX * FLOAT_SIZE
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * FLOAT_SIZE))
        glBindVertexArray(0)

    def _init_state(self):
        self._update_projections()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDisable(GL_CULL_FACE)

    def _ensure_simple_vbo(self, nbytes: int):
        if nbytes <= self._simple_vbo_size:
            return
        size = max(nbytes, 2 * self._simple_vbo_size)
        glBindBuffer(GL_ARRAY_BUFFER, self.simple_vbo)
        glBufferData(GL_ARRAY_BUFFER, size, None, GL_DYNAMIC_DRAW)
        self._simple_vbo_size = size

    # --- Proyecciones y estado por frame ---

    def _update_projections(self):
        near, far = config.DEPTH_NEAR, config.DEPTH_FAR
        w, h = self.screen_width, self.screen_height
        self.screen_projection = ortho_matrix(0, w, h, 0, near, far)
        z, cx, cy = self.camera_zoom, self.camera_x, self.camera_y
        self.world_projection = ortho_matrix(cx, cx + w / z, cy + h / z, cy, near, far)

    def resize(self, width: int, height: int):
        self.screen_width = width
        self.screen_height = height
        glViewport(0, 0, width, height)
        self._update_projections()

    def set_camera(self, x: float, y: float, zoom: float):
        self.camera_x, self.camera_y, self.camera_zoom = x, y, zoom
        self._update_projections()

    def begin_frame(self):
        self.draw_calls = self.batch.draw_calls
        self.batch.draw_calls = 0
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # --- Quads con textura ---

    def draw_quads(self, texture: Texture, vertices: np.ndarray, world: bool = True):
        """Dibuja quads generados con ``quad_vertices``, en coordenadas de mundo o de pantalla."""
        if len(vertices) == 0:
            return
        glUseProgram(self.shader_program)
        projection = self.world_projection if world else self.screen_projection
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection.T)
        glUniform1i(self.tex_loc, 0)
        self.batch.draw(texture, vertices)
        glUseProgram(0)

    # --- Primitivas simples (coordenadas de pantalla) ---

    def _draw_simple(self, vertices: np.ndarray, mode, vertex_count: int, depth_test: bool):
        self._ensure_simple_vbo(vertices.nbytes)
        glUseProgram(self.simple_shader)
        glUniformMatrix4fv(self.simple_proj_loc, 1, GL_FALSE, self.screen_projection.T)
        glBindBuffer(GL_ARRAY_BUFFER, self.simple_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glBindVertexArray(self.simple_vao)
        if not depth_test:
            glDisable(GL_DEPTH_TEST)
        glDrawArrays(mode, 0, vertex_count)
        self.batch.draw_calls += 1
        if not depth_test:
            glEnable(GL_DEPTH_TEST)
        glBindVertexArray(0)
        glUseProgram(0)

    def draw_lines(self, lines: List[Tuple[float, float, float, float]], color: Color):
        """Líneas ``(x1, y1, x2, y2)`` en píxeles de pantalla."""
        if not len(lines):
            return
        points = np.asarray(lines, dtype=np.float32).reshape(-1, 2)
        colors = np.broadcast_to(np.float32(_rgba(color)), (len(points), 4))
        vertices = np.ascontiguousarray(np.hstack([points, colors]), dtype=np.float32)
        self._draw_simple(vertices, GL_LINES, len(points), depth_test=True)

    def draw_rects(self, rects, color: Color):
        """Rectángulos rellenos ``(x, y, w, h)`` en píxeles de pantalla, sin depth test.

        Acepta una lista de tuplas o un array numpy (N, 4).
        """
        r = np.asarray(rects, dtype=np.float32).reshape(-1, 4)
        if not len(r):
            return
        x0, y0 = r[:, 0], r[:, 1]
        x1, y1 = x0 + r[:, 2], y0 + r[:, 3]
        # Dos triángulos por rectángulo: (0,0)(1,0)(1,1) y (0,0)(1,1)(0,1)
        xs = np.stack([x0, x1, x1, x0, x1, x0], axis=1).ravel()
        ys = np.stack([y0, y0, y1, y0, y1, y1], axis=1).ravel()
        colors = np.broadcast_to(np.float32(_rgba(color)), (len(xs), 4))
        vertices = np.ascontiguousarray(np.column_stack([xs, ys, colors]), dtype=np.float32)
        self._draw_simple(vertices, GL_TRIANGLES, len(xs), depth_test=False)

    # --- HUD ---

    def draw_text_lines(self, text_lines: List[str], x: int, y: int):
        """Panel de texto. Se regenera si cambia el contenido (salvo las líneas que
        empiezan por "FPS:") o cada ``HUD_REFRESH_FRAMES`` frames."""
        cache_key = "|".join(l for l in text_lines if not l.startswith("FPS:"))
        self._hud_frame_counter += 1
        if (self._hud_texture is None or cache_key != self._hud_cache_key
                or self._hud_frame_counter >= config.HUD_REFRESH_FRAMES):
            self._rebuild_hud(text_lines)
            self._hud_cache_key = cache_key
            self._hud_frame_counter = 0

        tex = self._hud_texture
        vertices = quad_vertices(x, y, tex.width, tex.height, HUD_DEPTH, *tex.uv_rect())
        glDisable(GL_DEPTH_TEST)
        self.draw_quads(tex, vertices, world=False)
        glEnable(GL_DEPTH_TEST)

    def _rebuild_hud(self, text_lines: List[str]):
        height = len(text_lines) * HUD_LINE_HEIGHT + 2 * HUD_PADDING
        img = Image.new("RGBA", (HUD_WIDTH, height), HUD_BACKGROUND)
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(text_lines):
            draw.text((HUD_PADDING, HUD_PADDING + i * HUD_LINE_HEIGHT), line,
                      font=self._hud_font, fill=(255, 255, 255, 255))
        if self._hud_texture is not None:
            self._hud_texture.delete()
        self._hud_texture = Texture.from_pil(img, add_border=False)
