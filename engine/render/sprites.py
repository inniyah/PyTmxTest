"""Sprites de personaje (planos verticales en los pies) y sus sombras en el suelo.

Profundidad por píxel con la misma cercanía que los tiles: un sprite es un plano vertical
en el Z de los pies, así que cada píxel tiene altura Y = pies + (lo alto que esté en el
sprite) y cercanía = Y · sz + Z · sy. La sombra es un plano horizontal a la altura del
suelo bajo los pies, con Z variando a lo largo del quad.
"""

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER, GL_TEXTURE0,
    GL_TEXTURE_2D, GL_TRIANGLES, GL_TRUE, GL_VERTEX_SHADER, glActiveTexture, glBindBuffer,
    glBindTexture, glBindVertexArray, glBufferData, glDeleteBuffers, glDeleteTextures,
    glDeleteVertexArrays, glDepthMask, glDrawArrays, glEnableVertexAttribArray, glGenBuffers,
    glGenVertexArrays, glGetUniformLocation, glUniform1f, glUniform1i, glUniform2f,
    glUniformMatrix4fv, glUseProgram, glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image

from .shaders import FACADE_GLSL, LEVEL_BIAS, LIGHTING_GLSL
from .world_renderer import FACADE_UNIFORMS, LIGHT_UNIFORMS, _texture, set_facade_uniforms

SPRITE_VERTEX = """
#version 330 core
layout(location = 0) in vec2 a_pos;      // píxeles de mundo
layout(location = 1) in vec2 a_uv;       // en la hoja (sprite) o en el disco de la sombra (-1..1)
layout(location = 2) in vec2 a_world;    // (Y, Z) en metros del punto
layout(location = 3) in vec3 a_tint;
layout(location = 4) in float a_bias;    // desempate con el suelo de su planta
layout(location = 5) in float a_level;   // planta de los pies (para ocultar lo de encima)
uniform mat4 u_projection;
out vec2 v_uv;
out vec2 v_world;
out vec2 v_pos;
out vec3 v_tint;
out float v_bias;
out float v_level;
void main() {
    v_uv = a_uv;
    v_world = a_world;
    v_pos = a_pos;
    v_tint = a_tint;
    v_bias = a_bias;
    v_level = a_level;
    gl_Position = u_projection * vec4(a_pos, 0.0, 1.0);
}
"""

SPRITE_FRAGMENT = """
#version 330 core
in vec2 v_uv;
in vec2 v_world;
in vec2 v_pos;
in vec3 v_tint;
in float v_bias;
in float v_level;
uniform float u_m_per_px;          // metros por píxel de pantalla en X
uniform sampler2D u_sheet;
uniform int u_shadow;
uniform vec2 u_scale;              // (sz, sy)
uniform vec2 u_closeness_range;
uniform int u_show_depth;
uniform float u_cut_level;         // ocultar lo que esté por encima de esta planta...
uniform vec2 u_cut_center;         // ...en este círculo (radio < 0: en todo el mapa)
uniform float u_cut_radius;
uniform float u_max_level;
""" + LIGHTING_GLSL + FACADE_GLSL + """
out vec4 frag;
void main() {
    if (v_level > u_max_level + 0.5) discard;
    bool in_circle = u_cut_radius < 0.0 || distance(v_pos, u_cut_center) < u_cut_radius;
    if (u_facade == 1) {
        // Fachada: los de las plantas de arriba cuya celda está oculta
        if (facade_mask(vec2(v_pos.x * u_m_per_px, v_world.y), int(floor(v_level + 0.5))) > 0.75)
            discard;
    } else if (v_level > u_cut_level + 0.5 && in_circle) {
        discard;
    }
    vec4 color;
    if (u_shadow == 1) {
        float r = length(v_uv);
        if (r > 1.0) discard;
        color = vec4(0.0, 0.0, 0.0, 0.35 * (1.0 - smoothstep(0.6, 1.0, r)));
    } else {
        color = texture(u_sheet, v_uv);
        if (color.a < 0.01) discard;
        color.rgb *= v_tint;
    }
    float closeness = v_world.x * u_scale.x + v_world.y * u_scale.y + v_bias;
    float depth = clamp(1.0 - (closeness - u_closeness_range.x)
                        / (u_closeness_range.y - u_closeness_range.x), 0.0, 1.0);
    gl_FragDepth = depth;
    if (u_lighting == 1 && u_shadow == 0) {
        // El sprite, de cara a la cámara (sin normales): luz "envolvente" para que también le
        // alumbre una luz que esté a un lado
        vec3 world = vec3(v_pos.x * u_m_per_px, v_world.x, v_world.y);
        vec3 n = normalize(vec3(0.0, 0.35, 1.0));
        color.rgb = to_srgb(to_linear(color.rgb) * light_at(world, n, 0.6));
    }
    frag = (u_show_depth == 1 && u_shadow == 0) ? vec4(vec3(1.0 - depth), 1.0) : color;
}
"""

FLOATS = 11          # pos (2), uv (2), mundo Y Z (2), tinte (3), desempate (1), planta (1)
WALK_FPS = 8.0


# Direcciones: índice 0 = este y sentido horario en pantalla (este, sur, oeste, norte con 4;
# este, sureste, sur… con 8). Cada hoja dice qué fila tiene cada dirección.
ROWS_4 = {0: 2, 1: 0, 2: 1, 3: 3}          # hoja actual: filas abajo, izquierda, derecha, arriba
HYSTERESIS_DEG = 20.0


def facing_index(dx: float, dz: float, previous: int, directions: int = 4,
                 hysteresis_deg: float = HYSTERESIS_DEG) -> int:
    """Dirección (0..directions-1) hacia la que mira algo que quiere moverse en (dx, dz).

    Sin movimiento se conserva la anterior. Con histéresis: la dirección actual se mantiene
    mientras el movimiento no se aleje de ella más de medio sector + ``hysteresis_deg``,
    así que una diagonal exacta (con 4 direcciones) no hace saltar el sprite.
    """
    if dx * dx + dz * dz < 1e-6:
        return previous
    sector = 360.0 / directions
    angle = np.degrees(np.arctan2(dz, dx)) % 360.0          # 0 = este, 90 = sur (Z hacia abajo)
    current = previous * sector
    off = abs((angle - current + 180.0) % 360.0 - 180.0)
    if off <= sector / 2 + hysteresis_deg:
        return previous
    return int(np.floor(angle / sector + 0.5)) % directions


@contextmanager
def _nothing():
    yield


def facing_indices(dx, dz, previous, directions: int = 4, hysteresis_deg: float = HYSTERESIS_DEG):
    """Versión vectorizada de ``facing_index`` para muchos personajes a la vez."""
    dx, dz, previous = np.asarray(dx, float), np.asarray(dz, float), np.asarray(previous)
    sector = 360.0 / directions
    angle = np.degrees(np.arctan2(dz, dx)) % 360.0
    off = np.abs((angle - previous * sector + 180.0) % 360.0 - 180.0)
    keep = (dx * dx + dz * dz < 1e-6) | (off <= sector / 2 + hysteresis_deg)
    return np.where(keep, previous, np.floor(angle / sector + 0.5).astype(int) % directions)


@dataclass
class SpriteSheet:
    """Hoja de frames. La actual (4 × 4): filas abajo, izquierda, derecha, arriba; en cada fila un
    ciclo de andar de cuatro frames: de pie, paso, de pie, el otro paso."""
    texture: int
    width: int
    height: int
    cols: int = 4
    rows: int = 4
    idle_col: int = 0
    walk_cycle: tuple = (1, 2, 3, 0)      # empieza por un paso, para que se note enseguida

    def walk_cols(self, walk_time, fps: float):
        """Versión vectorizada de ``walk_col``."""
        walk_time = np.asarray(walk_time, float)
        cycle = np.asarray(self.walk_cycle)
        return np.where(walk_time > 0, cycle[(walk_time * fps).astype(int) % len(cycle)], self.idle_col)

    def walk_col(self, walk_time: float, fps: float) -> int:
        """Columna del ciclo de andar tras ``walk_time`` segundos andando (o la de quieto)."""
        if walk_time <= 0:
            return self.idle_col
        return self.walk_cycle[int(walk_time * fps) % len(self.walk_cycle)]

    @classmethod
    def load(cls, path):
        with Image.open(path) as im:
            data = np.asarray(im.convert("RGBA"))
        return cls(_texture(data, False), data.shape[1], data.shape[0])

    @property
    def frame_size(self):
        return self.width // self.cols, self.height // self.rows

    def uv(self, row: int, col: int):
        fw, fh = 1 / self.cols, 1 / self.rows
        return col * fw, row * fh, (col + 1) * fw, (row + 1) * fh


class SpriteRenderer:
    """Dibuja todos los personajes (y sus sombras) en una llamada por hoja de sprites."""

    def __init__(self, geo):
        self.geo = geo
        self.program = compileProgram(compileShader(SPRITE_VERTEX, GL_VERTEX_SHADER),
                                      compileShader(SPRITE_FRAGMENT, GL_FRAGMENT_SHADER))
        self.u = {k: glGetUniformLocation(self.program, k) for k in (
            "u_projection", "u_sheet", "u_shadow", "u_scale", "u_closeness_range", "u_show_depth",
            "u_m_per_px", "u_cut_level", "u_cut_center", "u_cut_radius", "u_max_level", *LIGHT_UNIFORMS,
            *FACADE_UNIFORMS)}
        self.vao, self.vbo = glGenVertexArrays(1), glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        for loc, size, off in ((0, 2, 0), (1, 2, 2), (2, 2, 4), (3, 3, 6), (4, 1, 9), (5, 1, 10)):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, FLOATS * 4, ctypes.c_void_p(off * 4))
        glBindVertexArray(0)
        self.draw_calls = 0
        self.timer = None              # GpuTimer opcional: mide sombras y personajes

    # --- Vértices (numpy, sin OpenGL) ---

    def sprite_vertices(self, sheet: SpriteSheet, feet, body_h, rows, cols, tint, level):
        """(n·6, FLOATS): dos triángulos por personaje, de pie sobre sus pies."""
        g = self.geo
        feet = np.asarray(feet, float).reshape(-1, 3)
        n = len(feet)
        X, Y, Z = feet[:, 0], feet[:, 1], feet[:, 2]
        fw, fh = sheet.frame_size
        scale = np.asarray(body_h, float) * g.sy / fh
        w, h = fw * scale, fh * scale
        x = X * g.tile_w / g.cell_w
        s = Z * g.sz - Y * g.sy
        u0 = np.asarray(cols) / sheet.cols
        v0 = np.asarray(rows) / sheet.rows
        u1, v1 = u0 + 1 / sheet.cols, v0 + 1 / sheet.rows
        top = Y + h / g.sy
        bias = LEVEL_BIAS * np.floor(Y / g.cell_h + 1e-6) + 0.05
        corners = [  # arriba-izq, arriba-der, abajo-der, abajo-izq
            (x - w / 2, s - h, u0, v0, top), (x + w / 2, s - h, u1, v0, top),
            (x + w / 2, s, u1, v1, Y), (x - w / 2, s, u0, v1, Y)]
        quad = np.zeros((n, 4, FLOATS), np.float32)
        for k, (px, py, uu, vv, yy) in enumerate(corners):
            quad[:, k, 0], quad[:, k, 1], quad[:, k, 2], quad[:, k, 3] = px, py, uu, vv
            quad[:, k, 4], quad[:, k, 5] = yy, Z
        quad[:, :, 6:9] = np.asarray(tint, np.float32).reshape(-1, 1, 3)
        quad[:, :, 9] = bias[:, None]
        quad[:, :, 10] = np.asarray(level, np.float32)[:, None]
        return quad[:, [0, 1, 2, 0, 2, 3]].reshape(-1, FLOATS)

    def shadow_vertices(self, feet, ground, radius, level):
        g = self.geo
        feet = np.asarray(feet, float).reshape(-1, 3)
        ground = np.asarray(ground, float)
        X, Z = feet[:, 0], feet[:, 2]
        rx, rz = np.asarray(radius) * 1.3, np.asarray(radius) * 0.9
        quad = np.zeros((len(feet), 4, FLOATS), np.float32)
        for k, (du, dv) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            pz = Z + dv * rz
            quad[:, k, 0] = (X + du * rx) * g.tile_w / g.cell_w
            quad[:, k, 1] = pz * g.sz - ground * g.sy
            quad[:, k, 2], quad[:, k, 3] = du, dv
            quad[:, k, 4], quad[:, k, 5] = ground, pz
        quad[:, :, 9] = (LEVEL_BIAS * np.floor(ground / g.cell_h + 1e-6) + 0.02)[:, None]
        quad[:, :, 10] = np.asarray(level, np.float32)[:, None]
        return quad[:, [0, 1, 2, 0, 2, 3]].reshape(-1, FLOATS)

    # --- Dibujo ---

    def _draw(self, verts, texture, shadow, camera, show_depth, lights, cut, max_level):
        g = self.geo
        glUseProgram(self.program)
        glUniformMatrix4fv(self.u["u_projection"], 1, GL_FALSE, camera.projection().T)
        glUniform1i(self.u["u_sheet"], 0)
        glUniform1i(self.u["u_shadow"], int(shadow))
        glUniform2f(self.u["u_scale"], g.sz, g.sy)
        glUniform2f(self.u["u_closeness_range"], g.closeness_min, g.closeness_max)
        glUniform1i(self.u["u_show_depth"], int(show_depth))
        glUniform1f(self.u["u_m_per_px"], g.cell_w / g.tile_w)
        level, center, radius = cut if cut is not None else (1e9, (0.0, 0.0), -1.0)
        glUniform1f(self.u["u_cut_level"], float(level))
        glUniform2f(self.u["u_cut_center"], *center)
        glUniform1f(self.u["u_cut_radius"], float(radius))
        glUniform1f(self.u["u_max_level"], float(max_level))
        set_facade_uniforms(self.u, getattr(lights, "facade", None) if lights is not None else None)
        if lights is not None:
            lights.set_light_uniforms(self.u, camera)
        else:
            glUniform1i(self.u["u_lighting"], 0)
        if texture:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, texture)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, len(verts))
        glBindVertexArray(0)
        glUseProgram(0)
        self.draw_calls += 1

    def draw_batch(self, sheet: SpriteSheet, feet, ground, body_h, radius, rows, cols, tint, camera,
                   show_depth=False, lights=None, cut=None, max_level=1e9):
        """Todos los personajes de una hoja: sombras (sin escribir profundidad) y luego sprites.

        ``cut``: (planta, centro, radio) para ocultar lo que esté por encima, o None."""
        feet = np.asarray(feet, float).reshape(-1, 3)
        if len(feet) == 0:
            return
        g = self.geo
        level = np.floor(feet[:, 1] / g.cell_h + 1e-6)
        ground = np.asarray(ground, float)
        has_ground = ground > -1e8
        timed = self.timer if self.timer is not None else (lambda _name: _nothing())
        glDepthMask(GL_FALSE)
        if has_ground.any():
            sv = self.shadow_vertices(feet[has_ground], ground[has_ground], np.asarray(radius)[has_ground],
                                      level[has_ground])
            with timed("sombras"):
                self._draw(sv, 0, True, camera, show_depth, lights, cut, max_level)
        glDepthMask(GL_TRUE)
        verts = self.sprite_vertices(sheet, feet, body_h, rows, cols, tint, level)
        with timed("personajes"):
            self._draw(verts, sheet.texture, False, camera, show_depth, lights, cut, max_level)

    def draw(self, sheet: SpriteSheet, actors, camera, show_depth=False, lights=None):
        """Compatibilidad: ``actors`` como lista de (pies, suelo, alto, radio, fila, columna)."""
        if not actors:
            return
        feet, ground, h, r, rows, cols = (list(t) for t in zip(*actors))
        self.draw_batch(sheet, feet, ground, h, r, rows, cols, np.ones((len(feet), 3)), camera,
                        show_depth, lights)

    def delete(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])


def delete_sheet(sheet: SpriteSheet):
    glDeleteTextures([sheet.texture])
