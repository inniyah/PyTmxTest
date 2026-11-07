"""Sprites de personaje (planos verticales en los pies) y sus sombras en el suelo.

Profundidad por píxel con la misma cercanía que los tiles: un sprite es un plano vertical
en el Z de los pies, así que cada píxel tiene altura Y = pies + (lo alto que esté en el
sprite) y cercanía = Y · sz + Z · sy. La sombra es un plano horizontal a la altura del
suelo bajo los pies, con Z variando a lo largo del quad.
"""

import ctypes
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

from .shaders import LEVEL_BIAS, LIGHTING_GLSL
from .world_renderer import LIGHT_UNIFORMS, _texture

SPRITE_VERTEX = """
#version 330 core
layout(location = 0) in vec2 a_pos;      // píxeles de mundo
layout(location = 1) in vec2 a_uv;       // en la hoja (sprite) o en el disco de la sombra (-1..1)
layout(location = 2) in vec2 a_world;    // (Y, Z) en metros del punto
uniform mat4 u_projection;
out vec2 v_uv;
out vec2 v_world;
out float v_px;
void main() {
    v_uv = a_uv;
    v_world = a_world;
    v_px = a_pos.x;
    gl_Position = u_projection * vec4(a_pos, 0.0, 1.0);
}
"""

SPRITE_FRAGMENT = """
#version 330 core
in vec2 v_uv;
in vec2 v_world;
in float v_px;
uniform float u_m_per_px;          // metros por píxel de pantalla en X
uniform sampler2D u_sheet;
uniform int u_shadow;
uniform vec2 u_scale;              // (sz, sy)
uniform vec2 u_closeness_range;
uniform float u_bias;              // desempate con el suelo de la planta en la que está
uniform int u_show_depth;
""" + LIGHTING_GLSL + """
out vec4 frag;
void main() {
    vec4 color;
    if (u_shadow == 1) {
        float r = length(v_uv);
        if (r > 1.0) discard;
        color = vec4(0.0, 0.0, 0.0, 0.35 * (1.0 - smoothstep(0.6, 1.0, r)));
    } else {
        color = texture(u_sheet, v_uv);
        if (color.a < 0.01) discard;
    }
    float closeness = v_world.x * u_scale.x + v_world.y * u_scale.y + u_bias;
    float depth = clamp(1.0 - (closeness - u_closeness_range.x)
                        / (u_closeness_range.y - u_closeness_range.x), 0.0, 1.0);
    gl_FragDepth = depth;
    if (u_lighting == 1 && u_shadow == 0) {
        // El sprite, de cara a la cámara (sin normales): luz "envolvente" para que también le
        // alumbre una luz que esté a un lado
        vec3 world = vec3(v_px * u_m_per_px, v_world.x, v_world.y);
        vec3 n = normalize(vec3(0.0, 0.35, 1.0));
        color.rgb = to_srgb(to_linear(color.rgb) * light_at(world, n, 0.6));
    }
    frag = (u_show_depth == 1 && u_shadow == 0) ? vec4(vec3(1.0 - depth), 1.0) : color;
}
"""

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
    def __init__(self, geo):
        self.geo = geo
        self.program = compileProgram(compileShader(SPRITE_VERTEX, GL_VERTEX_SHADER),
                                      compileShader(SPRITE_FRAGMENT, GL_FRAGMENT_SHADER))
        self.u = {k: glGetUniformLocation(self.program, k) for k in (
            "u_projection", "u_sheet", "u_shadow", "u_scale", "u_closeness_range", "u_bias", "u_show_depth",
            "u_m_per_px", *LIGHT_UNIFORMS)}
        self.vao, self.vbo = glGenVertexArrays(1), glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        for loc, off in ((0, 0), (1, 2), (2, 4)):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(off * 4))
        glBindVertexArray(0)

    # Vértices: dos triángulos por quad, 6 floats (pos, uv, Y, Z)
    def _quad(self, corners):
        tri = [0, 1, 2, 0, 2, 3]
        return np.asarray(corners, np.float32)[tri]

    def sprite_quad(self, sheet: SpriteSheet, feet, body_h_m, row, col):
        """Quad de un sprite: ancho y alto escalados para que el cuerpo mida body_h_m."""
        g = self.geo
        X, Y, Z = feet
        fw, fh = sheet.frame_size
        scale = body_h_m * g.sy / fh
        w, h = fw * scale, fh * scale
        x = X * g.tile_w / g.cell_w
        s = Z * g.sz - Y * g.sy                       # pantalla (píxeles de mundo) de los pies
        u0, v0, u1, v1 = sheet.uv(row, col)
        top_y = Y + h / g.sy                          # altura del borde superior del sprite
        return self._quad([(x - w / 2, s - h, u0, v0, top_y, Z), (x + w / 2, s - h, u1, v0, top_y, Z),
                           (x + w / 2, s, u1, v1, Y, Z), (x - w / 2, s, u0, v1, Y, Z)])

    def shadow_quad(self, feet, ground_y, radius_m):
        g = self.geo
        X, _Y, Z = feet
        rx, rz = radius_m * 1.3, radius_m * 0.9
        corners = []
        for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            px = (X + du * rx) * g.tile_w / g.cell_w
            pz = Z + dv * rz
            corners.append((px, pz * g.sz - ground_y * g.sy, du, dv, ground_y, pz))
        return self._quad(corners)

    def _draw(self, verts, texture, shadow, bias, camera, show_depth, lights=None):
        g = self.geo
        glUseProgram(self.program)
        glUniform1f(self.u["u_m_per_px"], g.cell_w / g.tile_w)
        if lights is not None:
            lights.set_light_uniforms(self.u, camera)
        else:
            glUniform1i(self.u["u_lighting"], 0)
        glUniformMatrix4fv(self.u["u_projection"], 1, GL_FALSE, camera.projection().T)
        glUniform1i(self.u["u_sheet"], 0)
        glUniform1i(self.u["u_shadow"], int(shadow))
        glUniform2f(self.u["u_scale"], g.sz, g.sy)
        glUniform2f(self.u["u_closeness_range"], g.closeness_min, g.closeness_max)
        glUniform1f(self.u["u_bias"], bias)
        glUniform1i(self.u["u_show_depth"], int(show_depth))
        if texture:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, texture)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, len(verts))
        glBindVertexArray(0)
        glUseProgram(0)

    def draw(self, sheet: SpriteSheet, actors, camera, show_depth=False, lights=None):
        """``actors``: lista de (pies (X, Y, Z), suelo bajo los pies, alto, radio, fila, columna).
        ``lights``: el WorldRenderer, para iluminar con sus mismas luces (o None, sin luz)."""
        g = self.geo
        # Sombras primero, sin escribir profundidad (no deben tapar los pies)
        glDepthMask(GL_FALSE)
        for feet, ground, _h, radius, _row, _col in actors:
            if ground > -1e8:
                bias = LEVEL_BIAS * np.floor(ground / g.cell_h + 1e-6) + 0.02
                self._draw(self.shadow_quad(feet, ground, radius), 0, True, bias, camera, show_depth)
        glDepthMask(GL_TRUE)
        for feet, _ground, h, _radius, row, col in actors:
            bias = LEVEL_BIAS * np.floor(feet[1] / g.cell_h + 1e-6) + 0.05
            self._draw(self.sprite_quad(sheet, feet, h, row, col), sheet.texture, False, bias, camera, show_depth,
                       lights)

    def delete(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])


def delete_sheet(sheet: SpriteSheet):
    glDeleteTextures([sheet.texture])
