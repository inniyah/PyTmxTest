"""Texto sobre la vista (mensajes de los sucesos): Pillow pinta el texto en una imagen, que se
sube como textura y se dibuja como un rectángulo en píxeles de pantalla. Solo se vuelve a pintar
cuando cambian las líneas."""

import ctypes
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_DEPTH_TEST, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER, GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA, GL_RGBA, GL_RGBA8, GL_SRC_ALPHA, GL_STATIC_DRAW, GL_TEXTURE0,
    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TRIANGLES, GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER, glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray, glBlendFunc,
    glBufferData, glDeleteBuffers, glDeleteTextures, glDeleteVertexArrays, glDisable, glDrawArrays,
    glEnable, glEnableVertexAttribArray, glGenBuffers, glGenTextures, glGenVertexArrays,
    glGetUniformLocation, glTexImage2D, glTexParameteri, glUniform1i, glUniform2f, glUseProgram,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image, ImageDraw, ImageFont

VERTEX = """
#version 330 core
layout(location = 0) in vec2 a_pos;     // píxeles de pantalla, origen arriba a la izquierda
layout(location = 1) in vec2 a_uv;
uniform vec2 u_screen;
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos.x / u_screen.x * 2.0 - 1.0, 1.0 - a_pos.y / u_screen.y * 2.0, 0.0, 1.0);
}
"""
FRAGMENT = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_text;
out vec4 frag;
void main() { frag = texture(u_text, v_uv); }
"""


FONT = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "DejaVuSans.ttf"   # licencia al lado


def _font(size: int):
    """DejaVu Sans (incluida en assets/fonts, con todas las letras del español); si faltara, la
    de Pillow."""
    try:
        return ImageFont.truetype(str(FONT), size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


class TextOverlay:
    def __init__(self, size: int = 20):
        self.program = compileProgram(compileShader(VERTEX, GL_VERTEX_SHADER),
                                      compileShader(FRAGMENT, GL_FRAGMENT_SHADER))
        self.u_screen = glGetUniformLocation(self.program, "u_screen")
        self.u_text = glGetUniformLocation(self.program, "u_text")
        self.font = _font(size)
        self.texture = glGenTextures(1)
        self.vao, self.vbo = glGenVertexArrays(1), glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        for loc, off in ((0, 0), (1, 2)):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(off * 4))
        glBindVertexArray(0)
        self.lines = ()
        self.size = (0, 0)

    def render_lines(self, lines) -> Image.Image:
        """La imagen del texto: líneas centradas sobre una banda oscura semitransparente."""
        pad, gap = 12, 6
        boxes = [self.font.getbbox(t) for t in lines]
        w = max(b[2] - b[0] for b in boxes) + 2 * pad
        line_h = max(b[3] - b[1] for b in boxes) + gap
        h = line_h * len(lines) + 2 * pad - gap
        img = Image.new("RGBA", (w, h), (12, 14, 20, 170))
        d = ImageDraw.Draw(img)
        for i, (t, b) in enumerate(zip(lines, boxes)):
            x = (w - (b[2] - b[0])) // 2 - b[0]
            d.text((x + 1, pad + i * line_h - b[1] + 1), t, font=self.font, fill=(0, 0, 0, 200))
            d.text((x, pad + i * line_h - b[1]), t, font=self.font, fill=(245, 240, 225, 255))
        return img

    def draw(self, lines, screen_w: int, screen_h: int):
        lines = tuple(lines)
        if not lines:
            return
        if lines != self.lines:
            img = self.render_lines(lines)
            self.lines, self.size = lines, img.size
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         np.asarray(img).tobytes())
        w, h = self.size
        x0, y0 = (screen_w - w) / 2, screen_h - h - 24
        quad = np.array([[x0, y0, 0, 0], [x0 + w, y0, 1, 0], [x0 + w, y0 + h, 1, 1],
                         [x0, y0, 0, 0], [x0 + w, y0 + h, 1, 1], [x0, y0 + h, 0, 1]], np.float32)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self.program)
        glUniform2f(self.u_screen, float(screen_w), float(screen_h))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(self.u_text, 0)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glEnable(GL_DEPTH_TEST)

    def delete(self):
        glDeleteTextures(1, [self.texture])
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
