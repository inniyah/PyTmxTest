"""Halo del brillo propio: la escena se dibuja en una imagen intermedia con dos salidas (color y
brillo); el brillo se reduce a la mitad y a la cuarta parte, se difumina (gaussiano separable) y
se suma al color al pasar a la pantalla. A media y cuarta resolución, el difuminado es barato y el
halo se extiende lejos sin muchas muestras.
"""

from OpenGL.GL import (
    GL_CLAMP_TO_EDGE, GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_DEPTH_ATTACHMENT,
    GL_DEPTH_COMPONENT24, GL_DEPTH_TEST, GL_FLOAT, GL_FRAGMENT_SHADER, GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE, GL_LINEAR, GL_NEAREST, GL_RENDERBUFFER, GL_RGBA, GL_RGBA8, GL_RGBA16F,
    GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TRIANGLES, GL_UNSIGNED_BYTE, GL_VERTEX_SHADER, GL_BLEND,
    glActiveTexture, glBindFramebuffer, glBindRenderbuffer, glBindTexture, glBindVertexArray,
    glCheckFramebufferStatus, glDeleteFramebuffers, glDeleteRenderbuffers, glDeleteTextures,
    glDeleteVertexArrays, glDisable, glDrawArrays, glDrawBuffers, glEnable, glFramebufferRenderbuffer,
    glFramebufferTexture2D, glGenFramebuffers, glGenRenderbuffers, glGenTextures, glGenVertexArrays,
    glGetUniformLocation, glRenderbufferStorage, glTexImage2D, glTexParameteri, glUniform1f,
    glUniform1i, glUniform2f, glUseProgram, glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader

FULLSCREEN = """
#version 330 core
out vec2 v_uv;
void main() {   // un triángulo que cubre la pantalla, sin búfer de vértices
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
BLUR = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_step;          // un texel en la dirección del difuminado (o 0: copiar reduciendo)
out vec4 frag;
const float W[5] = float[5](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
void main() {
    vec3 c = texture(u_src, v_uv).rgb * W[0];
    for (int i = 1; i < 5; i++) {
        c += texture(u_src, v_uv + u_step * float(i)).rgb * W[i];
        c += texture(u_src, v_uv - u_step * float(i)).rgb * W[i];
    }
    frag = vec4(c, 1.0);
}
"""
COMPOSITE = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_scene;
uniform sampler2D u_half;
uniform sampler2D u_quarter;
uniform float u_strength;
out vec4 frag;
void main() {
    vec4 scene = texelFetch(u_scene, ivec2(gl_FragCoord.xy), 0);
    vec3 halo = (texture(u_half, v_uv).rgb * 0.7 + texture(u_quarter, v_uv).rgb * 0.9) * u_strength;
    if (max(halo.r, max(halo.g, halo.b)) < 1e-4) { frag = scene; return; }     // sin halo: tal cual
    frag = vec4(pow(pow(scene.rgb, vec3(2.2)) + halo, vec3(1.0 / 2.2)), 1.0);
}
"""
BLUR_PASSES = 2
STRENGTH = 1.0


def _tex(w, h, internal, fmt, kind, linear):
    t = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, t)
    glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, kind, None)
    f = GL_LINEAR if linear else GL_NEAREST
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, f)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, f)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return t


def _fbo(*textures):
    f = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, f)
    for i, t in enumerate(textures):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, t, 0)
    return f


class Bloom:
    def __init__(self):
        vs = compileShader(FULLSCREEN, GL_VERTEX_SHADER)
        self.blur = compileProgram(vs, compileShader(BLUR, GL_FRAGMENT_SHADER))
        self.composite = compileProgram(compileShader(FULLSCREEN, GL_VERTEX_SHADER),
                                        compileShader(COMPOSITE, GL_FRAGMENT_SHADER))
        self.vao = glGenVertexArrays(1)
        self.size = None
        self.textures, self.fbos, self.depth = [], [], None
        self.strength = STRENGTH

    def _release(self):
        if self.textures:
            glDeleteTextures(self.textures)
            glDeleteFramebuffers(len(self.fbos), self.fbos)
            glDeleteRenderbuffers(1, [self.depth])
        self.textures, self.fbos = [], []

    def _ensure(self, w, h):
        if self.size == (w, h):
            return
        self._release()
        self.size = (w, h)
        self.scene = _tex(w, h, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, False)
        self.glow = _tex(w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, True)
        self.scene_fbo = _fbo(self.scene, self.glow)
        self.depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("No se pudo crear la imagen intermedia del halo")
        self.levels = []
        for div in (2, 4):
            lw, lh = max(1, w // div), max(1, h // div)
            a = _tex(lw, lh, GL_RGBA16F, GL_RGBA, GL_FLOAT, True)
            b = _tex(lw, lh, GL_RGBA16F, GL_RGBA, GL_FLOAT, True)
            self.levels.append((lw, lh, a, b, _fbo(a), _fbo(b)))
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.textures = [self.scene, self.glow] + [t for lv in self.levels for t in lv[2:4]]
        self.fbos = [self.scene_fbo] + [f for lv in self.levels for f in lv[4:6]]

    def begin(self, w, h):
        """Dibujar la escena (color y brillo) en la imagen intermedia."""
        self._ensure(w, h)
        glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo)
        glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])

    def _pass(self, src, dst_fbo, size, step):
        glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo)
        glViewport(0, 0, *size)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src)
        glUniform2f(self.u_step, *step)
        glDrawArrays(GL_TRIANGLES, 0, 3)

    def end(self, w, h):
        """Difuminar el brillo y pasar la escena con su halo a la pantalla."""
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glBindVertexArray(self.vao)
        glUseProgram(self.blur)
        self.u_step = glGetUniformLocation(self.blur, "u_step")
        glUniform1i(glGetUniformLocation(self.blur, "u_src"), 0)
        src = self.glow
        for lw, lh, a, b, fa, fb in self.levels:
            self._pass(src, fa, (lw, lh), (0.0, 0.0))                  # reducir (filtro lineal)
            for _ in range(BLUR_PASSES):
                self._pass(a, fb, (lw, lh), (1.0 / lw, 0.0))
                self._pass(b, fa, (lw, lh), (0.0, 1.0 / lh))
            src = a
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, w, h)
        glUseProgram(self.composite)
        for unit, (name, tex) in enumerate((("u_scene", self.scene), ("u_half", self.levels[0][2]),
                                            ("u_quarter", self.levels[1][2]))):
            glActiveTexture((GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2)[unit])
            glBindTexture(GL_TEXTURE_2D, tex)
            glUniform1i(glGetUniformLocation(self.composite, name), unit)
        glUniform1f(glGetUniformLocation(self.composite, "u_strength"), self.strength)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)
        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)

    def delete(self):
        self._release()
        glDeleteVertexArrays(1, [self.vao])
