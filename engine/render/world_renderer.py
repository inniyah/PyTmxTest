"""Dibujo del mundo compilado con OpenGL 3.3.

- El atlas sube a la GPU una vez: color (RGBA8) y posición (RGBA16) por página.
- Cada chunk visible tiene su malla en la GPU, construida una sola vez al cargarlo.
- Cada frame solo se decide qué chunks se ven; la oclusión es por píxel (ver shaders.py).
"""

import ctypes
import io
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_ELEMENT_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER, GL_LEQUAL,
    GL_NEAREST, GL_ONE_MINUS_SRC_ALPHA, GL_RGBA, GL_RGBA8, GL_RGBA16, GL_SRC_ALPHA,
    GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TRIANGLES,
    GL_UNSIGNED_BYTE, GL_UNSIGNED_INT, GL_UNSIGNED_SHORT, GL_VERTEX_SHADER, glActiveTexture,
    glBindBuffer, glBindTexture, glBindVertexArray, glBufferData, glClear,
    glClearColor, glClearDepth, glDeleteBuffers, glDeleteTextures, glDeleteVertexArrays,
    glDepthFunc, glDrawElements, glEnable, glEnableVertexAttribArray, glGenBuffers,
    glGenTextures, glGenVertexArrays, glGetUniformLocation, glTexImage2D, glTexParameteri,
    glUniform1f, glUniform1i, glUniform2f, glUniform3f, glUniform3fv, glUniform4fv,
    glUniformMatrix4fv, glUseProgram,
    glVertexAttribPointer, glViewport,
)
from OpenGL.GL import GL_ONE, glBlendFuncSeparate
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image

from engine import png16
from engine.world.view import WorldView

from .geometry import FLOATS_PER_VERTEX, AtlasLayout, WorldGeometry, chunk_quads
from .shaders import LEVEL_BIAS, MAX_LIGHTS, TILE_FRAGMENT, TILE_VERTEX


def ortho(left, right, bottom, top) -> np.ndarray:
    m = np.identity(4, np.float32)
    m[0, 0], m[1, 1], m[2, 2] = 2 / (right - left), 2 / (top - bottom), -1
    m[0, 3], m[1, 3] = -(right + left) / (right - left), -(top + bottom) / (top - bottom)
    return m


LIGHT_UNIFORMS = ("u_lighting", "u_sky", "u_ground", "u_sun_dir", "u_sun_color", "u_num_lights",
                  "u_light_pos[0]", "u_light_col[0]", "u_lamps")


@dataclass
class Lighting:
    """Ambiente de cielo y suelo, sol y si la iluminación está activa. Colores lineales."""
    enabled: bool = True
    sky: tuple = (0.42, 0.46, 0.55)
    ground: tuple = (0.20, 0.18, 0.16)
    sun_dir: tuple = (-0.45, 0.80, 0.55)       # hacia el sol: alto, al suroeste (de cara a la cámara)
    sun_color: tuple = (0.85, 0.80, 0.70)
    lamps: float = 0.25                        # factor de las luces puntuales (de día, poco)

    @classmethod
    def day(cls, enabled=True):
        return cls(enabled=enabled)

    @classmethod
    def night(cls, enabled=True):
        return cls(enabled=enabled, sky=(0.035, 0.045, 0.08), ground=(0.015, 0.015, 0.02),
                   sun_dir=(0.3, 0.9, -0.2), sun_color=(0.06, 0.08, 0.14), lamps=1.0)


@dataclass
class Camera:
    """Esquina superior izquierda (píxeles de mundo) y zoom. La vista mide width × height px lógicos."""
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0
    width: int = 1280
    height: int = 720

    def bounds(self):
        return self.x, self.y, self.x + self.width / self.zoom, self.y + self.height / self.zoom

    def projection(self) -> np.ndarray:
        l, t, r, b = self.bounds()
        return ortho(l, r, b, t)

    def center_on(self, px: float, py: float):
        self.x, self.y = px - self.width / (2 * self.zoom), py - self.height / (2 * self.zoom)

    def zoom_at(self, factor: float, sx: float, sy: float, lo=0.05, hi=8.0):
        """Zoom manteniendo fijo el punto de pantalla (sx, sy)."""
        wx, wy = self.x + sx / self.zoom, self.y + sy / self.zoom
        self.zoom = float(np.clip(self.zoom * factor, lo, hi))
        self.x, self.y = wx - sx / self.zoom, wy - sy / self.zoom


def _texture(data: np.ndarray, sixteen_bits: bool):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    for p, v in ((GL_TEXTURE_MIN_FILTER, GL_NEAREST), (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                 (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
        glTexParameteri(GL_TEXTURE_2D, p, v)
    h, w = data.shape[:2]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16 if sixteen_bits else GL_RGBA8, w, h, 0, GL_RGBA,
                 GL_UNSIGNED_SHORT if sixteen_bits else GL_UNSIGNED_BYTE, np.ascontiguousarray(data))
    return tex


class GpuAtlas:
    """Páginas del atlas en la GPU: color y posición (siempre); albedo y normales (si las hay)."""

    def __init__(self, store, layout: AtlasLayout):
        self.color, self.position, self.albedo, self.normal = {}, {}, {}, {}
        for page in layout.page_sizes:
            with Image.open(io.BytesIO(store.atlas_page(page, "color"))) as im:
                self.color[page] = _texture(np.asarray(im.convert("RGBA")), False)
            raw = store.atlas_page(page, "position")
            if raw is None:
                raise ValueError("El mundo no tiene mapa de posición: hace falta para la profundidad por píxel")
            self.position[page] = _texture(png16.decode_rgba16(raw), True)
            for kind, dest in (("albedo", self.albedo), ("normal", self.normal)):
                raw = store.atlas_page(page, kind)
                if raw is not None:
                    with Image.open(io.BytesIO(raw)) as im:
                        dest[page] = _texture(np.asarray(im.convert("RGBA")), False)

    def delete(self):
        glDeleteTextures([t for d in (self.color, self.position, self.albedo, self.normal) for t in d.values()])


class ChunkMesh:
    """Mallas de un chunk en la GPU, una por página del atlas."""

    def __init__(self, quads: Dict[int, np.ndarray]):
        self.parts = []                                   # (página, vao, vbo, ebo, índices)
        for page, verts in quads.items():
            n = len(verts) // 4
            vao, vbo, ebo = glGenVertexArrays(1), glGenBuffers(1), glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
            stride = FLOATS_PER_VERTEX * 4
            for loc, size, off in ((0, 2, 0), (1, 2, 2), (2, 3, 4), (3, 1, 7), (4, 1, 8)):
                glEnableVertexAttribArray(loc)
                glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(off * 4))
            idx = (np.arange(n, dtype=np.uint32)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3], np.uint32)).ravel()
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
            glBindVertexArray(0)
            self.parts.append((page, vao, vbo, ebo, idx.size))
        self.quads = sum(len(v) // 4 for v in quads.values())

    def delete(self):
        for _page, vao, vbo, ebo, _n in self.parts:
            glDeleteVertexArrays(1, [vao])
            glDeleteBuffers(2, [vbo, ebo])


class WorldRenderer:
    def __init__(self, view: WorldView):
        self.view = view
        self.geo = WorldGeometry.from_meta(view.meta)
        self.layout = AtlasLayout.from_store(view.store)
        self.atlas = GpuAtlas(view.store, self.layout)
        self.meshes: Dict[Tuple[int, int], ChunkMesh] = {}
        self.max_level = view.level_max          # plantas por encima de esta no se dibujan
        self.show_depth = False
        self.height_tint = False
        # Ocultar lo que hay encima: celdas por encima de cut_level, en todo el mapa
        # (cut_radius < 0) o en un círculo de cut_radius px alrededor de cut_center
        self.cut_level = None
        self.cut_center = (0.0, 0.0)
        self.cut_radius = -1.0
        # Iluminación: luces del mundo (metros, color lineal) + luces dinámicas (el farol)
        self.lighting = Lighting()
        self.world_lights = np.array(view.store.lights(), dtype=np.float32).reshape(-1, 8)
        self.dynamic_lights = []
        self.draw_calls = 0

        self.program = compileProgram(compileShader(TILE_VERTEX, GL_VERTEX_SHADER),
                                      compileShader(TILE_FRAGMENT, GL_FRAGMENT_SHADER))
        self.u = {name: glGetUniformLocation(self.program, name) for name in (
            "u_projection", "u_color", "u_position", "u_pos_min", "u_pos_max", "u_scale",
            "u_closeness_range", "u_max_y", "u_show_depth", "u_level_bias", "u_cell_h",
            "u_height_tint", "u_cut_y", "u_cut_center", "u_cut_radius", "u_albedo", "u_normal",
            "u_has_albedo", "u_has_normal", "u_cell_w", *LIGHT_UNIFORMS)}

    # --- Qué chunks se ven ---

    def visible_chunks(self, camera: Camera):
        """Chunks del mapa que pueden tener algo dentro de la vista (teniendo en cuenta las plantas)."""
        g, v = self.geo, self.view
        left, top, right, bottom = camera.bounds()
        # Una celda de la fila y en la planta L se dibuja entre (y+1)·row_h − alto − L·level_h y (y+1)·row_h − L·level_h
        rise_max = (v.level_max + 1.25) * g.level_h
        rise_min = v.level_min * g.level_h
        row_lo = int(np.floor((top + rise_min) / g.row_h)) - 1
        row_hi = int(np.floor((bottom + rise_max) / g.row_h)) + 1
        col_lo, col_hi = int(np.floor(left / g.tile_w)), int(np.floor(right / g.tile_w))
        cx = range(max(col_lo, v.x0) >> v.shift, (min(col_hi, v.x0 + v.width - 1) >> v.shift) + 1)
        cy = range(max(row_lo, v.y0) >> v.shift, (min(row_hi, v.y0 + v.depth - 1) >> v.shift) + 1)
        return [(x, y) for y in cy for x in cx]

    def update(self, camera: Camera, keep_margin: int = 2):
        visible = self.visible_chunks(camera)
        for key in visible:
            if key not in self.meshes:
                chunk = self.view.chunk(*key)
                self.meshes[key] = ChunkMesh(chunk_quads(chunk, self.geo, self.layout) if chunk else {})
        if visible:                                       # liberar mallas lejanas
            xs, ys = [k[0] for k in visible], [k[1] for k in visible]
            for key in [k for k in self.meshes if not (min(xs) - keep_margin <= k[0] <= max(xs) + keep_margin
                                                       and min(ys) - keep_margin <= k[1] <= max(ys) + keep_margin)]:
                self.meshes.pop(key).delete()
        return visible

    # --- Dibujo ---

    def draw(self, camera: Camera, framebuffer_size):
        visible = self.update(camera)
        glViewport(0, 0, *framebuffer_size)
        glClearColor(0.07, 0.07, 0.09, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)            # en superficies coincidentes gana la que se dibuja después
        glEnable(GL_BLEND)
        # El alfa del framebuffer se queda opaco (las sombras y transparencias solo oscurecen el
        # color): necesario para capturas y para dibujar en texturas intermedias
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

        g = self.geo
        glUseProgram(self.program)
        glUniformMatrix4fv(self.u["u_projection"], 1, GL_FALSE, camera.projection().T)
        glUniform1i(self.u["u_color"], 0)
        glUniform1i(self.u["u_position"], 1)
        glUniform3f(self.u["u_pos_min"], *g.pos_min)
        glUniform3f(self.u["u_pos_max"], *g.pos_max)
        glUniform2f(self.u["u_scale"], g.sz, g.sy)
        glUniform2f(self.u["u_closeness_range"], g.closeness_min, g.closeness_max)
        glUniform1f(self.u["u_max_y"], (self.max_level + 0.5) * g.cell_h)
        glUniform1i(self.u["u_show_depth"], int(self.show_depth))
        glUniform1f(self.u["u_level_bias"], LEVEL_BIAS)
        glUniform1f(self.u["u_cell_h"], g.cell_h)
        glUniform1i(self.u["u_height_tint"], int(self.height_tint))
        cut_y = 1e9 if self.cut_level is None else (self.cut_level + 0.5) * g.cell_h
        glUniform1f(self.u["u_cut_y"], cut_y)
        glUniform2f(self.u["u_cut_center"], *self.cut_center)
        glUniform1f(self.u["u_cut_radius"], self.cut_radius)
        glUniform1i(self.u["u_albedo"], 2)
        glUniform1i(self.u["u_normal"], 3)
        glUniform1f(self.u["u_cell_w"], g.cell_w)
        self.set_light_uniforms(self.u, camera)

        self.draw_calls = 0
        for key in visible:                               # de norte a sur: mejor mezcla de transparencias
            for page, vao, _vbo, _ebo, count in self.meshes[key].parts:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self.atlas.color[page])
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, self.atlas.position[page])
                has_albedo, has_normal = page in self.atlas.albedo, page in self.atlas.normal
                glUniform1i(self.u["u_has_albedo"], int(has_albedo))
                glUniform1i(self.u["u_has_normal"], int(has_normal))
                if has_albedo:
                    glActiveTexture(GL_TEXTURE2)
                    glBindTexture(GL_TEXTURE_2D, self.atlas.albedo[page])
                if has_normal:
                    glActiveTexture(GL_TEXTURE3)
                    glBindTexture(GL_TEXTURE_2D, self.atlas.normal[page])
                glBindVertexArray(vao)
                glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, None)
                self.draw_calls += 1
        glBindVertexArray(0)
        glUseProgram(0)

    # --- Luces ---

    def active_lights(self, camera: Camera) -> np.ndarray:
        """Hasta MAX_LIGHTS luces (filas x, y, z, r, g, b, radio, intensidad), las más cercanas a la vista."""
        rows = [self.world_lights] + [np.asarray(self.dynamic_lights, np.float32).reshape(-1, 8)]
        lights = np.concatenate(rows) if any(len(r) for r in rows) else np.zeros((0, 8), np.float32)
        if len(lights) > MAX_LIGHTS:
            g = self.geo
            l, t, r, b = camera.bounds()
            cx, cy = (l + r) / 2, (t + b) / 2
            sx = lights[:, 0] * g.tile_w / g.cell_w
            sy = lights[:, 2] * g.sz - lights[:, 1] * g.sy
            order = np.argsort((sx - cx) ** 2 + (sy - cy) ** 2)
            lights = lights[order[:MAX_LIGHTS]]
        return lights

    def set_light_uniforms(self, u, camera: Camera):
        """Uniforms de iluminación, comunes a los tiles y a los sprites."""
        L = self.lighting
        glUniform1i(u["u_lighting"], int(L.enabled))
        glUniform3f(u["u_sky"], *L.sky)
        glUniform3f(u["u_ground"], *L.ground)
        sun = np.asarray(L.sun_dir, float)
        glUniform3f(u["u_sun_dir"], *(sun / np.linalg.norm(sun)))
        glUniform3f(u["u_sun_color"], *L.sun_color)
        glUniform1f(u["u_lamps"], L.lamps)
        lights = self.active_lights(camera)
        glUniform1i(u["u_num_lights"], len(lights))
        if len(lights):
            pos = np.concatenate([lights[:, 0:3], lights[:, 6:7]], axis=1).astype(np.float32)
            col = (lights[:, 3:6] * lights[:, 7:8]).astype(np.float32)
            glUniform4fv(u["u_light_pos[0]"], len(lights), pos)
            glUniform3fv(u["u_light_col[0]"], len(lights), col)

    def delete(self):
        for mesh in self.meshes.values():
            mesh.delete()
        self.meshes.clear()
        self.atlas.delete()
