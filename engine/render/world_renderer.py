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
    glDepthFunc, glEnable, glEnableVertexAttribArray, glGenBuffers,
    glGenTextures, glGenVertexArrays, glGetUniformLocation, glTexImage2D, glTexParameteri,
    glUniform1f, glUniform1i, glUniform2f, glUniform2i, glUniform3f, glUniform3fv, glUniformMatrix4fv, glUseProgram,
    glVertexAttribPointer, glViewport,
)
from OpenGL.GL import GL_COLOR, GLfloat, glClearBufferfv
from OpenGL.GL import (
    GL_UNPACK_ALIGNMENT, glPixelStorei,
    GL_NEAREST_MIPMAP_NEAREST, GL_ONE, GL_RGBA32F, GL_TEXTURE_BASE_LEVEL, GL_TEXTURE_MAX_LEVEL,
    glBlendFuncSeparate, glTexSubImage2D,
)
from OpenGL.raw.GL.VERSION.GL_1_1 import glDrawElements as _raw_draw_elements
from OpenGL.raw.GL.VERSION.GL_2_0 import glUniform1i as _raw_uniform1i
from OpenGL.raw.GL.VERSION.GL_2_0 import glUniform1iv as _raw_uniform1iv
from OpenGL.raw.GL.VERSION.GL_3_0 import glBindVertexArray as _raw_bind_vertex_array
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image

from engine import png16
from engine.world.view import WorldView

from .geometry import FLOATS_PER_VERTEX, AtlasLayout, WorldGeometry, chunk_quads
from .shaders import LEVEL_BIAS, MAX_CHUNK_LIGHTS, MAX_LIGHTS, TILE_FRAGMENT, TILE_VERTEX


def ortho(left, right, bottom, top) -> np.ndarray:
    m = np.identity(4, np.float32)
    m[0, 0], m[1, 1], m[2, 2] = 2 / (right - left), 2 / (top - bottom), -1
    m[0, 3], m[1, 3] = -(right + left) / (right - left), -(top + bottom) / (top - bottom)
    return m


LIGHT_UNIFORMS = ("u_lighting", "u_sky", "u_ground", "u_sun_dir", "u_sun_color", "u_num_lights",
                  "u_use_list", "u_light_list[0]", "u_lights", "u_lamps", "u_view_dir", "u_time",
                  "u_pulse_period", "u_glow_strength")
SURFACE_TEXTURE_UNIT = 6
PULSE_PERIOD = 2.0          # s: un latido del brillo con pulso
GLOW_STRENGTH = 1.6         # cuánto suma el brillo propio al color
LIGHT_TEXTURE_UNIT = 4
FACADE_UNIFORMS = ("u_facade", "u_cut_mask", "u_mask_origin", "u_mask_size", "u_mask_level",
                   "u_mask_floor", "u_cut_feet", "u_cut_keep", "u_facade_cell_w")
MASK_TEXTURE_UNIT = 5


def set_facade_uniforms(u, facade):
    """Corte de fachada: ``facade`` = dict(texture, origin, size, level, floor, feet, keep, cell_w)
    o None (sin corte)."""
    if facade is None:
        glUniform1i(u["u_facade"], 0)
        return
    glUniform1i(u["u_facade"], 1)
    glActiveTexture(GL_TEXTURE0 + MASK_TEXTURE_UNIT)
    glBindTexture(GL_TEXTURE_2D, facade["texture"])
    glUniform1i(u["u_cut_mask"], MASK_TEXTURE_UNIT)
    glUniform2i(u["u_mask_origin"], *facade["origin"])
    glUniform2i(u["u_mask_size"], *facade["size"])
    glUniform1i(u["u_mask_level"], int(facade["level"]))
    glUniform1i(u["u_mask_floor"], int(facade["floor"]))
    glUniform1f(u["u_cut_feet"], float(facade["feet"]))
    glUniform1f(u["u_cut_keep"], float(facade["keep"]))
    glUniform1f(u["u_facade_cell_w"], float(facade["cell_w"]))
    glActiveTexture(GL_TEXTURE0)


@dataclass
class Lighting:
    """Ambiente de cielo y suelo, sol y si la iluminación está activa. Colores lineales."""
    enabled: bool = True
    sky: tuple = (0.34, 0.38, 0.47)
    ground: tuple = (0.17, 0.15, 0.13)
    sun_dir: tuple = (-0.62, 0.60, 0.50)       # hacia el sol: al suroeste y no muy alto, para que
    sun_color: tuple = (1.00, 0.93, 0.80)      # el relieve se lea (las laderas al sol se aclaran)
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


MAX_MIP_LEVELS = 4


def _texture(levels, sixteen_bits: bool):
    """Textura con sus niveles (el 0, completo, y los reducidos), muestreo sin interpolar."""
    levels = levels if isinstance(levels, list) else [levels]
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    min_filter = GL_NEAREST_MIPMAP_NEAREST if len(levels) > 1 else GL_NEAREST
    for p, v in ((GL_TEXTURE_MIN_FILTER, min_filter), (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                 (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                 (GL_TEXTURE_BASE_LEVEL, 0), (GL_TEXTURE_MAX_LEVEL, len(levels) - 1)):
        glTexParameteri(GL_TEXTURE_2D, p, v)
    for i, data in enumerate(levels):
        h, w = data.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, i, GL_RGBA16 if sixteen_bits else GL_RGBA8, w, h, 0, GL_RGBA,
                     GL_UNSIGNED_SHORT if sixteen_bits else GL_UNSIGNED_BYTE, np.ascontiguousarray(data))
    return tex


UP_TOLERANCE = 0.15


def mip_chains(maps: Dict[str, np.ndarray], levels: int) -> Dict[str, list]:
    """Niveles reducidos (a la mitad cada vez) del color, el albedo y las normales de una página.

    Cada texel reducido promedia los texels opacos de su bloque de 2 × 2 que miran más hacia
    arriba (componente vertical de la normal dentro de UP_TOLERANCE de la mayor del bloque). En
    esta vista oblicua, las caras verticales de un tile suelen quedar tapadas por el tile de
    delante (por ejemplo, la cara sur de un tile de terreno con base elevada), y las superiores
    casi siempre se ven: un promedio de todo el bloque, o tomar cualquiera de sus texels, mete
    el color de una cara oculta en la arista visible y la oscurece (trazos a lo largo de las
    filas de tiles). Sin mapa de normales, se promedian todos los opacos. Las normales,
    renormalizadas. El mapa de posición no tiene niveles: de él salen la profundidad y la
    silueta exactas de cada píxel. El shader lee el texel reducido como (texel completo >> nivel):
    un píxel opaco cae siempre en un bloque que lo contiene, así que nunca lee uno vacío."""
    out = {k: [v] for k, v in maps.items() if k in ("color", "albedo", "normal", "surface")}

    def corners(x):
        return (x[0::2, 0::2], x[1::2, 0::2], x[0::2, 1::2], x[1::2, 1::2])
    for _ in range(levels):
        base = out["color"][-1]
        h, w = base.shape[0] // 2, base.shape[1] // 2
        if h == 0 or w == 0:
            break
        opaque = [c > 0 for c in corners(base[:2 * h, :2 * w, 3])]
        if "normal" in out:
            ny = [c.astype(np.float32) / 255.0 * 2.0 - 1.0 for c in corners(out["normal"][-1][:2 * h, :2 * w, 1])]
            top = np.maximum.reduce([np.where(o, y, -9.0) for o, y in zip(opaque, ny)])
            weight = [(o & (y >= top - UP_TOLERANCE)).astype(np.float32) for o, y in zip(opaque, ny)]
        else:
            weight = [o.astype(np.float32) for o in opaque]
        total = weight[0] + weight[1] + weight[2] + weight[3]
        any_opaque = total > 0
        for kind, chain in out.items():
            if kind == "surface":            # los cuatro canales son datos (A = pulso, no opacidad)
                parts = corners(chain[-1][:2 * h, :2 * w, :4].astype(np.float32))
                mean = sum(p * wgt[..., None] for p, wgt in zip(parts, weight)) / np.maximum(total, 1.0)[..., None]
                chain.append(np.clip(np.rint(mean), 0, 255).astype(np.uint8))
                continue
            parts = corners(chain[-1][:2 * h, :2 * w, :3].astype(np.float32))
            if kind == "normal":
                parts = [p / 255.0 * 2.0 - 1.0 for p in parts]
            mean = sum(p * wgt[..., None] for p, wgt in zip(parts, weight)) / np.maximum(total, 1.0)[..., None]
            if kind == "normal":
                mean = mean / np.maximum(np.linalg.norm(mean, axis=-1, keepdims=True), 1e-6)
                mean = (mean + 1.0) / 2.0 * 255.0
            level = np.empty((h, w, 4), np.uint8)
            level[..., :3] = np.clip(np.rint(mean), 0, 255)
            level[..., 3] = np.where(any_opaque, 255, 0)
            chain.append(level)
    return out


class GpuAtlas:
    """Páginas del atlas en la GPU: color y posición (siempre); albedo y normales (si las hay).

    Con regiones de borde suficiente (2^n píxeles), el color, el albedo y las normales llevan n
    niveles reducidos para leer de lejos sin saltar por todo el atlas; el mapa de posición, nunca
    (profundidad y silueta exactas). Con borde 1 (mundos antiguos), sin niveles."""

    def __init__(self, store, layout: AtlasLayout):
        self.color, self.position, self.albedo, self.normal, self.surface = {}, {}, {}, {}, {}
        borders = [int(r[6]) for r in store.atlas_regions()] or [1]
        self.levels = min(MAX_MIP_LEVELS, int(np.floor(np.log2(max(1, min(borders))))))
        for page in layout.page_sizes:
            maps = {}
            with Image.open(io.BytesIO(store.atlas_page(page, "color"))) as im:
                maps["color"] = np.asarray(im.convert("RGBA"))
            raw = store.atlas_page(page, "position")
            if raw is None:
                raise ValueError("El mundo no tiene mapa de posición: hace falta para la profundidad por píxel")
            maps["position"] = png16.decode_rgba16(raw)
            for kind in ("albedo", "normal", "surface"):
                raw = store.atlas_page(page, kind)
                if raw is not None:
                    with Image.open(io.BytesIO(raw)) as im:
                        maps[kind] = np.asarray(im.convert("RGBA"))
            chains = mip_chains(maps, self.levels)
            self.color[page] = _texture(chains["color"], False)
            self.position[page] = _texture(maps["position"], True)          # siempre completo
            for kind, dest in (("albedo", self.albedo), ("normal", self.normal), ("surface", self.surface)):
                if kind in chains:
                    dest[page] = _texture(chains[kind], False)

    def delete(self):
        glDeleteTextures([t for d in (self.color, self.position, self.albedo, self.normal, self.surface)
                          for t in d.values()])


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
            for loc, size, off in ((0, 2, 0), (1, 2, 2), (2, 3, 4), (3, 1, 7), (4, 1, 8), (5, 1, 9)):
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
        # Corte de fachada (si no es None, sustituye al de "por encima de cut_level"): ver
        # set_facade_uniforms; la máscara de celdas la sube set_cut_mask
        self.facade = None
        self.mask_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.mask_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # Iluminación: luces del mundo (metros, color lineal) + luces dinámicas (el farol)
        self.lighting = Lighting()
        self.world_lights = np.array(view.store.lights(), dtype=np.float32).reshape(-1, 8)
        self.dynamic_lights = []
        self.draw_calls = 0
        self.timer = None              # GpuTimer opcional: mide el dibujo del mundo
        # Luces de la vista en una textura (fila i: posición y radio; color); se sube cada frame
        self.light_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.light_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 2, MAX_LIGHTS, 0, GL_RGBA, GL_FLOAT, None)
        self.frame_lights = np.zeros((0, 8), np.float32)
        self.chunk_lights = {}
        self.per_chunk_lights = True   # False: cada tile recorre todas las luces de la vista (depurar)
        self.use_material_maps = True  # False: sin leer albedo ni normales (diagnóstico, tecla M)
        self.time = 0.0                # s: para el pulso del brillo propio
        self.use_surface = True        # False: sin capa de superficie (comparar, diagnóstico)
        self.glow_buffer = False       # True: se dibuja para el halo (hay una salida de brillo)
        self.use_mipmaps = True        # False: leer siempre el atlas completo (comparar)
        self.material_tint = False     # True: teñir el suelo según su material (depurar, tecla G)
        self.materials = None          # Materials del mundo (colores del tinte)
        meta_materials = view.store.meta().get("materials") if hasattr(view, "store") else None
        if meta_materials:
            from engine.world.materials import Materials
            self.materials = Materials.from_json(meta_materials)

        self.program = compileProgram(compileShader(TILE_VERTEX, GL_VERTEX_SHADER),
                                      compileShader(TILE_FRAGMENT, GL_FRAGMENT_SHADER))
        self.u = {name: glGetUniformLocation(self.program, name) for name in (
            "u_projection", "u_color", "u_position", "u_pos_min", "u_pos_max", "u_scale",
            "u_closeness_range", "u_max_y", "u_show_depth", "u_level_bias", "u_cell_h",
            "u_height_tint", "u_cut_y", "u_cut_center", "u_cut_radius", "u_albedo", "u_normal",
            "u_has_albedo", "u_has_normal", "u_cell_w", "u_lod", "u_material_tint", "u_surface", "u_has_surface",
            "u_material_colors[0]", *LIGHT_UNIFORMS, *FACADE_UNIFORMS)}

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
        if self.glow_buffer:                       # la salida de brillo, a negro (no al fondo)
            glClearBufferfv(GL_COLOR, 1, (GLfloat * 4)(0.0, 0.0, 0.0, 0.0))
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
        cut_y = 1e9 if self.cut_level is None or self.facade is not None else (self.cut_level + 0.5) * g.cell_h
        glUniform1f(self.u["u_cut_y"], cut_y)
        set_facade_uniforms(self.u, self.facade)
        glUniform2f(self.u["u_cut_center"], *self.cut_center)
        glUniform1f(self.u["u_cut_radius"], self.cut_radius)
        glUniform1f(self.u["u_lod"], self.lod(camera))
        tint = self.material_tint and self.materials is not None
        glUniform1i(self.u["u_material_tint"], int(tint))
        if tint:
            glUniform3fv(self.u["u_material_colors[0]"], 16, self.materials.colors.astype(np.float32))
        glUniform1i(self.u["u_albedo"], 2)
        glUniform1i(self.u["u_surface"], SURFACE_TEXTURE_UNIT)
        glUniform1i(self.u["u_normal"], 3)
        glUniform1f(self.u["u_cell_w"], g.cell_w)
        self.upload_lights(camera, visible)
        self.set_light_uniforms(self.u, camera)
        glUniform1i(self.u["u_use_list"], int(self.per_chunk_lights))   # los tiles: las de su chunk

        # Texturas y uniforms una vez por página del atlas; por chunk, solo su malla y el dibujo.
        # Dentro de cada página, los chunks van de norte a sur (mejor mezcla de transparencias).
        self.draw_calls = 0
        if self.timer is not None:
            with self.timer("mundo"):
                self._draw_pages(visible)
        else:
            self._draw_pages(visible)
        glBindVertexArray(0)
        glUseProgram(0)

    def _draw_pages(self, visible):
        by_page = {}
        for key in visible:
            for page, vao, _vbo, _ebo, count in self.meshes[key].parts:
                by_page.setdefault(page, []).append((vao, count, key))
        loc_num, loc_list, loc_use = self.u["u_num_lights"], self.u["u_light_list[0]"], self.u["u_use_list"]
        no_lights = (0, None)
        all_lights = len(self.frame_lights)
        for page, parts in by_page.items():
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.atlas.color[page])
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.atlas.position[page])
            maps = self.use_material_maps
            has_albedo, has_normal = maps and page in self.atlas.albedo, maps and page in self.atlas.normal
            glUniform1i(self.u["u_has_albedo"], int(has_albedo))
            glUniform1i(self.u["u_has_normal"], int(has_normal))
            if has_albedo:
                glActiveTexture(GL_TEXTURE2)
                glBindTexture(GL_TEXTURE_2D, self.atlas.albedo[page])
            if has_normal:
                glActiveTexture(GL_TEXTURE3)
                glBindTexture(GL_TEXTURE_2D, self.atlas.normal[page])
            has_surface = maps and self.use_surface and page in self.atlas.surface
            glUniform1i(self.u["u_has_surface"], int(has_surface))
            if has_surface:
                glActiveTexture(GL_TEXTURE0 + SURFACE_TEXTURE_UNIT)
                glBindTexture(GL_TEXTURE_2D, self.atlas.surface[page])
            # El bucle que crece con el zoom: funciones sin envoltorio (directas a OpenGL)
            for vao, count, key in parts:
                if self.per_chunk_lights:
                    entry = self.chunk_lights.get(key, no_lights)
                    if entry is None:                      # demasiadas: todas las de la vista
                        _raw_uniform1i(loc_use, 0)
                        _raw_uniform1i(loc_num, all_lights)
                    else:
                        n, indices = entry
                        _raw_uniform1i(loc_use, 1)
                        _raw_uniform1i(loc_num, n)
                        if n:
                            _raw_uniform1iv(loc_list, n, indices)
                _raw_bind_vertex_array(int(vao))
                _raw_draw_elements(GL_TRIANGLES, count, GL_UNSIGNED_INT, None)
                self.draw_calls += 1

    # --- Luces ---

    def active_lights(self, camera: Camera) -> np.ndarray:
        """Luces (filas x, y, z, r, g, b, radio, intensidad) que pueden alumbrar algo de la vista;
        si hay más de MAX_LIGHTS, las más cercanas al centro."""
        rows = [self.world_lights] + [np.asarray(self.dynamic_lights, np.float32).reshape(-1, 8)]
        lights = np.concatenate(rows) if any(len(r) for r in rows) else np.zeros((0, 8), np.float32)
        if not len(lights):
            return lights
        g = self.geo
        l, t, r, b = camera.bounds()
        sx = lights[:, 0] * g.tile_w / g.cell_w
        sy = lights[:, 2] * g.sz - lights[:, 1] * g.sy
        reach = lights[:, 6] * g.tile_w / g.cell_w + g.level_h            # radio en px y algo de altura
        inside = (sx > l - reach) & (sx < r + reach) & (sy > t - reach) & (sy < b + reach + g.level_h)
        lights, sx, sy = lights[inside], sx[inside], sy[inside]
        if len(lights) > MAX_LIGHTS:
            order = np.argsort((sx - (l + r) / 2) ** 2 + (sy - (t + b) / 2) ** 2)
            lights = lights[order[:MAX_LIGHTS]]
        return lights

    def set_cut_mask(self, mask: np.ndarray, origin, level: int, feet: float, keep: float, floor: int = None):
        """Sube la máscara de corte (alto, ancho, 4) uint8, canales = plantas level .. level + 3:
        0 se dibuja, 128 zócalo (lo que esté a más de ``keep`` sobre ``feet``), 255 se oculta.
        ``floor``: planta del suelo que se conserva (por defecto, ``level``)."""
        mask = np.ascontiguousarray(mask, np.uint8)
        glActiveTexture(GL_TEXTURE0 + MASK_TEXTURE_UNIT)
        glBindTexture(GL_TEXTURE_2D, self.mask_texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mask.shape[1], mask.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, mask)
        g = self.geo
        self.facade_state = {"texture": self.mask_texture, "origin": (int(origin[0]), int(origin[1])),
                       "size": (int(mask.shape[1]), int(mask.shape[0])), "level": int(level),
                       "floor": int(level if floor is None else floor),
                       "feet": float(feet), "keep": float(keep), "cell_w": g.cell_w}
        self.facade = self.facade_state

    @property
    def has_glow(self) -> bool:
        """¿Tiene el mundo capa de superficie (y por tanto, quizá, brillo propio)?"""
        return bool(self.atlas.surface)

    def lod(self, camera: Camera) -> float:
        """Nivel del atlas que se lee a este zoom: el que da como mucho dos texels por píxel."""
        if not self.use_mipmaps or self.atlas.levels == 0 or camera.zoom >= 0.5:
            return 0.0
        return float(min(self.atlas.levels, int(np.floor(np.log2(1.0 / camera.zoom)))))

    def chunk_light_lists(self, keys, lights: np.ndarray) -> dict:
        """Para cada chunk, las luces (índices en ``lights``) cuyo radio alcanza su caja en planta.
        Si son más de MAX_CHUNK_LIGHTS, el chunk se marca con None: recorrerá todas las de la vista
        (más lento, pero ninguna luz se pierde)."""
        if not len(keys) or not len(lights):
            return {}
        g, v = self.geo, self.view
        span = v.chunk_size * g.cell_w
        k = np.asarray(keys, float).reshape(-1, 2)
        x0, z0 = k[:, 0:1] * span, k[:, 1:2] * span
        lx, lz, lr = lights[None, :, 0], lights[None, :, 2], lights[None, :, 6]
        dx = np.maximum(np.maximum(x0 - lx, lx - (x0 + span)), 0.0)
        dz = np.maximum(np.maximum(z0 - lz, lz - (z0 + span)), 0.0)
        hits = dx * dx + dz * dz < lr * lr
        out = {}
        for i, key in enumerate(keys):
            idx = np.nonzero(hits[i])[0]
            if idx.size > MAX_CHUNK_LIGHTS:
                out[key] = None
            elif idx.size:
                out[key] = (int(idx.size), np.ascontiguousarray(idx, np.int32))
        return out

    def upload_lights(self, camera: Camera, visible):
        """Sube a la textura las luces de la vista y prepara la lista de cada chunk visible."""
        lights = self.active_lights(camera) if self.lighting.enabled else np.zeros((0, 8), np.float32)
        self.frame_lights = lights
        self.chunk_lights = self.chunk_light_lists(list(visible), lights)
        if len(lights):
            data = np.zeros((len(lights), 2, 4), np.float32)
            data[:, 0, :3], data[:, 0, 3] = lights[:, 0:3], lights[:, 6]
            data[:, 1, :3] = lights[:, 3:6] * lights[:, 7:8]
            # Su propia unidad: enlazarla en la que esté activa pisaría otra textura (la máscara)
            glActiveTexture(GL_TEXTURE0 + LIGHT_TEXTURE_UNIT)
            glBindTexture(GL_TEXTURE_2D, self.light_texture)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2, len(lights), GL_RGBA, GL_FLOAT, data)

    def set_light_uniforms(self, u, camera: Camera):
        """Uniforms de iluminación, comunes a los tiles y a los sprites. Por defecto se recorren
        todas las luces de la vista (los sprites); los tiles pasan a la lista de su chunk."""
        L = self.lighting
        glUniform1i(u["u_lighting"], int(L.enabled))
        glUniform3f(u["u_sky"], *L.sky)
        glUniform3f(u["u_ground"], *L.ground)
        sun = np.asarray(L.sun_dir, float)
        glUniform3f(u["u_sun_dir"], *(sun / np.linalg.norm(sun)))
        glUniform3f(u["u_sun_color"], *L.sun_color)
        glUniform1f(u["u_lamps"], L.lamps)
        g = self.geo
        view = np.array([0.0, g.sz, g.sy])                     # hacia la cámara (vista oblicua)
        glUniform3f(u["u_view_dir"], *(view / np.linalg.norm(view)))
        glUniform1f(u["u_time"], float(self.time))
        glUniform1f(u["u_pulse_period"], PULSE_PERIOD)
        glUniform1f(u["u_glow_strength"], GLOW_STRENGTH)
        glActiveTexture(GL_TEXTURE0 + LIGHT_TEXTURE_UNIT)
        glBindTexture(GL_TEXTURE_2D, self.light_texture)
        glUniform1i(u["u_lights"], LIGHT_TEXTURE_UNIT)
        glUniform1i(u["u_use_list"], 0)
        glUniform1i(u["u_num_lights"], len(self.frame_lights))

    def delete(self):
        for mesh in self.meshes.values():
            mesh.delete()
        self.meshes.clear()
        self.atlas.delete()
        glDeleteTextures(1, [self.light_texture])
        glDeleteTextures(1, [self.mask_texture])
