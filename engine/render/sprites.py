"""Sprites de personaje (planos verticales en los pies) y sus sombras en el suelo.

Profundidad por píxel con la misma cercanía que los tiles: un sprite es un plano vertical
en el Z de los pies, así que cada píxel tiene altura Y = pies + (lo alto que esté en el
sprite) y cercanía = Y · sz + Z · sy. La sombra es un plano horizontal a la altura del
suelo bajo los pies, con Z variando a lo largo del quad.
"""

import ctypes
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER, GL_TEXTURE0,
    GL_TEXTURE_2D, GL_TRIANGLES, GL_TRUE, GL_VERTEX_SHADER, glActiveTexture, glBindBuffer,
    glBindTexture, glBindVertexArray, glBufferData, glDeleteBuffers, glDeleteTextures,
    GL_GREATER, GL_LEQUAL, glDeleteVertexArrays, glDepthFunc, glDepthMask, glDrawArrays, glEnableVertexAttribArray, glGenBuffers,
    glGenVertexArrays, glGetUniformLocation, glUniform1f, glUniform1i, glUniform2f,
    glUniformMatrix4fv, glUseProgram, glVertexAttribPointer,
)
from OpenGL.GL import glUniform3f
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
layout(location = 6) in vec3 a_feet;     // (Y, Z) de los pies en metros y escala del modelo
layout(location = 7) in float a_radius;  // radio del cilindro de física (m)
uniform mat4 u_projection;
out vec2 v_uv;
out vec2 v_world;
out vec2 v_pos;
out vec3 v_tint;
out float v_bias;
out float v_level;
out vec3 v_feet;
out float v_radius;
void main() {
    v_uv = a_uv;
    v_feet = a_feet;
    v_radius = a_radius;
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
in vec3 v_feet;
in float v_radius;
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
uniform vec3 u_silhouette_color;
uniform sampler2D u_sheet_surface;
uniform int u_has_surface;
uniform int u_has_maps;            // la hoja trae albedo, normales y posición por píxel
uniform sampler2D u_sheet_albedo;
uniform sampler2D u_sheet_normal;
uniform sampler2D u_sheet_position;
uniform vec3 u_pos_min;            // rango en metros de la posición (respecto a los pies)
uniform vec3 u_pos_max;
const float SILHOUETTE_ALPHA = 0.55;
const float SPRITE_WRAP = 0.2;     // luz que "envuelve" un poco el borde de sombra
// Márgenes de profundidad (m a lo largo de la vista). El personaje gana los roces de menos de
// TOUCH_MARGIN con el escenario (un pie en una rampa, una mano contra una pared); lo que
// está delante de su cilindro queda siempre a más de ~0.25 m, así que no se ve a través de nada.
// La silueta solo sale donde algo lo tapa más de SILHOUETTE_MARGIN.
const float TOUCH_MARGIN = 0.12;
const float SILHOUETTE_MARGIN = 0.25;
""" + LIGHTING_GLSL + FACADE_GLSL + """
layout(location = 0) out vec4 frag;
layout(location = 1) out vec4 glow;    // los personajes tapan el brillo de lo que hay detrás
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
        if (u_shadow == 2) {
            // Silueta: lo que queda tapado del personaje (se dibuja con la prueba de profundidad
            // invertida), en un color plano semitransparente
            if (color.a < 0.5) discard;
            color = vec4(u_silhouette_color, SILHOUETTE_ALPHA);
        } else {
            color.rgb *= v_tint;
        }
    }
    // Altura y fondo del píxel: con mapa de posición, su punto 3D (los pies más lo que diga el
    // mapa); sin él, el plano vertical en los pies
    vec2 yz = v_world;
    bool maps = u_has_maps == 1 && u_shadow != 1;
    if (maps) {
        vec3 offset = mix(u_pos_min, u_pos_max, texture(u_sheet_position, v_uv).xyz) * v_feet.z;
        yz = v_feet.xy + offset.yz;
    }
    float closeness = yz.x * u_scale.x + yz.y * u_scale.y + v_bias;
    float ray = length(u_scale);           // cercanía por metro a lo largo de la vista
    if (maps) {
        // Nada del escenario entra en el cilindro de física: lo que quede detrás de su cara
        // norte (el fusil, la mochila contra una pared) se acerca hasta ella por su rayo
        float back = v_feet.y - v_radius;
        if (yz.y < back) closeness += (back - yz.y) * ray * ray / u_scale.y;
    }
    if (u_shadow != 1) closeness += TOUCH_MARGIN * ray;
    if (u_shadow == 2) closeness += SILHOUETTE_MARGIN * ray;
    float depth = clamp(1.0 - (closeness - u_closeness_range.x)
                        / (u_closeness_range.y - u_closeness_range.x), 0.0, 1.0);
    gl_FragDepth = depth;
    vec3 emitted = vec3(0.0);
    if (u_shadow == 0 && u_has_surface == 1) {
        vec4 surf = texture(u_sheet_surface, v_uv);
        emitted = to_linear(color.rgb) * surf.b * pulse(surf.a) * u_glow_strength;
    }
    if (u_lighting == 1 && u_shadow == 0 && maps) {
        // Con normales y albedo: se ilumina cada píxel en su punto, como un tile (con metal y
        // pulido si la hoja trae capa de superficie)
        vec3 world = vec3(v_pos.x * u_m_per_px, yz.x, yz.y);
        vec3 n = normalize(texture(u_sheet_normal, v_uv).rgb * 2.0 - 1.0);
        vec3 lin = to_linear(texture(u_sheet_albedo, v_uv).rgb * v_tint);
        vec4 surf = (u_has_surface == 1) ? texture(u_sheet_surface, v_uv) : vec4(0.0, 1.0, 0.0, 0.0);
        float metal = surf.r, rough = max(surf.g, 0.05);
        if (metal > 0.5 / 255.0 || rough < 254.5 / 255.0) {
            vec3 f0 = mix(vec3(0.04), lin, metal);
            vec3 spec;
            vec3 light = light_and_specular(world, n, SPRITE_WRAP, u_view_dir, rough, f0, spec);
            color.rgb = to_srgb(lin * (1.0 - metal * (1.0 - METAL_DIFFUSE)) * light + spec
                                + sky_reflection(n, u_view_dir, rough, f0) + emitted);
        } else {
            color.rgb = to_srgb(lin * light_at(world, n, SPRITE_WRAP) + emitted);
        }
    } else if (u_lighting == 1 && u_shadow == 0) {
        // El sprite, de cara a la cámara (sin normales): luz "envolvente" para que también le
        // alumbre una luz que esté a un lado
        vec3 world = vec3(v_pos.x * u_m_per_px, v_world.x, v_world.y);
        vec3 n = normalize(vec3(0.0, 0.35, 1.0));
        vec3 lin = to_linear(color.rgb);
        if (u_has_surface == 1) {
            vec4 surf = texture(u_sheet_surface, v_uv);
            float metal = surf.r, rough = max(surf.g, 0.05);
            vec3 f0 = mix(vec3(0.04), lin, metal);
            vec3 lit;
            if (metal > 0.5 / 255.0 || rough < 254.5 / 255.0) {
                vec3 spec;
                vec3 light = light_and_specular(world, n, 0.6, u_view_dir, rough, f0, spec);
                lit = lin * (1.0 - metal * (1.0 - METAL_DIFFUSE)) * light + spec
                      + sky_reflection(n, u_view_dir, rough, f0);
            } else {
                lit = lin * light_at(world, n, 0.6);
            }
            color.rgb = to_srgb(lit + emitted);
        } else {
            color.rgb = to_srgb(lin * light_at(world, n, 0.6));
        }
    } else if (u_shadow == 0 && u_has_surface == 1) {
        color.rgb = to_srgb(to_linear(color.rgb) + emitted);
    }
    frag = (u_show_depth == 1 && u_shadow == 0) ? vec4(vec3(1.0 - depth), 1.0) : color;
    glow = vec4(emitted, color.a);
}
"""

FLOATS = 15          # pos (2), uv (2), mundo Y Z (2), tinte (3), desempate (1), planta (1), pies (3), radio (1)
WALK_FPS = 8.0


# Direcciones: índice 0 = este y sentido horario en pantalla (este, sur, oeste, norte con 4;
# este, sureste, sur… con 8). Cada hoja dice qué fila tiene cada dirección.
ROWS_4 = {0: 2, 1: 0, 2: 1, 3: 3}          # hojas 4 × 4: filas abajo, izquierda, derecha, arriba
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
    directions = np.asarray(directions)                     # una para todos, o una por personaje
    sector = 360.0 / directions
    angle = np.degrees(np.arctan2(dz, dx)) % 360.0
    off = np.abs((angle - previous * sector + 180.0) % 360.0 - 180.0)
    keep = (dx * dx + dz * dz < 1e-6) | (off <= sector / 2 + hysteresis_deg)
    return np.where(keep, previous, np.floor(angle / sector + 0.5).astype(int) % directions)


SHEET_SURFACE_UNIT = 7                        # capa de superficie de la hoja (unidad de textura)
SHEET_ALBEDO_UNIT, SHEET_NORMAL_UNIT, SHEET_POSITION_UNIT = 8, 9, 10
RUN_FROM = 0.5      # se corre a partir de este punto entre la velocidad de andar y la de correr
SILHOUETTE_COLOR = (0.55, 0.85, 1.0)          # azul claro: se distingue sobre el escenario


def read_layout(path) -> dict:
    """Disposición de una hoja, sin OpenGL: la de su ``.json`` (la escriben ``tools.actors`` o el
    grafista) o, si no lo tiene, la de las hojas 4 × 4 de siempre. Claves de ``SpriteSheet``."""
    from pathlib import Path
    meta_path = Path(path).with_suffix(".json")
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    anims = meta.get("animations", {})
    idle = anims.get("quieto", {"cols": [0]})["cols"]
    walk = anims.get("andar")
    run = anims.get("correr")
    layout = {
        "animations": {name: (tuple(a["cols"]), float(a.get("fps", WALK_FPS)), bool(a.get("loop", True)))
                       for name, a in anims.items()},
        "cols": int(meta["cols"]), "rows": int(meta["rows"]),
        "directions": int(meta.get("directions", 4)),
        "direction_rows": tuple(meta.get("direction_rows", (2, 0, 1, 3))),
        "idle_col": int(idle[0]),
    }
    if walk:
        layout["walk_cycle"] = tuple(walk["cols"])
        layout["walk_fps"] = float(walk.get("fps", WALK_FPS))
        layout["walk_speed"] = float(walk.get("speed", 1.4))
    if run:
        layout["run_cycle"] = tuple(run["cols"])
        layout["run_fps"] = float(run.get("fps", WALK_FPS))
        layout["run_speed"] = float(run.get("speed", 4.0))
    if "anchor" in meta:
        layout["anchor"] = tuple(float(v) for v in meta["anchor"])
        layout["px_per_m"] = float(meta["px_per_m"])
        layout["model_height"] = float(meta["height"])
    if "position_range" in meta:
        lo, hi = zip(*meta["position_range"])
        layout["pos_min"], layout["pos_max"] = tuple(map(float, lo)), tuple(map(float, hi))
    return layout


@dataclass
class SpriteSheet:
    """Hoja de frames: una fila por dirección y, en cada fila, quieto y los ciclos de andar (y
    de correr, si lo tiene). Sin ``.json``, la de siempre (4 × 4): filas abajo, izquierda,
    derecha, arriba, y un ciclo de andar de cuatro frames: de pie, paso, de pie, el otro paso.

    Con ``anchor`` (píxel de los pies en el fotograma) y ``px_per_m``, el sprite se dibuja a la
    escala con la que se generó, multiplicada por (altura del personaje / ``model_height``);
    sin ellos, el fotograma entero mide lo que el personaje, con los pies abajo."""
    texture: int
    width: int
    height: int
    cols: int = 4
    rows: int = 4
    idle_col: int = 0
    walk_cycle: tuple = (1, 2, 3, 0)      # empieza por un paso, para que se note enseguida
    surface: int = 0                      # textura de la capa de superficie (hoja.s.png), 0 = sin ella
    directions: int = 4
    direction_rows: tuple = (2, 0, 1, 3)  # fila de cada dirección (0 = este, en sentido horario)
    walk_fps: float = WALK_FPS
    walk_speed: float = 1.4               # m/s a los que el ciclo de andar se ve natural
    run_cycle: tuple = ()
    run_fps: float = WALK_FPS
    run_speed: float = 4.0
    anchor: Optional[tuple] = None
    px_per_m: Optional[float] = None
    model_height: Optional[float] = None
    albedo: int = 0                       # hoja.a.png, hoja.n.png, hoja.p.png (0 = sin ellas)
    normal: int = 0
    position: int = 0
    pos_min: tuple = (0.0, 0.0, 0.0)
    pos_max: tuple = (1.0, 1.0, 1.0)
    animations: dict = field(default_factory=dict)   # nombre -> (columnas, fps, se repite)

    def anim_cols(self, name: str, anim_time):
        """Columna de cada personaje en la animación ``name`` tras ``anim_time`` segundos. Las
        que no se repiten (``morir``, ``granada``) se quedan en su último fotograma."""
        cols, fps, loop = self.animations[name]
        k = (np.asarray(anim_time, float) * fps).astype(int)
        k = k % len(cols) if loop else np.minimum(k, len(cols) - 1)
        return np.asarray(cols)[np.maximum(k, 0)]

    def anim_duration(self, name: str) -> float:
        cols, fps, _ = self.animations[name]
        return len(cols) / fps

    @property
    def has_maps(self) -> bool:
        return bool(self.albedo and self.normal and self.position)

    def row_of(self, facing):
        """Fila de la hoja para cada dirección (índices de ``facing_index`` con ``directions``)."""
        return np.asarray(self.direction_rows)[np.asarray(facing, int) % self.directions]

    def frame_cols(self, anim_time, speed=None):
        """Columna de cada personaje: quieto, o el ciclo de andar o de correr según su velocidad."""
        anim_time = np.asarray(anim_time, float)
        walk = np.asarray(self.walk_cycle)
        cols = walk[(anim_time * self.walk_fps).astype(int) % len(walk)]
        if self.run_cycle and speed is not None:
            run = np.asarray(self.run_cycle)
            threshold = self.walk_speed + RUN_FROM * (self.run_speed - self.walk_speed)
            cols = np.where(np.asarray(speed, float) > threshold,
                            run[(anim_time * self.run_fps).astype(int) % len(run)], cols)
        return np.where(anim_time > 0, cols, self.idle_col)

    def walk_cols(self, walk_time, fps: float = None):
        """Versión vectorizada de ``walk_col``."""
        walk_time = np.asarray(walk_time, float)
        cycle = np.asarray(self.walk_cycle)
        fps = self.walk_fps if fps is None else fps
        return np.where(walk_time > 0, cycle[(walk_time * fps).astype(int) % len(cycle)], self.idle_col)

    def walk_col(self, walk_time: float, fps: float = None) -> int:
        """Columna del ciclo de andar tras ``walk_time`` segundos andando (o la de quieto)."""
        if walk_time <= 0:
            return self.idle_col
        fps = self.walk_fps if fps is None else fps
        return self.walk_cycle[int(walk_time * fps) % len(self.walk_cycle)]

    @classmethod
    def load(cls, path):
        """La hoja y, si existen junto a ella: su disposición (``hoja.json``), su capa de
        superficie (``hoja.s.png``: R metal, G rugosidad, B brillo propio, A pulso) y sus mapas
        de albedo, normales y posición (``hoja.a.png``, ``hoja.n.png``, ``hoja.p.png`` de 16
        bits), todos con la misma disposición que la hoja."""
        from pathlib import Path

        from engine import png16
        path = Path(path)
        with Image.open(path) as im:
            data = np.asarray(im.convert("RGBA"))

        def layer(suffix, sixteen=False):
            p = path.with_name(path.stem + suffix)
            if not p.exists():
                return 0
            if sixteen:
                return _texture(png16.read_rgba16(p), True)
            with Image.open(p) as im:
                return _texture(np.asarray(im.convert("RGBA")), False)

        return cls(_texture(data, False), data.shape[1], data.shape[0], surface=layer(".s.png"),
                   albedo=layer(".a.png"), normal=layer(".n.png"), position=layer(".p.png", True),
                   **read_layout(path))

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
            "u_m_per_px", "u_cut_level", "u_cut_center", "u_cut_radius", "u_max_level",
            "u_silhouette_color", "u_sheet_surface", "u_has_surface", "u_has_maps", "u_sheet_albedo",
            "u_sheet_normal", "u_sheet_position", "u_pos_min", "u_pos_max", *LIGHT_UNIFORMS,
            *FACADE_UNIFORMS)}
        self.vao, self.vbo = glGenVertexArrays(1), glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        for loc, size, off in ((0, 2, 0), (1, 2, 2), (2, 2, 4), (3, 3, 6), (4, 1, 9), (5, 1, 10), (6, 3, 11),
                                  (7, 1, 14)):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, FLOATS * 4, ctypes.c_void_p(off * 4))
        glBindVertexArray(0)
        self.draw_calls = 0
        self.timer = None              # GpuTimer opcional: mide sombras y personajes

    # --- Vértices (numpy, sin OpenGL) ---

    def sprite_vertices(self, sheet: SpriteSheet, feet, body_h, rows, cols, tint, level, radius=0.25):
        """(n·6, FLOATS): dos triángulos por personaje, de pie sobre sus pies."""
        g = self.geo
        feet = np.asarray(feet, float).reshape(-1, 3)
        n = len(feet)
        X, Y, Z = feet[:, 0], feet[:, 1], feet[:, 2]
        fw, fh = sheet.frame_size
        body_h = np.asarray(body_h, float) * np.ones(n)
        if sheet.anchor is not None:
            # A la escala con la que se generó (un píxel de la hoja = un píxel de mundo si la
            # proyección es la misma), con los pies en su píxel
            model = body_h / sheet.model_height
            scale = model * (g.tile_w / g.cell_w) / sheet.px_per_m
            ax, ay = sheet.anchor
        else:
            model = np.ones(n)
            scale = body_h * g.sy / fh
            ax, ay = fw / 2, fh
        x = X * g.tile_w / g.cell_w
        s = Z * g.sz - Y * g.sy
        u0 = np.asarray(cols) / sheet.cols
        v0 = np.asarray(rows) / sheet.rows
        u1, v1 = u0 + 1 / sheet.cols, v0 + 1 / sheet.rows
        left, right = x - ax * scale, x + (fw - ax) * scale
        up, down = ay * scale, (fh - ay) * scale            # píxeles por encima y por debajo de los pies
        corners = [  # arriba-izq, arriba-der, abajo-der, abajo-izq
            (left, s - up, u0, v0, Y + up / g.sy), (right, s - up, u1, v0, Y + up / g.sy),
            (right, s + down, u1, v1, Y - down / g.sy), (left, s + down, u0, v1, Y - down / g.sy)]
        bias = LEVEL_BIAS * np.floor(Y / g.cell_h + 1e-6) + 0.05
        quad = np.zeros((n, 4, FLOATS), np.float32)
        for k, (px, py, uu, vv, yy) in enumerate(corners):
            quad[:, k, 0], quad[:, k, 1], quad[:, k, 2], quad[:, k, 3] = px, py, uu, vv
            quad[:, k, 4], quad[:, k, 5] = yy, Z
        quad[:, :, 6:9] = np.asarray(tint, np.float32).reshape(-1, 1, 3)
        quad[:, :, 9] = bias[:, None]
        quad[:, :, 10] = np.asarray(level, np.float32)[:, None]
        quad[:, :, 11], quad[:, :, 12], quad[:, :, 13] = Y[:, None], Z[:, None], model[:, None]
        quad[:, :, 14] = (np.asarray(radius, float) * np.ones(n))[:, None]
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

    def _draw(self, verts, sheet, shadow, camera, show_depth, lights, cut, max_level):
        g = self.geo
        glUseProgram(self.program)
        glUniformMatrix4fv(self.u["u_projection"], 1, GL_FALSE, camera.projection().T)
        glUniform1i(self.u["u_sheet"], 0)
        glUniform1i(self.u["u_shadow"], int(shadow))
        glUniform3f(self.u["u_silhouette_color"], *SILHOUETTE_COLOR)
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
        surface = sheet.surface if sheet is not None else 0
        if sheet is not None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, sheet.texture)
        glUniform1i(self.u["u_has_surface"], int(bool(surface) and shadow == 0))
        if surface:
            glActiveTexture(GL_TEXTURE0 + SHEET_SURFACE_UNIT)
            glBindTexture(GL_TEXTURE_2D, surface)
            glUniform1i(self.u["u_sheet_surface"], SHEET_SURFACE_UNIT)
        maps = sheet is not None and sheet.has_maps
        glUniform1i(self.u["u_has_maps"], int(maps))
        if maps:
            for name, unit, tex in (("u_sheet_albedo", SHEET_ALBEDO_UNIT, sheet.albedo),
                                    ("u_sheet_normal", SHEET_NORMAL_UNIT, sheet.normal),
                                    ("u_sheet_position", SHEET_POSITION_UNIT, sheet.position)):
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, tex)
                glUniform1i(self.u[name], unit)
            glUniform3f(self.u["u_pos_min"], *sheet.pos_min)
            glUniform3f(self.u["u_pos_max"], *sheet.pos_max)
        glActiveTexture(GL_TEXTURE0)
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
                self._draw(sv, None, True, camera, show_depth, lights, cut, max_level)
        glDepthMask(GL_TRUE)
        verts = self.sprite_vertices(sheet, feet, body_h, rows, cols, tint, level, radius)
        with timed("personajes"):
            self._draw(verts, sheet, False, camera, show_depth, lights, cut, max_level)

    def draw_silhouettes(self, sheet: SpriteSheet, feet, body_h, rows, cols, camera, lights=None,
                         cut=None, max_level=1e9, radius=0.25):
        """Siluetas de los personajes donde algo los tapa: se dibujan después del resto, solo en
        los píxeles en los que su profundidad queda detrás de lo ya dibujado."""
        feet = np.asarray(feet, float).reshape(-1, 3)
        if len(feet) == 0:
            return
        level = np.floor(feet[:, 1] / self.geo.cell_h + 1e-6)
        verts = self.sprite_vertices(sheet, feet, body_h, rows, cols, np.ones((len(feet), 3)), level, radius)
        glDepthMask(GL_FALSE)
        glDepthFunc(GL_GREATER)
        self._draw(verts, sheet, 2, camera, False, lights, cut, max_level)
        glDepthFunc(GL_LEQUAL)
        glDepthMask(GL_TRUE)

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
    glDeleteTextures([t for t in (sheet.texture, sheet.surface, sheet.albedo, sheet.normal, sheet.position) if t])
