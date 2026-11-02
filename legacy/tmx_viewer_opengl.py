#!/usr/bin/env python3

"""
TMX Map Explorer - OpenGL Only (Ultra Optimized - Culling 2D + Z-Order)
Versión optimizada con:
- Culling solo en X e Y (todas las alturas Z visibles)
- Renderizado layer por layer en orden: Y (lejos→cerca), X (izq→der), Z (abajo→arriba), N (0→max)
- FIX: Tile bleeding eliminado con NEAREST filtering y sin mipmaps
"""

import sys
import pygame
import numpy as np
from pathlib import Path
from tmx_manager import TiledMap, TileLayer, ObjectGroup, LayerGroup
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import ctypes
from collections import defaultdict

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GRAY = (64, 64, 64)


# ============================================================================
# SHADERS
# ============================================================================

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec4 aColor;
layout (location = 3) in float aDepth;
out vec2 TexCoord;
out vec4 Color;
uniform mat4 projection;
void main() {
    gl_Position = projection * vec4(aPos, aDepth, 1.0);
    TexCoord = aTexCoord;
    Color = aColor;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
in vec4 Color;
out vec4 FragColor;
uniform sampler2D texture0;
void main() {
    vec4 texColor = texture(texture0, TexCoord);
    FragColor = texColor * Color;
    if (FragColor.a < 0.01) discard;
}
"""

SIMPLE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec4 aColor;
out vec4 Color;
uniform mat4 projection;
void main() {
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
    Color = aColor;
}
"""

SIMPLE_FRAGMENT_SHADER = """
#version 330 core
in vec4 Color;
out vec4 FragColor;
void main() { FragColor = Color; }
"""


# ============================================================================
# TEXTURE (FIXED FOR TILE BLEEDING)
# ============================================================================

class Texture:
    def __init__(self, surface):
        self.width = surface.get_width()
        self.height = surface.get_height()
        texture_data = pygame.image.tostring(surface, "RGBA", False)
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id)

        # Use CLAMP_TO_EDGE to prevent texture wrapping
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # CRITICAL FIX: Use NEAREST_MIPMAP_NEAREST for zoom out without bleeding
        # NEAREST for magnification (zoom in)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        # Generate mipmaps for better quality at different zoom levels
        # Using NEAREST_MIPMAP_NEAREST prevents interpolation between tiles
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

    def bind(self, slot=0):
        glActiveTexture(GL_TEXTURE0 + slot)
        glBindTexture(GL_TEXTURE_2D, self.id)

    def __del__(self):
        if hasattr(self, 'id'):
            try:
                glDeleteTextures([self.id])
            except:
                pass


# ============================================================================
# SPRITE BATCH (WITH UV INSET FIX)
# ============================================================================

class SpriteBatch:
    def __init__(self, max_sprites=20000):
        self.max_sprites = max_sprites
        self.sprite_count = 0
        self.vertices = np.zeros(max_sprites * 4 * 9, dtype=np.float32)  # 9 floats: x,y, u,v, r,g,b,a, depth

        indices = []
        for i in range(max_sprites):
            offset = i * 4
            indices.extend([offset + 0, offset + 1, offset + 2,
                           offset + 0, offset + 2, offset + 3])
        self.indices = np.array(indices, dtype=np.uint32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, None, GL_DYNAMIC_DRAW)

        stride = 9 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes,
                     self.indices, GL_STATIC_DRAW)
        glBindVertexArray(0)
        self.current_texture = None

    def begin(self, texture):
        self.sprite_count = 0
        self.current_texture = texture

    def add_sprite(self, x, y, width, height, depth=0.0, color=(1, 1, 1, 1)):
        if self.sprite_count >= self.max_sprites:
            return False

        # Account for 1px border in texture
        # The actual tile content is in the center, surrounded by 1px border
        border = 1.0
        total_width = width + border * 2
        total_height = height + border * 2

        # UV coordinates that map to the center of the bordered texture
        u_min = border / total_width
        v_min = border / total_height
        u_max = (border + width) / total_width
        v_max = (border + height) / total_height

        idx = self.sprite_count * 4 * 9
        r, g, b, a = color

        # Top-left
        self.vertices[idx:idx+9] = [x, y, u_min, v_min, r, g, b, a, depth]
        # Top-right
        self.vertices[idx+9:idx+18] = [x + width, y, u_max, v_min, r, g, b, a, depth]
        # Bottom-right
        self.vertices[idx+18:idx+27] = [x + width, y + height, u_max, v_max, r, g, b, a, depth]
        # Bottom-left
        self.vertices[idx+27:idx+36] = [x, y + height, u_min, v_max, r, g, b, a, depth]

        self.sprite_count += 1
        return True

    def flush(self):
        if self.sprite_count == 0:
            return
        if self.current_texture:
            self.current_texture.bind(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        data_size = self.sprite_count * 4 * 9 * 4
        glBufferSubData(GL_ARRAY_BUFFER, 0, data_size, self.vertices[:self.sprite_count * 4 * 9])
        glBindVertexArray(self.vao)
        num_indices = self.sprite_count * 6
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        self.sprite_count = 0


# ============================================================================
# OPENGL RENDERER (Optimized)
# ============================================================================

class OpenGLRenderer:
    """Ultra-optimized OpenGL renderer"""

    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                       pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        self.screen = pygame.display.set_mode((screen_width, screen_height),
                                              pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)

        self.shader_program = self._compile_shaders()
        self.simple_shader = self._compile_simple_shaders()

        self.proj_loc = glGetUniformLocation(self.shader_program, "projection")
        self.tex_loc = glGetUniformLocation(self.shader_program, "texture0")
        self.simple_proj_loc = glGetUniformLocation(self.simple_shader, "projection")

        self.batch = SpriteBatch(max_sprites=20000)
        self._setup_simple_vao()

        self.projection = np.eye(4, dtype=np.float32)
        self.update_projection()

        self.camera_x = 0
        self.camera_y = 0
        self.camera_zoom = 1.0

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDisable(GL_CULL_FACE)

        self.font = pygame.font.Font(None, 18)

        # Cache de texturas
        self.texture_cache = {}

        # Texturas de UI
        self.ui_panel_texture = None
        self.ui_panel_cache_key = ""

        print(f"OpenGL Renderer: {glGetString(GL_VERSION).decode()}")

    def _compile_shaders(self):
        v = compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        f = compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        return compileProgram(v, f)

    def _compile_simple_shaders(self):
        v = compileShader(SIMPLE_VERTEX_SHADER, GL_VERTEX_SHADER)
        f = compileShader(SIMPLE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        return compileProgram(v, f)

    def _setup_simple_vao(self):
        self.simple_vao = glGenVertexArrays(1)
        self.simple_vbo = glGenBuffers(1)
        glBindVertexArray(self.simple_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.simple_vbo)
        glBufferData(GL_ARRAY_BUFFER, 10 * 1024 * 1024, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(8))
        glBindVertexArray(0)

    def _ortho_matrix(self, left, right, bottom, top, near, far):
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = 2.0 / (right - left)
        mat[1, 1] = 2.0 / (top - bottom)
        mat[2, 2] = -2.0 / (far - near)
        mat[3, 3] = 1.0
        mat[0, 3] = -(right + left) / (right - left)
        mat[1, 3] = -(top + bottom) / (top - bottom)
        mat[2, 3] = -(far + near) / (far - near)
        return mat

    def update_projection(self):
        # Proyección ortográfica con rango Z amplio para depth buffer
        self.projection = self._ortho_matrix(0, self.screen_width, self.screen_height, 0, -10000, 10000)

    def get_or_create_texture(self, surface):
        """Get texture from cache or create new one"""
        surf_id = id(surface)
        if surf_id not in self.texture_cache:
            self.texture_cache[surf_id] = Texture(surface)
        return self.texture_cache[surf_id]

    def preload_texture(self, gid, surface):
        """Pre-load texture with specific GID"""
        if gid not in self.texture_cache:
            self.texture_cache[gid] = Texture(surface)
        return self.texture_cache[gid]

    def set_camera(self, x, y, zoom):
        self.camera_x = x
        self.camera_y = y
        self.camera_zoom = zoom

    def begin_frame(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def end_frame(self):
        pygame.display.flip()

    def draw_sprite_immediate(self, texture, x, y, w, h):
        """Draw sprite immediately (for UI elements)"""
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection.T)
        glUniform1i(self.tex_loc, 0)

        self.batch.begin(texture)
        self.batch.add_sprite(x, y, w, h)
        self.batch.flush()

        glUseProgram(0)

    def draw_batched_tiles(self, tile_batches):
        """Draw all tiles grouped by texture with depth"""
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection.T)
        glUniform1i(self.tex_loc, 0)

        for texture, tiles in tile_batches.items():
            if not tiles:
                continue

            self.batch.begin(texture)
            for x, y, w, h, depth in tiles:
                # Apply camera transform
                screen_x = (x - self.camera_x) * self.camera_zoom
                screen_y = (y - self.camera_y) * self.camera_zoom
                screen_w = w * self.camera_zoom
                screen_h = h * self.camera_zoom

                self.batch.add_sprite(screen_x, screen_y, screen_w, screen_h, depth)

                if self.batch.sprite_count >= self.batch.max_sprites - 1:
                    self.batch.flush()
                    self.batch.begin(texture)

            self.batch.flush()

        glUseProgram(0)

    def draw_lines(self, lines, color):
        """Draw multiple lines in one call"""
        if not lines:
            return

        r, g, b = color[0]/255.0, color[1]/255.0, color[2]/255.0
        vertices = []
        for x1, y1, x2, y2 in lines:
            vertices.extend([x1, y1, r, g, b, 1.0])
            vertices.extend([x2, y2, r, g, b, 1.0])

        vertices = np.array(vertices, dtype=np.float32)

        glUseProgram(self.simple_shader)
        glUniformMatrix4fv(self.simple_proj_loc, 1, GL_FALSE, self.projection.T)
        glBindBuffer(GL_ARRAY_BUFFER, self.simple_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glBindVertexArray(self.simple_vao)
        glDrawArrays(GL_LINES, 0, len(lines) * 2)
        glBindVertexArray(0)
        glUseProgram(0)

    def draw_ui_panel(self, text_lines, x, y):
        """Draw UI panel with cached texture"""
        cache_key = "|".join(text_lines)

        # Only recreate texture if text changed
        if cache_key != self.ui_panel_cache_key:
            texts = []
            for text in text_lines:
                surf = self.font.render(text, True, WHITE)
                texts.append(surf)

            max_width = max(s.get_width() for s in texts)
            total_height = sum(s.get_height() + 2 for s in texts)

            panel_width = max_width + 16
            panel_height = total_height + 8
            panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
            panel_surface.fill((0, 0, 0, 180))

            y_offset = 4
            for surf in texts:
                panel_surface.blit(surf, (8, y_offset))
                y_offset += surf.get_height() + 2

            # Delete old texture
            if self.ui_panel_texture:
                del self.ui_panel_texture

            self.ui_panel_texture = Texture(panel_surface)
            self.ui_panel_cache_key = cache_key

        # Draw cached panel
        self.draw_sprite_immediate(self.ui_panel_texture, x, y,
                                   self.ui_panel_texture.width,
                                   self.ui_panel_texture.height)

    def resize(self, width, height):
        """Resize suave sin golpes"""
        self.screen_width = width
        self.screen_height = height
        glViewport(0, 0, width, height)
        self.update_projection()


# ============================================================================
# CAMERA
# ============================================================================

class Camera:
    def __init__(self, width, height):
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.width = width
        self.height = height

    def move(self, dx, dy):
        self.x += dx / self.zoom
        self.y += dy / self.zoom

    def set_zoom(self, zoom):
        center_x = self.x + self.width / (2 * self.zoom)
        center_y = self.y + self.height / (2 * self.zoom)
        self.zoom = max(0.1, min(5.0, zoom))
        self.x = center_x - self.width / (2 * self.zoom)
        self.y = center_y - self.height / (2 * self.zoom)

    def reset(self, map_width, map_height, tile_width, tile_height):
        self.x = 0
        self.y = 0
        zoom_x = self.width / (map_width * tile_width)
        zoom_y = self.height / (map_height * tile_height)
        self.zoom = min(zoom_x, zoom_y, 1.0)


# ============================================================================
# TILESET RENDERER
# ============================================================================

class TilesetRenderer:
    def __init__(self, tmx_map, tmx_path, gl_renderer):
        self.tmx_map = tmx_map
        self.tmx_path = Path(tmx_path).parent
        self.gl_renderer = gl_renderer
        self.tileset_surfaces = {}
        self.tile_surface_cache = {}
        self.tile_texture_cache = {}
        self.load_tilesets()

    def load_tilesets(self):
        print("\n=== Loading Tilesets ===")
        for tileset in self.tmx_map.tilesets:
            if tileset.source:
                tileset_base = self.tmx_path / Path(tileset.source).parent
            else:
                tileset_base = self.tmx_path

            if tileset.image:
                image_path = tileset_base / tileset.image.source
                try:
                    surface = pygame.image.load(str(image_path))
                    self.tileset_surfaces[tileset.firstgid] = {
                        'surface': surface,
                        'tileset': tileset
                    }
                    print(f"Loaded tileset: {tileset.name}")
                    self._preload_tileset_tiles(tileset, surface)
                except pygame.error as e:
                    print(f"Warning: {e}")
            elif tileset.tiles:
                print(f"Loading image collection: {tileset.name} ({len(tileset.tiles)} tiles)")
                for tile_id, tile in tileset.tiles.items():
                    if tile.image:
                        image_path = tileset_base / tile.image.source
                        try:
                            surface = pygame.image.load(str(image_path))
                            gid = tileset.firstgid + tile_id
                            self.tile_surface_cache[gid] = surface
                            self.tile_texture_cache[gid] = self.gl_renderer.preload_texture(gid, surface)
                        except pygame.error as e:
                            print(f"  Warning: {e}")

    def _preload_tileset_tiles(self, tileset, tileset_surface):
        """Pre-load and cache ALL tiles from a tileset with 1px border to prevent bleeding"""
        if tileset.columns <= 0:
            return

        tiles_loaded = 0
        for tile_id in range(tileset.tilecount):
            gid = tileset.firstgid + tile_id

            local_id = tile_id
            tile_x = (local_id % tileset.columns) * tileset.tilewidth
            tile_y = (local_id // tileset.columns) * tileset.tileheight

            tile_x += tileset.margin + (local_id % tileset.columns) * tileset.spacing
            tile_y += tileset.margin + (local_id // tileset.columns) * tileset.spacing

            # Create surface with 2px extra (1px border on each side)
            border = 1
            tile_surface = pygame.Surface((tileset.tilewidth + border * 2, tileset.tileheight + border * 2), pygame.SRCALPHA)

            # Blit the main tile in the center
            tile_surface.blit(tileset_surface, (border, border),
                            pygame.Rect(tile_x, tile_y, tileset.tilewidth, tileset.tileheight))

            # Extrude edges: copy border pixels to extend the tile
            # Top edge
            if tile_y > 0:
                tile_surface.blit(tileset_surface, (border, 0),
                                pygame.Rect(tile_x, tile_y, tileset.tilewidth, 1))
            else:
                tile_surface.blit(tileset_surface, (border, 0),
                                pygame.Rect(tile_x, tile_y, tileset.tilewidth, 1))

            # Bottom edge
            if tile_y + tileset.tileheight < tileset_surface.get_height():
                tile_surface.blit(tileset_surface, (border, tileset.tileheight + border),
                                pygame.Rect(tile_x, tile_y + tileset.tileheight - 1, tileset.tilewidth, 1))
            else:
                tile_surface.blit(tileset_surface, (border, tileset.tileheight + border),
                                pygame.Rect(tile_x, tile_y + tileset.tileheight - 1, tileset.tilewidth, 1))

            # Left edge
            if tile_x > 0:
                tile_surface.blit(tileset_surface, (0, border),
                                pygame.Rect(tile_x, tile_y, 1, tileset.tileheight))
            else:
                tile_surface.blit(tileset_surface, (0, border),
                                pygame.Rect(tile_x, tile_y, 1, tileset.tileheight))

            # Right edge
            if tile_x + tileset.tilewidth < tileset_surface.get_width():
                tile_surface.blit(tileset_surface, (tileset.tilewidth + border, border),
                                pygame.Rect(tile_x + tileset.tilewidth - 1, tile_y, 1, tileset.tileheight))
            else:
                tile_surface.blit(tileset_surface, (tileset.tilewidth + border, border),
                                pygame.Rect(tile_x + tileset.tilewidth - 1, tile_y, 1, tileset.tileheight))

            # Corners
            # Top-left
            tile_surface.blit(tileset_surface, (0, 0),
                            pygame.Rect(tile_x, tile_y, 1, 1))
            # Top-right
            tile_surface.blit(tileset_surface, (tileset.tilewidth + border, 0),
                            pygame.Rect(tile_x + tileset.tilewidth - 1, tile_y, 1, 1))
            # Bottom-left
            tile_surface.blit(tileset_surface, (0, tileset.tileheight + border),
                            pygame.Rect(tile_x, tile_y + tileset.tileheight - 1, 1, 1))
            # Bottom-right
            tile_surface.blit(tileset_surface, (tileset.tilewidth + border, tileset.tileheight + border),
                            pygame.Rect(tile_x + tileset.tilewidth - 1, tile_y + tileset.tileheight - 1, 1, 1))

            self.tile_surface_cache[gid] = tile_surface
            self.tile_texture_cache[gid] = self.gl_renderer.preload_texture(gid, tile_surface)
            tiles_loaded += 1

        print(f"  Pre-loaded {tiles_loaded} tiles (with 1px border)")

    def get_tile_texture(self, gid):
        """Get pre-loaded texture"""
        if gid == 0:
            return None
        return self.tile_texture_cache.get(gid)

    def get_tile_surface(self, gid):
        """Get tile surface"""
        if gid == 0:
            return None
        return self.tile_surface_cache.get(gid)


# ============================================================================
# MAP 3D STRUCTURE
# ============================================================================

class Map3DStructure:
    def __init__(self, tmx_map):
        self.tmx_map = tmx_map
        self.map_width = tmx_map.width
        self.map_height = tmx_map.height
        self.tile_width = tmx_map.tilewidth
        self.tile_height = tmx_map.tileheight
        self.W = self.map_width
        self.D = self.map_height

        self.layer_info = []
        self._extract_all_layers()

        self.levels = {}
        for _, level, _ in self.layer_info:
            self.levels[level] = True

        self.min_level = min(self.levels.keys()) if self.levels else 0
        self.max_level = max(self.levels.keys()) if self.levels else 0
        self.H = self.max_level - self.min_level + 1
        self.level_offset = -self.min_level
        self.N = len(self.layer_info)

        self.mapa = np.zeros((self.H, self.D, self.W, self.N), dtype=np.uint16)
        self.layer_names = []
        self.layer_levels = []

        print(f"\n=== 3D Map Structure ===")
        print(f"Dimensions: W={self.W}, D={self.D}, H={self.H}, N={self.N}")

        self._load_layers()
        self.tmx_map = None

    def _extract_all_layers(self):
        def process_layer(layer, layer_name=""):
            if isinstance(layer, TileLayer):
                level = 0
                if 'Z' in layer.properties:
                    level = layer.properties['Z'].value
                elif 'z' in layer.properties:
                    level = layer.properties['z'].value
                elif 'level' in layer.properties:
                    level = layer.properties['level'].value
                if isinstance(level, str):
                    level = int(level, 0)
                elif isinstance(level, float):
                    level = int(level)
                full_name = layer_name or layer.name
                self.layer_info.append((layer, level, full_name))
            elif isinstance(layer, LayerGroup):
                for sublayer in layer.layers:
                    subname = f"{layer_name}/{sublayer.name}" if layer_name else sublayer.name
                    process_layer(sublayer, subname)

        for layer in self.tmx_map.layers:
            process_layer(layer)

    def _load_layers(self):
        for layer_idx, (layer, level, layer_name) in enumerate(self.layer_info):
            z = level + self.level_offset
            if not (0 <= z < self.H):
                continue

            for y in range(min(layer.height, self.D)):
                for x in range(min(layer.width, self.W)):
                    gid = layer.get_tile_gid(x, y)
                    if gid > 0:
                        self.mapa[z, y, x, layer_idx] = gid

            self.layer_names.append(layer_name)
            self.layer_levels.append(level)

    def get_level_value(self, z):
        return z - self.level_offset


# ============================================================================
# TMX EXPLORER
# ============================================================================

class TMXExplorer:
    def __init__(self, source_path):
        pygame.init()

        source_path = Path(source_path)
        self.source_path = source_path

        self.screen_width = 1280
        self.screen_height = 720

        # Create renderer
        self.renderer = OpenGLRenderer(self.screen_width, self.screen_height)

        pygame.display.set_caption(f"TMX Explorer - {source_path.name}")

        # Load map
        print(f"\nLoading TMX: {source_path}")
        tmx_map = TiledMap.load(source_path)

        # Create tileset renderer
        self.tileset_renderer = TilesetRenderer(tmx_map, source_path, self.renderer)

        # Create map structure
        self.map_3d = Map3DStructure(tmx_map)

        print(f"\nTotal textures cached: {len(self.renderer.texture_cache)}")

        # Initialize camera
        self.camera = Camera(self.screen_width, self.screen_height)
        self.camera.reset(self.map_3d.map_width, self.map_3d.map_height,
                         self.map_3d.tile_width, self.map_3d.tile_height)

        self.current_z = self.map_3d.H - 1
        self.level_height_offset = 128
        self._adjust_camera_for_all_levels()

        # UI state
        self.show_grid = False
        self.show_info = True
        self.show_profiling = False
        self.layer_visibility = [True] * self.map_3d.N

        # Mouse panning
        self.panning = False
        self.pan_start_pos = (0, 0)
        self.pan_start_camera = (0, 0)

        self.clock = pygame.time.Clock()
        self.running = True

        # Profiling
        self.frame_times = {
            'collect': [],
            'render': [],
            'grid': [],
            'ui': [],
            'total': []
        }

        print("\n=== Ready! ===")
        print("Press 'P' to toggle profiling")

    def _adjust_camera_for_all_levels(self):
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

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen_width = event.w
                self.screen_height = event.h
                self.renderer.resize(event.w, event.h)
                self.camera.width = self.screen_width
                self.camera.height = self.screen_height
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.panning = True
                    self.pan_start_pos = event.pos
                    self.pan_start_camera = (self.camera.x, self.camera.y)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.panning = False
            elif event.type == pygame.MOUSEMOTION:
                if self.panning:
                    dx = event.pos[0] - self.pan_start_pos[0]
                    dy = event.pos[1] - self.pan_start_pos[1]
                    self.camera.x = self.pan_start_camera[0] - dx / self.camera.zoom
                    self.camera.y = self.pan_start_camera[1] - dy / self.camera.zoom
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.camera.reset(self.map_3d.map_width, self.map_3d.map_height,
                                    self.map_3d.tile_width, self.map_3d.tile_height)
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_p:
                    self.show_profiling = not self.show_profiling
                    print(f"Profiling: {'ON' if self.show_profiling else 'OFF'}")
                elif event.key == pygame.K_PAGEUP:
                    old_z = self.current_z
                    self.current_z = min(self.current_z + 1, self.map_3d.H - 1)
                    if old_z != self.current_z:
                        self._adjust_camera_for_all_levels()
                elif event.key == pygame.K_PAGEDOWN:
                    old_z = self.current_z
                    self.current_z = max(self.current_z - 1, 0)
                    if old_z != self.current_z:
                        self._adjust_camera_for_all_levels()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        self.level_height_offset += 16
                        self._adjust_camera_for_all_levels()
                    else:
                        self.camera.set_zoom(self.camera.zoom * 1.2)
                elif event.key == pygame.K_MINUS:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        self.level_height_offset = max(0, self.level_height_offset - 16)
                        self._adjust_camera_for_all_levels()
                    else:
                        self.camera.set_zoom(self.camera.zoom / 1.2)
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.camera.set_zoom(self.camera.zoom * 1.1)
                else:
                    self.camera.set_zoom(self.camera.zoom / 1.1)

        keys = pygame.key.get_pressed()
        move_speed = 10 / self.camera.zoom
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.camera.move(-move_speed, 0)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.camera.move(move_speed, 0)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.camera.move(0, -move_speed)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.camera.move(0, move_speed)

    def collect_visible_tiles_ordered(self):
        """
        OPTIMIZADO: Culling solo en X e Y
        Orden de renderizado con Z-buffer: offset + y + z + n * 0.1
        """
        tile_batches = defaultdict(list)

        # Culling 2D solo en X e Y (con margen)
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

        # Offset base negativo para que los valores más altos estén más cerca
        base_offset = -1000.0

        # Recolectar tiles con su profundidad Z calculada
        for y in range(start_y, end_y):  # Y: lejos a cerca
            for x in range(start_x, end_x):  # X: izquierda a derecha
                for z in range(0, self.current_z + 1):  # Z: abajo a arriba (TODAS las alturas)
                    level_y_offset = z * self.level_height_offset

                    for n in range(self.map_3d.N):  # N: 0 a max
                        if not self.layer_visibility[n]:
                            continue

                        # Verificar que el layer pertenece a este nivel Z
                        if self.map_3d.layer_levels[n] != self.map_3d.get_level_value(z):
                            continue

                        tile_id = self.map_3d.mapa[z, y, x, n]

                        if tile_id == 0:
                            continue

                        texture = self.tileset_renderer.get_tile_texture(tile_id)
                        if not texture:
                            continue

                        surface = self.tileset_renderer.get_tile_surface(tile_id)
                        if not surface:
                            continue

                        tile_height = surface.get_height()
                        world_x = x * self.map_3d.tile_width
                        world_y = (y + 1) * self.map_3d.tile_height - tile_height - level_y_offset

                        # Calcular profundidad Z: offset + y + z + n * 0.1
                        # Mayor valor = más cerca (se dibuja encima)
                        depth = base_offset + y + z + (n * 0.1)

                        tile_batches[texture].append((
                            world_x, world_y,
                            surface.get_width(), surface.get_height(),
                            depth
                        ))

        return tile_batches

    def draw_grid(self):
        if not self.show_grid:
            return

        start_x = max(0, int(self.camera.x / self.map_3d.tile_width))
        start_y = max(0, int(self.camera.y / self.map_3d.tile_height))
        end_x = min(self.map_3d.map_width, int((self.camera.x + self.screen_width / self.camera.zoom) /
                                           self.map_3d.tile_width) + 2)
        end_y = min(self.map_3d.map_height, int((self.camera.y + self.screen_height / self.camera.zoom) /
                                            self.map_3d.tile_height) + 2)

        lines = []
        for x in range(start_x, end_x + 1):
            world_x = x * self.map_3d.tile_width
            screen_x = (world_x - self.camera.x) * self.camera.zoom
            lines.append((screen_x, 0, screen_x, self.screen_height))

        for y in range(start_y, end_y + 1):
            world_y = y * self.map_3d.tile_height
            screen_y = (world_y - self.camera.y) * self.camera.zoom
            lines.append((0, screen_y, self.screen_width, screen_y))

        self.renderer.draw_lines(lines, DARK_GRAY)

    def draw_ui(self):
        if self.show_info:
            level_value = self.map_3d.get_level_value(self.current_z)

            info_lines = [
                f"Zoom: {self.camera.zoom:.2f}x | FPS: {int(self.clock.get_fps())}",
                f"Height: {self.current_z}/{self.map_3d.H-1} (level={level_value})",
                f"Textures: {len(self.renderer.texture_cache)}",
                f"Z-Buffer: ON | Depth = offset + y + z + n*0.1",
            ]

            if self.show_profiling:
                for key in ['collect', 'render', 'grid', 'ui']:
                    if self.frame_times[key]:
                        avg = sum(self.frame_times[key][-60:]) / min(60, len(self.frame_times[key]))
                        info_lines.append(f"{key}: {avg*1000:.1f}ms")

            self.renderer.draw_ui_panel(info_lines, 10, 10)

        help_lines = ["Mouse: Drag | Wheel: Zoom | PgUp/PgDn: Height | G: Grid | I: Info | P: Profile | ESC: Quit"]
        self.renderer.draw_ui_panel(help_lines, 10, self.screen_height - 35)

    def draw(self):
        import time
        frame_start = time.perf_counter()

        self.renderer.begin_frame()
        self.renderer.set_camera(self.camera.x, self.camera.y, self.camera.zoom)

        t1 = time.perf_counter()
        tile_batches = self.collect_visible_tiles_ordered()
        t2 = time.perf_counter()

        self.renderer.draw_batched_tiles(tile_batches)
        t3 = time.perf_counter()

        self.draw_grid()
        t4 = time.perf_counter()

        self.draw_ui()
        t5 = time.perf_counter()

        self.renderer.end_frame()

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
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source_path = sys.argv[1]

    if not Path(source_path).exists():
        print(f"Error: File '{source_path}' not found")
        sys.exit(1)

    try:
        explorer = TMXExplorer(source_path)
        explorer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
