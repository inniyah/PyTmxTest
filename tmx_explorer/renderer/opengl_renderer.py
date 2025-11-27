"""
OpenGL-based renderer for TMX maps
"""

import ctypes
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from typing import Dict, List, Tuple, Optional

from .texture import Texture
from .sprite_batch import SpriteBatch
from ..shaders.sources import (
    VERTEX_SHADER, FRAGMENT_SHADER,
    SIMPLE_VERTEX_SHADER, SIMPLE_FRAGMENT_SHADER
)

WHITE = (255, 255, 255)


class OpenGLRenderer:
    """Ultra-optimized OpenGL renderer for tile maps"""

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self._init_display()
        self._init_shaders()
        self._init_buffers()
        self._init_state()

    def _init_display(self):
        """Initialize pygame display with OpenGL context"""
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                       pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE
        )
        print(f"OpenGL Renderer: {glGetString(GL_VERSION).decode()}")

    def _init_shaders(self):
        """Compile and link shader programs"""
        self.shader_program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        self.simple_shader = compileProgram(
            compileShader(SIMPLE_VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(SIMPLE_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        
        self.proj_loc = glGetUniformLocation(self.shader_program, "projection")
        self.tex_loc = glGetUniformLocation(self.shader_program, "texture0")
        self.simple_proj_loc = glGetUniformLocation(self.simple_shader, "projection")

    def _init_buffers(self):
        """Initialize vertex buffers"""
        self.batch = SpriteBatch(max_sprites=20000)
        
        # Simple VAO for lines/shapes
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

    def _init_state(self):
        """Initialize rendering state"""
        self.projection = np.eye(4, dtype=np.float32)
        self.update_projection()
        
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_zoom = 1.0
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDisable(GL_CULL_FACE)
        
        self.texture_cache: Dict[int, Texture] = {}
        self.font = pygame.font.Font(None, 18)
        
        # UI texture cache
        self.ui_panel_texture: Optional[Texture] = None
        self.ui_panel_cache_key = ""

    def _ortho_matrix(self, left: float, right: float, bottom: float, 
                      top: float, near: float, far: float) -> np.ndarray:
        """Create orthographic projection matrix"""
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
        """Update projection matrix for current screen size"""
        self.projection = self._ortho_matrix(
            0, self.screen_width, self.screen_height, 0, -10000, 10000
        )

    def get_or_create_texture(self, surface: pygame.Surface) -> Texture:
        """Get texture from cache or create new one"""
        surf_id = id(surface)
        if surf_id not in self.texture_cache:
            self.texture_cache[surf_id] = Texture(surface)
        return self.texture_cache[surf_id]

    def preload_texture(self, gid: int, surface: pygame.Surface) -> Texture:
        """Pre-load texture with specific GID"""
        if gid not in self.texture_cache:
            self.texture_cache[gid] = Texture(surface)
        return self.texture_cache[gid]

    def set_camera(self, x: float, y: float, zoom: float):
        """Set camera position and zoom"""
        self.camera_x = x
        self.camera_y = y
        self.camera_zoom = zoom

    def begin_frame(self):
        """Start a new frame"""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def end_frame(self):
        """End current frame and swap buffers"""
        pygame.display.flip()

    def draw_sprite_immediate(self, texture: Texture, x: float, y: float, 
                              w: float, h: float):
        """Draw sprite immediately (for UI elements)"""
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection.T)
        glUniform1i(self.tex_loc, 0)
        
        self.batch.begin(texture)
        self.batch.add_sprite(x, y, w, h)
        self.batch.flush()
        
        glUseProgram(0)

    def draw_batched_tiles(self, tile_batches: Dict[Texture, List[Tuple]]):
        """Draw all tiles grouped by texture with depth"""
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection.T)
        glUniform1i(self.tex_loc, 0)

        for texture, tiles in tile_batches.items():
            if not tiles:
                continue

            self.batch.begin(texture)
            for x, y, w, h, depth in tiles:
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

    def draw_lines(self, lines: List[Tuple[float, float, float, float]], 
                   color: Tuple[int, int, int]):
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

    def draw_ui_panel(self, text_lines: List[str], x: int, y: int):
        """Draw UI panel with cached texture"""
        cache_key = "|".join(text_lines)

        if cache_key != self.ui_panel_cache_key:
            texts = [self.font.render(text, True, WHITE) for text in text_lines]
            
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

            if self.ui_panel_texture:
                del self.ui_panel_texture

            self.ui_panel_texture = Texture(panel_surface)
            self.ui_panel_cache_key = cache_key

        self.draw_sprite_immediate(
            self.ui_panel_texture, x, y,
            self.ui_panel_texture.width,
            self.ui_panel_texture.height
        )

    def resize(self, width: int, height: int):
        """Handle window resize"""
        self.screen_width = width
        self.screen_height = height
        glViewport(0, 0, width, height)
        self.update_projection()
