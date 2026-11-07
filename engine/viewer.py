"""Visor del mundo compilado con un personaje: el primer boceto del motor nuevo.

    python -m engine.viewer maps/course/course.cworld [--spawn X Y] [--zoom Z] [--no-player]
                            [--no-vsync] [--screenshot captura.png [--walk DX DZ SEGUNDOS]]

Controles: WASD/flechas: andar (sin personaje, mover la vista) · Espacio: saltar ·
arrastrar con el ratón: mover la vista (deja de seguir al personaje; F vuelve a seguirlo) ·
rueda o +/-: zoom · PgUp/PgDn: planta más alta que se dibuja · T: tinte por altura ·
P: profundidad por píxel · Inicio: centrar · ESC: salir.
"""

import argparse
import sys
import time
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import GL_RGBA, GL_TRUE, GL_UNSIGNED_BYTE, glReadPixels
from PIL import Image

from engine.physics import WalkerPhysics, Walkers
from engine.render.sprites import ROWS_4, SpriteRenderer, SpriteSheet, delete_sheet, facing_index
from engine.render.world_renderer import Camera, WorldRenderer
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView

PAN_SPEED = 900.0            # px de pantalla por segundo
KEY_ZOOM = 1.25
PHYSICS_HZ = 60
FOLLOW_RATE = 8.0            # 1/s: lo deprisa que la cámara alcanza al personaje
PLAYER_SHEET = Path(__file__).resolve().parent.parent / "assets" / "people" / "guy.png"
WALK_FPS = 8.0


class Viewer:
    def __init__(self, world_path, width=1280, height=720, visible=True, player=True,
                 spawn=None, vsync=True):
        if not glfw.init():
            raise RuntimeError("No se pudo iniciar GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.VISIBLE, GL_TRUE if visible else 0)
        self.window = glfw.create_window(width, height, "Motor nuevo", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("No se pudo crear la ventana")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if vsync else 0)
        glfw.poll_events()      # eventos del alta de la ventana (tamaño) antes de configurar la cámara

        self.view = WorldView(SQLiteWorldStore(world_path))
        self.renderer = WorldRenderer(self.view)
        w, h = glfw.get_window_size(self.window)
        self.camera = Camera(width=w, height=h)
        self.home()

        # Personaje
        self.physics = WalkerPhysics(self.view)
        self.sprites = SpriteRenderer(self.renderer.geo)
        self.sheet = SpriteSheet.load(PLAYER_SHEET) if player else None
        self.player = self._spawn(spawn) if player else None
        self.follow = player
        self.facing, self.anim_time, self.accumulator = 1, 0.0, 0.0     # mirando al sur
        if self.player is not None:
            self._center_on_player(1.0)

        self.keys, self.dragging, self.last_mouse = set(), False, (0.0, 0.0)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_window_size_callback(self.window, self._on_resize)
        self.world_path = Path(world_path)

    def _spawn(self, spawn):
        v, cw = self.view, self.renderer.geo.cell_w
        cx, cy = spawn if spawn else (v.x0 + v.width / 2, v.y0 + v.depth / 2)
        x, z = cx * cw, cy * cw
        top = (v.level_max + 1.5) * self.renderer.geo.cell_h
        y = float(self.physics.support(x, z, top, search=v.level_max - v.level_min + 3)[0])
        w = Walkers.create([[x, max(y, v.level_min * self.renderer.geo.cell_h), z]])
        w.grounded[:] = True
        return w

    def _player_screen(self):
        g = self.renderer.geo
        X, Y, Z = self.player.pos[0]
        return X * g.tile_w / g.cell_w, Z * g.sz - Y * g.sy - 0.5 * self.player.height[0] * g.sy

    def _center_on_player(self, amount):
        px, py = self._player_screen()
        cx = self.camera.x + self.camera.width / (2 * self.camera.zoom)
        cy = self.camera.y + self.camera.height / (2 * self.camera.zoom)
        self.camera.center_on(cx + (px - cx) * amount, cy + (py - cy) * amount)

    def home(self):
        v, g = self.view, self.renderer.geo
        self.camera.center_on((v.x0 + v.width / 2) * g.tile_w, (v.y0 + v.depth / 2) * g.row_h)

    # --- Entrada ---

    def _on_key(self, _w, key, _sc, action, _mods):
        if action == glfw.PRESS:
            self.keys.add(key)
            r = self.renderer
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)
            elif key == glfw.KEY_P:
                r.show_depth = not r.show_depth
            elif key == glfw.KEY_T:
                r.height_tint = not r.height_tint
            elif key == glfw.KEY_F and self.player is not None:
                self.follow = True
            elif key == glfw.KEY_SPACE and self.player is not None:
                self.player.jump[:] = True
            elif key == glfw.KEY_PAGE_UP:
                r.max_level = min(self.view.level_max, r.max_level + 1)
            elif key == glfw.KEY_PAGE_DOWN:
                r.max_level = max(self.view.level_min, r.max_level - 1)
            elif key == glfw.KEY_HOME:
                self.home()
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                self.camera.zoom_at(KEY_ZOOM, self.camera.width / 2, self.camera.height / 2)
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                self.camera.zoom_at(1 / KEY_ZOOM, self.camera.width / 2, self.camera.height / 2)
        elif action == glfw.RELEASE:
            self.keys.discard(key)

    def _on_button(self, _w, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS

    def _on_cursor(self, _w, x, y):
        if self.dragging:
            self.follow = False
            dx, dy = x - self.last_mouse[0], y - self.last_mouse[1]
            self.camera.x -= dx / self.camera.zoom
            self.camera.y -= dy / self.camera.zoom
        self.last_mouse = (x, y)

    def _on_scroll(self, _w, _dx, dy):
        self.camera.zoom_at(1.1 if dy > 0 else 1 / 1.1, *self.last_mouse)

    def _on_resize(self, _w, width, height):
        if width and height:
            self.camera.width, self.camera.height = width, height

    def _direction(self):
        k = self.keys
        dx = (glfw.KEY_D in k or glfw.KEY_RIGHT in k) - (glfw.KEY_A in k or glfw.KEY_LEFT in k)
        dy = (glfw.KEY_S in k or glfw.KEY_DOWN in k) - (glfw.KEY_W in k or glfw.KEY_UP in k)
        return dx, dy

    def _pan(self, dt):
        dx, dy = self._direction()
        self.camera.x += dx * PAN_SPEED * dt / self.camera.zoom
        self.camera.y += dy * PAN_SPEED * dt / self.camera.zoom

    def update(self, dt, wish=None):
        """Avanza la física a paso fijo, la animación y la cámara."""
        if self.player is None:
            self._pan(dt)
            return
        self.player.wish[0] = wish if wish is not None else self._direction()   # X este, Z sur
        # Mira hacia donde se quiere ir (no hacia donde resbala), con histéresis
        self.facing = facing_index(*self.player.wish[0], self.facing, directions=4)
        self.accumulator += dt
        step = 1.0 / PHYSICS_HZ
        while self.accumulator >= step:
            self.physics.step(self.player, step)
            self.accumulator -= step
        vx, vz = self.player.vel[0, 0], self.player.vel[0, 2]
        moving = np.hypot(vx, vz) > 0.3
        self.anim_time = self.anim_time + dt if moving else 0.0
        if self.follow:
            self._center_on_player(1 - np.exp(-FOLLOW_RATE * dt))

    # --- Bucle ---

    def frame(self):
        r = self.renderer
        r.draw(self.camera, glfw.get_framebuffer_size(self.window))
        if self.player is not None:
            pos = self.player.pos[0]
            level = int(np.floor(pos[1] / r.geo.cell_h + 1e-6))
            if level <= r.max_level:
                col = 1 + int(self.anim_time * WALK_FPS) % 3 if self.anim_time > 0 else 0
                self.sprites.draw(self.sheet, [(tuple(pos), float(self.player.support[0]),
                                                float(self.player.height[0]), float(self.player.radius[0]),
                                                ROWS_4[self.facing], col)],
                                  self.camera, r.show_depth)

    def screenshot(self, path):
        for _ in range(2):                    # el primer frame carga chunks y texturas
            self.frame()
            glfw.swap_buffers(self.window)
        self.frame()
        w, h = glfw.get_framebuffer_size(self.window)
        data = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE)
        Image.frombytes("RGBA", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM).save(path)

    def run(self):
        last, title_at, frames = time.perf_counter(), 0.0, 0
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            dt, last = min(now - last, 0.1), now
            glfw.poll_events()
            self.update(dt)
            self.frame()
            glfw.swap_buffers(self.window)
            frames += 1
            if now - title_at > 0.5:
                r = self.renderer
                extra = ""
                if self.player is not None:
                    X, Y, Z = self.player.pos[0]
                    extra = (f" · pies ({X:.1f}, {Y:.2f}, {Z:.1f}) m"
                             f"{'' if self.player.grounded[0] else ' en el aire'}")
                glfw.set_window_title(self.window,
                    f"{self.world_path.name} · {frames / max(now - title_at, 1e-6):.0f} fps · "
                    f"zoom {self.camera.zoom:.2f} · chunks {len(r.meshes)} · draw calls {r.draw_calls} · "
                    f"planta máx {r.max_level}{extra}{' · TINTE' if r.height_tint else ''}"
                    f"{' · PROFUNDIDAD' if r.show_depth else ''}")
                title_at, frames = now, 0

    def close(self):
        if self.sheet is not None:
            delete_sheet(self.sheet)
        self.sprites.delete()
        self.renderer.delete()
        glfw.terminate()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="engine.viewer", description=__doc__.split("\n\n")[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"), help="celda en el centro")
    parser.add_argument("--zoom", type=float)
    parser.add_argument("--depth", action="store_true", help="empezar mostrando la profundidad")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    parser.add_argument("--screenshot", type=Path, help="dibujar, guardar la captura y salir")
    parser.add_argument("--no-player", action="store_true", help="sin personaje: WASD mueve la vista")
    parser.add_argument("--spawn", type=float, nargs=2, metavar=("X", "Y"), help="celda de aparición")
    parser.add_argument("--no-vsync", action="store_true", help="sin sincronía vertical (medir fps reales)")
    parser.add_argument("--tint", action="store_true", help="empezar con el tinte por altura")
    parser.add_argument("--walk", type=float, nargs=3, action="append", metavar=("DX", "DZ", "S"),
                        help="con --screenshot: andar en esa dirección S segundos antes de capturar")
    args = parser.parse_args(argv)

    viewer = Viewer(args.world, *args.size, visible=args.screenshot is None,
                    player=not args.no_player, spawn=args.spawn, vsync=not args.no_vsync)
    try:
        if args.zoom:
            viewer.camera.zoom = args.zoom
        if args.center:
            g = viewer.renderer.geo
            viewer.camera.center_on(args.center[0] * g.tile_w, args.center[1] * g.row_h)
        elif args.zoom:
            viewer.home()
        viewer.renderer.show_depth = args.depth
        viewer.renderer.height_tint = args.tint
        if args.screenshot:
            for dx, dz, secs in args.walk or []:
                for _ in range(int(secs * PHYSICS_HZ)):
                    viewer.update(1.0 / PHYSICS_HZ, wish=(dx, dz))
            if viewer.player is not None and not args.center:
                viewer._center_on_player(1.0)
            viewer.screenshot(args.screenshot)
        else:
            viewer.run()
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
