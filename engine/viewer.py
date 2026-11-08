"""Visor del mundo compilado con personajes: el boceto del motor nuevo.

    python -m engine.viewer maps/course/course.cworld [--npcs 40] [--spawn X Y] [--zoom Z]
                            [--no-player] [--night] [--no-vsync]
                            [--screenshot captura.png [--walk DX DZ SEGUNDOS]]

Controles: WASD/flechas: andar (sin personaje, mover la vista) · Espacio: saltar ·
arrastrar con el ratón: mover la vista (deja de seguir al personaje; F vuelve a seguirlo) ·
rueda o +/-: zoom · PgUp/PgDn: planta más alta que se dibuja · C: cómo ocultar lo que hay
encima del personaje (círculo, planta entera o nada) · L: iluminación sí/no · N: día/noche ·
K: farol · T: tinte por altura · G: tinte por material del suelo · P: profundidad por píxel ·
M: sin mapas de albedo y normales
(diagnóstico de rendimiento) · Inicio: centrar · ESC: salir.
"""

import argparse
import sys
import time
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import GL_RGBA, GL_TRUE, GL_UNSIGNED_BYTE, glReadPixels
from PIL import Image

from engine.entities import BRAIN_NONE, BRAIN_WANDER, ActorTable, load_types
from tmx_explorer import __version__
from engine.entities.world_sim import PHYSICS_HZ, ActorSim
from engine.render.gpu_timer import GpuTimer
from engine.render.sprites import (
    ROWS_4, WALK_FPS, SpriteRenderer, SpriteSheet, delete_sheet, facing_index, facing_indices,
)
from engine.render.world_renderer import Camera, Lighting, WorldRenderer
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView

PAN_SPEED = 900.0            # px de pantalla por segundo
KEY_ZOOM = 1.25
FOLLOW_RATE = 8.0            # 1/s: lo deprisa que la cámara alcanza al personaje
ASSETS = Path(__file__).resolve().parent.parent / "assets"
CUT_MODES = ("círculo", "planta", "no")         # cómo se oculta lo que hay encima del personaje
CUT_RADIUS_CELLS = 2.5
LANTERN = {"color": (1.0, 0.62, 0.30), "radius": 5.0, "intensity": 1.3, "height": 1.5}
NPC_MIX = {"persona": 0.55, "niño": 0.15, "perro": 0.2, "guardia": 0.1}
EVICT_EVERY = 2.0            # segundos entre descargas de chunks lejanos
MAX_PHYSICS_STEPS = 4        # pasos de física por frame como mucho (a 15 fps o menos, va más lento)


class Viewer:
    def __init__(self, world_path, width=1280, height=720, visible=True, player=True,
                 spawn=None, vsync=True, spawn_level=None, npcs=0, seed=None):
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

        self.world_path = Path(world_path)
        self.view = WorldView(SQLiteWorldStore(world_path))
        self.renderer = WorldRenderer(self.view)
        w, h = glfw.get_window_size(self.window)
        self.camera = Camera(width=w, height=h)
        self.home()

        # Personajes: tabla, simulación (comportamiento + física) y dibujo
        self.rng = np.random.default_rng(seed)
        self.types = load_types()
        self.table = ActorTable(self.types)
        self.sim = ActorSim(self.view, self.table, self.rng)
        self.sprites = SpriteRenderer(self.renderer.geo)
        self.gpu_timer = GpuTimer()                       # tiempo de GPU por tramo (barra de título)
        self.renderer.timer = self.sprites.timer = self.gpu_timer
        self.cpu_ms = 0.0
        self.sheets = {}                                  # ruta -> SpriteSheet
        self.player_handle = None
        if player:
            self.player_handle = self.table.spawn("persona", self._spawn_point(spawn, spawn_level),
                                                  brain=BRAIN_NONE)
        if npcs:
            self.spawn_npcs(npcs)
        self.follow = player
        self.cut_mode = 0
        self.night = False
        self.lantern = True
        self.sim_ms = 0.0
        self._evict_timer = 0.0
        self.drawn_actors = 0
        if self.player is not None:
            self._center_on_player(1.0)

        self.keys, self.dragging, self.last_mouse = set(), False, (0.0, 0.0)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_window_size_callback(self.window, self._on_resize)

    # --- Personajes ---

    @property
    def player(self):
        """El personaje del jugador (una vista sobre su fila de la tabla), o None."""
        if self.player_handle is None or not self.table.valid(self.player_handle):
            return None
        return self.table.get(self.player_handle)

    def _spawn_point(self, spawn, level=None):
        """Sobre la superficie más alta de la columna, o de la planta ``level`` si se da."""
        v, g = self.view, self.renderer.geo
        cx, cy = spawn if spawn else (v.x0 + v.width / 2, v.y0 + v.depth / 2)
        x, z = cx * g.cell_w, cy * g.cell_w
        top = ((v.level_max if level is None else level) + 0.99) * g.cell_h
        y = float(self.sim.physics.support(x, z, top, search=v.level_max - v.level_min + 3)[0])
        return [x, max(y, v.level_min * g.cell_h), z]

    def spawn_npcs(self, count: int):
        v = self.view
        points = self.sim.spawn_points(count, self.rng, levels=list(range(v.level_min, v.level_max + 1)))
        names = list(NPC_MIX)
        kinds = self.rng.choice(names, len(points), p=list(NPC_MIX.values()))
        for p, kind in zip(points, kinds):
            self.table.spawn(kind, p, brain=BRAIN_WANDER)

    def sheet(self, path: str) -> SpriteSheet:
        if path not in self.sheets:
            self.sheets[path] = SpriteSheet.load(ASSETS / path)
        return self.sheets[path]

    def _player_screen(self):
        g = self.renderer.geo
        p = self.player
        X, Y, Z = p.pos
        h = self.table.height[self.table.row(self.player_handle)]
        return X * g.tile_w / g.cell_w, Z * g.sz - Y * g.sy - 0.5 * h * g.sy

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
            elif key == glfw.KEY_C:
                self.cut_mode = (self.cut_mode + 1) % len(CUT_MODES)
            elif key == glfw.KEY_L:
                r.lighting.enabled = not r.lighting.enabled
            elif key == glfw.KEY_N:
                self.set_night(not self.night)
            elif key == glfw.KEY_K:
                self.lantern = not self.lantern
            elif key == glfw.KEY_M:
                r.use_material_maps = not r.use_material_maps
            elif key == glfw.KEY_G:
                r.material_tint = not r.material_tint
            elif key == glfw.KEY_F and self.player is not None:
                self.follow = True
            elif key == glfw.KEY_SPACE and self.player is not None:
                self.player.request_jump()
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

    # --- Simulación ---

    def update(self, dt, wish=None):
        """Avanza comportamiento + física a paso fijo, la animación y la cámara."""
        t = self.table
        player = self.player
        if player is None:
            dx, dy = self._direction()
            self.camera.x += dx * PAN_SPEED * dt / self.camera.zoom
            self.camera.y += dy * PAN_SPEED * dt / self.camera.zoom
        else:
            player.wish = wish if wish is not None else self._direction()    # X este, Z sur
        t0 = time.perf_counter()
        focus = self.focus()
        self.sim.step(dt, focus, max_steps=MAX_PHYSICS_STEPS)
        self.sim_ms = 0.9 * self.sim_ms + 0.1 * (time.perf_counter() - t0) * 1000
        self._evict_timer += dt
        if self._evict_timer > EVICT_EVERY:
            self._evict_timer = 0.0
            self.evict_far_chunks(focus)

        rows = t.rows()
        speed = np.hypot(t.vel[rows, 0], t.vel[rows, 2])
        moving = speed > 0.3
        t.anim_time[rows] = np.where(moving, t.anim_time[rows] + dt, 0.0)
        # Dirección: los NPC miran hacia donde se mueven; el jugador, hacia donde quiere ir
        t.facing[rows] = facing_indices(np.where(moving, t.vel[rows, 0], 0), np.where(moving, t.vel[rows, 2], 0),
                                        t.facing[rows])
        if player is not None:
            row = t.row(self.player_handle)
            t.facing[row] = facing_index(*t.wish[row], int(t.facing[row]))
            if self.follow:
                self._center_on_player(1 - np.exp(-FOLLOW_RATE * dt))

    def focus(self):
        """Punto de interés de la simulación (X, Z en metros): el jugador, o el centro de la vista."""
        if self.player is not None:
            return float(self.player.pos[0]), float(self.player.pos[2])
        g = self.renderer.geo
        l, t, r, b = self.camera.bounds()
        return (l + r) / 2 * g.cell_w / g.tile_w, (t + b) / 2 / g.sz

    def evict_far_chunks(self, focus):
        """Descarga los chunks lejos del punto de interés y de la vista (memoria acotada)."""
        v, g = self.view, self.renderer.geo
        span = v.chunk_size * g.cell_w
        radius = self.sim.far + 2 * span
        keep = {k for k in v.loaded if np.hypot((k[0] + 0.5) * span - focus[0], (k[1] + 0.5) * span - focus[1]) < radius}
        keep |= set(self.renderer.visible_chunks(self.camera))
        if len(keep) < len(v.loaded):
            v.evict(keep)

    # --- Ocultar lo de encima y luces ---

    def covered(self) -> bool:
        """¿Tiene el personaje un techo encima (un forjado, un puente, un balcón)?

        Un techo tiene aire debajo: en el punto se busca el suelo transitable cerca de la altura
        del personaje (que no esté enterrado bajo la columna) y, encima, algo a menos de dos
        plantas con al menos 1 m de hueco. El terreno que sube a la planta siguiente es macizo
        hasta su superficie, así que no cuenta. Se mira en su posición y a una celda en cruz:
        cubierto si lo está su punto o al menos dos de los cuatro vecinos."""
        g, v, phys = self.renderer.geo, self.view, self.sim.physics
        X, Y, Z = self.player.pos
        d = g.cell_w
        xs = np.array([X, X - d, X + d, X, X])
        zs = np.array([Z, Z, Z, Z - d, Z + d])
        floor = phys.support(xs, zs, np.full(5, Y + 0.5))
        level = np.floor((floor + 0.05) / g.cell_h).astype(int)
        starts = np.array([v.column(int(np.floor(x / d)), int(np.floor(z / d)))[0] for x, z in zip(xs, zs)])
        ceiling = phys.ceiling(xs, zs, floor)
        roof = (starts <= level) & (floor > Y - 1.0) & (ceiling - floor >= 1.0) & (ceiling <= floor + 2 * g.cell_h)
        return bool(roof[0] or roof[1:].sum() >= 2)

    def update_cut(self):
        """Ocultar lo que hay encima del personaje según el modo (tecla C)."""
        r = self.renderer
        mode = CUT_MODES[self.cut_mode]
        if self.player is None or mode == "no" or not self.covered():
            r.cut_level = None
            return
        g = r.geo
        r.cut_level = int(np.floor((self.player.pos[1] + 0.05) / g.cell_h))
        r.cut_center = self._player_screen()
        r.cut_radius = CUT_RADIUS_CELLS * g.tile_w if mode == "círculo" else -1.0

    def set_night(self, night: bool):
        enabled = self.renderer.lighting.enabled
        self.night = night
        self.renderer.lighting = Lighting.night(enabled) if night else Lighting.day(enabled)

    def update_lights(self):
        """Luces dinámicas: el farol del personaje."""
        r = self.renderer
        r.dynamic_lights = []
        if self.player is not None and self.lantern:
            X, Y, Z = self.player.pos
            c = LANTERN["color"]
            r.dynamic_lights.append((X, Y + LANTERN["height"], Z, *c, LANTERN["radius"], LANTERN["intensity"]))

    # --- Dibujo ---

    def frame(self):
        r = self.renderer
        self.update_cut()
        self.update_lights()
        r.draw(self.camera, glfw.get_framebuffer_size(self.window))
        t = self.table
        rows = self.visible_actors()
        self.drawn_actors = int(rows.size)
        if rows.size == 0:
            return
        cut = None if r.cut_level is None else (r.cut_level, r.cut_center, r.cut_radius)
        self.sprites.draw_calls = 0
        types = np.array(t.type_names)[t.type[rows]]
        sheets = np.array([self.types[name].sheet for name in types])
        for path in np.unique(sheets):
            sel = rows[sheets == path]
            sheet = self.sheet(str(path))
            self.sprites.draw_batch(
                sheet, t.pos[sel], t.support[sel], t.height[sel], t.radius[sel],
                np.array([ROWS_4[int(f)] for f in t.facing[sel]]), sheet.walk_cols(t.anim_time[sel], WALK_FPS),
                t.tint[sel], self.camera, r.show_depth, lights=r, cut=cut, max_level=r.max_level)

    def visible_actors(self):
        """Filas de los personajes cuyo sprite puede verse (con margen para el más alto)."""
        t, g = self.table, self.renderer.geo
        rows = t.rows()
        if rows.size == 0:
            return rows
        X, Y, Z = t.pos[rows].T
        sx = X * g.tile_w / g.cell_w
        sy = Z * g.sz - Y * g.sy
        left, top, right, bottom = self.camera.bounds()
        margin = 3.0 * g.sy                                  # ~3 m: el sprite más alto y su sombra
        inside = (sx > left - margin) & (sx < right + margin) & (sy > top - margin) & (sy < bottom + margin)
        return rows[inside]

    def timing_text(self) -> str:
        """Tiempos del dibujo: GPU por tramo (mundo, sombras, personajes) y CPU de preparar el frame."""
        g = self.gpu_timer.ms
        parts = "+".join(f"{g.get(k, 0.0):.1f}" for k in ("mundo", "sombras", "personajes"))
        return f"GPU {sum(g.values()):.1f} ms ({parts} mundo/sombras/personajes) · CPU dibujo {self.cpu_ms:.1f} ms"

    def screenshot(self, path):
        for _ in range(2):                    # el primer frame carga chunks y texturas
            self.frame()
            glfw.swap_buffers(self.window)
        self.frame()
        w, h = glfw.get_framebuffer_size(self.window)
        data = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE)
        Image.frombytes("RGBA", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB").save(path)

    def run(self):
        last, title_at, frames = time.perf_counter(), 0.0, 0
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            dt, last = min(now - last, 0.1), now
            glfw.poll_events()
            self.update(dt)
            t0 = time.perf_counter()
            self.frame()
            self.cpu_ms = 0.9 * self.cpu_ms + 0.1 * (time.perf_counter() - t0) * 1000
            self.gpu_timer.frame_done()
            glfw.swap_buffers(self.window)
            frames += 1
            if now - title_at > 0.5:
                r = self.renderer
                extra = ""
                if self.player is not None:
                    X, Y, Z = self.player.pos
                    extra = f" · pies ({X:.1f}, {Y:.2f}, {Z:.1f}) m{'' if self.player.grounded else ' en el aire'}"
                glfw.set_window_title(self.window,
                    f"v{__version__} · {self.world_path.name} · {frames / max(now - title_at, 1e-6):.0f} fps · "
                    f"{len(self.table)} personajes ({'/'.join(map(str, self.sim.tier_counts))} cerca/medio/lejos), "
                    f"simulación {self.sim_ms:.1f} ms · {self.timing_text()} · "
                    f"chunks {len(self.view.loaded)} · zoom {self.camera.zoom:.2f} · "
                    f"planta máx {r.max_level}{extra} · ocultar encima: {CUT_MODES[self.cut_mode]}"
                    f"{' (activo)' if r.cut_level is not None else ''}"
                    f" · luz {'de noche' if self.night else 'de día'}{'' if r.lighting.enabled else ' (apagada)'}"
                    f"{' · TINTE' if r.height_tint else ''}{' · PROFUNDIDAD' if r.show_depth else ''}"
                    f"{'' if r.use_material_maps else ' · SIN MAPAS'}{' · MATERIALES' if r.material_tint else ''}")
                title_at, frames = now, 0

    def close(self):
        for sheet in self.sheets.values():
            delete_sheet(sheet)
        self.gpu_timer.delete()
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
    parser.add_argument("--spawn-level", type=int, help="planta de aparición (por defecto, la más alta)")
    parser.add_argument("--npcs", type=int, default=0, help="número de personajes que pasean")
    parser.add_argument("--seed", type=int, help="semilla de la aparición y del paseo de los NPC")
    parser.add_argument("--no-vsync", action="store_true", help="sin sincronía vertical (medir fps reales)")
    parser.add_argument("--tint", action="store_true", help="empezar con el tinte por altura")
    parser.add_argument("--materials", action="store_true", help="empezar con el tinte por material")
    parser.add_argument("--cut", choices=CUT_MODES, default=CUT_MODES[0],
                        help="cómo ocultar lo que hay encima del personaje")
    parser.add_argument("--night", action="store_true", help="empezar de noche")
    parser.add_argument("--no-lighting", action="store_true", help="sin iluminación (colores del tileset)")
    parser.add_argument("--walk", type=float, nargs=3, action="append", metavar=("DX", "DZ", "S"),
                        help="con --screenshot: andar en esa dirección S segundos antes de capturar")
    parser.add_argument("--simulate", type=float, default=0.0,
                        help="con --screenshot: segundos de simulación antes de capturar")
    args = parser.parse_args(argv)

    viewer = Viewer(args.world, *args.size, visible=args.screenshot is None,
                    player=not args.no_player, spawn=args.spawn, vsync=not args.no_vsync,
                    spawn_level=args.spawn_level, npcs=args.npcs, seed=args.seed)
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
        viewer.renderer.material_tint = args.materials
        viewer.cut_mode = CUT_MODES.index(args.cut)
        viewer.set_night(args.night)
        viewer.renderer.lighting.enabled = not args.no_lighting
        if args.screenshot:
            for dx, dz, secs in args.walk or []:
                for _ in range(int(secs * PHYSICS_HZ)):
                    viewer.update(1.0 / PHYSICS_HZ, wish=(dx, dz))
            for _ in range(int(args.simulate * PHYSICS_HZ)):
                viewer.update(1.0 / PHYSICS_HZ, wish=(0, 0))
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
