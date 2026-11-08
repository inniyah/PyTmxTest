"""Visor del mundo compilado con personajes: el boceto del motor nuevo.

    python -m engine.viewer maps/course/course.cworld [--npcs 40] [--spawn X Y] [--zoom Z]
                            [--no-player] [--night] [--no-vsync]
                            [--screenshot captura.png [--walk DX DZ SEGUNDOS]]

Controles: WASD/flechas: andar (sin personaje, mover la vista) · Espacio: saltar ·
arrastrar con el ratón: mover la vista (deja de seguir al personaje; F vuelve a seguirlo) ·
rueda o +/-: zoom · PgUp/PgDn: planta más alta que se dibuja · C: cómo ocultar lo que tapa al
personaje (fachada, círculo, planta entera o nada) · L: iluminación sí/no · N: día/noche ·
K: farol · E: usar · X: silueta del jugador · H: halo del brillo propio · T: tinte por altura · G: tinte por material del suelo · P: profundidad por píxel ·
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

from engine.entities import BRAIN_NONE, BRAIN_ROUTINE, BRAIN_WANDER, ActorTable, RoutineBrains, load_types
from tmx_explorer import __version__
from engine.entities.world_sim import PHYSICS_HZ, ActorSim
from engine.events import EventSystem
from engine.render.gpu_timer import GpuTimer
from engine.render.bloom import Bloom
from engine.render.overlay import TextOverlay
from engine.render.sprites import (
    ROWS_4, WALK_FPS, SpriteRenderer, SpriteSheet, delete_sheet, facing_index, facing_indices,
)
from engine.render.world_renderer import Camera, Lighting, WorldRenderer
from engine.world import cell as C
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView

PAN_SPEED = 900.0            # px de pantalla por segundo
KEY_ZOOM = 1.25
FOLLOW_RATE = 8.0            # 1/s: lo deprisa que la cámara alcanza al personaje
ASSETS = Path(__file__).resolve().parent.parent / "assets"
CUT_MODES = ("fachada", "círculo", "planta", "no")   # cómo se oculta lo que tapa al personaje
FACADE_KEEP = 0.5            # m: zócalo que se conserva de lo que hay delante (paredes de su planta)
MASK_HALF = 24               # la máscara de corte cubre 48 × 48 celdas alrededor del personaje
MAX_ROOM_CELLS = 1200        # una "habitación" más grande que esto se corta ahí
HOLE_SEARCH = 3              # celdas: bajo un agujero, hasta dónde se busca el techo de la habitación
ROOM_OPEN_EDGES = 2          # una habitación puede tener tantos bordes abiertos (puertas) ...
ROOM_OPEN_FRACTION = 0.25    # ... o esta fracción de su contorno; si no, no está cerrada: exterior
STEP_UP = 1.6                # m: una celda sin techo junto a la habitación cuenta si su suelo está a
                             # menos de esto de los pies (un hueco de escalera)
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
        self.routines = self._load_routines()
        g = self.renderer.geo
        self.events = EventSystem.from_store(self.view.store, g.cell_w, g.cell_h) \
            if self.view.store.events() else None
        self.overlay = TextOverlay()
        self.silhouette = True                 # silueta del jugador donde algo lo tapa (tecla X)
        self.bloom = Bloom()
        self.halo = True                       # halo del brillo propio (tecla H)
        self.clock = 0.0
        self._use = False
        self.sim = ActorSim(self.view, self.table, self.rng, routines=self.routines)
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
        self._facade_key, self._facade = None, None
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

    def _load_routines(self):
        """Rutinas (casa ↔ plaza) si el mundo tiene grafo de navegación y casas."""
        store = self.view.store
        if not store.has_nav():
            return None
        from engine.nav import NavGraph, load_places
        from engine.world.materials import Materials
        places = load_places(store)
        materials = store.meta().get("materials")
        if not materials or not any(p.kind == "casa" for p in places):
            return None
        graph = NavGraph(store, self.view.chunk_size, self.renderer.geo.cell_w)
        return RoutineBrains(graph, places, self.types, Materials.from_json(materials), self.rng,
                             self.renderer.geo.cell_h)

    def spawn_npcs(self, count: int):
        if self.routines is not None:
            return self.spawn_routine_npcs(count)
        v = self.view
        points = self.sim.spawn_points(count, self.rng, levels=list(range(v.level_min, v.level_max + 1)))
        names = list(NPC_MIX)
        kinds = self.rng.choice(names, len(points), p=list(NPC_MIX.values()))
        for p, kind in zip(points, kinds):
            self.table.spawn(kind, p, brain=BRAIN_WANDER)

    def spawn_routine_npcs(self, count: int):
        """Cada NPC en una casa (al azar), con la rutina casa ↔ plaza; empiezan esperando un poco."""
        R, t = self.routines, self.table
        houses = [p for p in R.places.values() if p.kind == "casa"]
        names = list(NPC_MIX)
        kinds = self.rng.choice(names, count, p=list(NPC_MIX.values()))
        for kind in kinds:
            home = houses[int(self.rng.integers(len(houses)))]
            nodes = R.cache.goal_nodes(home)
            if len(nodes) == 0:
                continue
            x, y, z = R.graph.position(int(self.rng.choice(nodes)))
            row = t.row(t.spawn(kind, [x, y, z], brain=BRAIN_ROUTINE))
            t.home[row] = t.target_place[row] = home.id
            t.routine_state[row] = 1
            t.routine_timer[row] = self.rng.uniform(0, 30)

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
            elif key == glfw.KEY_H:
                self.halo = not self.halo
            elif key == glfw.KEY_X:
                self.silhouette = not self.silhouette
            elif key == glfw.KEY_E:
                self._use = True                       # usar (sucesos "use")
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
        self.clock += dt
        self.renderer.time = self.clock
        focus = self.focus()
        self.sim.step(dt, focus, max_steps=MAX_PHYSICS_STEPS)
        if self.events is not None:
            self.events.update(t, dt, self.sim.physics, self.player_handle, use=self._use)
        self._use = False
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

    def roofed_cells(self, xs, ys, feet: float, below: float = 1.0, heights: bool = False):
        """Celdas con techo encima a la altura del personaje. En cuatro puntos de cada celda,
        lejos de las diagonales (una casilla partida tiene en la diagonal un borde de macizo que la
        búsqueda de suelo no ve, y parecería un techo con aire debajo): suelo de verdad cerca de
        sus pies (no enterrado bajo la columna ni dentro de un macizo) y, encima, algo a menos de
        dos plantas con al menos 1 m de hueco. Con techo, si lo tienen al menos 3 de los 4."""
        g, v, phys = self.renderer.geo, self.view, self.sim.physics
        cxs, cys = xs.ravel(), ys.ravel()
        starts = np.array([v.column(int(x), int(y))[0] for x, y in zip(cxs, cys)])
        votes = np.zeros(cxs.size, int)
        floors = np.zeros(cxs.size)
        for u, w in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
            px, pz = (cxs + u) * g.cell_w, (cys + w) * g.cell_w
            floor = phys.support(px, pz, np.full(px.size, feet + 0.5))
            level = np.floor((floor + 0.05) / g.cell_h).astype(int)
            here = v.cells(cxs, cys, np.floor((floor + 0.1) / g.cell_h).astype(int))
            buried = C.fill_contains(C.fill(here), u, w)
            ceiling = phys.ceiling(px, pz, floor)
            ok = ((starts <= level) & ~buried & (floor > feet - below) & (floor < feet + 1.0)
                  & (ceiling - floor >= 1.0) & (ceiling <= floor + 2 * g.cell_h))
            votes += ok
            floors = np.maximum(floors, np.where(ok, floor, -1e9))
        roofed = (votes >= 3).reshape(xs.shape)
        return (roofed, floors.reshape(xs.shape)) if heights else roofed

    @staticmethod
    def fill_holes(region: np.ndarray) -> np.ndarray:
        """Añade las celdas que quedan dentro de la región, entre celdas suyas en su fila y en su
        columna (un hueco de escalera, sin techo encima, también es del edificio)."""
        def spans(a):
            out = np.zeros_like(a)
            for i, row in enumerate(a):
                idx = np.nonzero(row)[0]
                if idx.size:
                    out[i, idx[0]:idx[-1] + 1] = True
            return out
        return region | (spans(region) & spans(region.T).T)

    def room(self, cx: int, cy: int, level: int, feet: float, half: int = MASK_HALF):
        """Habitación del personaje, si está en un interior: (x0, y0, máscara booleana de
        (2·half) × (2·half)); vacía al aire libre.

        Celdas con techo unidas sin cruzar paredes (ni macizos), más las celdas sin techo de
        dentro (huecos de escalera, agujeros del techo). Si su celda no tiene techo (está bajo
        un agujero), se empieza por la celda con techo más cercana a la que se llega sin cruzar
        paredes, y vale si su celda queda dentro. Solo es un interior si la región está cerrada:
        como mucho ROOM_OPEN_EDGES bordes, o la fracción ROOM_OPEN_FRACTION de su contorno, dan
        al exterior sin pared en medio (bajo un balcón o un puente, está abierta: exterior)."""
        v = self.view
        n = 2 * half
        x0, y0 = cx - half, cy - half
        ys, xs = np.mgrid[y0:y0 + n, x0:x0 + n]
        empty = np.zeros((n, n), bool)
        roofed = self.roofed_cells(xs, ys, feet)
        g, phys = self.renderer.geo, self.sim.physics
        floor_h = phys.support((xs.ravel() + 0.5) * g.cell_w, (ys.ravel() + 0.5) * g.cell_w,
                               np.full(xs.size, feet + 1.6)).reshape(xs.shape)
        cells = v.cells(xs, ys, level)
        wall_n, wall_w = C.wall_n(cells) != 0, C.wall_w(cells) != 0
        solid = C.fill(cells) == C.FILLS.index("solid")
        walkable = ~solid & (np.abs(floor_h - feet) < STEP_UP)

        def neighbours(j, i):
            """Vecinas a las que se pasa sin cruzar una pared: (fila, columna)."""
            for dj, di, blocked in ((-1, 0, wall_n[j, i]), (0, -1, wall_w[j, i]),
                                    (1, 0, wall_n[j + 1, i] if j + 1 < n else True),
                                    (0, 1, wall_w[j, i + 1] if i + 1 < n else True)):
                a, b = j + dj, i + di
                if not blocked and 0 <= a < n and 0 <= b < n:
                    yield a, b

        # Por dónde empezar: su celda o, bajo un agujero, la celda con techo más cercana
        seed = (half, half)
        if not roofed[seed]:
            seen, frontier, seed = {seed}, [seed], None
            for _ in range(HOLE_SEARCH):
                nxt = []
                for c in frontier:
                    for a in neighbours(*c):
                        if a not in seen and walkable[a]:
                            seen.add(a)
                            nxt.append(a)
                hits = [c for c in nxt if roofed[c]]
                if hits:
                    seed = hits[0]
                    break
                frontier = nxt
            if seed is None:
                return x0, y0, empty
        inside = np.zeros((n, n), bool)
        inside[seed] = True
        todo = [seed]
        while todo and inside.sum() < MAX_ROOM_CELLS:
            for a in neighbours(*todo.pop()):
                if not inside[a] and not solid[a] and roofed[a]:
                    inside[a] = True
                    todo.append(a)
        # Un paso más por celdas sin techo (un hueco de escalera dentro), y los huecos de dentro
        grown = inside.copy()
        for j, i in zip(*np.nonzero(inside)):
            for a in neighbours(j, i):
                if walkable[a] and not roofed[a]:
                    grown[a] = True
        region = self.fill_holes(grown)
        if not region[half, half]:
            return x0, y0, empty
        # ¿Cerrada? Bordes del contorno que dan fuera sin pared en medio (a celdas transitables)
        edges = opened = 0
        for j, i in zip(*np.nonzero(region)):
            for dj, di, blocked in ((-1, 0, wall_n[j, i]), (0, -1, wall_w[j, i]),
                                    (1, 0, wall_n[j + 1, i] if j + 1 < n else True),
                                    (0, 1, wall_w[j, i + 1] if i + 1 < n else True)):
                a, b = j + dj, i + di
                if 0 <= a < n and 0 <= b < n and region[a, b]:
                    continue
                edges += 1
                opened += (not blocked and 0 <= a < n and 0 <= b < n and walkable[a, b])
        if opened > max(ROOM_OPEN_EDGES, ROOM_OPEN_FRACTION * edges):
            return x0, y0, empty
        return x0, y0, region

    def facade_cut_mask(self, cx: int, cy: int, level: int, feet: float):
        """Máscara de corte, o None si no hay nada que ocultar: dict(origin, mask, base, floor,
        feet). Solo en un interior (algo sólido por encima del personaje): al aire libre nunca se
        oculta nada, aunque esté detrás de un edificio.

        En su habitación (celdas con techo unidas a la suya sin cruzar paredes, con los huecos de
        escalera) se oculta todo lo de encima de su planta, salvo la pared del fondo, que se
        conserva hacia arriba; en la fila de delante, la pared de delante queda en zócalo.
        ``mask`` (alto, ancho, 4): canales = plantas base .. base + 3; 0 se dibuja, 128 zócalo (más
        de FACADE_KEEP sobre ``feet``), 255 se oculta. El shader mira cada píxel 25 cm al sur en la
        planta ``floor`` y 25 cm al norte en las de encima."""
        x0, y0, inside = self.room(cx, cy, level, feet)
        if not inside.any():
            return None
        n = inside.shape[0]
        mask = np.zeros((n, n, 4), np.uint8)
        mask[inside, 1:] = 255
        front = np.zeros_like(inside)
        front[1:] = inside[:-1] & ~inside[1:]
        mask[front, 0] = 128
        return {"origin": (x0, y0), "mask": mask, "base": level, "floor": level, "feet": feet}

    def update_cut(self):
        """Ocultar lo que tapa al personaje según el modo (tecla C).

        - fachada (casa de muñecas): solo en un interior (algo sólido por encima), el techo y las
          plantas de arriba de su habitación y la pared de delante (en zócalo); las paredes del
          fondo y de los lados se quedan. Al aire libre, nada. Ver facade_cut_mask.
        - círculo / planta: lo que hay por encima de su planta, en un círculo o en todo el mapa;
          solo bajo techo."""
        r = self.renderer
        mode = CUT_MODES[self.cut_mode]
        r.facade = None
        if mode != "fachada":
            self._facade_key = None
        if self.player is None or mode == "no":
            r.cut_level = None
            return
        g = r.geo
        X, Y, Z = self.player.pos
        if mode == "fachada":
            level = int(np.floor((Y + 0.05) / g.cell_h))
            key = (int(np.floor(X / g.cell_w)), int(np.floor(Z / g.cell_w)), level)
            if key != self._facade_key:                 # solo al cambiar de celda o de planta
                self._facade_key = key
                self._facade = self.facade_cut_mask(*key, feet=float(Y))
                f = self._facade
                if f is not None:
                    r.set_cut_mask(f["mask"], f["origin"], f["base"], f["feet"], FACADE_KEEP, f["floor"])
            if self._facade is None:
                r.cut_level = None
                return
            r.facade = r.facade_state
            if self._facade["floor"] == level:             # su propio suelo: sigue a sus pies
                r.facade["feet"] = float(Y)
            r.cut_level = level
            return
        if not self.covered():
            r.cut_level = None
            return
        r.cut_level = int(np.floor((Y + 0.05) / g.cell_h))
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
        """Un frame: el mundo y los personajes (en la imagen intermedia del halo si hay brillo
        propio), el halo, y encima los mensajes."""
        r = self.renderer
        w, h = glfw.get_framebuffer_size(self.window)
        halo = self.halo and r.has_glow and not r.show_depth
        if halo:
            self.bloom.begin(w, h)
            r.glow_buffer = True
        self.draw_scene()
        if halo:
            r.glow_buffer = False
            with self.gpu_timer("halo"):
                self.bloom.end(w, h)
        self.draw_messages()

    def draw_scene(self):
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
        self.draw_player_silhouette(cut)

    def draw_player_silhouette(self, cut):
        """Silueta del jugador donde algo lo tapa (al aire libre no se oculta nada del escenario)."""
        if not self.silhouette or self.player is None or self.renderer.show_depth:
            return
        t, row = self.table, self.table.row(self.player_handle)
        sheet = self.sheet(self.types[t.type_names[t.type[row]]].sheet)
        self.sprites.draw_silhouettes(
            sheet, t.pos[row:row + 1], t.height[row:row + 1], np.array([ROWS_4[int(t.facing[row])]]),
            sheet.walk_cols(t.anim_time[row:row + 1], WALK_FPS), self.camera, lights=self.renderer,
            cut=cut, max_level=self.renderer.max_level)

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

    def routine_text(self) -> str:
        if self.routines is None:
            return ""
        rows = self.table.rows()
        mine = rows[self.table.brain[rows] == BRAIN_ROUTINE]
        moving = int((self.table.routine_state[mine] == 0).sum())
        return f" · rutinas: {moving} en camino, {len(mine) - moving} en un lugar, {self.routines.arrived} llegadas"

    def timing_text(self) -> str:
        """Tiempos del dibujo: GPU por tramo (mundo, sombras, personajes, halo) y CPU de preparar el frame."""
        g = self.gpu_timer.ms
        parts = "+".join(f"{g.get(k, 0.0):.1f}" for k in ("mundo", "sombras", "personajes", "halo"))
        return f"GPU {sum(g.values()):.1f} ms ({parts} mundo/sombras/personajes/halo) · CPU dibujo {self.cpu_ms:.1f} ms"

    def draw_messages(self):
        if self.events is not None and self.events.messages:
            w, h = glfw.get_framebuffer_size(self.window)
            self.overlay.draw([m.text for m in self.events.messages], w, h)

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
                    f"{len(self.table)} personajes ({'/'.join(map(str, self.sim.tier_counts))} cerca/medio/lejos)"
                    f"{self.routine_text()}, "
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
        self.overlay.delete()
        self.bloom.delete()
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
                        help="cómo ocultar lo que tapa al personaje")
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
