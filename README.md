# TMX Explorer

Visor OpenGL de mapas de [Tiled](https://www.mapeditor.org/) (TMX) con niveles de altura
(pseudo-3D), personajes animados, NPCs con IA simple, colisiones y soporte de mando.

## Instalación

Necesitas Python 3.8+ y una GPU con OpenGL 3.3.

```bash
pip install -r requirements.txt
# o, como paquete editable con tests y zstd:
pip install -e ".[dev,zstd]"
```

## Uso

```bash
python -m tmx_explorer maps/map01/VTiles1.tmx              # con jugador
python -m tmx_explorer maps/map01/VTiles1.tmx --npcs       # + NPCs de demo
python -m tmx_explorer maps/map01/VTiles1.tmx assets/people/kid.png
python -m tmx_explorer maps/map02/map02.tmx --no-player    # solo visor
python run_explorer.py maps/map01/VTiles1.tmx              # atajo: jugador + NPCs
```

| Control | Acción |
|---|---|
| WASD / flechas | Mover jugador (o cámara con `--no-player`) |
| Shift + WASD | Mover cámara |
| Q / E | Bajar / subir de nivel al jugador |
| Arrastrar ratón, rueda, `+`/`-` | Pan y zoom |
| Shift + `+`/`-` | Separación vertical entre niveles |
| PgUp / PgDn | Nivel más alto que se dibuja |
| 1 – 9 | Mostrar / ocultar la capa de Tiled número N |
| G / I / P / C | Grid / info / profiling / debug de colisión |
| Espacio | Reiniciar vista |
| ESC | Salir |

Mando: stick izquierdo o cruceta para mover, bumpers para la altura, Start para la info y
Back para el grid.

## Preparar mapas en Tiled

El motor trabaja con un **mundo 3D de celdas**. Las capas de Tiled son solo la forma
de editarlo: al cargar, todas las capas de una misma planta se integran en las celdas
de esa planta. Cada celda tiene 4 huecos apilados de abajo arriba, en el orden de las
capas en Tiled; los tiles vacíos no ocupan hueco.

**Plantas.** Añade una propiedad `Z` de tipo int a la capa o, mejor, al grupo que
contiene las capas de esa planta (las capas la heredan). Se admiten valores negativos
(sótanos). Sin `Z`, la capa va a la planta 0.

**Función de cada capa.** Ponla en la *clase* de la capa (campo Class de Tiled), o en
una propiedad `role`. También se hereda del grupo.

| Clase | Dibujo |
|---|---|
| `floor` | Plano, siempre por debajo de los personajes de su planta |
| `object` | Ordenado por fila con los personajes (por defecto) |
| `overhead` | Siempre por encima de los personajes (copas de árbol, arcos) |
| `roof` | Como `overhead` (en el futuro se ocultará al entrar en un edificio) |

Un tile puede imponer su función con su clase en el tileset: por ejemplo, una valla
con clase `object` lo es aunque se pinte en una capa de suelo.

**Colisión.** Propiedad bool `solid = true` en el tile, dentro del tileset.

**Otros detalles:**

- Las capas ocultas en Tiled no se cargan. La opacidad de capas y grupos se respeta.
- La altura en píxeles de cada planta se puede fijar con una propiedad int
  `level_height` en el mapa; si no, se usa 128.
- Si una celda recibe más de 4 tiles, se conservan los de más abajo y se avisa con las
  coordenadas.
- Se admiten mapas infinitos.

Ejemplo de organización:

```
ciudad.tmx                    (propiedad level_height = 92)
├── grupo "Calle"             Z = 0
│   ├── suelo                 class = floor
│   ├── transiciones          class = floor      (tiles semitransparentes)
│   ├── muros y vallas        class = object     (tiles con solid = true)
│   └── copas                 class = overhead
└── grupo "Casa, planta 1"    Z = 1
    ├── suelo                 class = floor
    └── muebles               class = object
```

## Estructura

```
tmx_manager.py              Lectura/escritura de TMX (independiente del visor)
tmx_explorer/
  __main__.py               CLI
  app.py                    Ventana GLFW, entrada, bucle y dibujo de la escena
  config.py                 Todas las constantes ajustables
  demo.py                   Escena de ejemplo (jugador + NPCs)
  camera.py                 Cámara 2D con zoom
  gamepad.py                Mandos (layout estándar y joysticks genéricos)
  map/
    world.py                World: tiles[z, y, x, hueco], roles, celdas (integra las capas)
    cells.py                Formato de celda de 16 bits (forma de colisión + flags)
    collision.py            CollisionMap: consultas de colisión sobre las celdas
    tileset_renderer.py     Tilesets -> atlas + tablas de consulta por GID
    visible_tiles.py        Recolección vectorizada (numpy) de los tiles visibles
  entities/
    sprite.py               Spritesheets 4x4 animados (frames y atlas compartidos)
    character.py            Movimiento con colisión, IA de NPC
    manager.py              Creación y actualización de personajes
  renderer/
    opengl_renderer.py      Quads en mundo (cámara en la proyección) y pantalla; HUD
    sprite_batch.py         quad_vertices (vectorizado, con flips) + dibujo por lotes
    atlas.py                Empaquetado de imágenes en atlas con borde anti-bleeding
    texture.py              Texturas OpenGL
  shaders/sources.py        GLSL
engine/                     Motor nuevo
  world/                    Celda de 32 bits, CellSpec, chunks, WorldStore (SQLite), WorldView
  compiler/                 Tiled (.tmx) -> mundo compilado (.cworld)
  render/                   Dibujo del mundo y de los personajes (OpenGL 3.3, profundidad por píxel)
  physics/                  Física de andar para cilindros, vectorizada
  viewer.py                 Visor: python -m engine.viewer MUNDO.cworld
  png16.py                  PNG de 16 bits por canal (mapas de posición)
tools/pieces/               Catálogo de piezas, .obj, render oblicuo, tileset y galería
tests/                      Tests sin GPU (pytest); se ejecutan en GitHub Actions
docs/                       architecture.md (diseño), tmx_manager.md (API del parser)
maps/  assets/              Mapas y sprites de ejemplo
```

## Uso como librería

```python
from tmx_explorer.app import TMXExplorer

explorer = TMXExplorer("maps/map01/VTiles1.tmx")
player = explorer.add_character("assets/people/guy.png", x=1920, y=13800, is_player=True)
explorer.entity_manager.create_npc_wanderer("assets/people/gal.png", x=2000, y=13900, radius=200)
explorer.entity_manager.create_npc_patrol("assets/people/kid.png",
                                          points=[(1800, 13700), (2100, 13700)])
explorer.entity_manager.create_npc_follower("assets/people/fat.png", target=player)
explorer.run()
```

`tmx_manager` se puede usar sin el visor:

```python
from tmx_manager import TiledMap

m = TiledMap.load("maps/map01/VTiles1.tmx")
m.layers[0].set_tile_gid(3, 4, 12)
m.save("copia.tmx")                                   # conserva la codificación de cada capa
m.save("copia.tmx", encoding="base64", compression="zlib")
```

Guardar no pierde nada: lo que `tmx_manager` no modela (capas de imagen, animaciones,
formas de colisión de tiles, polígonos, textos, `nextobjectid`...) se conserva tal cual.

## Motor nuevo: compilar mapas

El paquete `engine` es el motor nuevo. Los mapas de Tiled se compilan una vez a un mundo
de celdas de 32 bits (un archivo SQLite con chunks, paletas y el atlas), que el motor carga
sin saber nada de Tiled:

```bash
python -m engine.compiler maps/gallery/gallery.tmx          # -> maps/gallery/gallery.cworld
python -m engine.world info  maps/gallery/gallery.cworld    # parámetros e informe de compilación
python -m engine.world stats maps/gallery/gallery.cworld    # chunks, celdas, paletas
python -m engine.world cell  maps/gallery/gallery.cworld 1 22 0   # qué hay en una celda
```

El compilador avisa con coordenadas de los conflictos (dos suelos en la misma mitad, mezclar
las diagonales `/` y `\`…) y de las rampas que sobresalen hacia una celda ocupada; con
errores no escribe nada salvo con `--force`. En código:

```python
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView

world = WorldView(SQLiteWorldStore("maps/gallery/gallery.cworld"))
world.cells(xs, ys, zs)      # valores de 32 bits de muchas celdas a la vez (arrays de numpy)
world.visual(xs, ys, zs)     # sus 4 huecos visuales
```

### Ver un mundo compilado y andar por él

```bash
python -m engine.compiler maps/course/course.tmx
python -m engine.viewer maps/course/course.cworld          # con personaje
python -m engine.viewer maps/course/course.cworld --no-player --no-vsync
```

| Control | Acción |
|---|---|
| WASD / flechas | Andar (sin personaje: mover la vista) |
| Espacio | Saltar |
| Arrastrar con el ratón | Mover la vista (deja de seguir al personaje; F vuelve a seguirlo) |
| Rueda, `+` / `-` | Zoom |
| PgUp / PgDn | Planta más alta que se dibuja |
| C | Cómo ocultar lo que hay encima del personaje: círculo, planta entera o nada |
| L | Iluminación sí / no |
| N | Día / noche |
| K | Farol del personaje |
| T | Tinte por altura (más claro cuanto más alto) |
| P | Profundidad por píxel en escala de grises |
| Inicio | Centrar el mapa |
| ESC | Salir |

**Dibujo.** Profundidad por píxel: la cercanía de cada píxel a la cámara se calcula con su
altura (del mapa de posición del atlas, o de su posición dentro del sprite en los personajes)
y su posición en pantalla, así que la oclusión es exacta sin reglas de orden. En superficies
coincidentes gana la planta superior (0.1 px de cercanía por planta). La geometría de cada
chunk se construye una vez al cargarlo.

**Física de andar** (`engine/physics/walker.py`). Cilindros verticales con velocidad hacia una
objetivo (agarre en el suelo y en el aire), gravedad y salto; paredes como segmentos (bordes,
diagonales finas, lados de lo macizo) con deslizamiento en cualquier ángulo; escalón máximo,
techos bajos y rampas. Para apoyarse cuenta el centro del cilindro; para chocar con suelos y
techos, también su contorno. Paso fijo de 60 Hz, con las consultas al mundo agrupadas para
todos los cilindros a la vez (≈ 0.45 ms por paso con uno, ≈ 2.3 ms con cien). Un test de estrés
(200 personajes andando y saltando al azar) comprueba que nadie se mete en el suelo.

**Iluminación.** Por píxel y en espacio lineal: ambiente de cielo y suelo según la normal, un
sol y hasta 32 luces puntuales (las más cercanas a la vista), sobre el albedo (color sin luz)
y las normales del atlas; los personajes también se iluminan. Las luces se colocan en Tiled
como objetos punto de clase `light` en una capa de objetos dentro de cada planta, con
propiedades opcionales `color`, `radius` (m), `intensity` y `height` (m sobre el suelo de la
planta); el compilador las guarda en la tabla `lights` del mundo. De día las luces puntuales
se notan poco (25 %). Aún no hay sombras: la luz atraviesa las paredes.

**Ocultar lo que hay encima.** Cuando el personaje tiene algo por encima de su planta en su
celda o en las de alrededor (un forjado, un puente), se deja de dibujar lo que está por encima
de su planta: en un círculo a su alrededor o en todo el mapa (tecla C). La detección es
automática; no hay que marcar nada en Tiled.

Opciones útiles: `--spawn X Y` (celda de aparición) y `--spawn-level N` (su planta),
`--cut círculo|planta|no`, `--night`, `--no-lighting`, `--tint`, `--screenshot captura.png`
(dibujar, guardar y salir) y, con ella, `--walk DX DZ SEGUNDOS` para andar antes de capturar.

## Piezas de prueba (formato de celda de 32 bits)

`tools/pieces` genera, a partir de un catálogo de 106 piezas, todo lo necesario para
probar el formato de celda nuevo (paredes, rellenos, suelos, rampas y techos bajos):

```bash
python -m tools.pieces          # ~2 s
```

| Salida | Contenido |
|---|---|
| `assets/pieces/obj/*.obj`, `pieces.mtl` | Una malla mínima por pieza (metros; X este, Y arriba, Z sur) |
| `assets/pieces/pieces.png` | Atlas de color, renderizado desde los `.obj` en la proyección oblicua |
| `assets/pieces/pieces.n.png` | Normales por píxel, misma disposición |
| `assets/pieces/pieces.p.png` | Posición 3D por píxel (RGBA de 16 bits), misma disposición |
| `assets/pieces/pieces.a.png` | Albedo: el color del material sin luz, para la iluminación, misma disposición |
| `assets/pieces/pieces.tsx` | Tileset de Tiled: cada tile lleva su clase y su parte de celda como propiedades |
| `maps/gallery/gallery.tmx` | Galería con cada pieza en su celda y su nombre debajo |
| `maps/course/course.tmx` | Recorrido en tres plantas: plataforma con escalera, habitación, muro diagonal, túnel, terraza con rampa ancha y torre |
| `maps/levels/levels.tmx` | Dos niveles con aire debajo: puente, edificio de dos plantas con escalera interior y agujero, balcón |

Proyección (como los mapas existentes): celdas de 2 × 2 × 3 m, tiles de 128 px de ancho,
filas de 92 px y 133 px por planta. Las propiedades de cada tile (`walls`, `fill`, `floor`,
`floor_part`, `ceiling`, `ceiling_part`, `flags`) se leen con
`engine.world.CellSpec.from_properties`, y el formato de 32 bits está en
`engine/world/cell.py`.

## Tests

```bash
python -m pytest
```

## Rendimiento

Todos los tiles van en un atlas compartido y la recolección de tiles visibles es
vectorizada, así que el mapa se dibuja con una sola draw call (una por página del
atlas) y cada spritesheet de personaje con otra. El HUD (`I`) muestra las draw calls.

## Limitaciones conocidas

- De las formas de colisión de `cells.py` solo están implementadas vacío y bloque; el
  resto tiene el formato reservado y, de momento, cuenta como bloque.
- Los tejados (`roof`) aún no se ocultan al entrar en un edificio.
- Los personajes se dibujan después de los tiles: el borde semitransparente de un
  objeto que esté delante de un personaje se mezcla con lo que hay detrás del objeto,
  no con el personaje.
- No se soportan orientaciones distintas de ortogonal; el flag de rotación hexagonal
  se ignora.
- Los tiles animados de Tiled se conservan al guardar, pero se muestran estáticos.
