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
