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
| G / I / P / C | Grid / info / profiling / debug de colisión |
| Espacio | Reiniciar vista |
| ESC | Salir |

Mando: stick izquierdo o cruceta para mover, bumpers para la altura, Start para la info y
Back para el grid.

## Preparar mapas en Tiled

Cada capa de tiles se asigna a un nivel de altura con una propiedad custom `Z`
(también vale `z` o `level`) de tipo int. Las capas sin esa propiedad van al nivel 0.

Para que un tile bloquee el paso, añádele una propiedad bool `solid = true` en el tileset.

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
    structure.py            Map3DStructure: mapa[z, y, x, capa] + flips + colisión
    collision.py            CollisionMap: flags por celda, colisión de cajas
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
tests/                      Tests sin GPU (pytest)
docs/                       architecture.md (diseño), tmx_manager.md (API del parser)
legacy/                     Visores antiguos de una sola pieza (solo referencia)
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
m.save("copia.tmx", encoding="base64", compression="zlib")
```

## Tests

```bash
python -m pytest
```

## Rendimiento

Todos los tiles van en un atlas compartido y la recolección de tiles visibles es
vectorizada, así que el mapa se dibuja con una sola draw call (una por página del
atlas) y cada spritesheet de personaje con otra. El HUD (`I`) muestra las draw calls.

## Limitaciones conocidas

- No se soportan mapas infinitos (chunks) ni orientaciones distintas de ortogonal.
- El flag de rotación hexagonal de Tiled se ignora (solo aplica a mapas hexagonales).
- Los tiles animados de Tiled se muestran con su primer frame.
