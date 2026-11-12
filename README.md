# PyTmxTest

Motor de juego 2D en Python con mundos en celdas de varias plantas, editados en
[Tiled](https://www.mapeditor.org/) y compilados a un formato propio. Vista oblicua con
profundidad exacta por píxel, iluminación dinámica, física de personajes, multitudes de NPC y
navegación por caminos.

- **Mundos de celdas 3D**: celdas de 2 × 2 × 3 m apiladas en plantas, con paredes, suelos
  planos, rampas, pendientes suaves y techos bajos. Terrenos grandes tipo Populous (probado con
  1 × 1 km).
- **Compilador** de mapas de Tiled (`.tmx`) a un mundo compilado (`.cworld`, SQLite): chunks,
  atlas de texturas, luces, lugares y grafo de navegación.
- **Dibujo** con OpenGL 3.3: profundidad por píxel, iluminación por píxel con sol, ambiente y
  luces puntuales, día y noche, metal y superficies pulidas, brillo propio con halo y pulso,
  interiores como una casa de muñecas.
- **Personajes** como cilindros con física de andar, saltar y empujarse; sprites en 8 direcciones
  con normales y profundidad por píxel, generados por código mientras no haya arte; tipos definidos en
  datos; miles de NPC con niveles de detalle.
- **Navegación**: grafo verificado contra la física, costes por material del suelo y por
  pendiente, campos de flujo, lugares y rutinas (casa ↔ plaza).

El documento de diseño completo, con las decisiones y su porqué, está en
[Claude Docs](https://claude.ai/artifact/VaLKaPF4uJxTiFyzn9Amcd).

---

## Instalación

Hace falta Python 3.9 o posterior y una GPU con OpenGL 3.3.

```bash
pip install -r requirements.txt
# o, como paquete editable con las herramientas de desarrollo:
pip install -e ".[dev]"
```

Las únicas dependencias son **numpy, Pillow, PyOpenGL y glfw** (y nada más: el juego tiene que
poder distribuirse). Opcionales: `PyOpenGL_accelerate` (más rápido) y `zstandard` (mapas TMX
con compresión zstd).

## Inicio rápido

```bash
python -m tools.pieces                                # genera el tileset de prueba y 4 mapas (≈ 6 s)
python -m engine.compiler maps/terrain/terrain.tmx    # compila el terreno de 512 m (≈ 10 s)
python -m engine.viewer maps/terrain/terrain.cworld --npcs 300
```

Andas con WASD, saltas con Espacio y te mueves por el mapa con el ratón; los NPC van de su casa
a la plaza y vuelven, por los caminos. Prueba también:

```bash
python -m engine.compiler maps/levels/levels.tmx      # puente, edificio de dos plantas
python -m engine.viewer maps/levels/levels.cworld --night --npcs 40

python -m engine.compiler maps/course/course.tmx      # recorrido de prueba de la física
python -m engine.viewer maps/course/course.cworld

# Un kilómetro cuadrado con 3000 personajes:
python -m tools.pieces --terrain-size 512 --terrain maps/terrain1k/terrain1k.tmx
python -m engine.compiler maps/terrain1k/terrain1k.tmx
python -m engine.viewer maps/terrain1k/terrain1k.cworld --npcs 3000
```

Los mapas compilados (`.cworld`) no se guardan en el repositorio: se generan con el
compilador. Si cambias el formato de celda o las piezas, vuelve a generar y compilar.

---

## El visor

```
python -m engine.viewer MUNDO.cworld [opciones]
```

| Tecla | Acción |
|---|---|
| WASD / flechas | Andar (con `--no-player`, mover la vista) |
| Espacio | Saltar |
| Arrastrar con el ratón | Mover la vista (deja de seguir al personaje) |
| F | Volver a seguir al personaje |
| Rueda, `+` / `-` | Zoom |
| PgUp / PgDn | Planta más alta que se dibuja |
| C | Cómo ocultar lo que tapa al personaje: fachada, círculo, planta entera o nada |
| L | Iluminación sí / no |
| N | Día / noche |
| K | Farol del personaje |
| E | Usar (sucesos con `when = use`) |
| X | Silueta del jugador donde algo lo tapa |
| H | Halo del brillo propio |
| T | Tinte por altura |
| G | Tinte por material del suelo |
| P | Ver la profundidad por píxel |
| M | Sin mapas de albedo y normales (diagnóstico de rendimiento) |
| Inicio | Centrar la vista |
| ESC | Salir |

| Opción | |
|---|---|
| `--player-type T` | Tipo del jugador, de `types.json` (por ejemplo, `soldado`) |
| `--npc-type T` | Todos los NPC de ese tipo (por ejemplo, `enemigo`) |
| `--npcs N` | N personajes. Si el mundo tiene grafo de navegación y casas, siguen la rutina casa ↔ plaza; si no, pasean |
| `--spawn X Y`, `--spawn-level N` | Celda y planta donde aparece el jugador |
| `--no-player` | Sin jugador: WASD mueve la vista |
| `--zoom Z`, `--center X Y` | Zoom y celda en el centro |
| `--night`, `--no-lighting` | Empezar de noche, o sin iluminación |
| `--tint`, `--materials`, `--depth` | Empezar con el tinte por altura, por material, o viendo la profundidad |
| `--cut fachada\|círculo\|planta\|no` | Cómo ocultar lo que tapa al personaje (por defecto, fachada) |
| `--seed N` | Semilla de la aparición y del comportamiento de los NPC |
| `--no-vsync` | Sin sincronía vertical (para medir los fps reales) |
| `--size W H` | Tamaño de la ventana |
| `--screenshot F.png` | Dibujar, guardar una captura y salir (con `--walk DX DZ S` o `--simulate S` antes) |

**Ocultar lo que tapa al personaje.** Solo en interiores; al aire libre nunca se oculta nada.
Un interior es una habitación **cerrada**: las celdas con techo unidas a la del personaje sin
cruzar paredes (con los huecos de escalera y los agujeros del techo, así que bajo un agujero se
sigue dentro), con como mucho un par de bordes, o un cuarto de su contorno, abiertos al exterior
sin pared (las puertas). Bajo un balcón, un porche o un puente, la región está abierta: es
exterior. En un interior, por defecto como una casa de muñecas (modo *fachada*), se ocultan el
techo y las plantas de arriba de la habitación, salvo la pared del fondo, que se ve entera; la
pared de delante queda en un zócalo. Los modos *círculo* y *planta* ocultan todo lo que hay por
encima de su planta, en un círculo o en todo el mapa.

**Silueta.** Donde algo tapa al jugador (detrás de una casa, por ejemplo), se dibuja su silueta
en azul claro a través de lo que lo tapa (tecla X para quitarla).

La barra de título muestra la versión, los fps, los personajes por zona de detalle
(cerca/medio/lejos), el estado de las rutinas, y los tiempos: simulación, **GPU** por tramo
(mundo, sombras, personajes y halo; medido con consultas de OpenGL) y **CPU** de preparar el dibujo.

---

## Hacer mapas en Tiled

Lo más fácil es partir de lo que genera `python -m tools.pieces`: el tileset
`assets/pieces/pieces.tsx` (226 piezas con todas sus propiedades) y los mapas de
`maps/gallery`, `maps/course`, `maps/levels` y `maps/terrain`. La galería muestra cada pieza
con su nombre.

### El mundo: celdas en plantas

El mundo es una rejilla de celdas de **2 × 2 m** en planta y **3 m** de alto. Cada **planta**
del mapa es un **grupo de capas** de Tiled con una propiedad entera `Z` (0, 1, 2…; negativa para
sótanos). Todas las capas de tiles de un grupo se combinan en las celdas de esa planta: da igual
en qué capa pongas cada pieza, salvo por el orden de dibujo.

```
pueblo.tmx                      propiedad level_height = 133
├── grupo "Planta 0"            Z = 0
│   ├── suelo                   (tiles de suelo, rampas, terreno)
│   ├── piezas                  (paredes, bloques macizos)
│   ├── materiales              (marcadores de material, invisibles)
│   ├── luces                   (capa de objetos: luces)
│   └── lugares                 (capa de objetos: lugares)
└── grupo "Planta 1"            Z = 1
    └── …
```

Si una celda recibe más de 4 tiles visibles, el compilador avisa. Los tiles se pueden voltear
en horizontal (el compilador refleja su geometría).

### Propiedades del tileset

| Propiedad | Ejemplo | |
|---|---|---|
| `cell_width_m`, `cell_height_m` | 2, 3 | Tamaño de la celda en metros |
| `row_height_px`, `level_height_px` | 92, 133 | Píxeles por fila de celdas y por planta |
| `normal_map`, `position_map`, `albedo_map` | `pieces.n.png`… | Mapas que acompañan a la imagen del tileset, con la misma disposición: normales, posición en la celda (PNG de 16 bits, para la profundidad por píxel) y color sin sombrear |
| `position_range_m` | `x 0 2; y 0 3.75; z 0 2` | Rango del mapa de posición |
| `surface_map` | `pieces.s.png` | Opcional: capa de superficie (metal, rugosidad, brillo propio y pulso; ver abajo) |

### Capa de superficie: metal, pulido y brillo

Una imagen RGBA de 8 bits con la misma disposición que la del tileset, indicada con la propiedad
`surface_map`. Cada píxel dice cómo es la superficie del píxel correspondiente:

| Canal | Significado | Sin capa |
|---|---|---|
| R | **metal**: 0 no metálico, 255 metal (refleja con su propio color: cobre, oro, acero) | 0 |
| G | **rugosidad**: 0 pulido como un espejo, 255 mate | 255 |
| B | **brillo propio**: 0 nada, 255 máximo; brilla con el color del propio píxel, también a oscuras, y con un halo alrededor | 0 |
| A | **pulso**: 0 brillo fijo; otro valor, el momento de un latido lento (2 s) en el que ese píxel está en su máximo | 0 |

Con el pulso se anima el brillo sin fotogramas: el mismo valor en una zona hace que lata a la
vez; un degradado a lo largo de una línea hace que la luz la recorra (la energía de un circuito,
unas flechas que se encienden una tras otra). Basta con que tenga la capa algún tileset: los
tiles sin ella son mates, sin metal ni brillo, y se ven como siempre.

En las superficies curvas (tubos, columnas, cúpulas), el **mapa de normales** debe llevar la
curvatura suave, como sale de un modelo con normales suavizadas: si cada cara tiene su normal
plana, el reflejo va a escalones y la pieza se ve a barras. El generador de piezas de prueba lo
hace así: el tubo se exporta con grupo de suavizado y una normal por vértice, como lo exportaría
Blender, y al dibujarlo se interpola la normal y solo lleva contorno en la silueta.

Coste: los píxeles mates, sin metal ni brillo, se calculan como si no hubiera capa; los demás
recorren las luces una sola vez para la luz difusa y los reflejos. El halo solo se calcula
cuando hay a la vista algún tile (o sprite) con brillo propio, y se difumina a 1/4 y 1/8 de
resolución.

Los **sprites** pueden llevar la misma capa en un archivo junto a su hoja, `hoja.s.png` (ojos que
brillan, armaduras metálicas). Las piezas de prueba incluyen tubos de cobre y de acero, un suelo
pulido y un suelo con líneas que brillan y laten (en `maps/levels`).

### Propiedades de cada tile: su parte de la celda

Cada tile declara qué pone en su celda. El compilador combina los tiles de cada celda y avisa
de los conflictos (por ejemplo, dos suelos distintos en la misma mitad).

| Propiedad | Valores |
|---|---|
| `walls` | Bordes con pared, como letras: `N`, `S`, `E`, `W` (por ejemplo `NW`) |
| `fill` | `solid`, medias celdas macizas `half_nw`, `half_ne`, `half_se`, `half_sw`, paredes diagonales finas `thin_slash`, `thin_backslash` |
| `floor` | `flat 0` … `flat 0.75`; rampas rectas (escaleras, 37°) `ramp N 0`, `ramp N 0.5`…; rampas diagonales `ramp NE 0` … `ramp NW 0.75`; pendientes suaves (21°) `slope N 0` … `slope W 0.75`. El número es la altura de partida en fracciones de planta |
| `floor_part` | `all` (por defecto) o una mitad de la celda partida por una diagonal: `nw`, `se`, `ne`, `sw` |
| `ceiling`, `ceiling_part` | Techo bajo a `0.75`, `0.5` o `0.25` de la planta, en toda la celda o en una mitad |
| `flags` | `ladder` (escalera de mano), `active` (celda con suceso) |
| `material` | Material del suelo (ver abajo) |

### Materiales del suelo

`assets/materials.json` lista los materiales **ordenados de peor a mejor** (barro, arena, roca,
hierba, tierra, suelo, camino, adoquín), con su coste de andar y un color para depurar. Un tile
declara el suyo con la propiedad `material`. Si en una celda hay varios (hierba y, encima, barro
con transparencia para la transición), **cuenta el peor**. Los tiles sin material (una flor,
una piedra decorativa) no cuentan.

Para pintar material sin cambiar el aspecto hay **marcadores**: piezas `material_*` invisibles
que se pintan en una capa aparte; no ocupan huecos de la celda ni sitio en el atlas. La tecla G
del visor tiñe el suelo según su material.

### Luces

Objetos punto de clase `light` en una capa de objetos dentro del grupo de su planta, con
propiedades opcionales `color` (por ejemplo `#ffd9a0`), `radius` (m), `intensity` y `height`
(m sobre el suelo de la planta). De día las luces puntuales se notan poco; de noche, del todo.

### Lugares

Rectángulos de clase `place` en una capa de objetos, con su nombre y una propiedad `kind`
(`casa`, `plaza`, `taberna`…). Son los destinos de la navegación: las rutinas de los NPC
usan las casas y la plaza.

### Sucesos

Los sucesos se definen en Tiled, sin tocar código: **rectángulos de clase `event`** en una capa
de objetos dentro del grupo de su planta.

| Propiedad | Valores | Por defecto |
|---|---|---|
| `when` | `enter`, `exit`, `inside` (cada segundo dentro), `use` (tecla E dentro) | `enter` |
| `who` | `player`, `all` o un tipo de personaje (`perro`) | `player` |
| `times` | cuántas veces puede ocurrir; `0` = siempre | `0` |
| `if` | condición: `key`, `not key`, `levers >= 3` (también `<=`, `>`, `<`, `==`, `!=`), con `and` y `or` | (siempre) |
| `do` | acciones, una por línea (texto de varias líneas en Tiled) | |
| `enabled` | `false` para empezar desactivado | `true` |

| Acción | |
|---|---|
| `message "texto"` | muestra un mensaje; `{variable}` se sustituye por su valor |
| `teleport marcador` | lleva a quien disparó el suceso a un **marcador**: un objeto punto de clase `marker`, con nombre |
| `set variable [valor]` | pone una variable (por defecto, 1) |
| `clear variable` | la quita (vale 0) |
| `add variable n` | le suma n |
| `wait segundos` | espera antes de seguir con las acciones siguientes |
| `enable suceso`, `disable suceso` | activa o desactiva los sucesos con ese nombre |
| `call función [args…]` | llama a una función de Python que el juego registra con `EventSystem.on("función", f)` |

Por ejemplo, tres palancas y un paso que solo se abre con las tres:

```
lever_1   when: use    times: 1   do: add levers 1
                                      message "Palanca {levers} de 3"
gate      when: enter  if: levers >= 3
                                  do: message "Las tres palancas abren el paso"
                                      wait 1
                                      teleport upstairs
```

El compilador comprueba las acciones y las condiciones, y que existan los marcadores y sucesos
que se nombran; los fallos son errores con el nombre del suceso y la línea. Las condiciones no
usan `eval`: un mapa no puede ejecutar código. El mapa `maps/levels` trae ejemplos: una placa en
la calle que lleva al balcón, un mensaje al entrar en el edificio, tres palancas y un paso.

---

## Compilar

```
python -m engine.compiler MAPA.tmx [-o SALIDA.cworld] [--chunk-size 32] [--force] [--no-nav]
```

El compilador lee el mapa y sus tilesets, combina los tiles de cada celda, comprueba la
geometría y escribe el mundo compilado. Los **errores** (un material desconocido, piezas
incompatibles en una celda) impiden escribirlo salvo con `--force`; los **avisos** (más de 4
tiles en una celda, un suelo que sobresale de su celda con la celda de arriba ocupada) indican
la posición. Al terminar muestra un resumen: celdas, chunks, regiones del atlas, luces,
lugares, nodos y aristas de navegación, tiempos y tamaño.

El `.cworld` es una base de datos SQLite con:

- **Chunks** de 32 × 32 columnas (configurable). Cada columna guarda solo las plantas que
  ocupa; por encima hay aire y por debajo, macizo.
- **Atlas** de texturas: color, posición, normales y albedo, con un borde de 16 px por región
  para los niveles reducidos (mipmaps) que se usan de lejos.
- **Luces, lugares, materiales, sucesos y marcadores.**
- **Grafo de navegación** (salvo con `--no-nav`): superficies transitables, sus vecinos, su
  holgura y su material. Se calcula con la misma física que mueve a los personajes, así que lo
  que el grafo dice que se puede andar, se puede.

---

## Personajes y NPC

Los tipos de personaje están en `assets/actors/types.json` (persona, niño, perro, guardia, y los generados maniqui, soldado y enemigo):
radio y altura del cilindro, escalón máximo, masa, velocidades, hoja de sprites, tinte y, para
la navegación, ajustes de los costes por material (`material_costs`) y de la pendiente
(`uphill`, `downhill`). Lo que un tipo no indica se toma de `defecto`.

### Hojas de sprites

Una hoja es un PNG con una fila por dirección y, en cada fila, los fotogramas. Sin nada más es
una hoja de las de siempre (4 × 4: filas abajo, izquierda, derecha y arriba; un ciclo de andar
de cuatro fotogramas), y el fotograma entero mide lo que el personaje. Junto a ella pueden ir:

| Archivo | |
|---|---|
| `hoja.json` | Disposición: tamaño del fotograma, píxel de los pies (`anchor`), escala (`px_per_m`, `height`), direcciones (4 u 8) y la fila de cada una, y las animaciones por nombre con sus columnas, fotogramas por segundo, velocidad y si se repiten (`loop`) |
| `hoja.a.png` | Albedo: el color sin sombrear, que es lo que ilumina el motor |
| `hoja.n.png` | Normal por píxel, en ejes del mundo, como las piezas |
| `hoja.p.png` | Posición por píxel en metros respecto a los pies (16 bits, con `position_range` del .json) |
| `hoja.s.png` | Capa de superficie (metal, rugosidad, brillo propio), ver arriba |

Con `anchor`, el sprite se dibuja a la escala con la que se generó (un píxel de la hoja, un píxel
de mundo) y con los pies en su píxel. Con albedo, normales y posición, cada píxel se ilumina en
su punto 3D y tiene su propia profundidad, como un tile: una luz a un lado alumbra ese lado, y un
pie que asoma delante de una pared se ve delante. Lo que sobresale del cilindro de física (el
fusil, una mano) puede rozar el escenario: lo que quedaría detrás de la cara norte del cilindro
se acerca hasta ella (ahí nunca hay escenario), los roces de hasta 12 cm los gana el personaje
y la silueta solo sale donde algo lo tapa más de 25 cm. Con 8 direcciones, el personaje mira en
diagonal; si la hoja tiene `correr`, se usa cuando va más rápido que la mitad del camino entre
andar y correr. El resto de animaciones (`agachado`, `andar_agachado`, `granada`, `morir`…) las
pide el juego por su nombre con `SpriteSheet.anim_cols(nombre, tiempo)`; las que no se repiten
se quedan en su último fotograma.

### Personajes generados: `tools.actors`

```
python -m tools.actors [NOMBRE…]
```

Hasta que haya arte de verdad, los personajes se generan por código: un cuerpo hecho de formas
redondeadas (elipsoides y cápsulas) en las poses de cada animación, renderizado con la misma
cámara que las piezas del mundo. Cada descripción `assets/actors/models/NOMBRE.json` produce
`assets/actors/sheets/NOMBRE.*`: la hoja en dos tonos con contorno, y su albedo, normales,
posición, superficie (si algo es metálico) y disposición, en 8 direcciones. La descripción da:

- `altura` en metros y `proporciones` (`cabeza`, `piernas`, `brazos`: factores sobre las del maniquí);
- `materiales`, cada uno con su tono claro y su sombra (`luz`, `sombra`) y, opcionales, `metal` y `rugosidad`;
- `partes`: el material de cada parte del cuerpo (cabeza, orejas, torso, brazos, manos, piernas,
  botas) y del equipo; las manos y las botas, si no se dicen, son como el brazo y la pierna;
- `equipo`: `casco`, `mochila`, `fusil`. Con fusil, quieto y andar llevan el arma apuntando
  hacia donde mira;
- `animaciones` además de `quieto`: `andar`, `correr`, `agachado`, `andar_agachado`, `granada`, `morir`.

Hay tres: `maniqui` (el de la hoja original, ≈ 25 s), `soldado` (el jugador) y `enemigo` (≈ 70 s
cada uno). Al morir, el personaje gira y cae de espaldas hacia el este o el oeste: en esta cámara,
alguien tendido de norte a sur se vería casi igual que de pie. Por eso los fotogramas de estas
hojas son anchos (caben tendidos).

```bash
python -m tools.actors soldado                                   # regenerar una hoja
python -m tools.actors.preview assets/actors/sheets/soldado.png --out previas   # un GIF por animación
python -m engine.viewer maps/levels/levels.cworld --player-type soldado --npc-type enemigo --npcs 30 --night
```

Un grafista puede sustituir los PNG respetando la disposición del `.json`; para el motor no hay
diferencia.

### Simulación por zonas

La simulación tiene **niveles de detalle** alrededor del jugador: a menos de 60 m, física y
comportamiento completos a 60 Hz; hasta 160 m, física a 30 Hz; más lejos, sin física (los NPC
con rutina siguen avanzando por su camino según el tiempo). Si un frame tarda mucho, la física
da como mucho 4 pasos por frame.

## Usar el motor desde código

```python
import numpy as np
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from engine.entities import ActorTable, load_types, BRAIN_WANDER
from engine.entities.world_sim import ActorSim

world = WorldView(SQLiteWorldStore("maps/levels/levels.cworld"))
actors = ActorTable(load_types())
sim = ActorSim(world, actors)

for p in sim.spawn_points(20, np.random.default_rng(1)):
    actors.spawn("persona", p, brain=BRAIN_WANDER)          # NPC que pasean
hero = actors.get(actors.spawn("persona", [12.0, 0.0, 9.0], brain=0))

hero.wish = (1.0, 0.0)          # andar hacia el este (X), a toda velocidad
sim.step(2.0)                   # 2 s de simulación a paso fijo (60 Hz)
print(hero.pos, hero.grounded)
print(world.cells(6, 4, 0))     # valor de 32 bits de la celda (6, 4) de la planta 0
```

Coordenadas: X hacia el este, Z hacia el sur, Y hacia arriba, en metros. La celda (x, y) de la
planta z ocupa [2x, 2x+2) × [2y, 2y+2) en planta y [3z, 3z+3) en altura.

Para la navegación: `engine.nav.NavGraph` (el grafo del mundo), `load_places`, `CostProfile`,
`FieldCache` (campos de flujo hacia un lugar) y `engine.entities.RoutineBrains` (rutinas).

---

## Herramientas de prueba: `tools.pieces`

```
python -m tools.pieces [--terrain-size 256] [--houses N] [--seed N] [--terrain RUTA.tmx]
```

Genera el catálogo de piezas de prueba (geometría `.obj` y su render oblicuo: color, normales,
posición y albedo), el tileset `assets/pieces/pieces.tsx` y cuatro mapas:

| Mapa | |
|---|---|
| `maps/gallery` | Todas las piezas, con su nombre |
| `maps/course` | Recorrido para la física: escaleras, habitación, muro diagonal, túnel, rampas, torre |
| `maps/levels` | Dos plantas: puente sobre una calle, edificio con escalera interior y agujero, balcón |
| `maps/terrain` | Terreno tipo Populous con casas de dos plantas, caminos que siguen el terreno, plaza y farolas (256 × 256 celdas = 512 m; `--terrain-size 512` para 1 km) |

## Estructura del proyecto

```
engine/                     El motor
  world/                    Celda de 32 bits, CellSpec, materiales, chunks, almacén SQLite, vista del mundo
  compiler/                 Tiled (.tmx) -> mundo compilado (.cworld)
  render/                   Dibujo del mundo y de los personajes (OpenGL 3.3), medidas de GPU
  physics/                  Física de andar para cilindros, vectorizada, con contactos entre ellos
  entities/                 Tipos de personaje, tabla de personajes, paseo, rutinas, simulación por zonas
  nav/                      Grafo de navegación, costes, caminos mínimos, campos de flujo, lugares
  events/                   Sucesos: su pequeño lenguaje (acciones y condiciones) y el sistema en el juego
  viewer.py                 Visor: python -m engine.viewer
tools/pieces/               Catálogo de piezas, render oblicuo, tileset y mapas de prueba (terreno incluido)
tools/actors/               Personajes generados: cuerpo, poses, render con la cámara del motor, hojas
assets/                     Tileset de prueba, sprites, types.json, materials.json, fuente (DejaVu Sans)
maps/                       Mapas de ejemplo (los .cworld se generan)
tests/                      Tests (pytest)
docs/                       Documentación antigua del explorador y de tmx_manager
tmx_manager.py              Lectura y escritura de TMX, independiente del motor
tmx_explorer/               Explorador antiguo de mapas TMX (ver abajo)
```

## Tests

```bash
python -m pytest -q                              # sin pantalla, los tests de GPU se saltan
xvfb-run -a python -m pytest -q                  # con pantalla virtual: todos
```

Los tests de dibujo comparan la GPU píxel a píxel con una referencia en CPU; los de física y
navegación hacen andar a personajes de verdad (por ejemplo, recorren aristas al azar del grafo
para comprobar que coincide con la física). Para depurar llamadas a OpenGL, ejecuta con
`ENGINE_GL_DEBUG=1` (por defecto PyOpenGL no comprueba errores, porque es caro).

## Rendimiento (orientativo)

- Terreno de 1 km² con 90 casas: se genera en ≈ 25 s (casi todo es trazar la red de caminos),
  se compila en ≈ 40 s (más de la mitad, el grafo de navegación) y ocupa unos 5 MB; cargarlo
  entero son 0.3 s. El de 512 m: ≈ 6 s y ≈ 9 s.
- 3000 personajes repartidos por 1 km²: ≈ 4 ms por paso de simulación con niveles de detalle.
- Dibujo: a zoom normal, unos 12 ms de GPU para el mundo a 2560 × 1400; a zoom muy lejano
  (todo el mapa), unos 15 ms.
- Un campo de flujo en el terreno de 512 m: ≈ 50 ms, calculado repartido entre frames.

## Limitaciones conocidas

- La luz atraviesa las paredes (no hay sombras de las luces puntuales).
- Los personajes no se esquivan con antelación: se apartan al chocar.
- Aún no hay zonas que se muevan con un personaje, puertas que se abran ni objetos.
- A zoom muy lejano, cada chunk se dibuja con todo su detalle; falta un dibujo simplificado.

---

## Herramientas antiguas

Antes del motor, el proyecto era un visor de mapas TMX. Sigue funcionando y es útil para ver
mapas de Tiled cualquiera (sin compilar):

```bash
python -m tmx_explorer maps/map01/VTiles1.tmx              # con jugador
python -m tmx_explorer maps/map01/VTiles1.tmx --npcs       # y NPC de demostración
python -m tmx_explorer maps/map02/map02.tmx --no-player    # solo visor
```

Usa capas con clase `floor`, `object`, `overhead` o `roof`, la propiedad `Z` para las plantas y
`solid = true` en los tiles para la colisión; su documentación está en `docs/architecture.md`.

`tmx_manager` lee y escribe TMX sin perder nada de lo que no modela (capas de imagen,
animaciones, polígonos, textos…), y se puede usar por separado:

```python
from tmx_manager import TiledMap

m = TiledMap.load("maps/map01/VTiles1.tmx")
m.layers[0].set_tile_gid(3, 4, 12)
m.save("copia.tmx", encoding="base64", compression="zlib")
```

Su API está en `docs/tmx_manager.md`.
