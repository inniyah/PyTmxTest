"""Constantes compartidas. Cambia aquí en vez de buscar números mágicos por el código."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

# Ventana
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "TMX Explorer"
VSYNC = True

# Tiempo
MAX_FRAME_DT = 0.1             # s; un frame más largo (ventana arrastrada, pausa) se recorta
MAX_STEP_TILES = 0.25          # el movimiento se divide en pasos de como mucho esto

# Vista
LEVEL_HEIGHT_OFFSET = 128      # px que sube cada nivel en pantalla (el mapa puede fijarlo
                               # con la propiedad "level_height")
LEVEL_HEIGHT_STEP = 16         # ajuste con Shift +/-
INITIAL_FIT_FACTOR = 0.85      # margen al encuadrar el mapa al arrancar
KEY_ZOOM_FACTOR = 1.2
WHEEL_ZOOM_FACTOR = 1.1
CAMERA_KEY_SPEED = 500.0       # px de pantalla por segundo con Shift+WASD
TILE_CULL_MARGIN = 3           # tiles extra alrededor de la pantalla

# Profundidad: menor valor = más al fondo. Tiles y personajes comparten la escala:
#   depth = DEPTH_BASE + fila_y + z (+ 0.1 * capa para tiles, + 0.5 para personajes)
DEPTH_BASE = -1000.0
DEPTH_LAYER_STEP = 0.1
DEPTH_CHARACTER_BIAS = 0.5
DEPTH_NEAR, DEPTH_FAR = -10000.0, 10000.0

# Render
MAX_SPRITES_PER_BATCH = 20000
ATLAS_MAX_SIZE = 4096          # lado máximo de cada página del atlas (se limita al de la GPU)
TEXTURE_BORDER = 1             # px extruidos alrededor de cada tile (anti-bleeding)
HUD_REFRESH_FRAMES = 6         # el HUD se regenera cada N frames
HUD_FONT_SIZE = 14
HUD_FONT_CANDIDATES = (
    "DejaVuSansMono.ttf",                                   # si está en el path de PIL
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
    "/Library/Fonts/Menlo.ttc",                             # macOS
    "/System/Library/Fonts/Menlo.ttc",
    "C:/Windows/Fonts/consola.ttf",                         # Windows
)

# Personajes
SPRITE_COLS = 4
SPRITE_ROWS = 4
ANIMATION_FPS = 8.0
DIAGONAL_FACTOR = 0.7071       # 1/sqrt(2): misma velocidad en diagonal
HEIGHT_SPEED_FACTOR = 0.01     # velocidad en Z = speed * este factor (niveles/s)
DEFAULT_COLLISION_SIZE = (0.5, 0.5, 0.85)   # ancho, fondo, alto en tiles

# Entrada
GAMEPAD_DB = ASSETS_DIR / "gamecontrollerdb.txt"
GAMEPAD_DEADZONE = 0.15
GAMEPAD_ACTIVE_THRESHOLD = 0.1
GAMEPAD_RESCAN_SECONDS = 1.0   # cada cuánto buscar mando si no hay ninguno

# Bits de flip en los GID de Tiled (los 4 bits altos)
GID_FLIP_H = 0x80000000
GID_FLIP_V = 0x40000000
GID_FLIP_D = 0x20000000
GID_ROTATE_HEX = 0x10000000
GID_MASK = 0x0FFFFFFF
