#!/usr/bin/env python3
"""
TMX Explorer - GLFW Version

Uso:
    python run_explorer.py <archivo.tmx> [spritesheet.png]
    
Ejemplo:
    python run_explorer.py map01/VTiles1.tmx assets/people/guy.png
    
Controles:
    WASD/Flechas - Mover personaje
    Shift+WASD   - Mover cámara
    Rueda ratón  - Zoom
    Arrastrar    - Pan cámara
    PgUp/PgDn    - Cambiar nivel de altura
    G            - Mostrar/ocultar grid
    I            - Mostrar/ocultar info
    P            - Mostrar/ocultar profiling
    ESC/Q        - Salir

Requisitos:
    pip install glfw PyOpenGL PyOpenGL_accelerate pillow numpy
"""

import sys
from pathlib import Path

# Importar GLFW version
from tmx_explorer.app import TMXExplorer


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: Debes especificar un archivo TMX")
        sys.exit(1)
    
    tmx_path = sys.argv[1]
    
    if not Path(tmx_path).exists():
        print(f"Error: No se encuentra el archivo '{tmx_path}'")
        sys.exit(1)
    
    explorer = TMXExplorer(tmx_path)
    
    # Si se especifica un spritesheet, crear un personaje jugador
    if len(sys.argv) >= 3:
        sprite_path = sys.argv[2]
    else:
        # Buscar spritesheet por defecto
        default_paths = ["assets/people/guy.png"]
        sprite_path = None
        for path in default_paths:
            if Path(path).exists():
                sprite_path = path
                break
    
    if sprite_path and Path(sprite_path).exists():
        start_x = explorer.map_3d.map_width * explorer.map_3d.tile_width / 2
        start_y = explorer.map_3d.map_height * explorer.map_3d.tile_height / 2
        
        player = explorer.add_character(
            sprite_path,
            x=start_x,
            y=start_y,
            z=0,
            speed=150.0,
            is_player=True
        )
        print(f"\nPersonaje creado en ({start_x:.0f}, {start_y:.0f})")
        print("Usa WASD o flechas para mover el personaje")
    elif sprite_path:
        print(f"Advertencia: No se encuentra el spritesheet '{sprite_path}'")
    
    explorer.run()


if __name__ == "__main__":
    main()
