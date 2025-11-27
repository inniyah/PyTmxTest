#!/usr/bin/env python3
"""
TMX Explorer - Visor de mapas TMX con personajes

Uso:
    python run_explorer.py <archivo.tmx> [spritesheet.png]
    
Ejemplo:
    python run_explorer.py mapas/mi_mapa.tmx sprites/player.png
    
Controles:
    WASD/Flechas - Mover personaje (si hay) o cámara
    Shift+WASD   - Mover cámara (cuando hay personaje)
    Rueda ratón  - Zoom
    Arrastrar    - Pan cámara
    PgUp/PgDn    - Cambiar nivel de altura
    G            - Mostrar/ocultar grid
    I            - Mostrar/ocultar info
    P            - Mostrar/ocultar profiling
    ESC/Q        - Salir
"""

import sys
from pathlib import Path

from tmx_explorer import TMXExplorer


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
        # Buscar spritesheet por defecto en ubicaciones comunes
        default_paths = [
            "assets/people/guy.png",
        ]
        sprite_path = None
        for path in default_paths:
            if Path(path).exists():
                sprite_path = path
                break
    
    if sprite_path and Path(sprite_path).exists():
        # Posición inicial en el centro del mapa
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
        print("Shift+WASD para mover la cámara")
    elif sprite_path:
        print(f"Advertencia: No se encuentra el spritesheet '{sprite_path}'")
    
    explorer.run()


if __name__ == "__main__":
    main()