#!/usr/bin/env python3

"""
TMX Map Explorer - OpenGL Viewer for Tiled Maps

Usage:
    python -m tmx_explorer <map.tmx>
    
Controls:
    Mouse Drag  - Pan view
    Mouse Wheel - Zoom in/out
    WASD/Arrows - Pan view
    +/-         - Zoom in/out
    Shift +/-   - Adjust level height offset
    PgUp/PgDn   - Change visible height level
    G           - Toggle grid
    I           - Toggle info panel
    P           - Toggle profiling
    Space       - Reset view
    ESC/Q       - Quit
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source_path = sys.argv[1]

    if not Path(source_path).exists():
        print(f"Error: File '{source_path}' not found")
        sys.exit(1)

    try:
        from .app import TMXExplorer
        explorer = TMXExplorer(source_path)
        explorer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
