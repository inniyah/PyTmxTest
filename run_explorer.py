#!/usr/bin/env python3
"""Atajo compatible con la versión anterior: igual que ``python -m tmx_explorer --npcs``.

    python run_explorer.py maps/map01/VTiles1.tmx [assets/people/guy.png]
"""

import sys

from tmx_explorer.__main__ import main

if __name__ == "__main__":
    sys.exit(main(default_npcs=True))
