"""Punto de entrada: ``python -m tmx_explorer MAPA.tmx [opciones]``."""

import argparse
import logging
import sys
from pathlib import Path

from .demo import DEFAULT_PLAYER_SPRITE, add_demo_npcs, add_player


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tmx_explorer",
        description="Visor OpenGL de mapas Tiled (TMX) con niveles de altura y personajes.",
        epilog="Controles: WASD/flechas mover, Shift+WASD cámara, Q/E altura, rueda zoom, "
               "PgUp/PgDn nivel visible, G grid, I info, P profiling, C colisión, ESC salir.")
    parser.add_argument("map", type=Path, help="archivo .tmx")
    parser.add_argument("sprite", type=Path, nargs="?", default=DEFAULT_PLAYER_SPRITE,
                        help="spritesheet 4x4 del jugador (por defecto: %(default)s)")
    parser.add_argument("--no-player", action="store_true",
                        help="solo visor: sin personajes, WASD mueve la cámara")
    parser.add_argument("--npcs", action="store_true", help="añadir NPCs de demostración")
    parser.add_argument("-v", "--verbose", action="store_true", help="log detallado")
    return parser.parse_args(argv)


def main(argv=None, default_npcs: bool = False) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if not args.map.exists():
        print(f"Error: no se encuentra el archivo '{args.map}'", file=sys.stderr)
        return 1

    from .app import TMXExplorer   # importa GLFW/OpenGL solo si hace falta
    explorer = TMXExplorer(str(args.map))

    if not args.no_player:
        player = add_player(explorer, args.sprite)
        if player and (args.npcs or default_npcs):
            add_demo_npcs(explorer, player)

    explorer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
