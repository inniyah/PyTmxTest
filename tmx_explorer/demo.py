"""Escena de ejemplo: jugador en el centro del mapa y algunos NPCs alrededor."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .config import ASSETS_DIR

if TYPE_CHECKING:
    from .app import TMXExplorer
    from .entities import Character

log = logging.getLogger(__name__)

DEFAULT_PLAYER_SPRITE = ASSETS_DIR / "people" / "guy.png"
NPC_SPRITES = [ASSETS_DIR / "people" / name for name in ("gal.png", "kid.png", "fat.png")]


def add_player(explorer: "TMXExplorer", sprite_path: Path) -> Optional["Character"]:
    if not Path(sprite_path).exists():
        log.warning("No se encuentra el spritesheet '%s'", sprite_path)
        return None
    map_w, map_h = explorer.map_pixel_size
    player = explorer.add_character(str(sprite_path), x=map_w / 2, y=map_h / 2, z=0,
                                    speed=150.0, is_player=True)
    log.info("Jugador en (%.0f, %.0f)", player.x, player.y)
    return player


def add_demo_npcs(explorer: "TMXExplorer", player: "Character"):
    """Un seguidor, un vagabundo por sprite y una patrulla en rectángulo."""
    sprites = [str(p) for p in NPC_SPRITES if p.exists()]
    if not sprites:
        return
    em = explorer.entity_manager
    cx, cy = player.x, player.y

    em.create_npc_follower(sprites[0], target=player, x=cx + 100, y=cy + 100, speed=120.0)

    for i, sprite in enumerate(sprites):
        em.create_npc_wanderer(sprite, x=cx + (i - len(sprites) // 2) * 300, y=cy + 200,
                               radius=250.0, speed=50.0 + i * 10)

    if len(sprites) > 1:
        em.create_npc_patrol(sprites[1], points=[(cx - 200, cy - 200), (cx + 200, cy - 200),
                                                 (cx + 200, cy + 200), (cx - 200, cy + 200)],
                             speed=80.0)
    log.info("NPCs creados: %d", em.npc_count)
