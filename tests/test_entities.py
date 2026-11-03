from conftest import ROOT
from tmx_explorer.entities import Direction, EntityManager
from tmx_explorer.map.collision import FLAG_SOLID, CollisionMap
from tmx_explorer.renderer.atlas import TextureAtlas, layout_atlas

SPRITE = str(ROOT / "assets" / "people" / "guy.png")


def test_sprites_share_frames_and_gpu_atlas():
    em = EntityManager()
    a = em.create_character(SPRITE)
    b = em.create_npc(SPRITE)
    assert a.sprite is not b.sprite                 # estado de animación propio
    assert a.sprite.frames is b.sprite.frames       # imágenes compartidas
    assert a.sprite._gpu is b.sprite._gpu           # un solo atlas en GPU


def test_walk_cycle_skips_idle_frame():
    sprite = EntityManager().load_sprite(SPRITE)
    sprite.set_walking(True)
    seen = []
    for _ in range(7):
        sprite.update(1 / sprite.animation_speed)
        seen.append(sprite.current_frame)
    assert seen == [2, 3, 1, 2, 3, 1, 2]
    sprite.set_walking(False)
    sprite.update(0.1)
    assert sprite.current_frame == 0


def test_wall_sliding():
    collision = CollisionMap(10, 1, 10)
    collision.data[0, :, 5] = FLAG_SOLID            # pared vertical en x = 5
    em = EntityManager(tile_width=32, tile_height=32, collision_map=collision, max_z=1)
    c = em.create_character(SPRITE, x=4.5 * 32, y=4.5 * 32, speed=100)
    c.move(1, 1)
    for _ in range(60):
        c.update(1 / 60)
    assert c.x < 5 * 32                             # no atraviesa la pared...
    assert c.y > 4.5 * 32 + 30                      # ...pero se desliza en Y
    assert c.sprite.direction in (Direction.DOWN, Direction.RIGHT)


def test_patrol_visits_points():
    em = EntityManager()
    pts = [(0, 0), (100, 0)]
    npc = em.create_npc_patrol(SPRITE, points=pts, speed=200)
    for _ in range(200):
        npc.update(1 / 60)
    assert npc.patrol_index in (0, 1) and npc.x > 0


def test_long_frame_does_not_tunnel_through_walls():
    collision = CollisionMap(20, 1, 3)
    collision.data[0, :, 10] = FLAG_SOLID                # pared de un tile en x = 10
    em = EntityManager(tile_width=32, tile_height=32, collision_map=collision, max_z=1)
    c = em.create_character(SPRITE, x=8.5 * 32, y=1.5 * 32, speed=150)
    c.move(1, 0)
    c.update(1.5)                                         # frame de 1.5 s
    assert c.x < 10 * 32


def test_characters_above_view_level_are_not_drawn():
    em = EntityManager(max_z=3)
    low = em.create_character(SPRITE, z=0)
    em.create_character(SPRITE, z=2)
    sprite = low.sprite
    _, regions = layout_atlas({(d, i): f for d, fr in sprite.frames.items()
                               for i, f in enumerate(fr)}, 1024)
    sprite._gpu.atlas = TextureAtlas([object()], regions)     # atlas falso, sin GPU
    count = lambda max_z: sum(len(v) for v in em.collect_render_data(0, max_z=max_z).values()) // 4
    assert count(None) == 2 and count(2) == 2 and count(1) == 1
