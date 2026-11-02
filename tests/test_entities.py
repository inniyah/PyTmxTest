from conftest import ROOT
from tmx_explorer.entities import Character, Direction, EntityManager
from tmx_explorer.map.collision import FLAG_SOLID, CollisionMap

SPRITE = str(ROOT / "assets" / "people" / "guy.png")


def test_sprites_share_frames_and_texture_cache():
    em = EntityManager()
    a = em.create_character(SPRITE)
    b = em.create_npc(SPRITE)
    assert a.sprite is not b.sprite                 # estado de animación propio
    assert a.sprite.frames is b.sprite.frames       # imágenes compartidas
    assert a.sprite.textures is b.sprite.textures   # texturas GPU compartidas


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
