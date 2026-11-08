"""Sucesos definidos en Tiled: el lenguaje, el compilador y el sistema en el juego."""
import numpy as np
import pytest

from engine.compiler import compile_tmx
from engine.entities import ActorTable, load_types
from engine.events import EventSystem, ScriptError, parse_actions, parse_condition
from engine.events.system import Event
from engine.physics import WalkerPhysics
from engine.world.store import SQLiteWorldStore
from engine.world.view import WorldView
from tools.pieces.catalog import build_catalog
from tools.pieces.course import CourseBuilder
from tools.pieces.levels import build_levels
from tools.pieces.render import DEFAULT_PROJECTION

CW, CH = 2.0, 3.0


# --- El lenguaje ----------------------------------------------------------------------------

def test_actions_parse_and_report_the_bad_line():
    actions = parse_actions('message "Hola {x}"\n\n# comentario\nteleport tower\nadd x -1\nwait 0.5\ncall f a b')
    assert [a.name for a in actions] == ["message", "teleport", "add", "wait", "call"]
    assert actions[1].line == 4 and actions[-1].args == ("f", "a", "b")
    for bad, line in (("message ok\nfly away", 2), ("add x", 1), ("wait -1", 1), ('message "abierta', 1),
                      ("set 3x", 1)):
        with pytest.raises(ScriptError) as e:
            parse_actions(bad)
        assert e.value.line == line


def test_conditions():
    c = parse_condition("levers >= 3 and not trap or key")
    assert c({"levers": 3}) and not c({"levers": 3, "trap": 1}) and c({"key": 1}) and not c({})
    assert parse_condition("")({}) and parse_condition("a != 2")({}) and not parse_condition("a")({"a": 0})
    for bad in ("levers >=", "levers > tres", "a xor b", "and"):
        with pytest.raises(ScriptError):
            parse_condition(bad)


# --- El compilador ------------------------------------------------------------------------

def _map_with(tmp_path, tsx, events, markers=()):
    b = CourseBuilder(6, 4, build_catalog())
    for y in range(4):
        for x in range(6):
            b.put(0, x, y, "floor_flat_000")
    for args, props in events:
        b.event(*args, **props)
    for m in markers:
        b.marker(*m)
    tmx = tmp_path / "e.tmx"
    b.to_map(tsx, tmx, DEFAULT_PROJECTION).save(tmx)
    return compile_tmx(tmx, tmp_path / "e.cworld", nav=False)


@pytest.mark.parametrize("props, wanted", [
    ({"do": "fly away"}, "acción desconocida"),
    ({"do": "teleport nowhere"}, "marcador 'nowhere'"),
    ({"do": "enable ghost"}, "suceso 'ghost'"),
    ({"when": "sometimes", "do": "set a"}, "when"),
    ({"if": "a >", "do": "set a"}, "condición"),
])
def test_compiler_rejects_bad_events(tmp_path, shared_tileset, props, wanted):
    report = _map_with(tmp_path, shared_tileset, [((0, 1, 1, 2, 2, "bad"), props)])
    assert not report.ok and any(wanted in e and "'bad'" in e for e in report.errors), report.errors


def test_compiler_stores_events_and_markers(tmp_path, shared_tileset):
    report = _map_with(tmp_path, shared_tileset,
                       [((0, 1, 1, 3, 2, "door"), {"when": "enter", "times": 2, "if": "key", "do": "teleport out"})],
                       [(0, 4.5, 2.5, "out")])
    assert report.ok and report.stats["sucesos"] == 1 and report.stats["marcadores"] == 1
    store = SQLiteWorldStore(tmp_path / "e.cworld")
    (row,) = store.events()
    assert row[1:7] == ("door", 0, 1, 1, 3, 2) and row[7:11] == ("enter", "player", 2, "key")
    assert store.markers() == [("out", 0, 4.5 * CW, 2.5 * CW)]


# --- El sistema en el juego ----------------------------------------------------------------

def _system(*events, markers=None):
    evs = [Event(i + 1, name, 0, x0, y0, x1, y1, when, who, times, parse_condition(cond), parse_actions(do), enabled)
           for i, (name, (x0, y0, x1, y1), when, who, times, cond, do, enabled) in enumerate(events)]
    return EventSystem(evs, markers or {}, CW, CH)


def _ev(name, rect, do, when="enter", who="player", times=0, cond="", enabled=True):
    return (name, rect, when, who, times, cond, do, enabled)


def _place(table, handle, cx, cy):
    table.pos[table.row(handle)] = ((cx + 0.5) * CW, 0.0, (cy + 0.5) * CW)


@pytest.fixture
def actors():
    t = ActorTable(load_types())
    return t, t.spawn("persona", [0.0, 0.0, 0.0], brain=0), t.spawn("perro", [0.0, 0.0, 0.0], brain=0)


def test_enter_exit_inside_and_who(actors):
    t, player, dog = actors
    s = _system(_ev("in", (2, 2, 4, 4), "add enters 1"), _ev("out", (2, 2, 4, 4), "add exits 1", when="exit"),
                _ev("tick", (2, 2, 4, 4), "add ticks 1", when="inside"),
                _ev("dogs", (2, 2, 4, 4), "add dogs 1", who="perro"), _ev("all", (2, 2, 4, 4), "add all 1", who="all"))
    s.update(t, 0.1, player=player)
    assert not s.variables                                     # nadie dentro
    _place(t, player, 2, 2)
    _place(t, dog, 3, 3)
    s.update(t, 0.1, player=player)
    assert s.variables == {"enters": 1, "ticks": 1, "dogs": 1, "all": 2}
    for _ in range(25):                                        # 2.5 s dentro
        _place(t, player, 3, 2)                                # moverse dentro no vuelve a entrar
        s.update(t, 0.1, player=player)
    assert s.variables["enters"] == 1 and s.variables["ticks"] == 3
    _place(t, player, 6, 6)
    s.update(t, 0.1, player=player)
    assert s.variables["exits"] == 1


def test_use_times_condition_and_enable(actors):
    t, player, _dog = actors
    s = _system(_ev("lever", (1, 1, 2, 2), "add levers 1", when="use", times=2),
                _ev("gate", (5, 5, 6, 6), "set opened", cond="levers >= 2"),
                _ev("trap", (7, 7, 8, 8), "set caught", enabled=False),
                _ev("switch", (9, 9, 10, 10), "enable trap"))
    _place(t, player, 1, 1)
    for _ in range(4):
        s.update(t, 0.1, player=player, use=True)
    s.update(t, 0.1, player=player)                            # sin pulsar: no cuenta
    assert s.variables["levers"] == 2                          # times = 2
    _place(t, player, 7, 7)
    s.update(t, 0.1, player=player)
    assert "caught" not in s.variables                         # desactivado
    _place(t, player, 9, 9)
    s.update(t, 0.1, player=player)
    _place(t, player, 7, 7)
    s.update(t, 0.1, player=player)
    assert s.variables.get("caught") == 1                      # activado por "switch"
    _place(t, player, 5, 5)
    s.update(t, 0.1, player=player)
    assert s.variables.get("opened") == 1


def test_wait_messages_variables_and_call(actors):
    t, player, _dog = actors
    called = []
    s = _system(_ev("talk", (1, 1, 2, 2), 'set n 3\nmessage "Hay {n} y {nada}"\nwait 1\nclear n\n'
                                          'add m 2.5\ncall greet hola'))
    s.on("greet", lambda system, actor, word: called.append((actor, word)))
    _place(t, player, 1, 1)
    s.update(t, 0.1, player=player)
    assert [m.text for m in s.messages] == ["Hay 3 y 0"] and s.variables == {"n": 3.0}
    for _ in range(12):
        s.update(t, 0.1, player=player)
    assert s.variables == {"m": 2.5} and called == [(player, "hola")]
    for _ in range(40):
        s.update(t, 0.1, player=player)
    assert not s.messages                                      # los mensajes caducan


# --- El mapa de prueba -------------------------------------------------------------------

def test_levels_map_plate_and_gate(tmp_path, shared_tileset):
    build_levels(tmp_path / "l.tmx", shared_tileset)
    assert compile_tmx(tmp_path / "l.tmx", tmp_path / "l.cworld", nav=False).ok
    store = SQLiteWorldStore(tmp_path / "l.cworld")
    physics = WalkerPhysics(WorldView(store))
    s = EventSystem.from_store(store, CW, CH)
    t = ActorTable(load_types())
    player = t.spawn("persona", [7.0, 0.0, 29.0], brain=0)
    s.update(t, 0.1, physics, player)
    _place(t, player, 3, 12)                                   # la placa
    s.update(t, 0.1, physics, player)
    assert np.allclose(t.pos[t.row(player)], (31.8, 3.0, 21.0))   # en el balcón, sobre su suelo
    _place(t, player, 23, 14)
    s.update(t, 0.1, physics, player)
    assert "faltan" in s.messages[-1].text
    s.variables["levers"] = 3
    _place(t, player, 20, 14)
    s.update(t, 0.1, physics, player)
    _place(t, player, 23, 14)
    for _ in range(15):
        s.update(t, 0.1, physics, player)
    assert t.pos[t.row(player)][1] == pytest.approx(3.0)       # arriba, en la planta 1
