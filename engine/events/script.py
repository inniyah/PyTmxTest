"""El pequeño lenguaje de los sucesos: acciones (propiedad ``do``) y condiciones (``if``).

Acciones, una por línea (las líneas vacías y las que empiezan por # se ignoran):

    message "texto"          muestra un mensaje; {variable} se sustituye por su valor
    teleport marcador        lleva a quien disparó el suceso a un marcador (objeto marker)
    set variable [valor]     pone una variable (por defecto, 1)
    clear variable           la quita (vale 0)
    add variable n           le suma n (puede ser negativo)
    wait segundos            espera antes de seguir con las siguientes acciones
    enable suceso            activa los sucesos con ese nombre
    disable suceso           los desactiva
    call función [args…]     llama a una función de Python registrada por el juego

Condiciones: ``llave``, ``not llave``, ``palancas >= 3`` (también <=, >, <, ==, !=), unidas con
``and`` y ``or`` (and va antes que or). Una variable sin poner vale 0.

No se usa eval: un mapa no puede ejecutar código arbitrario.
"""

import re
import shlex
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ACTIONS = {            # nombre: (mínimo de argumentos, máximo; None = sin límite)
    "message": (1, 1),
    "teleport": (1, 1),
    "set": (1, 2),
    "clear": (1, 1),
    "add": (2, 2),
    "wait": (1, 1),
    "enable": (1, 1),
    "disable": (1, 1),
    "call": (1, None),
}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class ScriptError(ValueError):
    def __init__(self, message: str, line: int = 0):
        super().__init__(f"línea {line}: {message}" if line else message)
        self.line = line


@dataclass(frozen=True)
class Action:
    name: str
    args: Tuple[str, ...]
    line: int = 0


def _number(text: str, what: str, line: int) -> float:
    try:
        return float(text)
    except ValueError:
        raise ScriptError(f"{what}: '{text}' no es un número", line) from None


def parse_actions(text: str) -> List[Action]:
    out = []
    for i, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line)
        except ValueError as e:
            raise ScriptError(f"comillas sin cerrar ({e})", i) from None
        name, args = tokens[0].lower(), tuple(tokens[1:])
        if name not in ACTIONS:
            raise ScriptError(f"acción desconocida '{tokens[0]}' (hay: {', '.join(ACTIONS)})", i)
        lo, hi = ACTIONS[name]
        if len(args) < lo or (hi is not None and len(args) > hi):
            want = f"{lo}" if lo == hi else (f"{lo} o más" if hi is None else f"{lo} o {hi}")
            raise ScriptError(f"'{name}' lleva {want} argumento(s), no {len(args)}", i)
        if name in ("set", "clear", "add", "enable", "disable", "teleport", "call") and not NAME.match(args[0]):
            raise ScriptError(f"'{args[0]}' no es un nombre válido", i)
        if name == "add":
            _number(args[1], "add", i)
        if name == "wait" and _number(args[0], "wait", i) < 0:
            raise ScriptError("wait no puede ser negativo", i)
        out.append(Action(name, args, i))
    return out


# --- Condiciones ---------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(>=|<=|==|!=|>|<|-?\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_.]*|\S)")
_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
        "<": lambda a, b: a < b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


@dataclass
class Condition:
    """Or de ands de comparaciones: [[(variable, operador, número, negada), …], …]."""
    clauses: List[List[tuple]] = field(default_factory=list)
    text: str = ""

    def __call__(self, variables: dict) -> bool:
        if not self.clauses:
            return True

        def value(name):
            v = variables.get(name, 0)
            try:
                return float(v)
            except (TypeError, ValueError):
                return 1.0 if v else 0.0
        for clause in self.clauses:
            if all((not _OPS[op](value(name), num)) if neg else _OPS[op](value(name), num)
                   for name, op, num, neg in clause):
                return True
        return False


def parse_condition(text: Optional[str]) -> Condition:
    text = (text or "").strip()
    if not text:
        return Condition([], "")
    tokens = [t for t in _TOKEN.findall(text) if t]
    clauses, current, i = [], [], 0

    def take():
        nonlocal i
        if i >= len(tokens):
            raise ScriptError(f"condición incompleta: '{text}'")
        i += 1
        return tokens[i - 1]
    while i < len(tokens):
        neg = False
        tok = take()
        if tok == "not":
            neg, tok = True, take()
        if not NAME.match(tok) or tok in ("and", "or", "not"):
            raise ScriptError(f"en la condición '{text}': se esperaba un nombre y hay '{tok}'")
        if i < len(tokens) and tokens[i] in _OPS:
            op = take()
            num = take()
            try:
                num = float(num)
            except ValueError:
                raise ScriptError(f"en la condición '{text}': '{num}' no es un número") from None
            current.append((tok, op, num, neg))
        else:
            current.append((tok, "!=", 0.0, neg))                  # verdadera si no es 0
        if i < len(tokens):
            joiner = take()
            if joiner == "or":
                clauses.append(current)
                current = []
            elif joiner != "and":
                raise ScriptError(f"en la condición '{text}': se esperaba and/or y hay '{joiner}'")
            elif i >= len(tokens):
                raise ScriptError(f"condición incompleta: '{text}'")
    clauses.append(current)
    return Condition(clauses, text)
