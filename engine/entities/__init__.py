"""Personajes: tipos en datos, tabla con identificadores estables, comportamiento de paseo."""

from .brains import BRAIN_NONE, BRAIN_WANDER, WanderBrains
from .table import Actor, ActorTable
from .types import ActorType, load_types

__all__ = ["Actor", "ActorTable", "ActorType", "load_types", "WanderBrains", "BRAIN_NONE", "BRAIN_WANDER"]
