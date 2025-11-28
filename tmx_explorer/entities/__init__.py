"""
Entity system for characters and NPCs
"""

from .sprite import AnimatedSprite, Direction, AnimationState
from .character import Character, NPCBehavior
from .manager import EntityManager

__all__ = [
    "AnimatedSprite",
    "Direction", 
    "AnimationState",
    "Character",
    "NPCBehavior",
    "EntityManager",
]
