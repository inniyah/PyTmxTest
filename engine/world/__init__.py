"""Formato del mundo: celdas de 32 bits, su especificación legible, almacenamiento y consultas."""

from . import cell
from .cellspec import CellConflict, CellSpec

__all__ = ["cell", "CellSpec", "CellConflict"]
