"""Core data types and orientation math for the graph solver.

Conventions (used everywhere in this package)
--------------------------------------------
* Sides / directions are numbered clockwise: 0=TOP, 1=RIGHT, 2=BOTTOM, 3=LEFT.
* A piece's ``sides`` and ``core`` are in its *own* (as-scanned) orientation.
* A placement ``rot`` = number of clockwise quarter turns applied to the piece.
  After rotating by ``rot``, the piece's own side ``s`` faces world direction
  ``(s + rot) % 4``.
* Grid coordinates are ``(row, col)``; row grows downward.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
DIR_DELTA = {TOP: (-1, 0), RIGHT: (0, 1), BOTTOM: (1, 0), LEFT: (0, -1)}


class SideKind(IntEnum):
    FLAT = 0
    TAB = 1
    BLANK = 2


def complementary(k1: int, k2: int) -> bool:
    """Only a tab and a blank can interlock."""
    return {int(k1), int(k2)} == {SideKind.TAB, SideKind.BLANK}


@dataclass
class Piece:
    id: int
    core: np.ndarray  # (T, T, C) float image of the square body (no tabs)
    sides: tuple[SideKind, SideKind, SideKind, SideKind]  # TOP, RIGHT, BOTTOM, LEFT

    @property
    def n_flat(self) -> int:
        return sum(1 for s in self.sides if s == SideKind.FLAT)

    @property
    def is_border(self) -> bool:
        return self.n_flat >= 1

    @property
    def is_corner(self) -> bool:
        return self.n_flat >= 2


def facing_dir(side: int, rot: int) -> int:
    """World direction that own side ``side`` faces after ``rot`` cw turns."""
    return (side + rot) % 4


def side_facing(direction: int, rot: int) -> int:
    """Own side that ends up facing world ``direction`` after ``rot`` cw turns."""
    return (direction - rot) % 4


def place_neighbor(r: int, c: int, rot: int, a: int, b: int) -> tuple[int, int, int]:
    """Piece at (r, c, rot) meets another piece using its side ``a`` against the
    other's side ``b``. Return where the other piece must sit: (r2, c2, rot2)."""
    d = facing_dir(a, rot)
    dr, dc = DIR_DELTA[d]
    rot2 = ((d + 2) - b) % 4  # so that facing_dir(b, rot2) == opposite of d
    return r + dr, c + dc, rot2


def rotate_vec_cw(dr: int, dc: int, times: int) -> tuple[int, int]:
    """Rotate a grid offset clockwise by ``times`` quarter turns."""
    for _ in range(times % 4):
        dr, dc = dc, -dr
    return dr, dc


def rotated_core(piece: Piece, rot: int) -> np.ndarray:
    return np.rot90(piece.core, k=-rot)  # negative k = clockwise
