"""Graph-based jigsaw solver.

Pieces are nodes, candidate side-to-side matches are weighted edges.
Geometry (flat/tab/blank + border rule) prunes impossible edges, visual
continuity scores the rest, and assembly is a constrained spanning tree
followed by trim + refill cleanup.
"""

from .assembly import assemble
from .candidates import CandidateGraph, Edge, build_candidate_graph, compute_loop_support
from .geometry import classify_sides, find_core_box, piece_from_mask
from .pieces import BOTTOM, LEFT, RIGHT, TOP, Piece, SideKind
from .solve import SolveResult, neighbor_accuracy, render, solve

__all__ = [
    "Piece", "SideKind", "TOP", "RIGHT", "BOTTOM", "LEFT",
    "classify_sides", "find_core_box", "piece_from_mask",
    "CandidateGraph", "Edge", "build_candidate_graph", "compute_loop_support",
    "assemble", "solve", "render", "neighbor_accuracy", "SolveResult",
]
