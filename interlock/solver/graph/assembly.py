"""Assembly as a constrained spanning tree (Kruskal-style cluster merging).

* Every piece starts as its own cluster with its own local frame.
* Edges are processed best-first: loop-supported edges first, then by
  confidence.
* For edge (i, a, j, b) with i, j in different clusters, the edge fixes where
  j sits (and how it is rotated) in i's frame. That gives one rigid transform
  (rotation + translation) that maps all of j's cluster into i's frame.
* The merge is rejected if
    - two pieces would land in the same cell,
    - the merged cluster no longer fits the puzzle's dimensions,
    - any newly adjacent pair is not tab<->blank,
    - a flat side would have something beyond it (border lines violated).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .candidates import CandidateGraph, Edge
from .pieces import (DIR_DELTA, Piece, SideKind, complementary, facing_dir,
                     place_neighbor, rotate_vec_cw, side_facing)

Placement = dict[int, tuple[int, int, int]]  # piece id -> (row, col, rot)


@dataclass
class Cluster:
    members: Placement = field(default_factory=dict)
    cells: dict[tuple[int, int], int] = field(default_factory=dict)

    def add(self, pid: int, r: int, c: int, rot: int) -> None:
        self.members[pid] = (r, c, rot)
        self.cells[(r, c)] = pid


def fits(members: Placement, rows: int, cols: int) -> bool:
    rs = [r for r, _, _ in members.values()]
    cs = [c for _, c, _ in members.values()]
    h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
    return (h <= rows and w <= cols) or (h <= cols and w <= rows)


def border_lines_ok(members: Placement, pieces: list[Piece]) -> bool:
    """A piece whose flat side faces TOP must be in the cluster's top-most row
    (nothing may be above it anywhere), etc."""
    rs = [r for r, _, _ in members.values()]
    cs = [c for _, c, _ in members.values()]
    rmin, rmax, cmin, cmax = min(rs), max(rs), min(cs), max(cs)
    for pid, (r, c, rot) in members.items():
        for s, kind in enumerate(pieces[pid].sides):
            if kind != SideKind.FLAT:
                continue
            d = facing_dir(s, rot)
            if (d == 0 and r != rmin) or (d == 2 and r != rmax) or \
               (d == 1 and c != cmax) or (d == 3 and c != cmin):
                return False
    return True


def transform_cluster(e: Edge, A: Cluster, B: Cluster) -> Placement:
    """Where every member of B lands in A's frame, according to edge e
    (e.i in A, e.j in B)."""
    ri, ci, roti = A.members[e.i]
    tr, tc, trot = place_neighbor(ri, ci, roti, e.a, e.b)
    rj, cj, rotj = B.members[e.j]
    R = (trot - rotj) % 4
    out = {}
    for pid, (r, c, rot) in B.members.items():
        dr, dc = rotate_vec_cw(r - rj, c - cj, R)
        out[pid] = (tr + dr, tc + dc, (rot + R) % 4)
    return out


def merge_ok(A: Cluster, moved: Placement, pieces: list[Piece], rows: int, cols: int) -> bool:
    for pid, (r, c, rot) in moved.items():
        if (r, c) in A.cells:
            return False
    union = {**A.members, **moved}
    if not fits(union, rows, cols):
        return False
    # every new contact must be tab<->blank
    for pid, (r, c, rot) in moved.items():
        for d, (dr, dc) in DIR_DELTA.items():
            q = A.cells.get((r + dr, c + dc))
            if q is None:
                continue
            _, _, rq = A.members[q]
            if not complementary(pieces[pid].sides[side_facing(d, rot)],
                                 pieces[q].sides[side_facing((d + 2) % 4, rq)]):
                return False
    return border_lines_ok(union, pieces)


def assemble(g: CandidateGraph, rows: int, cols: int) -> list[Cluster]:
    pieces = g.pieces
    N = len(pieces)
    owner = list(range(N))
    clusters: dict[int, Cluster] = {}
    for p in range(N):
        clusters[p] = Cluster()
        clusters[p].add(p, 0, 0, 0)

    order = sorted(g.edges, key=lambda e: (e.loop_support == 0, -e.confidence))
    for e in order:
        ca, cb = owner[e.i], owner[e.j]
        if ca == cb:
            continue
        # move the smaller cluster into the bigger one's frame
        if len(clusters[cb].members) > len(clusters[ca].members):
            e = Edge(e.j, e.b, e.i, e.a, e.score, e.confidence, e.loop_support)
            ca, cb = cb, ca
        A, B = clusters[ca], clusters[cb]
        moved = transform_cluster(e, A, B)
        if not merge_ok(A, moved, pieces, rows, cols):
            continue
        for pid, (r, c, rot) in moved.items():
            A.add(pid, r, c, rot)
            owner[pid] = ca
        del clusters[cb]
        if len(clusters) == 1:
            break

    return sorted(clusters.values(), key=lambda cl: -len(cl.members))
