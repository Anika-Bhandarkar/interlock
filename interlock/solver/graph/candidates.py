"""Build the candidate graph.

Node  = a puzzle piece.
Edge  = candidate match between side a of piece i and side b of piece j.
        (a, b) fully determine j's rotation relative to i, so every rotation
        of every piece is covered without storing rotated copies.
Weight = continuity score (lower is better) + confidence
         confidence = 1 - best_score / second_best_score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .continuity import score_tensor
from .geometry import validity_tensor
from .pieces import Piece, SideKind, place_neighbor, side_facing


@dataclass
class Edge:
    i: int
    a: int
    j: int
    b: int
    score: float
    confidence: float = 0.0
    loop_support: int = 0  # number of consistent 2x2 loops this edge is part of


@dataclass
class CandidateGraph:
    pieces: list[Piece]
    scores: np.ndarray            # (N,4,N,4), inf where invalid
    edges: list[Edge]
    ranked: dict[tuple[int, int], list[tuple[int, int]]] = field(default_factory=dict)
    # ranked[(i, a)] = kept (j, b) candidates for side a of piece i, best first

    def rank_of(self, i: int, a: int, j: int, b: int) -> int:
        lst = self.ranked.get((i, a), [])
        try:
            return lst.index((j, b))
        except ValueError:
            return 10**9

    def is_top(self, i, a, j, b, k: int) -> bool:
        """(j,b) is in i's top-k for side a, or vice versa."""
        return self.rank_of(i, a, j, b) < k or self.rank_of(j, b, i, a) < k


def build_candidate_graph(pieces: list[Piece], keep_frac: float = 0.5,
                          min_keep: int = 3) -> CandidateGraph:
    """For each piece p, for each non-flat side a:
        * candidates = pieces with the complementary side kind (all rotations),
          passing the border rule
        * score each by visual continuity
        * keep the best ``keep_frac`` (at least ``min_keep``)
    """
    assert all(p.id == k for k, p in enumerate(pieces)), "piece ids must be 0..N-1"
    N = len(pieces)
    raw = score_tensor(pieces)
    valid = validity_tensor(pieces)
    scores = np.where(valid, raw, np.inf)

    flat = scores.reshape(N * 4, N * 4)
    order = np.argsort(flat, axis=1)
    n_valid = np.isfinite(flat).sum(1)

    ranked: dict[tuple[int, int], list[tuple[int, int]]] = {}
    best = np.full(N * 4, np.inf)
    second = np.full(N * 4, np.inf)
    best_idx = np.full(N * 4, -1)
    edge_keys: set[tuple[int, int]] = set()

    for x in range(N * 4):
        nv = int(n_valid[x])
        if nv == 0:
            continue
        i, a = divmod(x, 4)
        keep = min(nv, max(min_keep, math.ceil(keep_frac * nv)))
        kept = order[x, :keep]
        ranked[(i, a)] = [divmod(int(y), 4) for y in kept]
        best[x], best_idx[x] = flat[x, order[x, 0]], order[x, 0]
        if nv > 1:
            second[x] = flat[x, order[x, 1]]
        for y in kept:
            edge_keys.add((min(x, int(y)), max(x, int(y))))

    edges = []
    for x, y in edge_keys:
        s = float(flat[x, y])
        # best alternative score for either endpoint, excluding this pairing
        alt_x = second[x] if best_idx[x] == y else best[x]
        alt_y = second[y] if best_idx[y] == x else best[y]
        alt = min(alt_x, alt_y)
        conf = 1.0 if not np.isfinite(alt) or alt <= 0 else 1.0 - s / alt
        i, a = divmod(x, 4)
        j, b = divmod(y, 4)
        edges.append(Edge(i, a, j, b, s, conf))
    edges.sort(key=lambda e: -e.confidence)
    return CandidateGraph(pieces, scores, edges, ranked)


def compute_loop_support(g: CandidateGraph, top_k: int = 3) -> None:
    """Small-cycle check: an edge i-j is supported if there are pieces k, l
    forming a 2x2 square i-j-l-k where all four edges are top-k candidates and
    all positions / rotations agree. Writes ``edge.loop_support``."""
    P = g.pieces
    top = {key: v[:top_k] for key, v in g.ranked.items()}
    for e in g.edges:
        if not g.is_top(e.i, e.a, e.j, e.b, top_k):
            continue
        rj, cj, rotj = place_neighbor(0, 0, 0, e.a, e.b)  # i fixed at (0,0,rot 0)
        support = 0
        for a2 in ((e.a + 1) % 4, (e.a + 3) % 4):
            if P[e.i].sides[a2] == SideKind.FLAT:
                continue
            for k, bk in top.get((e.i, a2), []):
                if k in (e.i, e.j):
                    continue
                _, _, rotk = place_neighbor(0, 0, 0, a2, bk)
                sj = side_facing(a2, rotj)  # j's side pointing the same way as i->k
                for l, bl in top.get((e.j, sj), []):
                    if l in (e.i, e.j, k):
                        continue
                    _, _, rotl = place_neighbor(rj, cj, rotj, sj, bl)
                    sk = side_facing(e.a, rotk)          # k's side pointing toward l
                    sl = side_facing((e.a + 2) % 4, rotl)  # l's side pointing toward k
                    if np.isfinite(g.scores[k, sk, l, sl]) and g.is_top(k, sk, l, sl, top_k):
                        support += 1
        e.loop_support = support
