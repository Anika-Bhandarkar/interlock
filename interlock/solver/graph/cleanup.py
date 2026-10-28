"""Cleanup after greedy merging.

Trim:   keep the largest connected region in which every internal seam is an
        "agreeing" edge (top-k candidate for at least one of its two sides).
        Wrong greedy merges (e.g. red->green when green really belongs next to
        light purple) usually create seams that nobody ranks highly, so they
        get cut off here.
Anchor: place the trimmed region inside the rows x cols frame, using flat
        sides to decide where the borders are.
Refill: put every removed / unplaced piece back, one at a time, always into
        the empty cell with the most placed neighbours, choosing the piece +
        rotation with the lowest continuity cost that obeys the geometry.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .candidates import CandidateGraph
from .pieces import DIR_DELTA, Piece, SideKind, complementary, facing_dir, side_facing

Placement = dict[int, tuple[int, int, int]]


def _seams(placement: Placement):
    cells = {(r, c): p for p, (r, c, _) in placement.items()}
    for p, (r, c, rot) in placement.items():
        for d in (1, 2):  # right, bottom: each seam once
            dr, dc = DIR_DELTA[d]
            q = cells.get((r + dr, c + dc))
            if q is not None:
                rq = placement[q][2]
                yield p, side_facing(d, rot), q, side_facing((d + 2) % 4, rq)


def trim(g: CandidateGraph, placement: Placement, top_k: int = 2) -> Placement:
    adj: dict[int, list[int]] = {p: [] for p in placement}
    for p, sp, q, sq in _seams(placement):
        if g.is_top(p, sp, q, sq, top_k):
            adj[p].append(q)
            adj[q].append(p)
    seen, best = set(), []
    for start in placement:
        if start in seen:
            continue
        comp, dq = [], deque([start])
        seen.add(start)
        while dq:
            u = dq.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        if len(comp) > len(best):
            best = comp
    return {p: placement[p] for p in best}


def _cell_ok(piece: Piece, rot: int, r: int, c: int, H: int, W: int) -> int:
    """Number of frame violations: flat sides must face the outside, non-flat
    sides must face the inside."""
    bad = 0
    for s, kind in enumerate(piece.sides):
        dr, dc = DIR_DELTA[facing_dir(s, rot)]
        outside = not (0 <= r + dr < H and 0 <= c + dc < W)
        bad += (kind == SideKind.FLAT) != outside
    return bad


def anchor(placement: Placement, pieces: list[Piece], rows: int, cols: int):
    """Shift the region into a frame; returns (placement, H, W)."""
    rs = [r for r, _, _ in placement.values()]
    cs = [c for _, c, _ in placement.values()]
    r0, c0 = min(rs), min(cs)
    h, w = max(rs) - r0 + 1, max(cs) - c0 + 1
    best = None
    frames = {(rows, cols), (cols, rows)}
    for H, W in frames:
        for oy in range(H - h + 1):
            for ox in range(W - w + 1):
                cost = sum(_cell_ok(pieces[p], rot, r - r0 + oy, c - c0 + ox, H, W)
                           for p, (r, c, rot) in placement.items())
                if best is None or cost < best[0]:
                    best = (cost, H, W, oy, ox)
    _, H, W, oy, ox = best
    out = {p: (r - r0 + oy, c - c0 + ox, rot) for p, (r, c, rot) in placement.items()}
    return out, H, W


def refill(g: CandidateGraph, placement: Placement, H: int, W: int,
           violation_penalty: float = 1e6) -> Placement:
    pieces = g.pieces
    placement = dict(placement)
    cells = {(r, c): p for p, (r, c, _) in placement.items()}
    pool = [p.id for p in pieces if p.id not in placement]
    finite = g.scores[np.isfinite(g.scores)]
    bad_pair = float(finite.max()) * 2 if finite.size else 1.0

    while pool:
        empties = [(r, c) for r in range(H) for c in range(W) if (r, c) not in cells]
        if not empties:
            break

        def n_nb(rc):
            r, c = rc
            return sum((r + dr, c + dc) in cells for dr, dc in DIR_DELTA.values())

        most = max(n_nb(rc) for rc in empties)
        frontier = [rc for rc in empties if n_nb(rc) == most]

        best = None
        for (r, c) in frontier:
            for pid in pool:
                for rot in range(4):
                    cost = violation_penalty * _cell_ok(pieces[pid], rot, r, c, H, W)
                    total, n = 0.0, 0
                    for d, (dr, dc) in DIR_DELTA.items():
                        q = cells.get((r + dr, c + dc))
                        if q is None:
                            continue
                        sp = side_facing(d, rot)
                        sq = side_facing((d + 2) % 4, placement[q][2])
                        if not complementary(pieces[pid].sides[sp], pieces[q].sides[sq]):
                            cost += violation_penalty
                        s = g.scores[pid, sp, q, sq]
                        total += s if np.isfinite(s) else bad_pair
                        n += 1
                    cost += total / max(n, 1)
                    if best is None or cost < best[0]:
                        best = (cost, pid, r, c, rot)
        _, pid, r, c, rot = best
        placement[pid] = (r, c, rot)
        cells[(r, c)] = pid
        pool.remove(pid)
    return placement
