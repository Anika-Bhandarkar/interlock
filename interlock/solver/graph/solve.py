"""End-to-end pipeline: candidate graph -> loop check -> assembly -> trim -> refill."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assembly import assemble
from .candidates import build_candidate_graph, compute_loop_support
from .cleanup import anchor, refill, trim
from .pieces import DIR_DELTA, Piece, facing_dir, rotated_core, side_facing


@dataclass
class SolveResult:
    grid: np.ndarray        # (H, W) piece id per cell, -1 if empty
    rotations: np.ndarray   # (H, W) clockwise quarter turns per cell
    placement: dict[int, tuple[int, int, int]]
    stats: dict = field(default_factory=dict)


def solve(pieces: list[Piece], rows: int, cols: int, *, keep_frac: float = 0.5,
          min_keep: int = 3, loop_top_k: int = 3, trim_top_k: int = 2,
          verbose: bool = False) -> SolveResult:
    g = build_candidate_graph(pieces, keep_frac=keep_frac, min_keep=min_keep)
    compute_loop_support(g, top_k=loop_top_k)
    clusters = assemble(g, rows, cols)
    main = clusters[0].members
    kept = trim(g, main, top_k=trim_top_k)
    framed, H, W = anchor(kept, pieces, rows, cols)
    final = refill(g, framed, H, W)

    grid = -np.ones((H, W), dtype=int)
    rots = np.zeros((H, W), dtype=int)
    for p, (r, c, rot) in final.items():
        grid[r, c], rots[r, c] = p, rot

    stats = {
        "n_edges": len(g.edges),
        "n_loop_supported": sum(e.loop_support > 0 for e in g.edges),
        "n_clusters_after_assembly": len(clusters),
        "largest_cluster": len(main),
        "kept_after_trim": len(kept),
        "refilled": len(final) - len(kept),
    }
    if verbose:
        for k, v in stats.items():
            print(f"{k:>28}: {v}")
    return SolveResult(grid, rots, final, stats)


def render(result: SolveResult, pieces: list[Piece]) -> np.ndarray:
    """Stitch the solved bodies into one image (empty cells left black)."""
    T = pieces[0].core.shape[0]
    C = pieces[0].core.shape[2] if pieces[0].core.ndim == 3 else 1
    H, W = result.grid.shape
    out = np.zeros((H * T, W * T, C), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            p = result.grid[r, c]
            if p >= 0:
                out[r * T:(r + 1) * T, c * T:(c + 1) * T] = rotated_core(pieces[p], result.rotations[r, c])
    return out


def neighbor_accuracy(result: SolveResult, truth: dict[int, tuple[int, int, int]]) -> float:
    """Fraction of seams in the solution that are correct seams in the ground
    truth (same two pieces, same two sides). Invariant to global rotation."""
    cells_t = {(r, c): p for p, (r, c, _) in truth.items()}
    good = total = 0
    H, W = result.grid.shape
    for r in range(H):
        for c in range(W):
            p = result.grid[r, c]
            if p < 0:
                continue
            for d in (1, 2):
                dr, dc = DIR_DELTA[d]
                rr, cc = r + dr, c + dc
                if not (rr < H and cc < W) or result.grid[rr, cc] < 0:
                    continue
                q = result.grid[rr, cc]
                sp = side_facing(d, result.rotations[r, c])
                sq = side_facing((d + 2) % 4, result.rotations[rr, cc])
                total += 1
                tr, tc, trot = truth[p]
                D = facing_dir(sp, trot)
                tdr, tdc = DIR_DELTA[D]
                q_true = cells_t.get((tr + tdr, tc + tdc))
                if q_true == q and facing_dir(sq, truth[q][2]) == (D + 2) % 4:
                    good += 1
    return good / max(total, 1)
