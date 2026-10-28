# Graph solver

Pieces are **nodes**; a candidate match between side `a` of piece `i` and side `b` of piece `j` is a weighted **edge**. Because `(a, b)` fixes j's rotation relative to i, every rotation is covered without storing rotated copies.

| File | Stage |
|---|---|
| `pieces.py` | `Piece`, `SideKind`, orientation math (sides numbered clockwise: 0 top, 1 right, 2 bottom, 3 left; `rot` = clockwise quarter turns) |
| `geometry.py` | Classify each side FLAT / TAB / BLANK from a mask; border rule; `piece_from_mask` adapter |
| `continuity.py` | Visual continuity: extrapolate the edge gradient one pixel outward and score the prediction error, both directions |
| `candidates.py` | Build the candidate graph, keep top 50% per side, `confidence = 1 - best / second_best`, 2×2 loop check |
| `assembly.py` | Constrained spanning tree: merge clusters best-edge-first, reject collisions / oversize / bad geometry |
| `cleanup.py` | Trim to the largest agreeing region, anchor in the frame, refill removed pieces |
| `solve.py` | `solve()` pipeline, `render()`, `neighbor_accuracy()` |
| `synthetic.py` | Synthetic puzzles for testing |

## Rules implemented

* **Tab + blank only.** Flat sides never get edges.
* **Border rule.** If side `a` of i meets side `b` of j, the sides perpendicular to the seam must agree on flatness: `flat(i, a+1) == flat(j, b-1)` and `flat(i, a+3) == flat(j, b+1)`.
* **Merge checks.** No two pieces in one cell, bounding box fits `rows×cols` (either orientation), every new contact is tab↔blank, and nothing lies beyond a flat side.
* **Loop check.** An edge is loop-supported if it closes a consistent 2×2 square of top-k candidates. Loop-supported edges are merged first, then the rest by confidence.
* **Trim + refill.** Seams that neither side ranks in its top-k are cut; the largest region left is kept, anchored using flat sides, and the rest are refilled greedily into the cell with the most placed neighbours.

## Open question from the plan: gradients on tabs

We score only the square body (the green region). Every side then has the same width, so strips flatten to equal-length vectors. The bulb (purple region) varies in width and is mostly background on the neighbour's side, so it's left to the geometry rules. If you later want it, add a second term that compares the tab bulb pixels against the neighbour's blank cavity border.

## Using real pieces

```python
from interlock.solver.graph import piece_from_mask, solve
pieces = [piece_from_mask(k, rgb, mask, core_size=64) for k, (rgb, mask) in enumerate(cutouts)]
result = solve(pieces, rows, cols, verbose=True)
```

If `data_collection/cutting.py` already knows each piece's body box and side types, build `Piece(id, core, sides)` directly instead of inferring them.

## Results on synthetic puzzles

Neighbour accuracy 1.0 on 4×4 up to 20×25 (500 pieces, ~7 s). With heavy pixel noise (σ = 0.1) a 10×12 puzzle drops to ~0.73, which is where better scoring (e.g. MGC normalisation) would help next.
