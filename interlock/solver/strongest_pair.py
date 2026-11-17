"""Greedy growth on an unbounded integer lattice, using scores only (no board metadata)."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from interlock.data_collection.pairs import OFFSETS


@dataclass
class FreeAssembly:
    coordinates: np.ndarray  # indexed by piece ID; arbitrary translation, upright orientation
    order: list[int]
    seed: dict


def strongest_pair(logits: np.ndarray) -> FreeAssembly:
    """Seed with the global best directed pair; commit unused pieces to the best frontier slot.

    Every candidate is scored by the arithmetic mean of raw logits against ALL occupied
    neighbors of its proposed slot. No rectangle, dimensions, corner labels, ground truth,
    score floor, refinement, or Hungarian assignment is used. Ties are deterministic.
    """
    scores = np.asarray(logits, dtype=float).copy()
    if scores.ndim != 3 or scores.shape[0] != 4 or scores.shape[1] != scores.shape[2] or scores.shape[1] < 2:
        raise ValueError('Expected (4, N, N) scores with at least two pieces')
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError('Scores must be finite or negative infinity')
    n = scores.shape[1]
    for relation in range(4):
        np.fill_diagonal(scores[relation], -np.inf)
    relation, first, second = np.unravel_index(scores.argmax(), scores.shape)
    if not np.isfinite(scores[relation, first, second]):
        raise ValueError('No finite matching pair')
    coords = np.zeros((n, 2), dtype=int)
    coords[second] = OFFSETS[int(relation)]
    occupied = {(0, 0): int(first), tuple(coords[second]): int(second)}
    order = [int(first), int(second)]
    remaining = np.ones(n, dtype=bool)
    remaining[order] = False
    seed = {'pieces':[int(first), int(second)], 'relation':int(relation), 'score':float(scores[relation,first,second])}
    frontier = set()

    def extend(point):
        x, y = point
        frontier.discard(point)
        for dx, dy in OFFSETS.values():
            neighbor = (x+dx, y+dy)
            if neighbor not in occupied:
                frontier.add(neighbor)

    for point in occupied:
        extend(point)
    while remaining.any():
        candidates = np.flatnonzero(remaining)
        best_score, best = -np.inf, None
        for x, y in sorted(frontier, key=lambda p:(p[1],p[0])):
            evidence = [scores[r, occupied[(x-dx,y-dy)], candidates]
                        for r,(dx,dy) in OFFSETS.items() if (x-dx,y-dy) in occupied]
            means = np.mean(evidence, axis=0)
            j = int(means.argmax())
            if means[j] > best_score:
                best_score, best = float(means[j]), (int(candidates[j]), x, y)
        if best is None:
            raise ValueError('No finite frontier placement for the remaining pieces')
        piece, x, y = best
        coords[piece] = (x,y)
        occupied[(x,y)] = piece
        remaining[piece] = False
        order.append(piece)
        extend((x,y))
    return FreeAssembly(coords, order, seed)


def aligned_metrics(coordinates: np.ndarray, truth: np.ndarray) -> dict:
    """Translation alignment is evaluation-only. Report precision AND recall of adjacency."""
    coords, truth = np.asarray(coordinates), np.asarray(truth)
    if coords.shape != truth.shape or coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
        raise ValueError('Coordinates and truth must be matching nonempty (N, 2) arrays')
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError('Pieces cannot occupy the same coordinate')
    votes = Counter(map(tuple, truth-coords))
    shift, count = sorted(votes.items(), key=lambda item:(-item[1],item[0]))[0]
    occupied = {tuple(point): i for i,point in enumerate(coords)}
    actual, correct = 0, 0
    for i,(x,y) in enumerate(coords):
        for delta in ((1,0),(0,1)):
            j = occupied.get((x+delta[0], y+delta[1]))
            if j is not None:
                actual += 1
                correct += int(np.array_equal(truth[j]-truth[i], delta))
    truth_set = set(map(tuple,truth))
    expected = sum((x+dx,y+dy) in truth_set for x,y in truth_set for dx,dy in ((1,0),(0,1)))
    return {'direct': count/len(coords), 'perfect':float(count == len(coords)),
            'neighbors':correct/actual if actual else 0.,
            'neighbor_recall':correct/expected if expected else 0.,
            'correct_neighbors':correct, 'assembly_neighbors':actual, 'expected_neighbors':expected,
            'placed':len(coords), 'correct_count':count, 'alignment_shift':[int(v) for v in shift]}
