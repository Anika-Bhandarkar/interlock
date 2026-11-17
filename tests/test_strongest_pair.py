import numpy as np
import pytest

from interlock.data_collection.pairs import OFFSETS
from interlock.solver.strongest_pair import aligned_metrics, strongest_pair


def perfect_scores(rows=3, cols=3):
    positions = np.array([(x,y) for y in range(rows) for x in range(cols)])
    scores = np.full((4,len(positions),len(positions)),-100.)
    for relation,delta in OFFSETS.items():
        for i,point in enumerate(positions):
            for j,other in enumerate(positions):
                if np.array_equal(other-point,delta):
                    scores[relation,i,j] = 20 + (i+j)/100
    return scores, positions


def test_internal_seed_grows_in_all_directions_without_truth_or_dimensions():
    scores, truth = perfect_scores()
    scores[0,4,5] = scores[2,5,4] = 50
    for relation in range(4):
        np.fill_diagonal(scores[relation],1000)  # Self-pairs must never seed growth.
    result = strongest_pair(scores)
    assert result.order[:2] == [4,5]
    assert result.seed['score'] == 50
    assert result.coordinates.min() < 0
    assert len(np.unique(result.coordinates,axis=0)) == 9
    assert sorted(result.order) == list(range(9))
    metrics = aligned_metrics(result.coordinates,truth)
    assert metrics['direct'] == metrics['perfect'] == metrics['neighbors'] == metrics['neighbor_recall'] == 1


def test_growth_averages_all_touching_neighbors_instead_of_ignoring_conflicts():
    scores = np.full((4,4,4),-100.)
    def edge(r,a,b,value):
        scores[r,a,b] = scores[(r+2)%4,b,a] = value
    edge(0,0,1,10)
    edge(1,0,2,9)
    edge(0,2,3,8)
    edge(1,1,3,-90)
    edge(2,0,3,6)
    result = strongest_pair(scores)
    assert result.order == [0,1,2,3]
    assert result.coordinates[3].tolist() == [-1,0]


def test_alignment_is_translation_only_and_does_not_hide_missing_joins():
    truth = np.array([[0,0],[1,0],[0,1],[1,1]])
    line = np.array([[0,0],[1,0],[2,0],[3,0]])
    metrics = aligned_metrics(line,truth)
    assert metrics['direct'] == .5
    assert metrics['perfect'] == 0
    assert metrics['neighbors'] == pytest.approx(2/3)
    assert metrics['neighbor_recall'] == .5
    moved = aligned_metrics(line+[100,-70],truth)
    for name in ('direct','perfect','neighbors','neighbor_recall'):
        assert moved[name] == metrics[name]
    assert aligned_metrics(truth[:,::-1],truth)['perfect'] == 0  # No rotation/reflection alignment.


def test_ties_are_deterministic_and_every_piece_is_used_once():
    scores = np.zeros((4,20,20))
    a,b = strongest_pair(scores),strongest_pair(scores)
    np.testing.assert_array_equal(a.coordinates,b.coordinates)
    assert a.order == b.order
    assert sorted(a.order) == list(range(20))
    assert len(np.unique(a.coordinates,axis=0)) == 20


@pytest.mark.parametrize('scores',[np.zeros((3,3,3)),np.full((4,3,3),np.nan),np.full((4,3,3),-np.inf)])
def test_reject_invalid_or_unusable_scores(scores):
    with pytest.raises(ValueError):
        strongest_pair(scores)
