import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('updater_metrics',Path(__file__).resolve().parents[1]/'frontend/updater_metrics.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def state(label,grid,locked=None):
    return {'label':label,'grid':grid,'locked':locked or [False]*4}


def test_cohort_keeps_same_cases_in_later_rounds_and_exposes_recoveries_and_regressions():
    correct=[0,1,2,3]
    wrong=[1,0,2,3]
    base={'truth':correct,'rows':2,'cols':2,'piece_count':4}
    a=base|{'frames':[state('stage 2',wrong),state('stage 3, round 1',correct),state('stage 3, round 2',wrong),state('stage 4',correct)]}
    b=base|{'frames':[state('stage 2',correct),state('stage 3, round 1',wrong),state('stage 4',wrong)]}
    report=module.summarize_updater([a,b])
    assert report['evaluated']==2
    assert [row['solve_rate'] for row in report['rows']]==[.5,.5,0,.5]
    assert [row['accuracy'] for row in report['rows']]==[.75,.75,.5,.75]
    assert report['rescued']==report['broken']==1


def test_final_assignment_does_not_turn_every_wrong_assignment_into_an_updater_lock():
    puzzle={'truth':[0,1,2,3],'rows':2,'cols':2,'piece_count':4}
    puzzle['frames']=[state('stage 3, round 1',[1,0,2,3],[True,False,False,False]),
                      state('stage 4',[1,0,2,3],[True]*4)]
    metrics=module.snapshot_metrics(puzzle,puzzle['frames'][-1])
    assert metrics['wrong_locks']==1
    assert metrics['accuracy']==.5


def test_partial_board_does_not_treat_empty_slots_as_pieces_or_neighbors():
    puzzle={'truth':[0,1,2,3],'rows':2,'cols':2,'piece_count':4,'frames':[]}
    metrics=module.snapshot_metrics(puzzle,state('stage 2',[0,1,-1,-1]))
    assert metrics['accuracy']==.5
    assert metrics['neighbors']==pytest.approx(.25)
    assert not metrics['perfect']


def test_skips_free_growth_and_missing_stage_traces():
    assert module.summarize_updater([{'strategy':'strongest'},{'frames':[]}])=={'evaluated':0,'rows':[]}
