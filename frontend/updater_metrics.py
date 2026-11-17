"""Evaluation of existing solver snapshots; never used to choose placements."""
import numpy as np


def snapshot_metrics(puzzle, frame):
    grid=np.asarray(frame['grid'])
    truth=np.asarray(puzzle['truth'])
    n=puzzle['piece_count']
    correct=grid==truth
    present=grid>=0
    locked=np.asarray(frame['locked'],dtype=bool)
    if frame['label']=='stage 4':
        previous=[s for s in puzzle['frames'] if s['label'].startswith('stage 3')]
        locked=np.asarray(previous[-1]['locked'],dtype=bool) if previous else np.zeros(grid.size,dtype=bool)
    positions=np.empty((n,2),dtype=int)
    for slot,piece in enumerate(truth):
        positions[piece]=(slot%puzzle['cols'],slot//puzzle['cols'])
    correct_joins=0
    total=(puzzle['rows']*(puzzle['cols']-1)+(puzzle['rows']-1)*puzzle['cols'])
    for slot,piece in enumerate(grid):
        if piece<0:continue
        for neighbor,delta in ((slot+1,(1,0)),(slot+puzzle['cols'],(0,1))):
            if delta==(1,0) and slot%puzzle['cols']==puzzle['cols']-1:continue
            if neighbor<len(grid) and grid[neighbor]>=0:
                correct_joins+=int(np.array_equal(positions[grid[neighbor]]-positions[piece],delta))
    return {'accuracy':float((correct&present).sum()/n),'neighbors':correct_joins/total if total else 0.,
            'perfect':bool(correct.all() and present.all()),'wrong_locks':int((locked&present&~correct).sum())}


def summarize_updater(puzzles):
    """Compare the same puzzles at each round; early-stopped cases hold their last stage-3 state."""
    cases=[]
    for puzzle in puzzles:
        if puzzle.get('strategy','grid')!='grid':continue
        initial=[f for f in puzzle['frames'] if f['label']=='stage 2']
        rounds=[f for f in puzzle['frames'] if f['label'].startswith('stage 3')]
        final=[f for f in puzzle['frames'] if f['label']=='stage 4']
        if not initial or not final:continue
        cases.append((snapshot_metrics(puzzle,initial[-1]),
                      [snapshot_metrics(puzzle,f) for f in rounds],snapshot_metrics(puzzle,final[-1])))
    if not cases:return {'evaluated':0,'rows':[]}
    max_rounds=max(len(c[1]) for c in cases)
    rows=[]
    for step in range(max_rounds+2):
        values=[initial if step==0 else final if step==max_rounds+1 else rounds[min(step,len(rounds))-1] if rounds else initial
                for initial,rounds,final in cases]
        solved=sum(v['perfect'] for v in values)
        rows.append({'key':'initial' if step==0 else 'final' if step==max_rounds+1 else f'round-{step}',
                     'label':'Initial placement' if step==0 else 'Final assignment' if step==max_rounds+1 else f'Round {step}',
                     'solved':solved,'solve_rate':solved/len(cases),
                     'accuracy':float(np.mean([v['accuracy'] for v in values])),
                     'neighbors':float(np.mean([v['neighbors'] for v in values])),
                     'wrong_locks':sum(v['wrong_locks'] for v in values)})
    rescued=sum(not initial['perfect'] and final['perfect'] for initial,_,final in cases)
    broken=sum(initial['perfect'] and not final['perfect'] for initial,_,final in cases)
    return {'evaluated':len(cases),'rows':rows,'rescued':rescued,'broken':broken}
