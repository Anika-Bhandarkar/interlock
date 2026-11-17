"""Integration checks against the running local viewer; generates real replay caches."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:8765'


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as response:
        return json.load(response)


def replay(category, index=0, anchor=0, model='robust', source=None, grid=4, strategy='grid'):
    path = f'/api/puzzle?model={model}&category={category}&index={index}&anchor={anchor}&strategy={strategy}'
    if source:
        path += f'&source={source}&grid={grid}'
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        result = get(path)
        assert result['status'] != 'error', result
        if result['status'] == 'ready':
            break
        time.sleep(.5)
    else:
        raise AssertionError('Replay timed out')
    data = get(result['url'])
    assert data['index'] == index and data['val_index'] == (None if source else index+100)
    if source and strategy == 'grid':
        assert data['mode'] == 'custom' and data['rows'] == grid and data['cols'] == grid
    assert data['anchored'] == bool(anchor)
    assert len(data['frames']) >= data['piece_count']
    assert sorted(p for p in data['frames'][-1]['grid'] if p>=0) == list(range(data['piece_count']))
    if strategy == 'strongest':
        import numpy as np
        from interlock.solver.strongest_pair import aligned_metrics
        assert data['strategy'] == 'strongest' and not data['anchored']
        assert [sum(p>=0 for p in f['grid']) for f in data['frames']] == [0]+list(range(2,data['piece_count']+1))
        for frame in data['frames']:
            ids=[p for p in frame['grid'] if p>=0]
            assert len(ids)==len(set(ids))
        recomputed=aligned_metrics(np.array(data['coordinates']),np.array(data['true_positions']))
        for metric in ('direct','neighbors','neighbor_recall','perfect'):
            assert data['metrics'][metric] == recomputed[metric]
    if anchor:
        assert all(frame['grid'][0] == data['truth'][0] and frame['locked'][0] for frame in data['frames'][1:])
    for asset in ('pieces.png', 'reference.png'):
        with urllib.request.urlopen(BASE+result['url'].replace('result.json',asset)) as image:
            assert image.read(8) == b'\x89PNG\r\n\x1a\n'
    print(model, category, index, anchor, data['metrics'], flush=True)
    return data


if __name__ == '__main__':
    meta = get('/api/meta')
    for category in meta['categories']:
        replay(category['id'])
    for category in ('flat_even','flat_diagonal'):
        replay(category, anchor=1)
        growth = replay(category, strategy='strongest')
        grid = replay(category)
        assert growth['metrics']['top1'] == grid['metrics']['top1']
        summary = get(f'/api/results?model=robust&category={category}&strategy=strongest')
        assert summary['evaluated'] >= 1
        assert any(f['index']==0 for f in summary['failures']) == (not growth['metrics']['perfect'])
        replay(category, source='gallery-0',strategy='strongest')
    replay('flat_even', index=39, anchor=1)
    for model in ('original','single_tab'):
        replay('clean', model=model)
    gallery = get('/api/gallery')
    assert len(gallery) == 12
    with urllib.request.urlopen(BASE+gallery[0]['thumbnail']) as response:
        assert response.read().startswith(b'\xff\xd8')
    for category in ('clean','flat_even','flat_diagonal'):
        replay(category, source='gallery-0', anchor=int(category.startswith('flat_')))
    image_path = next((Path(__file__).resolve().parents[1]/'data/processed/val').glob('*.png'))
    request = urllib.request.Request(BASE+'/api/upload', data=image_path.read_bytes(), headers={'Content-Type':'image/png'},method='POST')
    with urllib.request.urlopen(request) as response:
        uploaded = json.load(response)
    replay('flat_diagonal', source=uploaded['id'],anchor=1)
    for category, expected in [('flat_even',{1,3,4,19}), ('flat_diagonal',{1,3})]:
        summary = get(f'/api/results?model=robust&category={category}&anchor=1')
        assert summary['evaluated'] >= 30 and summary['total'] == 100
        assert summary['failed'] == len(summary['failures'])
        assert summary['solved'] + summary['failed'] == summary['evaluated']
        assert expected <= {f['index'] for f in summary['failures']}
        assert all(f['number'] == f['index']+1 and f['accuracy'] < 1 for f in summary['failures'])
    summary = get('/api/results?model=original&category=clean&anchor=0')
    assert summary['evaluated'] >= 1
    assert not summary['failures']  # This checkpoint's tested clean puzzle solves fully.
    try:
        urllib.request.urlopen(urllib.request.Request(BASE+'/api/upload',data=b'not an image',method='POST'))
        raise AssertionError('Invalid upload accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400
    for query in ('model=robust&category=clean&index=100', 'model=robust&category=clean&index=0&anchor=1', 'model=missing&category=clean&index=0',
                  'model=robust&category=clean&index=0&source=gallery-99',
                  'model=robust&category=flat_even&index=0&source=gallery-0&grid=99',
                  'model=robust&category=clean&index=0&strategy=strongest',
                  'model=robust&category=flat_even&index=0&strategy=strongest&anchor=1',
                  'model=robust&category=flat_even&index=0&strategy=unknown',
                  'model=robust&category=clean&index=0&source=upload-../../server.py'):
        try:
            get('/api/puzzle?'+query)
            raise AssertionError('Invalid query was accepted')
        except urllib.error.HTTPError as error:
            assert error.code == 400
    print('All replay integration checks passed.', flush=True)
