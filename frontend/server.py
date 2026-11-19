"""Local validation viewer. Run: PYTHONPATH=. .venv/bin/python frontend/server.py"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import inspect
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from experiment import CONDITIONS, FlatPuzzles, apply_condition, score_all_pairs, flat_cut
from interlock.data_collection.cutting import PAD, cut
from interlock.data_collection.dataset import PuzzleDataset, load_manifest
from interlock.data_collection.pairs import cell_size, side_types
from interlock.models.edge_comparator import load_comparator, neighbor_ranks, score_matrix
from interlock.solver.updaters import assemble, placement_metrics, puzzle_shape, shuffle_puzzle, solve, top_left_anchor, true_grid
from interlock.solver.strongest_pair import strongest_pair, aligned_metrics
from interlock.solver.shrink_frame import grid_coordinates, prepare_scores, shrink_frame, swap_polish
from interlock.solver.bounded_growth import bounded_growth
from updater_metrics import summarize_updater

WEB = ROOT / 'frontend'
CACHE = ROOT / 'artifacts/viewer_cache'
CACHE.mkdir(parents=True, exist_ok=True)
UPLOADS = CACHE / 'uploads'
UPLOADS.mkdir(exist_ok=True)
MODELS = {'robust': ('Robust comparator', ROOT / 'checkpoints/robust_comparator_hf.pt'),
          'original': ('Original local comparator', ROOT / 'checkpoints/edge_comparator.pt'),
          'single_tab': ('Single-tab comparator', ROOT / 'checkpoints/edge_comparator_single_tab_full.pt')}
CATEGORIES = [('clean', 'Clean pieces'), ('shrink_1px_half', 'Shrunk outlines'),
              ('uneven_outline', 'Uneven outlines'), ('smoothed_outline', 'Smoothed outlines'),
              ('pixel_noise', 'Pixel noise'), ('flat_even', 'Straight edges'),
              ('flat_diagonal', 'Slanted edges'), ('color_removed', 'Color removed')]
FREE_STRATEGIES = ('strongest', 'shrink', 'bounded')  # solvers that place pieces on an open lattice, scored with alignment
jobs = {}
pool = ThreadPoolExecutor(max_workers=1)
models = {}
hashes = {}
updater_cache = {}


def identity(model):
    path = MODELS[model][1]
    if model not in hashes:
        hashes[model] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources = [WEB / 'experiment.py', ROOT / 'interlock/solver/updaters.py', ROOT / 'interlock/solver/strongest_pair.py',
               ROOT / 'interlock/data_collection/cutting.py', ROOT / 'interlock/data_collection/pairs.py',
               ROOT / 'interlock/data_collection/grid.py', ROOT / 'interlock/models/edge_comparator.py',
               ROOT / 'data/processed/manifest.csv']
    return hashlib.sha256((hashes[model] + hashlib.sha256(inspect.getsource(build).encode()).hexdigest() + ''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sources)).encode()).hexdigest()[:16]



def replay_suffix(strategy, anchored):
    """Last part of a replay's cache key. Shrinking-frame and bounded-growth replays also carry a hash of their
    solver files, so editing a solver only redoes its replays; the CNN scores (keyed by identity) stay cached."""
    if strategy in ('shrink', 'bounded'):
        files = ['shrink_frame.py'] + (['bounded_growth.py'] if strategy == 'bounded' else [])
        code = b''.join((ROOT / 'interlock/solver' / name).read_bytes() for name in files)
        return strategy + hashlib.sha256(code).hexdigest()[:8]
    return strategy if strategy in FREE_STRATEGIES else str(int(anchored))


def validation_results(model, category, anchored, strategy='grid'):
    """List only measured cases, scoped to checkpoint, cuts, and anchor mode."""
    fingerprint = identity(model)
    measured = {}
    # These saved flat-cut results use the same images, seeds, model, and solver.
    original = ROOT / 'artifacts/anchored_flat/results.json'
    if strategy == 'grid' and model == 'robust' and category.startswith('flat_') and original.exists():
        saved = json.loads(original.read_text())
        manifest = hashlib.sha1((ROOT/'data/processed/manifest.csv').read_bytes()).hexdigest()[:12]
        if saved.get('checkpoint_sha256') == hashes[model] and saved.get('manifest_sha1') == manifest:
            for record in saved.get('conditions',{}).get(category,{}).get('puzzles',[]):
                index = record['val_index'] - 100
                if 0 <= index < 100:
                    measured[index] = record['anchored' if anchored else 'unanchored']
    suffix = replay_suffix(strategy, anchored)
    for path in CACHE.glob(f'{fingerprint}-{model}-{category}-*-{suffix}/result.json'):
        try:
            result = json.loads(path.read_text())
            if result.get('mode','validation') != 'validation' or result.get('val_index') is None:
                continue
            measured[result['index']] = result['metrics']
        except (OSError, ValueError, KeyError):
            continue  # An in-progress cache write is not an evaluated result yet.
    failures = [{'index':index, 'number':index+1, 'accuracy':metrics['direct']}
                for index,metrics in sorted(measured.items()) if not metrics['perfect']]
    return {'evaluated':len(measured), 'total':100, 'failed':len(failures),
            'solved':len(measured)-len(failures), 'failures':failures}


def build(key, model_id, category, index, anchored, custom=None, strategy='grid'):
    job = jobs[key]
    folder = CACHE / key
    folder.mkdir(exist_ok=True)
    try:
        job.update(message='Preparing validation pieces')
        rows_meta = load_manifest('val')[100:200]
        flat = category.startswith('flat_')
        if custom:
            source, side = custom['source'], custom['grid']
            path = UPLOADS / f'{source[7:]}.png' if source.startswith('upload-') else ROOT / 'data/processed' / rows_meta[int(source[8:])]['path']
            with Image.open(path) as image:
                image = ImageOps.fit(ImageOps.exif_transpose(image).convert('RGB'), (side*(400//side),)*2, method=Image.Resampling.LANCZOS)
                pixels = np.array(image)
            puzzle = flat_cut(pixels, 0, side, side, .2 if category == 'flat_diagonal' else 0) if flat else cut(pixels, 0, side, side)
        else:
            dataset = FlatPuzzles(rows_meta, category, seed_offset=100) if flat else PuzzleDataset(rows_meta, seed_offset=100)
            puzzle = dataset[index]
        reference = assemble(puzzle[0], puzzle[1], true_grid(puzzle[2], *puzzle_shape(puzzle[2])))
        pieces, masks, positions = shuffle_puzzle(*puzzle, np.random.default_rng(10000 + index))
        if not flat:
            pieces, masks = apply_condition(category, pieces, masks, np.random.default_rng([10000, CONDITIONS.index(category), index]))
        source_key = f'{custom["source"]}-{custom["grid"]}' if custom else str(index)
        score_path = CACHE / f'scores-{identity(model_id)}-{category}-{source_key}.npz'
        legacy = ROOT / f'artifacts/anchored_flat/{category}_{index:02d}.npz'
        manifest = hashlib.sha1((ROOT / 'data/processed/manifest.csv').read_bytes()).hexdigest()[:12]
        legacy_ok = (not custom and model_id == 'robust' and flat and index < 30 and legacy.exists()
                     and manifest == 'cb08f6f90cc3'
                     and hashes[model_id] == '27ba0f5a5d9da995354a64b6da35bbdf2c6119f4c274d5d0dbab92457382c0db')
        if score_path.exists() or legacy_ok:
            with np.load(score_path if score_path.exists() else legacy) as cached:
                if not np.array_equal(cached['positions'], positions):
                    raise ValueError('Cached puzzle positions do not match this validation image')
                logits = cached['logits']
        else:
            job.update(message='Scoring piece pairs · first load may take a minute for large puzzles')
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            if model_id not in models:
                models[model_id] = load_comparator(MODELS[model_id][1], device)
            model = models[model_id]
            logits = score_all_pairs(model, pieces, masks, batch_size=128) if flat else score_matrix(model, pieces, masks, batch_size=128)
            np.savez_compressed(score_path, logits=logits, positions=positions)
        job.update(message='Recording solver steps')
        rows, cols = puzzle_shape(positions)
        trace = []
        if strategy == 'strongest':
            growth = strongest_pair(logits)
            metrics = aligned_metrics(growth.coordinates, positions)
            normalized = growth.coordinates - growth.coordinates.min(axis=0)
            cols, rows = (normalized.max(axis=0)+1).tolist()
            grid = np.full((rows,cols),-1,dtype=int)
            frames = [{'label':'Ready','grid':grid.ravel().tolist(),'locked':[False]*grid.size}]
            # The first event commits the seed pair; every later event adds one piece.
            for step, piece in enumerate(growth.order):
                x,y = normalized[piece]
                grid[y,x] = piece
                if step == 0:
                    continue
                frames.append({'label':'Strongest pair' if step == 1 else 'Outward growth',
                               'grid':grid.ravel().tolist(),'locked':(grid>=0).ravel().tolist()})
            frames[-1]['label'] = 'Free assembly complete'
            ground_truth = [-1]*grid.size  # Ground truth is never passed to the growth solver.
        elif strategy == 'shrink':
            # Board size comes from the piece count (square boards); true positions are never passed in.
            shrunk = shrink_frame(logits)
            growth = shrunk.growth
            # Draw on the first frame's box, which holds the growth and every later frame.
            fx0, fy0, fx1, fy1 = shrunk.steps[0].frame
            origin = np.array([fx0, fy0])
            normalized = growth.coordinates - origin
            cols, rows = fx1-fx0+1, fy1-fy0+1
            grid = np.full((rows,cols),-1,dtype=int)
            frames = [{'label':'Ready','grid':grid.ravel().tolist(),'locked':[False]*grid.size}]
            for step, piece in enumerate(growth.order):
                x,y = normalized[piece]
                grid[y,x] = piece
                if step == 0:
                    continue
                frames.append({'label':'Strongest pair' if step == 1 else 'Outward growth',
                               'grid':grid.ravel().tolist(),'locked':(grid>=0).ravel().tolist()})
            for frame_step in shrunk.steps:
                grid = np.full((rows,cols),-1,dtype=int)
                local = frame_step.coordinates - origin
                grid[local[:,1], local[:,0]] = np.arange(len(local))
                x0,y0,x1,y1 = frame_step.frame
                frames.append({'label':frame_step.label,'grid':grid.ravel().tolist(),'locked':(grid>=0).ravel().tolist(),
                               'frame':[int(x0-fx0),int(y0-fy0),int(x1-fx0),int(y1-fy0)],
                               'moved':[int(p) for p in frame_step.moved]})
            frames[-1]['label'] = 'Frame closed'
            final_coordinates = shrunk.steps[-1].coordinates - origin
            metrics = aligned_metrics(final_coordinates, positions) | {
                'exact_direct': float((shrunk.coordinates == positions).all(axis=1).mean()),
                'before_polish_direct': aligned_metrics(shrunk.before_polish, positions)['direct'],
                'growth_direct': aligned_metrics(growth.coordinates, positions)['direct'],
                'evicted': shrunk.num_evicted, 'swaps': shrunk.num_swaps}
            ground_truth = [-1]*grid.size
        elif strategy == 'bounded':
            # Board size comes from the piece count (square boards); true positions are never passed in.
            growth = bounded_growth(logits)
            origin = growth.coordinates.min(axis=0)
            normalized = growth.coordinates - origin
            cols, rows = (normalized.max(axis=0) + 1).tolist()
            grid = np.full((rows,cols),-1,dtype=int)
            frames = [{'label':'Ready','grid':grid.ravel().tolist(),'locked':[False]*grid.size}]
            for step, piece in enumerate(growth.order):
                x,y = normalized[piece]
                grid[y,x] = piece
                if step == 0:
                    continue
                frames.append({'label':'Strongest pair' if step == 1 else 'Bounded growth',
                               'grid':grid.ravel().tolist(),'locked':(grid>=0).ravel().tolist()})
            passes = swap_polish(prepare_scores(logits), grid)
            for polished, _ in passes:
                frames.append({'label':'Swap polish','grid':polished.ravel().tolist(),'locked':[True]*polished.size})
            final_grid = passes[-1][0] if passes else grid
            frames[-1]['label'] = 'Board complete'
            final_coordinates = grid_coordinates(final_grid)
            metrics = aligned_metrics(final_coordinates, positions) | {
                'growth_direct': aligned_metrics(normalized, positions)['direct'],
                'swaps': sum(len(swapped) // 2 for _, swapped in passes)}
            ground_truth = [-1]*grid.size
        else:
            _, _, final = solve(logits, np.zeros((len(pieces), 4), dtype=np.int8) if flat else side_types(masks),
                                trace=trace, rows=rows, cols=cols, straight_edges=flat,
                                anchor_piece=top_left_anchor(positions) if anchored else None)
            metrics = placement_metrics(final.grid, positions)
            frames = [{'label':'Ready','grid':[-1]*len(pieces),'locked':[False]*len(pieces)}] + [
                {'label':s.label,'grid':s.grid.ravel().tolist(),'locked':s.locked.ravel().tolist()} for s in trace]
            ground_truth = true_grid(positions, rows, cols).ravel().tolist()
        box = pieces.shape[1]
        atlas_cols = min(12, len(pieces))
        atlas = Image.new('RGBA', (atlas_cols * box, ((len(pieces)+atlas_cols-1)//atlas_cols)*box))
        for i, (piece, mask) in enumerate(zip(pieces, masks)):
            rgba = np.dstack([piece, mask.astype(np.uint8)*255])
            atlas.paste(Image.fromarray(rgba), ((i % atlas_cols)*box, (i//atlas_cols)*box))
        atlas.save(folder / 'pieces.png')
        Image.fromarray(reference).save(folder / 'reference.png')
        ranks = np.concatenate(list(neighbor_ranks(logits, positions).values()))
        payload = {'key': key, 'category': category, 'model': model_id, 'index': index,
                   'val_index': None if custom else index+100,
                   'image': custom['source'] if custom else rows_meta[index]['path'], 'anchored': anchored,
                   'mode': 'custom' if custom else 'validation',
                   'strategy':strategy,
                   'rows': rows, 'cols': cols, 'cell': cell_size(box), 'pad': PAD,
                   'box': box, 'atlas_cols': atlas_cols, 'piece_count': len(pieces),
                   'truth':ground_truth,
                   'metrics':metrics | {'top1':float((ranks==1).mean())}, 'frames':frames}
        if strategy in FREE_STRATEGIES:
            payload['seed'] = growth.seed | {'slots':[(int(normalized[p][1])*cols+int(normalized[p][0])) for p in growth.seed['pieces']]}
            payload['coordinates'] = (growth.coordinates if strategy == 'strongest' else final_coordinates).tolist()
            payload['true_positions'] = positions.tolist()  # evaluation / audit only, not solver input
        (folder / 'result.json').write_text(json.dumps(payload))
        job.update(status='ready', url=f'/cache/{key}/result.json')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        job.update(status='error', message=str(exc))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/api/updater':
            try:
                q=parse_qs(url.query)
                model,category=q['model'][0],q['category'][0]
                anchored=q.get('anchor',['0'])[0]=='1'
                if model not in MODELS or category not in dict(CATEGORIES) or anchored and not category.startswith('flat_'):
                    raise ValueError('Invalid model, category, or anchor mode')
                fingerprint=identity(model)
                paths=sorted(CACHE.glob(f'{fingerprint}-{model}-{category}-*-{int(anchored)}/result.json'))
                signature=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths)
                scope=(model,category,anchored)
                if scope not in updater_cache or updater_cache[scope][0]!=signature:
                    puzzles=[]
                    for path in paths:
                        try:
                            puzzle=json.loads(path.read_text())
                            if puzzle.get('mode','validation')=='validation':puzzles.append(puzzle)
                        except (OSError,ValueError):continue
                    updater_cache[scope]=(signature,summarize_updater(puzzles))
                return self.json(updater_cache[scope][1])
            except (KeyError,ValueError,OSError) as exc:
                return self.json({'message':str(exc)},400)
        if url.path == '/api/batch-status':
            try:
                state = json.loads((CACHE/'batch-status.json').read_text())
                current = {key:identity(key) for key,(_,path) in MODELS.items() if path.exists()}
                if state.get('fingerprints') != current:
                    return self.json({'status':'stale','message':'Cached batch belongs to different weights or inference code; restart precompute.py.'})
                return self.json(state)
            except (OSError,ValueError):
                return self.json({'status':'idle'})
        if url.path == '/api/results':
            try:
                q = parse_qs(url.query)
                model, category = q['model'][0], q['category'][0]
                anchored = q.get('anchor',['0'])[0] == '1'
                strategy = q.get('strategy',['grid'])[0]
                if model not in MODELS or not MODELS[model][1].exists() or category not in dict(CATEGORIES):
                    raise ValueError('Invalid model or category')
                if anchored and not category.startswith('flat_'):
                    raise ValueError('Anchoring is available for straight and slanted cuts')
                if strategy not in ('grid',) + FREE_STRATEGIES or strategy in FREE_STRATEGIES and (anchored or not category.startswith('flat_')):
                    raise ValueError('Strongest pair and shrinking frame are separate modes for flat cuts; disable the top-left anchor')
                return self.json(validation_results(model, category, anchored, strategy))
            except (KeyError, ValueError) as exc:
                return self.json({'message':str(exc)},400)
        if url.path == '/api/gallery':
            return self.json([{'id':f'gallery-{i}', 'label':row['path'].split('/')[-1], 'thumbnail':f'/gallery/{i}.jpg'}
                              for i,row in enumerate(load_manifest('val')[100:112])])
        if url.path.startswith('/gallery/'):
            try:
                index = int(url.path.split('/')[-1].removesuffix('.jpg'))
                if not 0 <= index < 12:
                    raise ValueError('Invalid gallery image')
                row = load_manifest('val')[100+index]
                with Image.open(ROOT/'data/processed'/row['path']) as image:
                    image = ImageOps.fit(image.convert('RGB'), (120,120))
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=75)
                body = buffer.getvalue()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self.wfile.write(body)
                return
            except (ValueError, OSError):
                return self.json({'message':'Gallery image not found'},404)
        if url.path == '/api/meta':
            return self.json({'models': [{'id': k, 'label': v[0], 'fingerprint':identity(k)} for k,v in MODELS.items() if v[1].exists()],
                              'categories': [{'id': k, 'label': v, 'count': 100} for k,v in CATEGORIES],
                              'cohort': 'SA-1B · validation 100–199'})
        if url.path == '/api/puzzle':
            try:
                q = parse_qs(url.query)
                model, category = q['model'][0], q['category'][0]
                index = int(q['index'][0])
                anchor = q.get('anchor', ['0'])[0] == '1'
                strategy = q.get('strategy',['grid'])[0]
                custom = None
                if q.get('source'):
                    source = q['source'][0]
                    side = int(q.get('grid',['6'])[0])
                    gallery_ok = source.startswith('gallery-') and source[8:].isdigit() and 0 <= int(source[8:]) < 12
                    upload_id = source.removeprefix('upload-')
                    upload_ok = source.startswith('upload-') and len(upload_id)==32 and all(c in '0123456789abcdef' for c in upload_id) and (UPLOADS/f'{upload_id}.png').exists()
                    if not (gallery_ok or upload_ok) or side not in (4,6,8,10) or category not in ('clean','flat_even','flat_diagonal'):
                        raise ValueError('Choose a valid image, cut type, and grid size')
                    custom = {'source':source, 'grid':side}
                if model not in MODELS or not MODELS[model][1].exists() or category not in dict(CATEGORIES) or not 0 <= index < 100:
                    raise ValueError('Invalid model, category, or image number')
                if anchor and not category.startswith('flat_'):
                    raise ValueError('Anchoring is available for straight and slanted cuts')
                if strategy not in ('grid',) + FREE_STRATEGIES or strategy in FREE_STRATEGIES and (anchor or not category.startswith('flat_')):
                    raise ValueError('Strongest pair and shrinking frame are separate modes for flat cuts; disable the top-left anchor')
                source_key = f'{custom["source"]}-{custom["grid"]}' if custom else str(index)
                suffix = replay_suffix(strategy, anchor)
                key = f'{identity(model)}-{model}-{category}-{source_key}-{suffix}'
                if (CACHE/key/'result.json').exists():
                    return self.json({'status':'ready', 'url':f'/cache/{key}/result.json'})
                if key not in jobs or jobs[key]['status'] == 'error':
                    jobs[key] = {'status': 'loading', 'message':'Queued for local inference'}
                    pool.submit(build, key, model, category, index, anchor, custom, strategy)
                return self.json(jobs[key])
            except (KeyError, ValueError) as exc:
                return self.json({'status':'error', 'message':str(exc)}, 400)
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != '/api/upload':
            return self.json({'message':'Unknown endpoint'},404)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 12*1024*1024:
                return self.json({'message':'Choose an image smaller than 12 MB'},413)
            raw = self.rfile.read(length)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width*image.height > 40_000_000:
                    raise ValueError('Choose an image smaller than 40 megapixels')
                image = ImageOps.exif_transpose(image).convert('RGB')
                if min(image.size) < 120:
                    raise ValueError('Use an image at least 120 pixels wide and tall')
                image = ImageOps.fit(image,(400,400),method=Image.Resampling.LANCZOS)
                digest = hashlib.sha256(image.tobytes()).hexdigest()[:32]
                image.save(UPLOADS/f'{digest}.png')
            return self.json({'id':f'upload-{digest}'})
        except (ValueError, OSError, Image.DecompressionBombError, UnidentifiedImageError) as exc:
            return self.json({'message':f'Cannot use this image: {exc}'},400)

    def translate_path(self, path):
        clean = urlparse(path).path
        if clean.startswith('/cache/'):
            target = (CACHE/clean[len('/cache/'):]).resolve()
            if target.is_relative_to(CACHE.resolve()) and target.name in ('pieces.png','reference.png','result.json'):
                return str(target)
            return str(WEB/'not-found')
        # Serve only browser assets, never Python source or checkpoints.
        if clean not in ('/', '/index.html', '/app.js', '/style.css', '/placement.js'):
            return str(WEB/'not-found')
        return super().translate_path(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH', help='Add a compatible EdgeComparator checkpoint to the model menu')
    args = parser.parse_args()
    for i, spec in enumerate(args.checkpoint):
        label, separator, filename = spec.partition('=')
        path = Path(filename).expanduser().resolve()
        if not separator or not label.strip() or not path.is_file():
            parser.error('--checkpoint must be NAME=PATH pointing to an existing checkpoint')
        load_comparator(path)  # Validate architecture and state dict before advertising this model.
        MODELS[f'custom_{i}'] = (label.strip(),path)
    torch.set_num_threads(4)
    print(f'Interlock viewer: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
