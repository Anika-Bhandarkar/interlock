"""Resume a full, matched validation cache through the viewer's single inference queue.

Run with the local viewer running: .venv/bin/python frontend/precompute.py
Ctrl-C stops scheduling after the current request; rerunning skips completed replays.
"""
import argparse
import fcntl
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

CACHE = Path(__file__).resolve().parents[1]/'artifacts/viewer_cache'


def atomic_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--count',type=int,default=100,choices=range(1,101),metavar='1..100')
    args = parser.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    lock = (CACHE/'precompute.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('A validation batch is already running')
    def get(path):
        with urllib.request.urlopen(args.url+path, timeout=120) as response:
            return json.load(response)
    meta=get('/api/meta')
    fingerprints={m['id']:m['fingerprint'] for m in meta['models']}
    # Walk matched image indices first, so all models get coverage throughout the run.
    tasks=[]
    for index in range(args.count):
        for category in meta['categories']:
            for model in meta['models']:
                modes=[(0,'grid'),(1,'grid'),(0,'strongest')] if category['id'].startswith('flat_') else [(0,'grid')]
                for anchor,strategy in modes:
                    tasks.append({'model':model['id'],'category':category['id'],'index':index,'anchor':anchor,'strategy':strategy})
    state={'status':'running','fingerprints':fingerprints,'total':len(tasks),'completed':0,
           'failed':0,'errors':[],'count_per_combination':args.count,'started':datetime.now(timezone.utc).isoformat()}
    status_path=CACHE/'batch-status.json'
    summary_path=CACHE/'batch-summary.json'
    measured={}
    started=time.monotonic()
    def save():
        state['updated']=datetime.now(timezone.utc).isoformat()
        state['elapsed_seconds']=round(time.monotonic()-started)
        atomic_json(status_path,state)
        atomic_json(summary_path,{'fingerprints':fingerprints,'combinations':measured})
    save()
    try:
        for query in tasks:
            state['current']=query
            save()
            path='/api/puzzle?'+urllib.parse.urlencode(query)
            try:
                while True:
                    result=get(path)
                    if result['status']=='error':raise RuntimeError(result['message'])
                    if result['status']=='ready':break
                    time.sleep(1)
                data=get(result['url'])
                key=f'{query["model"]}/{query["category"]}/{"strongest" if query["strategy"]=="strongest" else "anchored" if query["anchor"] else "unanchored"}'
                row=measured.setdefault(key,{'evaluated':0,'solved':0,'failed_indices':[], 'location_accuracy_sum':0., 'neighbor_accuracy_sum':0.})
                row['evaluated']+=1
                row['solved']+=int(data['metrics']['perfect'])
                row['location_accuracy_sum']+=data['metrics']['direct']
                row['neighbor_accuracy_sum']+=data['metrics']['neighbors']
                if not data['metrics']['perfect']:row['failed_indices'].append(query['index'])
                row['solve_rate']=row['solved']/row['evaluated']
                row['mean_location_accuracy']=row['location_accuracy_sum']/row['evaluated']
                row['mean_neighbor_accuracy']=row['neighbor_accuracy_sum']/row['evaluated']
                state['completed']+=1
            except (OSError,ValueError,RuntimeError) as exc:
                state['failed']+=1
                state['errors'].append({'query':query,'message':str(exc)})
                # A disconnected server should not burn through the entire remaining task list.
                if isinstance(exc,OSError):raise
            save()
            print(f'{state["completed"]}/{state["total"]} cached; {state["failed"]} errors; {query}',flush=True)
        state['status']='complete' if not state['failed'] else 'incomplete'
    except KeyboardInterrupt:
        state['status']='paused'
    except Exception as exc:
        state.update(status='error',message=str(exc))
    finally:
        save()
        print(f'Batch {state["status"]}: {state["completed"]}/{state["total"]} cached. Results: {summary_path}',flush=True)


if __name__=='__main__':
    main()
