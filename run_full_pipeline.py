#!/usr/bin/env python3
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT = Path(_file_).resolve().parent
CATEGORIES = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
STAGES = ['mcl', 'pairs', 'csda']

def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def commands(category, python):
    return [[python, '-u', str(ROOT / 'train_mcl.py'), '--category', category, '--epochs', '6100', '--resume'], [python, '-u', str(ROOT / 'generate_pairs.py'), '--category', category, '--embeddings', str(ROOT / 'stage1_embeddings' / category / 'embeddings_final.pt')], [python, '-u', str(ROOT / 'train_eval_csda.py'), '--category', category, '--masks', str(ROOT / 'stage2_pairs'), '--output', str(ROOT / 'stage5_full'), '--evaluate']]

def atomic_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def gpu_occupancy():

    def query(fields, kind):
        result = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
        return [line.strip().split(', ') for line in result.stdout.splitlines() if line.strip()]
    mapping = {uuid: int(index) for (index, uuid) in query('index,uuid', 'gpu')}
    occupied = {index: [] for index in mapping.values()}
    for (uuid, pid) in query('gpu_uuid,pid', 'compute-apps'):
        if uuid not in mapping:
            raise RuntimeError(f'Unrecognized GPU UUID in compute process query: {uuid}')
        occupied[mapping[uuid]].append(int(pid))
    return occupied

def main():
    parser = argparse.ArgumentParser(description=_doc_)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--state-dir', type=Path, default=ROOT / 'logs_full_pipeline')
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or args.poll_seconds <= 0:
        parser.error('GPU indices must be unique and polling interval positive')
    if len(set(args.categories)) != len(args.categories):
        parser.error('Categories must be unique')
    if args.dry_run:
        print(json.dumps({'gpus': args.gpus, 'state_dir': str(args.state_dir), 'commands': {c: commands(c, args.python) for c in args.categories}}, indent=2))
        return 0
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    if args.background:
        argv = [v for v in sys.argv[1:] if v != '--background']
        with (args.state_dir / 'scheduler.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(_file_).resolve()), *argv], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        time.sleep(1)
        if child.poll() is not None:
            print(f"Scheduler exited ({child.returncode}); inspect {args.state_dir / 'scheduler.log'}")
            return child.returncode or 0
        print(f'Background scheduler PID {child.pid}; logs: {args.state_dir}')
        return 0
    lock = (ROOT / '._full_pipeline.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another  full pipeline scheduler or its surviving child holds the lock.', flush=True)
        return 2
    state_path = args.state_dir / 'state.json'
    pid_path = args.state_dir / 'scheduler.pid'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'categories': {}}
    for category in args.categories:
        job = state['categories'].setdefault(category, {'next_stage': 0, 'status': 'pending', 'history': []})
        if job['status'] == 'running':
            job['status'] = 'pending'
            job['history'].append({'event': 'interrupted_scheduler_recovery', 'time': timestamp()})
        if args.retry_failed and job['status'] == 'failed':
            job['status'] = 'pending'
    pid_path.write_text(str(os.getpid()) + '\n')
    state['scheduler'] = {'pid': os.getpid(), 'started': timestamp(), 'gpus': args.gpus}
    active = {}
    stopping = False

    def save():
        state['updated'] = timestamp()
        atomic_json(state_path, state)

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        print(f'Received signal {signum}; draining current stages without launching new ones.', flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    save()
    try:
        while True:
            for (gpu, (process, category, logfile)) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                logfile.close()
                job = state['categories'][category]
                job['history'][-1].update(finished=timestamp(), returncode=code)
                if code == 0:
                    job['next_stage'] += 1
                    job['status'] = 'completed' if job['next_stage'] == len(STAGES) else 'pending'
                else:
                    job['status'] = 'failed'
                job.pop('pid', None)
                print(f"{category} GPU {gpu}: stage exit={code}, status={job['status']}", flush=True)
                del active[gpu]
                save()
            pending = [c for c in args.categories if state['categories'][c]['status'] == 'pending']
            pending.sort(key=lambda c: state['categories'][c]['next_stage'] != 0)
            if not active and (stopping or not pending):
                break
            if not stopping and pending:
                try:
                    occupied = gpu_occupancy()
                    missing = set(args.gpus) - set(occupied)
                    if missing:
                        raise RuntimeError(f'Requested GPUs missing: {missing}')
                except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as exc:
                    print(f'GPU query failed; no work launched: {exc}', flush=True)
                else:
                    for gpu in args.gpus:
                        if gpu in active or occupied[gpu] or (not pending) or stopping:
                            continue
                        category = pending.pop(0)
                        job = state['categories'][category]
                        index = job['next_stage']
                        command = commands(category, args.python)[index]
                        log_path = args.state_dir / f'{category}.{STAGES[index]}.log'
                        logfile = log_path.open('a')
                        event = {'stage': STAGES[index], 'gpu': gpu, 'command': command, 'started': timestamp(), 'log': str(log_path)}
                        job['history'].append(event)
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
                        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                        try:
                            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=logfile, stderr=logfile, start_new_session=True, pass_fds=(lock.fileno(),))
                        except OSError as exc:
                            logfile.close()
                            job['status'] = 'failed'
                            event.update(finished=timestamp(), error=str(exc))
                        else:
                            job.update(status='running', pid=process.pid)
                            event['pid'] = process.pid
                            active[gpu] = (process, category, logfile)
                            print(f'Launched {category}/{STAGES[index]} GPU {gpu} PID {process.pid}', flush=True)
                        save()
            time.sleep(args.poll_seconds)
        state['scheduler']['finished'] = timestamp()
        state['scheduler']['status'] = 'drained' if stopping else 'finished'
        save()
    finally:
        pid_path.unlink(missing_ok=True)
        lock.close()
    return 1 if any((state['categories'][c]['status'] == 'failed' for c in args.categories)) else 0
if _name_ == '_main_':
    raise SystemExit(main())
