"""Fixed-plan, resumable experiments. Workers use separate Python processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from pathlib import Path
import re
import subprocess
import sys

from .common import digest, load_config, read_json, write_json
from .data import load_prepared
from .engine import load_checkpoint, training_signature


def load_plan(path):
    path = Path(path).resolve()
    plan = read_json(path)
    jobs = []
    seen = set()
    reference = None
    manifest = None
    for item in plan['jobs']:
        if not re.fullmatch(r'[a-zA-Z0-9_]+', item['variant']):
            raise ValueError('Variant must be a simple identifier')
        config_path = (path.parent / item['config']).resolve()
        cfg = load_config(config_path)
        if 'time_budget_seconds' in cfg['train']:
            raise ValueError('Use compute_study for time-budget experiments; this study requires fixed steps')
        identity = (item['variant'], cfg['train']['seed'])
        if identity in seen:
            raise ValueError(f'Duplicate variant/seed: {identity}')
        seen.add(identity)
        if item['kind'] not in ('joint', 'sequential'):
            raise ValueError('Unknown model kind')
        comparable = copy.deepcopy(cfg)
        comparable.pop('output_dir')
        comparable['train'].pop('seed')
        comparable['tokenizer'].pop('field_order')
        if reference is None:
            reference = comparable
            manifest = load_prepared(cfg)
        elif reference != comparable:
            raise ValueError('Study changes more than seed, field order or output directory')
        else:
            other = load_prepared(cfg, verify_files=False)
            if other['fingerprint'] != manifest['fingerprint']:
                raise ValueError('Prepared data differs across jobs')
        jobs.append({**item, 'config_path': str(config_path), 'cfg': cfg,
                     'id': f"{item['variant']}_seed{cfg['train']['seed']}"})
    if not jobs:
        raise ValueError('Empty study')
    for variant in {j['variant'] for j in jobs}:
        selected = [j for j in jobs if j['variant'] == variant]
        if sorted(j['cfg']['train']['seed'] for j in selected) != sorted(plan['seeds']):
            raise ValueError('Every variant must cover exactly the planned seeds')
        if len({(j['kind'], tuple(j['cfg']['tokenizer']['field_order'])) for j in selected}) != 1:
            raise ValueError('Model kind/order must be fixed within a variant')
    locations = [(j['cfg']['output_dir'], j['kind']) for j in jobs]
    if len(set(locations)) != len(locations):
        raise ValueError('Jobs share a checkpoint directory')
    root = (path.parent / plan['output_dir']).resolve()
    fingerprint = digest({'plan': plan, 'configs': [j['cfg'] for j in jobs],
                          'data': manifest['fingerprint']})
    return plan, jobs, manifest, root, fingerprint


def train_study(path, workers=2):
    if not 1 <= workers <= 2:
        raise ValueError('Use one or two workers to bound CPU and memory demand')
    plan, jobs, manifest, root, fingerprint = load_plan(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    progress_path = root / 'progress.json'
    if progress_path.exists() and read_json(progress_path)['fingerprint'] != fingerprint:
        raise ValueError('Study plan changed; use a new study output directory')
    states = {}
    pending = []
    for job in jobs:
        cp_path = Path(job['cfg']['output_dir']) / job['kind'] / 'last.pt'
        if cp_path.exists():
            cp = load_checkpoint(cp_path)
            if cp['signature'] != training_signature(job['cfg'], manifest, job['kind']):
                raise ValueError(f"Checkpoint mismatch: {job['id']}")
            if cp['step'] == job['cfg']['train']['max_steps']:
                best = load_checkpoint(cp_path.with_name('best.pt'))
                if best['signature'] != cp['signature']:
                    raise ValueError('Best checkpoint mismatch')
                states[job['id']] = 'complete_reused'
                continue
        states[job['id']] = 'pending'
        pending.append(job)

    def save():
        write_json(progress_path, {'fingerprint': fingerprint, 'workers': workers,
                                  'data_fingerprint': manifest['fingerprint'], 'jobs': states})

    def execute(job):
        cp = Path(job['cfg']['output_dir']) / job['kind'] / 'last.pt'
        command = [sys.executable, '-u', '-m', 'taq_lab', 'train', '--config',
                   job['config_path'], '--model', job['kind']]
        if cp.exists():
            command.append('--resume')
        with (root / 'logs' / f"{job['id']}.log").open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{job['id']} failed; inspect its log")
        last = load_checkpoint(cp)
        if last['step'] != job['cfg']['train']['max_steps']:
            raise RuntimeError(f"Incomplete training: {job['id']}")
        return job['id']

    save()
    print(f"Study: {len(jobs) - len(pending)} reused; {len(pending)} training jobs; {workers} workers", flush=True)
    # Each child owns its RNG and torch thread pool; do not train models in threads.
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, job): job for job in pending}
        for job in pending:
            states[job['id']] = 'submitted'
        save()
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                states[job['id']] = 'complete'
            except Exception as exc:
                states[job['id']] = 'failed'
                errors.append(str(exc))
            save()
            print(f"{job['id']}: {states[job['id']]}", flush=True)
    if errors:
        raise RuntimeError('; '.join(errors))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate', 'run', 'status'])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'status':
        _, jobs, _, root, fingerprint = load_plan(args.plan)
        progress = read_json(root / 'progress.json') if (root / 'progress.json').exists() else {'jobs': {}}
        if progress.get('fingerprint', fingerprint) != fingerprint:
            raise ValueError('Progress belongs to a different study')
        for job in jobs:
            history = Path(job['cfg']['output_dir']) / job['kind'] / 'history.jsonl'
            step = 0
            if history.exists():
                import json
                lines = history.read_text().splitlines()
                if lines:
                    step = json.loads(lines[-1])['step']
            print(f"{job['id']}: saved step {step}/{job['cfg']['train']['max_steps']}; {progress['jobs'].get(job['id'], 'not submitted')}")
        return
    if args.command in ('train', 'run'):
        train_study(args.plan, args.workers)
    if args.command in ('evaluate', 'run'):
        from .study_report import evaluate_study
        evaluate_study(args.plan)


if __name__ == '__main__':
    main()
