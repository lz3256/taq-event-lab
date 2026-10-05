"""Serial, equal training-time comparison; not a hardware-independent FLOP benchmark."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import html
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from .baselines import transition_counts, select_strength, shrinkage_probabilities
from .common import digest, file_hash, load_config, read_json, write_json
from .data import load_prepared
from .dense_report import coverage
from .engine import (device_for, environment, evaluate_model, fit_baseline,
                     load_checkpoint, training_signature, windows)
from .model import EventTransformer
from .study_report import evaluation_pairs, record_scores


def load_compute_plan(path):
    path = Path(path).resolve()
    plan = read_json(path)
    if sorted(j['kind'] for j in plan['jobs']) != ['joint', 'sequential']:
        raise ValueError('Time pilot must contain exactly joint and sequential')
    jobs, reference, manifest = [], None, None
    for item in plan['jobs']:
        config_path = (path.parent / item['config']).resolve()
        cfg = load_config(config_path)
        if cfg['train'].get('time_budget_seconds') is None or cfg['train'].get('supervision') != 'dense':
            raise ValueError('Explicit dense supervision and time budget required')
        if cfg['train']['device'] != 'cpu':
            raise ValueError('This experiment is a CPU timing pilot')
        comparable = copy.deepcopy(cfg)
        comparable.pop('output_dir')
        if reference is None:
            reference, manifest = comparable, load_prepared(cfg)
        elif comparable != reference:
            raise ValueError('Paired models must use identical settings, including seed and time budget')
        elif load_prepared(cfg, verify_files=False)['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Prepared data mismatch')
        jobs.append({**item, 'cfg': cfg, 'config_path': str(config_path)})
    root = (path.parent / plan['output_dir']).resolve()
    fingerprint = digest({'plan': plan, 'configs': [j['cfg'] for j in jobs], 'data': manifest['fingerprint']})
    return plan, jobs, manifest, root, fingerprint


def completed(job, manifest):
    folder = Path(job['cfg']['output_dir']) / job['kind']
    cp = load_checkpoint(folder / 'last.pt')
    summary = read_json(folder / 'training_summary.json')
    if cp['signature'] != training_signature(job['cfg'], manifest, job['kind']):
        raise ValueError('Time-budget checkpoint signature mismatch')
    budget = job['cfg']['train']['time_budget_seconds']
    if not summary['complete'] or cp['train_seconds'] < budget:
        raise ValueError('Time budget not completed')
    if summary['step'] != cp['step'] or summary['train_seconds'] != cp['train_seconds']:
        raise ValueError('Summary/checkpoint mismatch')
    if summary['budget_overshoot_seconds'] > summary['final_update_seconds'] + 1e-8:
        raise ValueError('Time budget exceeded by more than one optimizer update')
    return cp, summary


def train_compute(path):
    plan, jobs, manifest, root, fingerprint = load_compute_plan(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    progress_path = root / 'progress.json'
    if progress_path.exists() and read_json(progress_path)['fingerprint'] != fingerprint:
        raise ValueError('Time plan changed; choose a new output directory')
    progress = {'fingerprint': fingerprint, 'order': [j['kind'] for j in jobs],
                'execution': 'serial subprocesses within this study; external contention not measured', 'jobs': {}}
    for job in jobs:
        kind = job['kind']
        folder = Path(job['cfg']['output_dir']) / kind
        if (folder / 'last.pt').exists() and (folder / 'training_summary.json').exists():
            if read_json(folder / 'training_summary.json')['complete']:
                completed(job, manifest)
                progress['jobs'][kind] = {'status': 'complete_reused'}
                write_json(progress_path, progress)
                continue
        progress['jobs'][kind] = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat()}
        write_json(progress_path, progress)
        command = [sys.executable, '-u', '-m', 'taq_lab', 'train', '--config', job['config_path'], '--model', kind]
        if (folder / 'last.pt').exists():
            command.append('--resume')
        with (root / 'logs' / f'{kind}.log').open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            progress['jobs'][kind]['status'] = 'failed'
            write_json(progress_path, progress)
            raise RuntimeError(f'{kind} failed; inspect its log')
        _, summary = completed(job, manifest)
        progress['jobs'][kind].update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
                                     steps=summary['step'], train_seconds=summary['train_seconds'])
        write_json(progress_path, progress)
        print(f"{kind}: {summary['step']} steps; {summary['train_seconds']:.3f} training seconds", flush=True)
    return root


def evaluate_compute(path):
    plan, jobs, manifest, root, fingerprint = load_compute_plan(path)
    for job in jobs:
        completed(job, manifest)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'evaluation').mkdir(exist_ok=True)
    cfg = jobs[0]['cfg']
    device = device_for(cfg['train']['device'])
    torch.set_num_threads(cfg['train']['threads'])
    datasets = {s: windows(cfg, manifest, s) for s in ('val', 'test', 'heldout')}
    references = {s: evaluation_pairs(data, cfg['tokenizer']['bins'], cfg['evaluation']['max_windows'])
                  for s, data in datasets.items()}
    prior = fit_baseline(cfg, manifest)
    training_data = windows(cfg, manifest, 'train', True)
    counts = transition_counts(training_data, cfg['tokenizer']['bins'])
    _, previous, targets = references['val']
    strength, grid = select_strength(counts, prior, previous, targets, plan['markov_strengths'])
    markov = shrinkage_probabilities(counts, prior, strength)
    write_json(root / 'baseline_parameters.json', {'data_fingerprint': manifest['fingerprint'],
               'strength': strength, 'validation_grid': grid, 'training_target_exposures': int(counts.sum())})
    metrics, session_rows, training_rows, curves, coverage_rows = [], [], [], [], []
    for name, probability in [('frequency_full', prior), ('markov_full', markov)]:
        for split, data in datasets.items():
            ids, previous, targets = references[split]
            losses = -np.log(probability[targets] if probability.ndim == 1 else probability[previous, targets])
            metric, sessions = record_scores(root, name, None, split, data, ids, losses)
            metrics.append(metric); session_rows.extend(sessions)
    for job in jobs:
        c, kind = job['cfg'], job['kind']
        last, summary = completed(job, manifest)
        folder = Path(c['output_dir']) / kind
        cp_path = folder / 'best.pt'
        cp = load_checkpoint(cp_path)
        if cp['signature'] != last['signature'] or cp['step'] > last['step']:
            raise ValueError('Best checkpoint mismatch')
        validation_records = [r for r in last['history'] if r['val_nll_per_event'] is not None]
        if not np.isclose(cp['best_val_nll'], min(r['val_nll_per_event'] for r in validation_records)):
            raise ValueError('Selection did not minimize validation loss')
        covered = coverage(training_data, c['train']['seed'], last['step'], c['train']['batch_size'],
                           {cp['step'], last['step']})
        covered = covered[covered.supervision == 'dense'].copy()
        covered['variant'] = kind
        coverage_rows.extend(covered.to_dict('records'))
        unique_targets = int(covered.loc[covered.step == last['step'], 'unique_targets'].iloc[0])
        training_rows.append({'variant': kind, 'seed': c['train']['seed'], 'step': last['step'],
              'best_step': cp['step'], 'best_train_seconds': cp['train_seconds'], 'train_seconds': last['train_seconds'],
              'loop_wall_seconds': last['wall_seconds'],
              'budget_seconds': c['train']['time_budget_seconds'], 'overshoot_seconds': summary['budget_overshoot_seconds'],
              'final_update_seconds': summary['final_update_seconds'], 'validation_checkpoints': len(validation_records),
              'target_exposures': summary['target_events_seen'], 'unique_targets': unique_targets,
              'parameters': summary['parameters']['total'],
              'checkpoint_sha256': file_hash(cp_path), 'last_checkpoint_sha256': file_hash(folder / 'last.pt')})
        curves.extend({'variant': kind, **r} for r in last['history'])
        model = EventTransformer(c, kind).to(device)
        model.load_state_dict(cp['model'])
        for split, data in datasets.items():
            ids, losses = evaluate_model(model, data, c, device)
            if not np.array_equal(ids, references[split][0]):
                raise ValueError('Evaluation windows differ')
            metric, sessions = record_scores(root, kind, c['train']['seed'], split, data, ids, losses,
                                             {'selected_checkpoint_step': cp['step']})
            metrics.append(metric); session_rows.extend(sessions)
        del model, cp, last
    pd.DataFrame(metrics).to_csv(root / 'metrics.csv', index=False)
    pd.DataFrame(session_rows).to_csv(root / 'session_metrics.csv', index=False)
    pd.DataFrame(training_rows).to_csv(root / 'training_runs.csv', index=False)
    pd.DataFrame(curves).to_csv(root / 'learning_curves.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(root / 'target_coverage.csv', index=False)
    write_json(root / 'evaluation_metadata.json', {'fingerprint': fingerprint, 'data_fingerprint': manifest['fingerprint'],
              'plan': plan, 'configs': [j['cfg'] for j in jobs], 'environment': environment(device),
              'provenance': manifest['provenance'], 'selection': 'lowest validation last-event NLL, with validation every 10% of time budget',
              'code_sha256': {n: file_hash(Path(__file__).with_name(n)) for n in ['engine.py','model.py','compute_study.py']}})
    return make_report(root)


def make_report(root):
    from .report import plt
    root = Path(root)
    metrics = pd.read_csv(root / 'metrics.csv')
    training = pd.read_csv(root / 'training_runs.csv')
    curves = pd.read_csv(root / 'learning_curves.csv')
    meta = read_json(root / 'evaluation_metadata.json')
    table = metrics.pivot(index='variant', columns='split', values='nll_per_event')[['val','test','heldout']]
    table.to_csv(root / 'summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12,4.5), layout='constrained')
    for kind, group in curves.groupby('variant'):
        valid = group.dropna(subset=['val_nll_per_event'])
        axes[0].plot(valid.train_seconds, valid.val_nll_per_event, marker='o', label=kind)
        axes[1].plot(group.train_seconds, group.learning_rate, label=kind)
    axes[0].set(xlabel='Accumulated training seconds', ylabel='Validation last-event NLL (nats)')
    axes[1].set(xlabel='Accumulated training seconds', ylabel='Learning rate')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(root / 'time_comparison.png', dpi=160)
    plt.close(fig)
    notes = [
        f"两种模型从零训练，seed={meta['configs'][0]['train']['seed']}，各自训练时间预算 {meta['configs'][0]['train']['time_budget_seconds']:g} 秒；同设备、线程、精度和 batch。",
        '时间包含采样、数据转移、前向、反向与优化器更新；不含验证、保存和启动。预算在完整更新后检查，因此最多超出一次更新。',
        'loop_wall_seconds 另计训练循环的实际经过时间（含验证和此前保存；不含启动及最后一次保存），便于区分纯训练预算与实际等待时间。',
        '学习率按累计训练时间 warmup 和余弦衰减；每经过 10% 预算验证一次，仅据验证集选择最佳检查点。训练顺序预先固定，串行执行。',
        '这衡量当前 CPU 与实现的训练时间效率，不是等 FLOPs，也不直接代表 GPU 或其他实现的效率。系统负载、温度与固定运行顺序仍有影响。',
        '本轮只有一个配对 seed；达到预算的更新步数和采样事件数不同，不能据此独立证明某种编码的样本效率。墙钟调度受实测耗时影响，真实运行不能保证逐位复现。',
        '目标曝光含重复；不同目标位置由实际采样流重建，也不代表独立统计样本。',
        '旧两天测试日期已被查看；这是探索性比较，不是新日期确认。未验证迁移、报价、成交执行或交易收益。',
    ]
    title = '等训练时间：CPU 单种子诊断'
    banner = '真实 TAQ' if meta['provenance'] == 'real_taq' else 'SYNTHETIC DATA — PIPELINE TEST ONLY'
    training_table = training.drop(columns=['checkpoint_sha256','last_checkpoint_sha256'])
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#24353d;background:#f5f7f8}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}img{{width:100%}}</style>
<h1>{title}</h1><p>{banner}</p><section><h2>末事件 NLL，越低越好</h2>{table.to_html(float_format=lambda x:f'{x:.4f}')}</section>
<section><img src="time_comparison.png" alt="Validation loss and learning rate by elapsed training time"></section>
<section><h2>预算与检查点</h2><div class="scroll">{training_table.to_html(index=False,float_format=lambda x:f'{x:.3f}')}</div></section>
<section><h2>解释范围</h2><ul>{''.join('<li>'+html.escape(n)+'</li>' for n in notes)}</ul></section></html>'''
    (root/'report.html').write_text(body)
    lines = [f'# {title}', '', banner, '', '| 方法 | 验证 | 未来日期 | NVDA |','|---|---:|---:|---:|']
    lines += [f'| {name} | {r.val:.4f} | {r.test:.4f} | {r.heldout:.4f} |' for name,r in table.iterrows()]
    lines += ['', '![时间曲线](time_comparison.png)', ''] + ['- '+n for n in notes]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(f"Compute report: {root/'report.html'}", flush=True)
    return root/'report.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train','evaluate','run'])
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    if args.command in ('train','run'):
        train_compute(args.plan)
    if args.command in ('evaluate','run'):
        evaluate_compute(args.plan)


if __name__ == '__main__':
    main()
