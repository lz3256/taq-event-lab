"""Evaluate a fixed study on identical windows and report seed variability."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import transition_counts, select_strength, shrinkage_probabilities
from .common import file_hash, read_json, write_json
from .engine import (device_for, evaluate_model, fit_baseline, load_checkpoint,
                     training_signature, windows)
from .model import EventTransformer
from .study import load_plan
from .tokenization import joint_encode


def evaluation_pairs(data, bins, maximum):
    ids = data.evaluation_indices(maximum)
    pairs = np.stack([data[int(i)][-2:] for i in ids])
    return ids, joint_encode(pairs[:, 0], bins), joint_encode(pairs[:, 1], bins)


def sampled_transition_counts(data, bins, seed, steps, batch_size, supervision='last_event'):
    """Exactly the target index stream used by engine.train, including replacement."""
    generator = torch.Generator().manual_seed(seed + 1000)
    indices = np.concatenate([torch.randint(len(data), (batch_size,), generator=generator).numpy()
                              for _ in range(steps)])
    vocab = bins ** 3
    counts = np.zeros((vocab, vocab), dtype=np.int64)
    if supervision not in ('last_event', 'dense'):
        raise ValueError('Unknown supervision')
    for start in range(0, len(indices), 1024):
        sequences = np.stack([data[int(i)] if supervision == 'dense' else data[int(i)][-2:]
                              for i in indices[start:start + 1024]])
        codes = joint_encode(sequences, bins)
        keys = (codes[:, :-1] * vocab + codes[:, 1:]).ravel()
        counts += np.bincount(keys, minlength=vocab ** 2).reshape(vocab, vocab)
    return counts


def prepare_baselines(path):
    plan, jobs, manifest, root, fingerprint = load_plan(path)
    cfg = jobs[0]['cfg']
    root.mkdir(parents=True, exist_ok=True)
    params_path = root / 'baseline_parameters.json'
    if params_path.exists():
        params = read_json(params_path)
        if params['fingerprint'] != fingerprint:
            raise ValueError('Baseline cache belongs to a different study')
        if file_hash(root / 'baseline_probabilities.npz') != params['probabilities_sha256']:
            raise ValueError('Baseline probability checksum mismatch')
        with np.load(root / 'baseline_probabilities.npz', allow_pickle=False) as archive:
            probabilities = {k: archive[k] for k in archive.files}
        return probabilities, params
    bins = cfg['tokenizer']['bins']
    training = windows(cfg, manifest, 'train', True)
    validation = windows(cfg, manifest, 'val')
    _, previous, targets = evaluation_pairs(validation, bins, cfg['evaluation']['max_windows'])
    prior = fit_baseline(cfg, manifest)
    probabilities = {'frequency_full': prior}
    params = {'fingerprint': fingerprint, 'data_fingerprint': manifest['fingerprint'],
              'selection': 'shrinkage strength selected ONLY by validation NLL', 'models': {}}
    candidates = [('markov_full', None, transition_counts(training, bins), prior)]
    for seed in plan['seeds']:
        counts = sampled_transition_counts(training, bins, seed, cfg['train']['max_steps'], cfg['train']['batch_size'],
                                           cfg['train'].get('supervision', 'last_event'))
        small_prior = counts.sum(0).astype(float) + 1.0
        small_prior /= small_prior.sum()
        candidates.append((f'markov_matched_seed{seed}', seed, counts, small_prior))
    for name, seed, counts, prior in candidates:
        strength, scores = select_strength(counts, prior, previous, targets, plan['markov_strengths'])
        probabilities[name] = shrinkage_probabilities(counts, prior, strength)
        params['models'][name] = {'seed': seed, 'training_target_exposures': int(counts.sum()),
                                  'strength': strength, 'validation_grid': scores}
    temp = root / 'baseline_probabilities.tmp.npz'
    np.savez_compressed(temp, **probabilities)
    temp.replace(root / 'baseline_probabilities.npz')
    params['probabilities_sha256'] = file_hash(root / 'baseline_probabilities.npz')
    write_json(params_path, params)
    return probabilities, params


def record_scores(root, variant, seed, split, data, ids, losses, extra=None):
    if len(ids) != len(losses) or not np.isfinite(losses).all():
        raise ValueError('Invalid evaluation scores')
    rows = []
    for index, loss in zip(ids, losses):
        sid, start = data.locate(int(index))
        session = data.sessions[sid]
        rows.append({'window_index': int(index), 'session': session['id'], 'symbol': session['symbol'],
                     'date': session['date'], 'target_event_index': start + data.context, 'nll_per_event': float(loss)})
    frame = pd.DataFrame(rows)
    suffix = f'seed{seed}' if seed is not None else 'deterministic'
    frame.to_csv(root / 'evaluation' / f'{variant}_{suffix}_{split}.csv', index=False)
    grouped = frame.groupby(['session', 'symbol', 'date']).nll_per_event.agg(['mean', 'count']).reset_index()
    sessions = [{'variant': variant, 'seed': seed, 'split': split, **r} for r in grouped.to_dict('records')]
    metric = {'variant': variant, 'seed': seed, 'split': split, 'nll_per_event': float(np.mean(losses)),
              'session_macro_nll': float(grouped['mean'].mean()), 'evaluated_events': len(ids), **(extra or {})}
    return metric, sessions


def evaluate_study(path):
    plan, jobs, manifest, root, fingerprint = load_plan(path)
    # Require every planned training run to finish before any test scoring.
    for job in jobs:
        cp = load_checkpoint(Path(job['cfg']['output_dir']) / job['kind'] / 'last.pt')
        if cp['step'] != job['cfg']['train']['max_steps'] or cp['signature'] != training_signature(job['cfg'], manifest, job['kind']):
            raise ValueError(f"Incomplete or mismatched job: {job['id']}")
    cfg = jobs[0]['cfg']
    torch.set_num_threads(cfg['train']['threads'])
    device = device_for(cfg['train']['device'])
    probabilities, baseline_params = prepare_baselines(path)
    if file_hash(root / 'baseline_probabilities.npz') != baseline_params['probabilities_sha256']:
        raise ValueError('Baseline probability checksum mismatch')
    (root / 'evaluation').mkdir(parents=True, exist_ok=True)
    metrics, session_rows, training_rows = [], [], []
    datasets = {s: windows(cfg, manifest, s) for s in ('val', 'test', 'heldout')}
    references = {s: evaluation_pairs(data, cfg['tokenizer']['bins'], cfg['evaluation']['max_windows'])
                  for s, data in datasets.items()}
    for name, probability in probabilities.items():
        seed = None
        variant = name
        if name.startswith('markov_matched_seed'):
            seed = int(name.split('seed')[-1])
            variant = 'markov_matched'
        for split, data in datasets.items():
            ids, previous, targets = references[split]
            losses = -np.log(probability[targets] if probability.ndim == 1 else probability[previous, targets])
            metric, sessions = record_scores(root, variant, seed, split, data, ids, losses)
            metrics.append(metric)
            session_rows.extend(sessions)
    for job in jobs:
        c = job['cfg']
        seed = c['train']['seed']
        folder = Path(c['output_dir']) / job['kind']
        cp_path = folder / 'best.pt'
        cp = load_checkpoint(cp_path)
        if cp['signature'] != training_signature(c, manifest, job['kind']):
            raise ValueError('Best checkpoint mismatch')
        model = EventTransformer(c, job['kind']).to(device)
        model.load_state_dict(cp['model'])
        summary = read_json(folder / 'training_summary.json')
        training_rows.append({'variant': job['variant'], 'seed': seed, 'best_step': cp['step'],
                              'train_seconds': summary['train_seconds'], 'parameters': summary['parameters']['total'],
                              'target_exposures': summary['target_events_seen'], 'checkpoint_sha256': file_hash(cp_path),
                              'source_output_dir': c['output_dir']})
        for split, data in datasets.items():
            ids, losses = evaluate_model(model, data, c, device)
            if not np.array_equal(ids, references[split][0]):
                raise ValueError('Evaluation windows differ between variants')
            metric, sessions = record_scores(root, job['variant'], seed, split, data, ids, losses,
                                             {'selected_checkpoint_step': cp['step']})
            metrics.append(metric)
            session_rows.extend(sessions)
        print(f"Evaluated {job['id']}", flush=True)
        del model, cp
    pd.DataFrame(metrics).to_csv(root / 'metrics.csv', index=False)
    pd.DataFrame(session_rows).to_csv(root / 'session_metrics.csv', index=False)
    pd.DataFrame(training_rows).to_csv(root / 'training_runs.csv', index=False)
    write_json(root / 'evaluation_metadata.json', {
        'fingerprint': fingerprint, 'data_fingerprint': manifest['fingerprint'], 'plan': plan,
        'configs': [j['cfg'] for j in jobs], 'seeds': plan['seeds'],
        'code_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in ['engine.py', 'model.py', 'baselines.py', 'study.py', 'study_report.py']},
        'provenance': manifest['provenance'], 'test_policy': 'Plan fixed before new training; prior seed42 test results were already seen; no test-driven selection in this study.',
        'runtime_note': ('New jobs trained serially; durations still depend on machine load and are not matched compute.'
                         if (root / 'progress.json').exists() and read_json(root / 'progress.json').get('workers') == 1
                         else 'Training concurrency is not controlled here; durations are not isolated hardware benchmarks.')})
    return make_report(root)


def make_report(root):
    from .report import plt
    root = Path(root)
    metrics = pd.read_csv(root / 'metrics.csv')
    meta = read_json(root / 'evaluation_metadata.json')
    grouped = metrics.groupby(['variant', 'split']).nll_per_event.agg(['mean', 'std', 'count']).reset_index()
    grouped.to_csv(root / 'summary.csv', index=False)
    paired = []
    for split in ('val', 'test', 'heldout'):
        frame = metrics[(metrics.split == split) & metrics.seed.notna()].pivot(index='seed', columns='variant', values='nll_per_event')
        for left, right in [('seq_tpv', 'joint'), ('seq_ptv', 'seq_tpv'), ('seq_tpv', 'markov_matched')]:
            if left not in frame or right not in frame:
                continue
            difference = frame[left] - frame[right]
            paired.append({'split': split, 'comparison': f'{left} minus {right}', 'mean_difference': difference.mean(),
                           'seed_std': difference.std(ddof=1), 'seeds': len(difference), 'negative_in_seeds': int((difference < 0).sum())})
    pd.DataFrame(paired).to_csv(root / 'paired_seed_differences.csv', index=False)
    order = ['frequency_full', 'markov_full', 'markov_matched', 'joint', 'seq_tpv', 'seq_ptv']
    order = [name for name in order if name in set(grouped.variant)]
    overview = []
    for variant in order:
        row = {'方法': variant}
        for split, label in [('val', '验证'), ('test', '未来日期'), ('heldout', 'NVDA')]:
            x = grouped[(grouped.variant == variant) & (grouped.split == split)].iloc[0]
            single = '（固定基线）' if variant in ('frequency_full', 'markov_full') else '（单次运行）'
            row[label] = f"{x['mean']:.4f}" + (f" ± {x['std']:.4f}" if x['count'] > 1 else single)
        overview.append(row)
    table = pd.DataFrame(overview)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, split in zip(axes, ('test', 'heldout')):
        subset = grouped[grouped.split == split].set_index('variant').loc[order]
        ax.barh(order, subset['mean'], xerr=subset['std'].fillna(0), capsize=3,
                color=['#a0a9ad', '#7c8790', '#687f91', '#16838e', '#bd673f', '#8364a7'])
        ax.invert_yaxis()
        ax.set(title=split, xlabel='NLL / complete event (nats; lower is better)')
    fig.suptitle('Mean ± seed standard deviation; not a market-generalization confidence interval')
    fig.savefig(root / 'seed_comparison.png', dpi=160)
    plt.close(fig)
    settings = meta['configs'][0]['train']
    targets_per_window = meta['configs'][0]['model']['context_events'] if settings.get('supervision') == 'dense' else 1
    seed_count = len(meta['seeds'])
    notes = [
        'joint：联合 token；seq_tpv：间隔→价格→成交量；seq_ptv：价格→间隔→成交量。',
        f"每种神经网络设置使用相同 {seed_count} 个 seed、每轮 {settings['max_steps']:,} 步和 {settings['max_steps'] * settings['batch_size'] * targets_per_window:,} 次目标事件曝光；同 seed 的采样目标流相同。每窗口监督 {targets_per_window} 个完整事件，评价只取末事件。",
        '控制的是历史事件数、目标曝光和 Transformer 主体结构，并非等 FLOPs 或等墙钟时间。逐字段序列更长；两种编码的词嵌入、输出头和位置嵌入参数量也不同。',
        'frequency_full 和 markov_full 使用全部合格训练目标；markov_matched 使用与同 seed 神经网络完全相同的目标曝光及前一事件。',
        'Markov 仅在同股票同日内统计相邻事件转移；未见状态回退到训练频率。平滑强度从预设网格中仅按验证集选择。',
        '模型检查点仅由验证损失选择。各方法评价相同窗口；测试和 NVDA 不参与梯度训练、分箱或平滑参数选择。',
        '显示均值 ± 随机种子的样本标准差（单次运行不计算标准差）。它只衡量初始化/采样随机性，不是跨日期/市场泛化的置信区间；窗口重叠，测试日期很少。',
        '上一轮 seed42 测试结果已查看，因此这些日期不再是整个研究过程中从未见过的最终保留集。后续确认性研究需新日期。',
        meta['runtime_note'],
        '目标是成交似然，未验证交易收益、报价/完整订单簿、交易成本或实盘能力。',
    ]
    title = '密集监督：多随机种子与 Markov 基线' if settings.get('supervision') == 'dense' else '多随机种子、字段顺序与 Markov 基线'
    banner = '真实 TAQ：初步稳健性实验' if meta['provenance'] == 'real_taq' else 'SYNTHETIC DATA — PIPELINE TEST ONLY'
    content = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1150px;margin:35px auto;padding:0 24px;background:#f5f7f8;color:#24353d}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}.scroll{{overflow:auto}}.banner{{padding:14px;background:#dceeed}}li{{margin:8px 0}}</style>
<h1>{title}</h1><p class="banner">{banner}</p><section><h2>事件 NLL，越低越好</h2><div class="scroll">{table.to_html(index=False,border=0)}</div></section>
<section><img src="seed_comparison.png" alt="Seed means and standard deviations"></section><section><h2>同 seed 配对差值</h2><p>负数表示前一种设置的损失更低。</p><div class="scroll">{pd.DataFrame(paired).to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)}</div></section>
<section><h2>方法与解释范围</h2><ul>{''.join('<li>'+html.escape(n)+'</li>' for n in notes)}</ul><p>明细：metrics.csv、session_metrics.csv、training_runs.csv、paired_seed_differences.csv、baseline_parameters.json 和 evaluation/。</p></section></html>'''
    (root / 'report.html').write_text(content)
    markdown = [f'# {title}', '', banner, '', '| 方法 | 验证 | 未来日期 | NVDA |', '|---|---:|---:|---:|']
    markdown += ['| ' + ' | '.join(row.values()) + ' |' for row in overview]
    markdown += ['', '![种子比较](seed_comparison.png)', ''] + ['- '+n for n in notes] + ['']
    (root / 'report.md').write_text('\n'.join(markdown))
    print(f"Study report: {root / 'report.html'}", flush=True)
    return root / 'report.html'
