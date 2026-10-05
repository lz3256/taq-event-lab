"""Compare dense and last-event training while keeping evaluation unchanged."""
import argparse
import copy
import html
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import transition_counts, select_strength, shrinkage_probabilities
from .common import digest, file_hash, load_config, read_json, write_json
from .data import load_prepared
from .engine import fit_baseline, load_checkpoint, training_signature, windows
from .study_report import evaluation_pairs, sampled_transition_counts


def coverage(data, seed, steps, batch_size, checkpoints):
    """Count repeated exposures AND distinct target positions in actual sampled windows."""
    generator = torch.Generator().manual_seed(seed + 1000)
    sparse = [np.zeros(len(a), dtype=bool) for a in data.arrays]
    dense = [np.zeros(len(a), dtype=bool) for a in data.arrays]
    counts = {'sparse': 0, 'dense': 0}
    rows = []
    for step in range(1, steps + 1):
        indices = torch.randint(len(data), (batch_size,), generator=generator).numpy()
        for index in indices:
            sid, start = data.locate(int(index))
            target = start + data.context
            counts['sparse'] += int(not sparse[sid][target])
            sparse[sid][target] = True
            selected = dense[sid][start + 1:target + 1]
            counts['dense'] += int((~selected).sum())
            selected[:] = True
        if step in checkpoints:
            for mode, factor in [('sparse', 1), ('dense', data.context)]:
                rows.append({'supervision': mode, 'step': step, 'sampled_windows': step * batch_size,
                             'target_exposures': step * batch_size * factor, 'unique_targets': counts[mode]})
    return pd.DataFrame(rows)


def run(path):
    path = Path(path).resolve()
    plan = read_json(path)
    cfgs = {'sparse': load_config(path.parent / plan['reference_config']),
            'dense': load_config(path.parent / plan['dense_config'])}
    if cfgs['sparse']['train'].get('supervision', 'last_event') != 'last_event' or cfgs['dense']['train'].get('supervision') != 'dense':
        raise ValueError('Require last-event reference and dense candidate')
    normalized = []
    for cfg in cfgs.values():
        c = copy.deepcopy(cfg)
        c.pop('output_dir')
        c['train'].pop('supervision', None)
        c['train'].pop('extra_eval_steps', None)
        normalized.append(c)
    if normalized[0] != normalized[1]:
        raise ValueError('Pilot changes more than supervision, validation cadence or output')
    root = (path.parent / plan['output_dir']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = cfgs['dense']
    manifest = load_prepared(cfg)
    torch.set_num_threads(cfg['train']['threads'])
    rows, histories, checkpoints = [], [], []
    references = {}
    sources = []
    for mode, config in cfgs.items():
        output = Path(config['output_dir'])
        evaluation = read_json(output / 'evaluation_metadata.json')
        if evaluation['config_fingerprint'] != digest(config) or evaluation['data_fingerprint'] != manifest['fingerprint']:
            raise ValueError('Evaluation metadata mismatch')
        metrics = pd.read_csv(output / 'metrics.csv')
        for kind in ('joint', 'sequential'):
            folder = output / kind
            last = load_checkpoint(folder / 'last.pt')
            best = load_checkpoint(folder / 'best.pt')
            if last['step'] != config['train']['max_steps'] or any(
                    cp['signature'] != training_signature(config, manifest, kind) for cp in (last, best)):
                raise ValueError('Incomplete/mismatched model')
            summary = read_json(folder / 'training_summary.json')
            factor = config['model']['context_events'] if mode == 'dense' else 1
            if summary['target_events_seen'] != config['train']['max_steps'] * config['train']['batch_size'] * factor:
                raise ValueError('Supervised target counter is incorrect')
            variant = f'{mode}_{kind}'
            sources.append({'variant': variant, 'best_step': best['step'],
                            'best_target_exposures': best['step'] * config['train']['batch_size'] * factor,
                            'final_target_exposures': summary['target_events_seen'], 'train_seconds': summary['train_seconds'],
                            'checkpoint_sha256': file_hash(folder / 'best.pt'), 'source_output_dir': str(output)})
            for split in ('val', 'test', 'heldout'):
                score = metrics[(metrics.model == kind) & (metrics.split == split)].iloc[0]
                detail = pd.read_csv(output / 'evaluation' / f'{kind}_{split}_events.csv')
                keys = detail[['window_index', 'session', 'symbol', 'date', 'target_event_index']]
                if split not in references:
                    references[split] = keys
                else:
                    pd.testing.assert_frame_equal(references[split], keys)
                np.testing.assert_allclose(detail.nll_per_event.mean(), score.nll_per_event, rtol=0, atol=1e-6)
                rows.append({'variant': variant, 'split': split, 'nll_per_event': float(score.nll_per_event),
                             'evaluated_events': len(detail), 'selected_checkpoint_step': best['step']})
            for record in last['history']:
                if record['val_nll_per_event'] is not None:
                    histories.append({'variant': variant, 'supervision': mode, **record})
                    histories[-1]['supervision'] = mode
                    checkpoints.append(record['step'])
    training = windows(cfg, manifest, 'train', True)
    covered = coverage(training, cfg['train']['seed'], cfg['train']['max_steps'], cfg['train']['batch_size'], set(checkpoints))
    covered.to_csv(root / 'target_coverage.csv', index=False)
    curves = pd.DataFrame(histories).merge(covered[['supervision', 'step', 'unique_targets']], on=['supervision', 'step'], validate='many_to_one')
    curves.to_csv(root / 'learning_curves.csv', index=False)
    datasets = {s: windows(cfg, manifest, s) for s in ('val', 'test', 'heldout')}
    pairs = {s: evaluation_pairs(d, cfg['tokenizer']['bins'], cfg['evaluation']['max_windows']) for s, d in datasets.items()}
    bins = cfg['tokenizer']['bins']
    prior = fit_baseline(cfg, manifest)
    probability = {'frequency': prior}
    baseline_metadata = {}
    candidates = [('markov_full', transition_counts(training, bins), prior)]
    for mode, supervision in [('sparse', 'last_event'), ('dense', 'dense')]:
        counts = sampled_transition_counts(training, bins, cfg['train']['seed'], cfg['train']['max_steps'], cfg['train']['batch_size'], supervision)
        marginal = counts.sum(0).astype(float) + 1.
        marginal /= marginal.sum()
        candidates.append((f'markov_matched_{mode}', counts, marginal))
    for name, counts, marginal in candidates:
        strength, scores = select_strength(counts, marginal, pairs['val'][1], pairs['val'][2], plan['markov_strengths'])
        probability[name] = shrinkage_probabilities(counts, marginal, strength)
        baseline_metadata[name] = {'target_exposures': int(counts.sum()), 'strength': strength, 'validation_grid': scores}
    for name, p in probability.items():
        for split, (ids, previous, targets) in pairs.items():
            np.testing.assert_array_equal(ids, references[split].window_index)
            nll = -np.log(p[targets] if p.ndim == 1 else p[previous, targets])
            rows.append({'variant': name, 'split': split, 'nll_per_event': float(nll.mean()), 'evaluated_events': len(ids)})
    pd.DataFrame(rows).to_csv(root / 'metrics.csv', index=False)
    pd.DataFrame(sources).to_csv(root / 'training_runs.csv', index=False)
    write_json(root / 'baseline_parameters.json', baseline_metadata)
    write_json(root / 'comparison_metadata.json', {'plan': plan, 'configs': cfgs, 'data_fingerprint': manifest['fingerprint'],
               'seed': cfg['train']['seed'], 'context_events': cfg['model']['context_events'],
               'train_feature_events': sum(len(a) for a in training.arrays), 'eligible_last_targets': len(training),
               'provenance': manifest['provenance'], 'code_sha256': {name: file_hash(Path(__file__).with_name(name))
               for name in ('model.py', 'engine.py', 'study_report.py', 'dense_report.py')},
               'warning': 'Single-seed fixed-step pilot on previously inspected dates; not an independent confirmation or scaling-law fit.'})
    make_report(root)


def make_report(root):
    from .report import plt
    root = Path(root)
    metrics = pd.read_csv(root / 'metrics.csv')
    curves = pd.read_csv(root / 'learning_curves.csv')
    coverage_table = pd.read_csv(root / 'target_coverage.csv')
    metadata = read_json(root / 'comparison_metadata.json')
    order = ['frequency', 'markov_full', 'markov_matched_sparse', 'markov_matched_dense',
             'sparse_joint', 'dense_joint', 'sparse_sequential', 'dense_sequential']
    summary = metrics.pivot(index='variant', columns='split', values='nll_per_event').loc[order, ['val', 'test', 'heldout']]
    summary.to_csv(root / 'summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = {'sparse_joint': '#16838e', 'dense_joint': '#06515b', 'sparse_sequential': '#ce8b64', 'dense_sequential': '#943f1c'}
    for axis, key, label in zip(axes.flat, ['step', 'target_events_seen', 'unique_targets', 'train_seconds'],
                               ['Optimizer updates', 'Supervised event exposures (with repetition)', 'Distinct supervised event positions', 'Training seconds (historical runs; not controlled timing)']):
        for variant, group in curves.groupby('variant'):
            axis.plot(group[key], group.val_nll_per_event, marker='.', label=variant, color=colors[variant])
        axis.axhline(summary.loc['markov_full', 'val'], color='#777777', linestyle='--', label='full Markov')
        axis.set_xscale('log')
        axis.set(xlabel=label, ylabel='Validation LAST-event NLL (nats)')
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Dense supervision vs last-event supervision — single-seed pilot')
    fig.savefig(root / 'dense_comparison.png', dpi=160)
    plt.close(fig)
    final_coverage = coverage_table[coverage_table.step == coverage_table.step.max()]
    notes = [
        '旧训练只监督末事件；新训练对窗口中的每个下一事件计算完整事件 NLL。逐字段损失在字段内做有效类别 softmax，三个字段求和，再对事件和 batch 取平均。',
        '两种编码使用相同随机初始化规则、相同抽样窗口、相同优化器/学习率计划和步数；两种监督方式使用同一数据分箱和模型结构。新增早期验证点帮助观察曲线，不参与梯度更新。',
        f"所有评价继续只计算具有完整 {metadata['context_events']} 个历史事件的末事件 NLL，使用同一组确定性评价窗口。训练中的早期目标历史较短；因此这不是仅改变标签数量的纯消融。",
        '目标曝光次数不是独特样本数，也不是完整 epoch。独特目标位置通过实际随机采样流逐一计数；相同目标可在不同窗口/上下文中被重复监督。',
        '相同优化步数不等于相同计算量。两种编码 token 长度不同，密集损失也会增加部分计算；历史耗时还受当时系统负载影响。图中的曲线不是经过控制的 scaling law 或等计算预算排名。',
        '全量 Markov 使用全部合格末事件训练目标；匹配基线分别使用各监督方式的实际目标曝光及前一事件。其平滑强度只由验证集从固定网格选择。',
        '这是 seed42 的初步诊断，尚未完成密集监督的多种子验证。测试日期此前已查看，不能作为新的最终确认。',
        '本轮未改下游标签或重新跑迁移；此前迁移负结果仅适用于旧监督预算和旧标签，不能推出一般的预训练无效。',
    ]
    heading = '密集因果监督：与末事件监督的初步对照'
    banner = '真实 TAQ / seed42 / 探索性诊断' if metadata['provenance'] == 'real_taq' else 'SYNTHETIC DATA — PIPELINE CHECK'
    doc = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{heading}</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1160px;margin:36px auto;padding:0 24px;background:#f5f7f8;color:#23343d}}section{{background:white;border-radius:12px;padding:22px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}img{{width:100%}}.scroll{{overflow:auto}}li{{margin:8px 0}}</style>
<h1>{heading}</h1><p>{banner}</p><section><h2>末事件 NLL，越低越好</h2><div class="scroll">{summary.to_html(float_format=lambda x:f'{x:.4f}')}</div></section>
<section><h2>训练覆盖</h2>{final_coverage.to_html(index=False)}<p>每种监督模式下，两个编码模型共享同一抽样流和覆盖计数。</p></section>
<section><img src="dense_comparison.png" alt="Validation NLL by updates, exposures, unique targets, and time"></section>
<section><h2>解释范围</h2><ul>{''.join('<li>'+html.escape(n)+'</li>' for n in notes)}</ul></section></html>'''
    (root / 'report.html').write_text(doc)
    lines = [f'# {heading}', '', banner, '', '| 方法 | 验证 | 未来日期 | NVDA |', '|---|---:|---:|---:|']
    lines += [f'| {variant} | {row.val:.4f} | {row.test:.4f} | {row.heldout:.4f} |' for variant, row in summary.iterrows()]
    lines += ['', '![密集监督对照](dense_comparison.png)', ''] + ['- '+n for n in notes]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f"Dense comparison report: {root / 'report.html'}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    run(args.plan)
