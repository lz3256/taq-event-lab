"""Transparent summaries for exploratory downstream transfer experiments."""
import html
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_json


def report(root):
    from .report import plt
    root = Path(root)
    metrics = pd.read_csv(root / 'metrics.csv')
    metadata = read_json(root / 'dataset_metadata.json')
    plan = metadata['plan']
    summary = metrics.groupby(['label_budget', 'mode', 'split']).agg(
        nll_mean=('nll', 'mean'), nll_std=('nll', 'std'),
        accuracy_mean=('accuracy', 'mean'), balanced_accuracy_mean=('balanced_accuracy', 'mean'),
        macro_f1_mean=('macro_f1', 'mean'), seed_count=('seed', 'count')).reset_index()
    summary.to_csv(root / 'summary.csv', index=False)
    pairs = []
    for (budget, split), frame in metrics.groupby(['label_budget', 'split']):
        wide = frame.pivot(index='seed', columns='mode', values='nll')
        for right in ('scratch', 'logistic', 'prior'):
            difference = wide.pretrained - wide[right]
            pairs.append({'label_budget': budget, 'split': split, 'comparison': f'pretrained minus {right}',
                          'mean_nll_difference': difference.mean(), 'seed_std': difference.std(),
                          'improved_seeds': int((difference < 0).sum()), 'seeds': len(difference)})
    paired = pd.DataFrame(pairs)
    paired.to_csv(root / 'paired_differences.csv', index=False)
    examples = pd.read_csv(root / 'examples.csv')
    distribution = examples.groupby(['split', 'label']).size().unstack(fill_value=0)
    distribution.to_csv(root / 'class_distribution.csv')
    stats = examples.groupby('split').horizon_seconds.agg(['min', 'median', 'max'])
    stats.to_csv(root / 'horizon_seconds.csv')
    modes = ['prior', 'logistic', 'scratch', 'pretrained']
    display_rows = []
    for budget in plan['budgets']:
        for mode in modes:
            line = {'标注数': budget, '方法': mode}
            for split, label in [('test', '未来日期 NLL'), ('heldout', 'NVDA NLL')]:
                row = summary[(summary.label_budget == budget) & (summary['mode'] == mode) & (summary.split == split)].iloc[0]
                line[label] = f"{row.nll_mean:.4f}" + (f" ± {row.nll_std:.4f}" if row.seed_count > 1 else '')
            display_rows.append(line)
    table = pd.DataFrame(display_rows)
    classification = summary[summary.split != 'val'][[
        'label_budget', 'mode', 'split', 'accuracy_mean', 'balanced_accuracy_mean', 'macro_f1_mean']]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    budgets = plan['budgets']
    for axis, split in zip(axes, ('test', 'heldout')):
        for offset, mode in enumerate(modes):
            subset = summary[(summary.split == split) & (summary['mode'] == mode)].set_index('label_budget').loc[budgets]
            axis.errorbar(np.arange(len(budgets)) + (offset - 1.5) * .055,
                          subset.nll_mean, yerr=subset.nll_std.fillna(0), marker='o', capsize=3, label=mode)
        axis.set_xticks(np.arange(len(budgets)), [str(x) for x in budgets])
        axis.set(title=split, xlabel='Unique labeled examples', ylabel='Classification NLL (lower is better)')
        axis.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Pretrained vs scratch — mean ± seed SD; exploratory reused dates')
    fig.savefig(root / 'transfer_comparison.png', dpi=160)
    plt.close(fig)
    notes = [
        f"任务：仅用过去 {metadata['context_events']} 个成交特征事件，预测未来 {plan['horizon_events']} 笔成交后的价格方向。累计对数收益低于 −{plan['neutral_band_bps']} bp 为 down，高于 +{plan['neutral_band_bps']} bp 为 up，其余 flat。",
        '预测的是成交价而非中间价，可能包含买卖价跳动效应。事件时长可变，不等于固定秒数预测，也没有可执行收益或成本测算。',
        '每个窗口的输入和未来标签都在同一股票同一天内；相邻候选窗口连同标签区间不重叠，但市场样本仍非独立。日期/股票切分沿用预训练实验。',
        '同 seed 的两个神经网络使用同一结构、相同的新分类头、同一标注子集、同一小批次顺序、同一优化器和步数；所有编码器参数均微调。不同标注预算嵌套抽样，每次都重新初始化，未从小预算继续训练。',
        f"每组 {plan['train']['steps']} 步 × batch {plan['train']['batch_size']}，每个 epoch 内随机无放回取样。标注预算不同但优化步数相同；更大预算不保证模型充分收敛。预训练额外使用了无标注数据与计算量，因此不是相同总训练成本对照。",
        'prior 只用对应标注子集的类别计数并加一平滑；logistic 只用相同子集的 21 个历史统计特征，其标准化也仅在该子集拟合。逻辑回归 L2 系数固定 0.001。',
        '神经网络检查点仅按验证 NLL 选择（含第 0 步）。学习率/训练预算固定，未针对各初始化条件分别调优；结论限定于这套固定微调方案。',
        '预训练模型和字段顺序沿用此前实验选择。此前已经看过这些日期的事件建模测试结果，因此本轮只能作为探索性迁移实验，不能称为独立确认。需要新日期进行最终验证。',
        '均值 ± 标准差只反映三个种子的波动，不是市场泛化置信区间。完整 balanced accuracy、macro F1、Brier、混淆矩阵和每股票/日期指标已另存。',
    ]
    heading = '下游方向预测：预训练与随机初始化'
    banner = '探索性真实 TAQ 实验：复用此前已查看的测试日期' if metadata['provenance'] == 'real_taq' else 'SYNTHETIC — PIPELINE VALIDATION ONLY'
    doc = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{heading}</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1150px;margin:36px auto;padding:0 22px;color:#23333b;background:#f5f7f8}}section{{background:white;padding:22px;border-radius:12px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:nth-child(2),th:nth-child(2){{text-align:left}}img{{width:100%}}.scroll{{overflow:auto}}li{{margin:8px 0}}.banner{{background:#fff0d2;padding:14px}}</style>
<h1>{heading}</h1><p class="banner">{banner}</p><section><h2>分类 NLL，越低越好</h2><div class="scroll">{table.to_html(index=False,border=0)}</div><img src="transfer_comparison.png" alt="Transfer experiment comparison"></section>
<section><h2>类别不平衡下的分类表现</h2><p>以下为跨种子均值。balanced_accuracy 是三类召回率的平均值；始终预测多数类时为 1/3（各类都有样本）。较高的普通 accuracy 并不一定代表能识别涨跌。</p><div class="scroll">{classification.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')}</div></section>
<section><h2>配对差值</h2><p>负数表示预训练模型损失更低。improved_seeds 是改善的种子数。</p><div class="scroll">{paired.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')}</div></section>
<section><h2>方法与边界</h2><ul>{''.join('<li>'+html.escape(n)+'</li>' for n in notes)}</ul></section>
<section><h2>标签分布：0 下跌，1 平稳，2 上涨</h2>{distribution.to_html()}<h2>未来事件实际时长（秒）</h2>{stats.to_html(float_format=lambda x:f'{x:.4f}')}</section></html>'''
    (root / 'report.html').write_text(doc)
    lines = [f'# {heading}', '', banner, '', '| 标注数 | 方法 | 未来日期 NLL | NVDA NLL |', '|---|---|---:|---:|']
    lines += ['| ' + ' | '.join(map(str, r.values())) + ' |' for r in display_rows]
    lines += ['', '分类表现（跨种子均值）：', '', '| 标注数 | 方法 | split | accuracy | balanced accuracy | macro F1 |', '|---|---|---|---:|---:|---:|']
    lines += [f'| {r.label_budget} | {r.mode} | {r.split} | {r.accuracy_mean:.4f} | {r.balanced_accuracy_mean:.4f} | {r.macro_f1_mean:.4f} |' for r in classification.itertuples()]
    lines += ['', '![迁移对照](transfer_comparison.png)', ''] + ['- ' + n for n in notes]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f"Downstream report: {root / 'report.html'}", flush=True)
    return root / 'report.html'
