"""Offline development report; never promote explored dates to a final test."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from .common import file_hash, read_json, write_json

METHODS = {'frequency': '类别频率', 'stats': '历史统计', 'random': '统计＋随机表征', 'pretrained': '统计＋预训练表征'}
TASK_NAMES = {'T1': 'T1 活跃度', 'T2': 'T2 波动率', 'T3': 'T3 方向'}
SPLIT_NAMES = {'train': '训练', 'val': '调参日期', 'test': '同股票未来日期（开发）', 'heldout': 'NVDA（开发）'}


def make_report(root):
    from .report import plt
    root = Path(root)
    meta = read_json(root / 'task_metadata.json')
    for name in ('probe_metadata.json', 'field_metadata.json'):
        info = read_json(root / name)
        for artifact, sha in info['artifacts'].items():
            if file_hash(root / artifact) != sha:
                raise ValueError('Report source checksum mismatch')
    if read_json(root / 'probe_metadata.json')['task_signature'] != meta['signature']:
        raise ValueError('Probe task signature mismatch')
    metrics = pd.read_csv(root / 'probe_metrics.csv', dtype={'budget': str})
    fields = pd.read_csv(root / 'field_nll.csv')
    classes = pd.read_csv(root / 'class_distribution.csv')
    search = pd.read_csv(root / 'probe_search.csv')
    examples = pd.read_csv(root / 'examples.csv')
    expected = len(meta['sources']) * 3 * len(meta['plan']['budgets']) * 4 * 3
    if len(metrics) != expected or metrics.duplicated(['task','method','budget','seed','split']).any():
        raise ValueError('Incomplete or duplicate probe grid')
    if not search.converged.all():
        raise ValueError('Unconverged selected/search fits')
    paired = []
    keys = ['task', 'budget', 'seed', 'split']
    pt = metrics[metrics.method == 'pretrained']
    for baseline in ('random', 'stats'):
        joined = pt.merge(metrics[metrics.method == baseline], on=keys, suffixes=('_pt','_base'), validate='one_to_one')
        for row in joined.to_dict('records'):
            paired.append({**{k: row[k] for k in keys}, 'comparison': f'pretrained_minus_{baseline}',
                           'delta_nll': row['nll_pt'] - row['nll_base'],
                           'delta_balanced_accuracy': row['balanced_accuracy_pt'] - row['balanced_accuracy_base']})
    paired = pd.DataFrame(paired)
    paired.to_csv(root / 'probe_paired_differences.csv', index=False)
    pair_summary = paired.groupby(['task','budget','split','comparison']).agg(
        delta_nll=('delta_nll','mean'), seed_std=('delta_nll','std'),
        wins=('delta_nll',lambda v:int((v<0).sum())), seeds=('seed','count'),
        delta_balanced_accuracy=('delta_balanced_accuracy','mean')).reset_index()
    pair_summary.to_csv(root / 'probe_paired_summary.csv', index=False)
    summary = metrics.groupby(['task','budget','split','method']).agg(
        nll=('nll','mean'), seed_std=('nll','std'), balanced_accuracy=('balanced_accuracy','mean'),
        macro_f1=('macro_f1','mean'), n=('n','first'), seeds=('seed','count')).reset_index()
    summary.to_csv(root / 'probe_summary.csv', index=False)
    fs = fields.groupby(['variant','split','field']).agg(nll=('nll','mean'), seed_std=('nll','std')).reset_index()
    base = fs[fs.variant == 'markov_full'][['split','field','nll']].rename(columns={'nll':'markov_nll'})
    gains = fs[fs.variant != 'markov_full'].merge(base,on=['split','field'],validate='many_to_one')
    gains['gain_nats'] = gains.markov_nll - gains.nll
    gains.to_csv(root / 'field_gains.csv', index=False)
    classes_total = classes.groupby(['task','split'])[['n','class_0','class_1','class_2']].sum().reset_index()
    for k in range(3):
        classes_total[f'fraction_{k}'] = classes_total[f'class_{k}'] / classes_total.n
    classes_total.to_csv(root / 'class_summary.csv', index=False)
    budgets = [str(b) for b in meta['plan']['budgets']]
    fig, axes = plt.subplots(3,2,figsize=(12,12),layout='constrained')
    colors = {'frequency':'#939ba8','stats':'#d78b28','random':'#477ec0','pretrained':'#159675'}
    for i, task in enumerate(TASK_NAMES):
        for j, split in enumerate(('test','heldout')):
            ax = axes[i,j]
            for method in METHODS:
                v = summary[(summary.task==task)&(summary.split==split)&(summary.method==method)].set_index('budget').loc[budgets]
                ax.errorbar(v.n, v.nll, yerr=v.seed_std, marker='o', capsize=3, color=colors[method],label=method)
            ax.set_xscale('log')
            ax.set_xticks([int(meta['split_sizes']['train']) if b=='all' else int(b) for b in budgets],budgets)
            ax.set_title(f'{task} / {"future dates (development)" if split=="test" else "NVDA (development)"}')
            ax.set_xlabel('Fitting labels (shared calibration/validation additional)')
            ax.set_ylabel('NLL (nats / anchor)')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(root/'probe_learning_curves.png',dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax, split in zip(axes,('test','heldout')):
        for j, variant in enumerate(('joint','sequential')):
            v = gains[(gains.variant==variant)&(gains.split==split)].set_index('field').loc[['interval','price','size']]
            ax.bar(np.arange(3)+(j-.5)*.34,v.gain_nats,width=.34,label=variant)
        ax.axhline(0,color='black',lw=.7)
        ax.set_xticks(np.arange(3),['Interval','Price | interval','Size | interval, price'])
        ax.tick_params(axis='x',labelsize=8)
        ax.set_title(f'{split} (development)')
        ax.set_ylabel('Markov NLL minus model NLL (nats/event)')
        ax.legend()
    fig.savefig(root/'field_gains.png',dpi=160)
    plt.close(fig)
    primary = summary[(summary.task=='T1')&(summary.budget=='1024')&(summary.split=='test')].copy()
    primary['method'] = primary.method.map(METHODS)
    primary = primary[['method','nll','seed_std','balanced_accuracy']]
    comparison = pair_summary[(pair_summary.task=='T1')&(pair_summary.budget=='1024')&(pair_summary.split=='test')]
    context = examples.groupby('split').context_seconds.agg(['median','min','max']).reset_index()
    exclusions = {}
    for s in meta['session_audit']:
        for name,n in s['excluded'].items():
            exclusions[name] = exclusions.get(name,0)+n
    def table(frame):
        return frame.to_html(index=False,border=0,float_format=lambda v:f'{v:.5f}',escape=True)
    primary_delta = comparison.set_index('comparison').loc['pretrained_minus_random']
    stats_delta = comparison.set_index('comparison').loc['pretrained_minus_stats']
    conclusion = (f"主检验 T1 / 1,024 标注 / 同股票未来日期开发集：预训练相对随机表征的配对 NLL 差值为 "
                  f"{primary_delta.delta_nll:+.5f} nats/anchor，{int(primary_delta.wins)}/{int(primary_delta.seeds)} 个种子改善。"
                  f"相对仅历史统计的差值为 {stats_delta.delta_nll:+.5f}，{int(stats_delta.wins)}/{int(stats_delta.seeds)} 个种子改善。"
                  "负值表示预训练更好。这是冻结探针结果，尚未检验端到端微调收益。")
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>时钟任务与冻结表征 · 开发研究</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 22px;color:#203143;line-height:1.65;background:#f6f8fb}}h1,h2{{color:#10374d}}section{{background:white;padding:24px;margin:22px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #dee5ec;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}.flag{{border-left:5px solid #d78b28;padding:16px;background:#fff4de}}.scroll{{overflow:auto}}a{{color:#176aac}}</style>
<h1>时钟任务与冻结表征</h1><p>三任务 × 四个标注预算 × 三种子 · 阶段一</p>
<p class="flag">全部既有评价日期都是开发数据。未运行全量微调；没有新日期最终确认；不报告交易收益。</p>
<section><h2>事先指定的主检验</h2><p>{html.escape(conclusion)}</p>{table(primary)}<p>各组分别按调参日期 NLL 选择正则。标准差仅表示三个训练种子的离散程度，不是市场置信区间。</p>{table(comparison)}</section>
<section><h2>任务与数据审计</h2><p>T1：10 秒相对成交速率；T2：30 秒相对网格波动；T3：平滑 VWAP 收益 / 历史波动。40 秒锚点，300 秒历史，最近 128 个成交特征事件。</p>
<p>训练标签阈值：{html.escape(str(dict(zip(TASK_NAMES,meta['thresholds']))))}。阈值仅在训练样本拟合；各预算另共享这些校准标签及验证标签。</p>
<p>样本数：{html.escape(str(meta['split_sizes']))}。排除：{html.escape(str(exclusions))}。三任务共享锚点；终点无成交的排除限制了适用样本。</p>
<div class="scroll">{table(classes_total)}</div><p>128 笔事件覆盖的秒数：</p>{table(context)}<p>所有非频率方法共享 13 项因果历史特征。未来标签窗口不重叠，但历史窗口重叠、股票共同波动，样本不独立。</p></section>
<section><h2>标注效率</h2><img src="probe_learning_curves.png"><p>误差棒是种子标准差。比较的是模型拟合标注预算；预训练成本及共享校准/验证成本不计入横轴。</p>
<div class="scroll">{table(pair_summary[pair_summary.split!='val'])}</div></section>
<section><h2>上游字段 NLL 拆解</h2><img src="field_gains.png"><p>三个模型都采用相同的时间→价格→成交量条件分解。价格项知道目标时间间隔，成交量项知道目标时间间隔和价格。因此这些改善不直接等于未来活跃度、波动或方向预测能力。</p>{table(gains)}</section>
<section><h2>复现与边界</h2><p>{len(search)} 次候选线性拟合全部通过梯度阈值；最大最终梯度 {search.gradient_max.max():.3g}。每组 5 个 L2 候选，仅 val 选优。冻结表征关闭 dropout，使用缓存，拟合可恢复，数据/权重/产物有 SHA256 检查。</p>
<p>用三个 1,000 步密集监督联合模型，不混用单种子 6,376 步模型。T3 使用成交 VWAP，仍含微观结构噪声。结论是表征增量价值，不能替代预训练与从零训练的全量微调比较。</p>
<p><a href="../../CLOCK_PROBE_PROTOCOL.md">事先固定的方案</a> · <a href="probe_summary.csv">全部汇总</a> · <a href="probe_paired_differences.csv">配对种子差值</a> · <a href="probe_session_metrics.csv">按股票日期结果</a> · <a href="class_distribution.csv">完整类别分布</a></p></section></html>'''
    (root/'report.html').write_text(body,encoding='utf-8')
    lines=['# 时钟任务与冻结表征：阶段一开发结果','',conclusion,'',
           '全部旧日期作为开发数据；没有端到端微调、新日期确认或交易收益证据。','',
           f"训练/调参/未来日期/NVDA 锚点数：{meta['split_sizes']}。",'',
           '| 方法 | NLL 均值 | 种子标准差 | 平衡准确率 |','|---|---:|---:|---:|']
    for row in primary.itertuples(index=False):
        lines.append(f'| {row.method} | {row.nll:.5f} | {row.seed_std:.5f} | {row.balanced_accuracy:.5f} |')
    lines += ['', '字段改善（Markov − 模型，nats/事件；三种子均值）：','',
              '| 模型 | 开发集 | 时间 | 价格条件项 | 成交量条件项 |','|---|---|---:|---:|---:|']
    for (variant,split),v in gains[gains.split!='val'].groupby(['variant','split']):
        d=v.set_index('field').gain_nats
        lines.append(f'| {variant} | {split} | {d["interval"]:.5f} | {d["price"]:.5f} | {d["size"]:.5f} |')
    lines += ['', f'{len(search)} 个正则候选拟合均收敛；所有方法使用相同嵌套子集。', '',
              '256/1024 等数字只计算模型拟合标签；全部训练标签另用于任务阈值校准，验证标签另用于选正则。', '',
              '下游标签基于成交数据；没有 NBBO。输入历史重叠和跨股票相关性仍存在，种子标准差不是市场置信区间。', '',
              '下一阶段：预先固定端到端微调对照、实测训练预算，再做新日期和新股票的一次性确认。', '',
              '[完整离线报告](runs/clock_probes/report.html) · [实验方案](CLOCK_PROBE_PROTOCOL.md)']
    (root.parents[1]/'CLOCK_PROBE_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(root/'report_metadata.json',{'primary':primary_delta.to_dict(), 'candidate_fits':len(search),
                 'artifacts':{n:file_hash(root/n) for n in ('report.html','probe_learning_curves.png','field_gains.png','probe_summary.csv','probe_paired_summary.csv')}})
    print(conclusion,flush=True)
    return root/'report.html'
