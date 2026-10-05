"""Join the sparse, dense three-seed and single-seed time-budget experiments."""
import argparse
import copy
import html
from pathlib import Path

import numpy as np
import pandas as pd

from .common import file_hash, read_json, write_json


def run(sparse, dense, timed, output):
    roots = {name: Path(path).resolve() for name, path in [('sparse',sparse),('dense',dense),('timed',timed)]}
    metas = {name: read_json(root/'evaluation_metadata.json') for name,root in roots.items()}
    if len({m['data_fingerprint'] for m in metas.values()}) != 1:
        raise ValueError('Experiments use different prepared data')
    if metas['sparse']['seeds'] != metas['dense']['seeds'] or len(metas['dense']['seeds']) < 2:
        raise ValueError('Require matching multi-seed studies')
    reference_config = None
    for stage in ('sparse', 'dense'):
        for job, cfg in zip(metas[stage]['plan']['jobs'], metas[stage]['configs'], strict=True):
            if job['variant'] not in ('joint', 'seq_tpv'):
                continue
            expected = 'dense' if stage == 'dense' else 'last_event'
            if cfg['train'].get('supervision', 'last_event') != expected:
                raise ValueError('Supervision mode mismatch')
            comparable = copy.deepcopy(cfg)
            comparable.pop('output_dir')
            for key in ('seed', 'supervision', 'extra_eval_steps'):
                comparable['train'].pop(key, None)
            if reference_config is None:
                reference_config = comparable
            elif reference_config != comparable:
                raise ValueError('Fixed-step studies change more than supervision or seed')
    for cfg in metas['timed']['configs']:
        if any(cfg[key] != reference_config[key] for key in ('data','split','tokenizer','model','evaluation')):
            raise ValueError('Time study uses a different model/data/evaluation setup')
    scores = {name: pd.read_csv(root/'metrics.csv') for name,root in roots.items()}
    seeds = metas['dense']['seeds']
    if scores['dense'].duplicated(['variant','seed','split']).any():
        raise ValueError('Duplicate study scores')
    reference = {}
    for stage in ('sparse','dense'):
        for variant in ('joint','seq_tpv'):
            for seed in seeds:
                for split in ('val','test','heldout'):
                    row = scores[stage].query('variant == @variant and seed == @seed and split == @split')
                    if len(row) != 1:
                        raise ValueError('Missing or duplicate seed/model/split')
                    frame = pd.read_csv(roots[stage]/'evaluation'/f'{variant}_seed{seed}_{split}.csv')
                    keys = frame[['window_index','session','symbol','date','target_event_index']]
                    if split not in reference:
                        reference[split] = keys
                    pd.testing.assert_frame_equal(reference[split], keys)
                    np.testing.assert_allclose(frame.nll_per_event.mean(), row.nll_per_event.iloc[0], rtol=0, atol=1e-6)
    time_seed = metas['timed']['configs'][0]['train']['seed']
    for variant in ('joint','sequential'):
        for split in ('val','test','heldout'):
            frame = pd.read_csv(roots['timed']/'evaluation'/f'{variant}_seed{time_seed}_{split}.csv')
            pd.testing.assert_frame_equal(reference[split], frame[reference[split].columns])
            row = scores['timed'].query('variant == @variant and split == @split')
            np.testing.assert_allclose(frame.nll_per_event.mean(), row.nll_per_event.iloc[0], rtol=0, atol=1e-6)
    rows = []
    for stage in ('sparse','dense'):
        for variant in ('joint','seq_tpv'):
            for split in ('val','test','heldout'):
                values = scores[stage].query('variant == @variant and split == @split').nll_per_event
                rows.append({'supervision':stage,'variant':variant,'split':split,'mean':values.mean(),
                             'seed_std':values.std(ddof=1),'seeds':len(values)})
    summary = pd.DataFrame(rows)
    paired = []
    for split in ('val','test','heldout'):
        panels = {stage:scores[stage].query('split == @split and variant in ["joint","seq_tpv"]').pivot(index='seed',columns='variant',values='nll_per_event')
                  for stage in ('sparse','dense')}
        for seed in seeds:
            old = panels['sparse'].loc[seed]
            new = panels['dense'].loc[seed]
            paired.append({'seed':seed,'split':split,'sparse_joint_minus_sequential':old['joint']-old['seq_tpv'],
                           'dense_joint_minus_sequential':new['joint']-new['seq_tpv'],
                           'joint_nll_improvement':old['joint']-new['joint'],
                           'sequential_nll_improvement':old['seq_tpv']-new['seq_tpv']})
    paired = pd.DataFrame(paired)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root/'summary.csv',index=False)
    paired.to_csv(root/'paired_differences.csv',index=False)
    overview = []
    for stage in ('sparse','dense'):
        for variant in ('joint','seq_tpv'):
            r = {'Method':f'{stage} / {variant}'}
            for split in ('val','test','heldout'):
                s = summary.query('supervision == @stage and variant == @variant and split == @split').iloc[0]
                r[split] = f'{s["mean"]:.4f} ± {s.seed_std:.4f}'
            overview.append(r)
    table = pd.DataFrame(overview)
    baseline_rows = []
    findings = []
    for split, label in [('test','同股票未来日期'),('heldout','未见股票 NVDA')]:
        group = paired[paired.split == split]
        old_gap = group.sparse_joint_minus_sequential.mean()
        new_gap = group.dense_joint_minus_sequential.mean()
        findings.append(f'{label}：逐字段相对联合 token 的平均优势从 {old_gap:.4f} 缩小到 {new_gap:.4f} nats/event（缩小 {(1-new_gap/old_gap)*100:.1f}%）；密集模式中 {(group.dense_joint_minus_sequential>0).sum()}/{len(seeds)} 个种子逐字段领先。')
        baselines = [scores[s].query('variant == "markov_full" and split == @split').nll_per_event.iloc[0] for s in roots]
        np.testing.assert_allclose(baselines, baselines[0], rtol=0, atol=1e-10)
        for variant in ('joint','seq_tpv'):
            s = summary.query('supervision == "dense" and variant == @variant and split == @split').iloc[0]
            baseline_rows.append({'split':split,'variant':variant,'full_markov_nll':baselines[0],
                                  'dense_mean_nll':s['mean'],'improvement':baselines[0]-s['mean']})
    improvements = pd.DataFrame(baseline_rows)
    improvements.to_csv(root/'baseline_improvements.csv',index=False)
    timing = pd.read_csv(roots['timed']/'training_runs.csv')
    timed_table = scores['timed'].pivot(index='variant',columns='split',values='nll_per_event')[['val','test','heldout']]
    settings = metas['dense']['configs'][0]
    steps = settings['train']['max_steps']
    exposures = steps * settings['train']['batch_size'] * settings['model']['context_events']
    time_budget = metas['timed']['configs'][0]['train']['time_budget_seconds']
    notes = [
        f'固定步数比较使用 {len(seeds)} 个配对种子，每模型 {steps:,} 步与 {exposures:,} 次密集目标曝光；误差条为种子标准差，不是市场泛化置信区间。',
        f'等时间实验另行从零训练，只含 seed{time_seed}；两种编码各获得 {time_budget:g} 秒训练时间，并按共同的时间比例验证和调节学习率。它比较当前 CPU 实现，不能推广为等 FLOPs 或 GPU 性能。',
        '评价均为同一组窗口的最后一个完整事件 NLL。密集监督相对旧模式同时改变目标数量、梯度批量和训练历史长度分布。',
        '三阶段完整事件位置逐项核对，完整 Markov 基线分数一致。所有选择使用验证集；测试日期此前已查看，仍需要新日期确认。',
        '数据集总量应与梯度训练使用的训练集规模区分。未重跑时钟时间标签、线性探针或下游微调，不宣称迁移或交易收益。',
    ]
    from .report import plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,split in zip(axes,('test','heldout')):
        selected = summary[summary.split == split]
        labels = [f'{r.supervision} / {r.variant}' for r in selected.itertuples()]
        ax.barh(labels,selected['mean'],xerr=selected.seed_std,capsize=3,color=['#8bb7bd','#dbb095','#16838e','#bd673f'])
        ax.invert_yaxis(); ax.set(title=split,xlabel='Last-event NLL: mean ± seed SD (nats)')
    fig.savefig(root/'supervision_seed_comparison.png',dpi=160)
    plt.close(fig)
    title = '密集监督复核：多种子结果与等训练时间诊断'
    banner = '真实 TAQ / 探索性研究' if metas['dense']['provenance'] == 'real_taq' else 'SYNTHETIC DATA — PIPELINE TEST ONLY'
    doc = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;background:#f5f7f8;color:#24353d}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}img{{width:100%}}</style>
<h1>{title}</h1><p>{banner}</p><section><h2>固定步数：多种子</h2><ul>{''.join('<li>'+html.escape(n)+'</li>' for n in findings)}</ul>{table.to_html(index=False)}<img src="supervision_seed_comparison.png" alt="Multi-seed sparse and dense comparison"></section>
<section><h2>相对全量 Markov 的改善</h2>{improvements.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</section>
<section><h2>等训练时间：单种子</h2>{timed_table.to_html(float_format=lambda x:f'{x:.4f}')}<div class="scroll">{timing.drop(columns=['checkpoint_sha256','last_checkpoint_sha256']).to_html(index=False,float_format=lambda x:f'{x:.3f}')}</div></section>
<section><h2>解释范围</h2><ul>{''.join('<li>'+html.escape(n)+'</li>' for n in notes)}</ul></section></html>'''
    (root/'report.html').write_text(doc)
    write_json(root/'comparison_metadata.json', {'data_fingerprint':metas['dense']['data_fingerprint'], 'seeds':seeds,
               'timed_seed':time_seed,'sources':{k:{'root':str(v),'metrics_sha256':file_hash(v/'metrics.csv'),
               'metadata_sha256':file_hash(v/'evaluation_metadata.json')} for k,v in roots.items()},'findings':findings,'notes':notes})
    lines = [f'# {title}', ''] + findings + ['', '| Method | val | test | heldout |','|---|---:|---:|---:|']
    lines += ['| '+' | '.join(r.values())+' |' for r in overview]
    lines += ['', '## 等训练时间（单种子）', '', '| Method | val | test | heldout |','|---|---:|---:|---:|']
    lines += [f'| {name} | {r.val:.4f} | {r.test:.4f} | {r.heldout:.4f} |' for name,r in timed_table.iterrows()]
    lines += [''] + ['- '+n for n in notes]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(f"Combined report: {root/'report.html'}",flush=True)
    return root/'report.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sparse','dense','timed','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    run(args.sparse,args.dense,args.timed,args.output)


if __name__ == '__main__':
    main()
