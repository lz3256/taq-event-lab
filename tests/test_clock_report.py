import pandas as pd
import pytest

from taq_lab.common import file_hash, write_json
from taq_lab.clock_report import make_report


def test_complete_grid_report_and_missing_job_rejection(tmp_path):
    root=tmp_path/'runs'/'clock_probes'
    root.mkdir(parents=True)
    rows=[]
    for task in ('T1','T2','T3'):
        for budget in ('256','1024','4096','all'):
            for seed in (42,43,44):
                for method in ('frequency','stats','random','pretrained'):
                    for split in ('val','test','heldout'):
                        rows.append({'task':task,'budget':budget,'seed':seed,'method':method,'split':split,
                                     'nll':1.0 if method=='pretrained' else 1.1,'balanced_accuracy':.4,
                                     'macro_f1':.4,'n':5000 if budget=='all' else int(budget)})
    pd.DataFrame(rows).to_csv(root/'probe_metrics.csv',index=False)
    pd.DataFrame([{'converged':True,'gradient_max':1e-8}]).to_csv(root/'probe_search.csv',index=False)
    fields=[]
    for variant in ('markov_full','joint','sequential'):
        for split in ('val','test','heldout'):
            for field in ('interval','price','size'):
                fields.append({'variant':variant,'split':split,'field':field,'nll':1.5 if variant=='markov_full' else 1.4})
    pd.DataFrame(fields).to_csv(root/'field_nll.csv',index=False)
    pd.DataFrame([{'task':t,'split':s,'n':30,'class_0':10,'class_1':10,'class_2':10}
                  for t in ('T1','T2','T3') for s in ('train','val','test','heldout')]).to_csv(root/'class_distribution.csv',index=False)
    pd.DataFrame([{'split':s,'context_seconds':10} for s in ('train','val','test','heldout')]).to_csv(root/'examples.csv',index=False)
    write_json(root/'task_metadata.json',{'signature':'example','sources':[{}, {}, {}],
               'plan':{'budgets':[256,1024,4096,'all']},'split_sizes':{'train':5000,'val':30,'test':30,'heldout':30},
               'thresholds':[[.5,1],[.5,1],[-.3,.3]],'session_audit':[{'excluded':{}}]})
    write_json(root/'probe_metadata.json',{'task_signature':'example','artifacts':{'probe_metrics.csv':file_hash(root/'probe_metrics.csv')}})
    write_json(root/'field_metadata.json',{'artifacts':{'field_nll.csv':file_hash(root/'field_nll.csv')}})
    page=make_report(root)
    assert '尚未检验端到端微调收益' in page.read_text()
    assert (tmp_path/'CLOCK_PROBE_RESULTS.md').exists()
    summary=pd.read_csv(root/'probe_paired_summary.csv')
    assert (summary.wins==3).all()
    assert (root/'probe_learning_curves.png').stat().st_size>1000
    pd.DataFrame(rows[:-1]).to_csv(root/'probe_metrics.csv',index=False)
    write_json(root/'probe_metadata.json',{'task_signature':'example','artifacts':{'probe_metrics.csv':file_hash(root/'probe_metrics.csv')}})
    with pytest.raises(ValueError,match='Incomplete'):
        make_report(root)
