import numpy as np
import pytest
import torch

from taq_lab.clock_data import clock_session, fit_thresholds, apply_thresholds
from taq_lab.nll_fields import joint_field_nll, model_field_nll
from taq_lab.model import EventTransformer


def raw_session():
    t = np.arange(0, 1200, .5)
    p = 100 * np.exp(.001 * np.sin(t / 13) + t * 1e-6)
    size = 1 + np.arange(len(t)) % 20
    codes = np.column_stack([np.arange(len(t)-1) % 8] * 3).astype('uint8')
    settings = {'anchor_seconds': 40, 'max_trade_age_seconds': 5, 'past_rv_floor': 1e-16}
    return t, p, size, codes, settings


def test_clock_alignment_and_manual_labels():
    t, p, size, codes, settings = raw_session()
    x, z, y, rows, audit = clock_session(t, p, size, codes, settings, 128, 1200)
    a, r = rows[0]['anchor_seconds'], rows[0]['last_raw_trade_index']
    assert a == 300 and r == 600
    np.testing.assert_array_equal(x[0], codes[r-128:r])
    assert all(b['anchor_seconds'] >= a['future_end_seconds'] for a,b in zip(rows,rows[1:]))
    assert all(r['future_end_seconds'] <= 1200 for r in rows)
    assert audit['candidate_anchors'] == audit['retained'] + sum(audit['excluded'].values())
    assert y[0,0] == 1
    grid_p = p[np.searchsorted(t, np.arange(a-300,a+31), side='right')-1]
    rets = np.diff(np.log(grid_p))
    past_rv = (rets[:300]**2).sum()
    np.testing.assert_allclose(y[0,1], np.sqrt((rets[300:]**2).sum()/(.1*past_rv)))
    mask = (t > a+25) & (t <= a+35)
    end = np.average(p[mask],weights=size[mask])
    origin = np.average(p[r-9:r+1],weights=size[r-9:r+1])
    np.testing.assert_allclose(y[0,2],np.log(end/origin)/np.sqrt(.1*past_rv), atol=1e-8)


def test_no_future_leak_into_inputs():
    t,p,size,codes,settings=raw_session()
    base=clock_session(t,p,size,codes,settings,128,1200)
    p[t>300] *= 1.1
    size[t>300] *= 5
    changed=clock_session(t,p,size,codes,settings,128,1200)
    np.testing.assert_array_equal(base[0][0],changed[0][0])
    np.testing.assert_array_equal(base[1][0],changed[1][0])
    assert base[2][0,2] != changed[2][0,2]


def test_missing_future_and_zero_volatility_audit():
    t,p,size,codes,settings=raw_session()
    keep=~((t>325)&(t<=335))
    result=clock_session(t[keep],p[keep],size[keep],codes[:keep.sum()-1],settings,128,1200)
    assert result[4]['excluded']['empty_future_vwap'] == 1
    zero=clock_session(t,np.ones(len(t)),size,codes,settings,128,1200)
    assert zero[4]['retained']==0
    assert zero[4]['excluded']['zero_past_volatility']==zero[4]['candidate_anchors']


def test_train_only_thresholds_and_ties():
    train=np.arange(90).reshape(30,3)
    q=fit_thresholds(train)
    assert (np.bincount(apply_thresholds(train,q)[:,0])==[10,10,10]).all()
    np.testing.assert_array_equal(apply_thresholds(np.array([q[:,0]]),q),[[1,1,1]])
    with pytest.raises(ValueError,match='Degenerate'):
        fit_thresholds(np.ones((30,3)))


@pytest.mark.parametrize('kind',['joint','sequential'])
def test_field_sum_matches_existing_loss(cfg,kind):
    torch.set_num_threads(1)
    model=EventTransformer(cfg,kind).eval()
    events=torch.randint(cfg['tokenizer']['bins'],(5,model.context+1,3))
    losses=model_field_nll(model,events)
    torch.testing.assert_close(losses.sum(1),model.event_nll(events))
    if kind=='joint':
        first=model_field_nll(model,events)
        events[:,-1,1:] = (events[:,-1,1:]+1)%model.bins
        torch.testing.assert_close(first[:,0],model_field_nll(model,events)[:,0])


def test_chain_rule_conditioning():
    p=torch.arange(1,9,dtype=torch.float64).reshape(1,8)
    p/=p.sum()
    y=torch.tensor([[1,0,1]])
    loss=joint_field_nll(p.log(),y,2)
    expected=torch.tensor([[-np.log(26/36),-np.log(11/26),-np.log(6/11)]])
    torch.testing.assert_close(loss,expected)


def test_probe_scaler_subset_and_head_convergence():
    from taq_lab.clock_probes import fit_head, head_predict, select_head
    torch.set_num_threads(1)
    rng=np.random.default_rng(4)
    x=rng.normal(size=(150,5))
    y=np.tile(np.arange(3),50)
    x[:,0]+=y*2
    head,diagnostic=fit_head(x[:90],y[:90],.01)
    assert diagnostic['converged']
    np.testing.assert_allclose(head['mean'],x[:90].mean(0))
    p=head_predict(head,x)
    np.testing.assert_allclose(p.sum(1),1)
    assert (p.argmax(1)==y).mean()>.5
    features={'train':x[:120], 'val':x[120:]}
    labels={'train':y[:120], 'val':y[120:]}
    best,strength,grid=select_head(features,labels,np.arange(90),[.001,.01,.1],
                                  {'probe_max_iter':500,'probe_gradient_tolerance':1e-5})
    assert strength==min(grid,key=lambda r:r['val_nll'])['lambda']
    np.testing.assert_allclose(best['mean'],x[:90].mean(0))


def test_frozen_cache_dropout_and_corruption(cfg,tmp_path):
    from pathlib import Path
    from taq_lab.clock_probes import frozen_embeddings
    from taq_lab.common import file_hash
    torch.set_num_threads(1)
    seed=cfg['train']['seed']
    torch.manual_seed(seed)
    model=EventTransformer(cfg,'joint')
    cp=tmp_path/'same_initialization.pt'
    torch.save({'model':model.state_dict()},cp)
    source={'seed':seed,'cfg':cfg,'checkpoint':str(cp),'sha256':file_hash(cp)}
    rng=np.random.default_rng(5)
    data={f'{s}_x':rng.integers(0,cfg['tokenizer']['bins'],(7,cfg['model']['context_events'],3),dtype=np.uint8)
          for s in ('train','val','test','heldout')}
    plan={'batch_size':3}
    meta={'signature':'synthetic-test'}
    random=frozen_embeddings(source,'random',plan,tmp_path,meta,data)
    same=frozen_embeddings(source,'pretrained',plan,tmp_path,meta,data)
    cached=frozen_embeddings(source,'random',plan,tmp_path,meta,data)
    for s in random:
        np.testing.assert_array_equal(random[s],same[s])
        np.testing.assert_array_equal(random[s],cached[s])
    archive=tmp_path/'embeddings'/f'random_seed{seed}.npz'
    with archive.open('ab') as f:
        f.write(b'corrupted')
    with pytest.raises(ValueError,match='cache mismatch'):
        frozen_embeddings(source,'random',plan,tmp_path,meta,data)
