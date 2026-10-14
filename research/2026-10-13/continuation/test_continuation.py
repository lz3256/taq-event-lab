import unittest
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import core
from clock_pool import recover_nanoseconds,older_ends
from warm_start import model_from_probe


class ContinuationChecks(unittest.TestCase):
    def test_exact_nanosecond_reconstruction(self):
        times=np.array([123456789,124000000,125111111,200000000,200000001,200000010],dtype=np.int64)
        f=np.zeros((len(times)-1,3),dtype=np.float64);f[:,0]=np.log1p(np.diff(times)/1e9)
        rows=pd.DataFrame({'last_raw_trade_index':[2,4,5],'last_trade_seconds':times[[2,4,5]]/1e9})
        np.testing.assert_array_equal(recover_nanoseconds(f,rows),times)
        rows.loc[1,'last_trade_seconds']+=1e-8
        with self.assertRaises(ValueError):recover_nanoseconds(f,rows)

    def test_clock_causality_and_future_invariance(self):
        times=np.arange(0,61,dtype=np.int64)*1_000_000_000
        current=np.array([40]);anchor=np.array([40])
        clock=older_ends(times,anchor,current,'clock',context=4,seconds=20)
        event=older_ends(times,anchor,current,'event',context=4,event_offset=4)
        self.assertEqual(clock[0],20);self.assertEqual(event[0],36)
        changed=times.copy();changed[41:]+=10_000_000_000
        np.testing.assert_array_equal(older_ends(changed,anchor,current,'clock',context=4,seconds=20),clock)

    def test_clock_ties_and_insufficient_history(self):
        times=np.array([0,1,2,3,3,4,5])*1_000_000_000
        self.assertEqual(older_ends(times,np.array([5]),np.array([6]),'clock',context=2,seconds=2)[0],4)
        with self.assertRaises(ValueError):older_ends(times,np.array([2]),np.array([2]),'event',context=3,event_offset=1)

    def test_warm_objective_updates_eval_mode_encoder(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        cfg={'model':{'context_events':4,'d_model':8,'heads':2,'layers':1,'dropout':.1},'tokenizer':{'bins':3,'field_order':[0,1,2]}}
        head={'mean':np.zeros(21),'scale':np.ones(21),'weight':np.random.default_rng(3).normal(size=(3,21))*.1,'bias':np.zeros(3)}
        model=model_from_probe({'checkpoint':{'config':cfg}},head,3)
        initial=model.encoder.embedding.weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.)
        model.train();model.encoder.eval()
        x=torch.randint(0,3,(6,4,3));z=torch.randn(6,13);y=torch.tensor([0,1,2,0,1,2])
        objective=F.cross_entropy(model(x,z),y)+.01*model.classifier.weight.square().sum()
        objective.backward();optimizer.step()
        self.assertFalse(model.encoder.training)
        self.assertGreater(float((initial-model.encoder.embedding.weight).detach().abs().max()),1e-8)


if __name__=='__main__':unittest.main()
