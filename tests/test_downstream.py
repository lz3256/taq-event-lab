from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from taq_lab.common import write_json
from taq_lab.data import prepare
from taq_lab.downstream import DirectionClassifier, metrics, run, train_one
from taq_lab.downstream_data import engineered_features, future_labels, prepare_task
from taq_lab.engine import load_checkpoint, train


def task_plan(cfg, tmp_path):
    prepare(cfg)
    train(cfg, 'sequential')
    p = {'sources': [{'config': 'config.json'}], 'output_dir': 'downstream',
         'horizon_events': 2, 'neutral_band_bps': .1, 'budgets': [8, 16], 'evaluation_examples': 8,
         'train': {'steps': 4, 'batch_size': 4, 'learning_rate': .001, 'weight_decay': .01,
                   'warmup_steps': 1, 'eval_every': 2, 'threads': 1}}
    path = tmp_path / 'downstream.json'
    write_json(path, p)
    return path


def test_future_labels_exclude_anchor_and_inputs_exclude_future():
    f = np.zeros((12, 3))
    f[:, 1] = [0, 0, 0, 50, -2, -1, 0, 3, 0, 0, 1, -1]
    anchors = np.array([3, 6, 9])
    y, r = future_labels(f, anchors, 2, 1.)
    np.testing.assert_allclose(r, [-3., 3., 0.])
    np.testing.assert_array_equal(y, [0, 2, 1])
    past = f[:4][None].copy()
    before = engineered_features(past)
    f[4:, 1] = 1000
    np.testing.assert_array_equal(engineered_features(f[:4][None]), before)
    assert future_labels(f, np.array([3]), 2, 1.)[0][0] == 2


def test_classification_metrics_include_all_classes():
    y = np.array([0, 1, 2])
    p = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
    result = metrics(y, p)
    assert result['accuracy'] == result['balanced_accuracy'] == result['macro_f1'] == 1.
    assert result['nll'] == pytest.approx(-np.log(.8))
    assert result['brier'] == pytest.approx(.06)
    assert result['confusion'] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_transfer_loading_identical_head_and_full_backprop(cfg, tmp_path):
    prepare(cfg)
    train(cfg, 'sequential')
    checkpoint = Path(cfg['output_dir']) / 'sequential/best.pt'
    torch.manual_seed(cfg['train']['seed'])
    scratch = DirectionClassifier(cfg)
    torch.manual_seed(cfg['train']['seed'])
    pretrained = DirectionClassifier(cfg, checkpoint)
    assert torch.equal(scratch.classifier.weight, pretrained.classifier.weight)
    state = load_checkpoint(checkpoint)['model']
    for key, tensor in pretrained.encoder.state_dict().items():
        assert torch.equal(tensor, state[key])
    assert not torch.equal(scratch.encoder.embedding.weight, pretrained.encoder.embedding.weight)
    x = torch.randint(0, 8, (4, cfg['model']['context_events'], 3))
    loss = torch.nn.functional.cross_entropy(pretrained(x), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    assert pretrained.encoder.embedding.weight.grad.abs().sum() > 0
    assert pretrained.encoder.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='history only'):
        pretrained(torch.cat([x, x[:, :1]], dim=1))


def test_downstream_end_to_end_and_dataset_integrity(cfg, tmp_path):
    path = task_plan(cfg, tmp_path)
    run(path)
    plan, sources, root, meta = prepare_task(path)
    frame = pd.read_csv(root / 'examples.csv')
    for _, session in frame.groupby('session'):
        assert (session.input_start.to_numpy()[1:] > session.label_end.to_numpy()[:-1]).all()
        assert (session.label_end - session.anchor == plan['horizon_events']).all()
    summary = pd.read_csv(root / 'metrics.csv')
    assert len(summary) == 24  # 4 methods × 2 budgets × 3 splits × 1 seed
    assert np.isfinite(summary.nll).all()
    for split in ('val', 'test', 'heldout'):
        files = list((root / 'predictions').glob(f'*_{split}.csv'))
        reference = None
        for file in files:
            keys = pd.read_csv(file)[['session', 'anchor', 'label', 'example_index']]
            if reference is None:
                reference = keys
            else:
                pd.testing.assert_frame_equal(keys, reference)
    training = pd.read_csv(root / 'training_runs.csv')
    assert (training.groupby(['seed', 'label_budget']).selected_examples_sha256.nunique() == 1).all()
    assert (root / 'report.html').exists()
    # Dataset is content checked on reuse, including continuous label-source features.
    manifest = __import__('json').loads((Path(cfg['data']['prepared_dir']) / 'manifest.json').read_text())
    feature_path = Path(cfg['data']['prepared_dir']) / manifest['sessions'][0]['features']
    features = np.load(feature_path)
    features[0, 1] += 1
    np.save(feature_path, features)
    with pytest.raises(ValueError, match='dataset changed'):
        prepare_task(path)


def test_finetune_resume_and_paired_batches(cfg, tmp_path, monkeypatch):
    path = task_plan(cfg, tmp_path)
    plan, sources, root, meta = prepare_task(path)
    with np.load(root / 'dataset.npz') as z:
        data = {k: z[k] for k in z.files}
    captured = []
    original = DirectionClassifier.forward
    def capture(self, x):
        if self.training:
            captured.append(x.detach().clone())
        return original(self, x)
    monkeypatch.setattr(DirectionClassifier, 'forward', capture)
    train_one(sources[0], 8, 'scratch', plan, root / 'full', meta, data)
    full_batches = captured.copy()
    captured.clear()
    train_one(sources[0], 8, 'pretrained', plan, root / 'paired', meta, data)
    assert len(captured) == len(full_batches) == plan['train']['steps']
    assert all(torch.equal(a, b) for a, b in zip(captured, full_batches))
    train_one(sources[0], 8, 'scratch', plan, root / 'resumed', meta, data, stop_after=2)
    train_one(sources[0], 8, 'scratch', plan, root / 'resumed', meta, data)
    name = f"scratch_n8_seed{sources[0]['seed']}/last.pt"
    a = load_checkpoint(root / 'full/models' / name)
    b = load_checkpoint(root / 'resumed/models' / name)
    for key in a['model']:
        assert torch.equal(a['model'][key], b['model'][key])
