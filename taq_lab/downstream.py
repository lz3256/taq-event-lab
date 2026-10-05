"""Paired small-label fine-tuning from pretrained and random backbones."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .common import digest, file_hash, read_json, write_json
from .downstream_data import prepare_task
from .engine import load_checkpoint, save_checkpoint, rng_state, restore_rng
from .model import EventTransformer


class DirectionClassifier(nn.Module):
    def __init__(self, cfg, checkpoint=None):
        super().__init__()
        self.encoder = EventTransformer(cfg, 'sequential')
        self.encoder.head = nn.Identity()
        if checkpoint is not None:
            state = load_checkpoint(checkpoint)['model']
            self.encoder.load_state_dict({k: v for k, v in state.items() if k != 'head.weight'}, strict=True)
        # Seeded independently so both initialization conditions get identical new heads.
        with torch.random.fork_rng():
            torch.manual_seed(cfg['train']['seed'] + 5000)
            self.classifier = nn.Linear(cfg['model']['d_model'], 3)
            nn.init.normal_(self.classifier.weight, std=.02)
            nn.init.zeros_(self.classifier.bias)

    def forward(self, events):
        if events.shape[1] != self.encoder.context:
            raise ValueError('Classifier accepts observed history only')
        tokens = self.encoder.encode(events)
        return self.classifier(self.encoder(tokens)[:, -1])


def metrics(y, probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1., rtol=1e-5, atol=1e-6)
    predicted = p.argmax(1)
    confusion = np.bincount(3 * y + predicted, minlength=9).reshape(3, 3)
    tp, support, guesses = np.diag(confusion), confusion.sum(1), confusion.sum(0)
    recall = np.divide(tp, support, out=np.zeros(3, dtype=float), where=support > 0)
    f1 = np.divide(2 * tp, support + guesses, out=np.zeros(3, dtype=float), where=(support + guesses) > 0)
    return {'nll': float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
            'accuracy': float((predicted == y).mean()), 'balanced_accuracy': float(recall.mean()),
            'macro_f1': float(f1.mean()), 'brier': float(((p - np.eye(3)[y]) ** 2).sum(1).mean()),
            'examples': len(y), 'confusion': confusion.tolist(), 'class_counts': support.tolist()}


@torch.no_grad()
def predict(model, x, batch_size=64):
    model.eval()
    values = []
    for offset in range(0, len(x), batch_size):
        batch = torch.as_tensor(np.asarray(x[offset:offset + batch_size]), dtype=torch.long)
        values.append(model(batch).float().softmax(-1).numpy())
    return np.concatenate(values)


def job_signature(source, budget, mode, plan, meta):
    return digest({'source': source, 'budget': budget, 'mode': mode, 'plan': plan,
                   'dataset': meta['signature'], 'code_sha256': file_hash(Path(__file__))})


def train_one(source, budget, mode, plan, root, meta, data, stop_after=None):
    if mode not in ('scratch', 'pretrained'):
        raise ValueError('Unknown initialization condition')
    seed, settings = source['seed'], plan['train']
    folder = root / 'models' / f'{mode}_n{budget}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    signature = job_signature(source, budget, mode, plan, meta)
    torch.set_num_threads(settings['threads'])
    torch.manual_seed(seed)
    model = DirectionClassifier(source['cfg'], source['checkpoint'] if mode == 'pretrained' else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    selected = data[f'train_selection_seed{seed}'][:budget]
    x, y = data['train_x'][selected], data['train_y'][selected]
    sampler = torch.Generator().manual_seed(seed + 6000)
    torch.manual_seed(seed + 7000)  # matched dropout stream, independent of construction/loading
    permutation, cursor = torch.randperm(budget, generator=sampler), 0
    step, best, history, elapsed = 0, float('inf'), [], 0.
    if (folder / 'last.pt').exists():
        cp = load_checkpoint(folder / 'last.pt')
        if cp['signature'] != signature:
            raise ValueError('Downstream resume configuration/data/code changed')
        if cp['step'] == settings['steps'] and (folder / 'training_summary.json').exists():
            print(f"Reuse {folder.name}", flush=True)
            return
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        sampler.set_state(cp['sampler_rng'])
        restore_rng(cp['rng'])
        step, best, history, elapsed = cp['step'], cp['best_val_nll'], cp['history'], cp['train_seconds']
        permutation, cursor = cp['permutation'], cp['cursor']

    def checkpoint(validation):
        nonlocal best
        improved = validation < best
        best = min(best, validation)
        cp = {'signature': signature, 'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
              'sampler_rng': sampler.get_state(), 'rng': rng_state(), 'permutation': permutation, 'cursor': cursor,
              'best_val_nll': best, 'history': history, 'train_seconds': elapsed}
        if improved:
            save_checkpoint(folder / 'best.pt', cp)
        save_checkpoint(folder / 'last.pt', cp)
        write_json(folder / 'history.json', history)

    if step == 0 and not history:
        score = metrics(data['val_y'], predict(model, data['val_x']))['nll']
        history.append({'step': 0, 'val_nll': score})
        checkpoint(score)
    limit = min(settings['steps'], stop_after) if stop_after is not None else settings['steps']
    while step < limit:
        start = time.perf_counter()
        # Shuffle without replacement within epochs, keeping every full minibatch.
        pieces, needed = [], settings['batch_size']
        while needed:
            take = min(needed, budget - cursor)
            pieces.append(permutation[cursor:cursor + take])
            cursor += take
            needed -= take
            if cursor == budget:
                permutation, cursor = torch.randperm(budget, generator=sampler), 0
        index = torch.cat(pieces).numpy()
        model.train()
        next_step = step + 1
        warmup = settings['warmup_steps']
        factor = next_step / warmup if warmup and next_step <= warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (next_step - warmup) / max(1, settings['steps'] - warmup)))
        for group in optimizer.param_groups:
            group['lr'] = settings['learning_rate'] * factor
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(torch.as_tensor(x[index], dtype=torch.long)), torch.as_tensor(y[index], dtype=torch.long))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite downstream loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        step = next_step
        elapsed += time.perf_counter() - start
        if step % settings['eval_every'] == 0 or step == limit:
            score = metrics(data['val_y'], predict(model, data['val_x']))['nll']
            history.append({'step': step, 'train_nll': float(loss.detach()), 'val_nll': score, 'train_seconds': elapsed})
            checkpoint(score)
            print(f"{folder.name} {step}/{settings['steps']} val_nll={score:.4f}", flush=True)
    if step == settings['steps']:
        write_json(folder / 'training_summary.json', {'seed': seed, 'mode': mode, 'label_budget': budget,
                   'step': step, 'labeled_exposures': step * settings['batch_size'], 'train_seconds': elapsed,
                   'parameters': sum(p.numel() for p in model.parameters()), 'signature': signature,
                   'selected_examples_sha256': digest(selected.tolist()), 'pretraining_sha256': source['sha256'] if mode == 'pretrained' else None})


def fit_logistic(x, y):
    mean, scale = x.mean(0), x.std(0)
    scale = np.maximum(scale, 1e-6)
    features = torch.tensor((x - mean) / scale, dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.long)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = nn.Linear(x.shape[1], 3, dtype=torch.float64)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100, line_search_fn='strong_wolfe', tolerance_grad=1e-8)
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(model(features), target) + .001 * model.weight.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    return model, mean, scale


def evaluate(plan, sources, root, meta, data):
    # Do not inspect task test predictions until every planned training run is complete.
    for source in sources:
        for budget in plan['budgets']:
            for mode in ('scratch', 'pretrained'):
                cp = load_checkpoint(root / 'models' / f"{mode}_n{budget}_seed{source['seed']}" / 'last.pt')
                if cp['step'] != plan['train']['steps'] or cp['signature'] != job_signature(source, budget, mode, plan, meta):
                    raise ValueError('Incomplete or mismatched downstream run')
    examples = pd.read_csv(root / 'examples.csv')
    rows, session_rows, confusion_rows, training_rows = [], [], [], []
    (root / 'predictions').mkdir(exist_ok=True)
    def record(mode, seed, budget, split, p):
        y = data[f'{split}_y']
        score = metrics(y, p)
        details = examples[examples.split == split].copy().reset_index(drop=True)
        np.testing.assert_array_equal(details.label, y)
        details[['p_down', 'p_flat', 'p_up']] = p
        details['predicted_label'] = p.argmax(1)
        details['nll'] = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))
        details.to_csv(root / 'predictions' / f'{mode}_n{budget}_seed{seed}_{split}.csv', index=False)
        key = {'mode': mode, 'seed': seed, 'label_budget': budget, 'split': split}
        confusion_rows.append({**key, **score})
        for name in ('confusion', 'class_counts'):
            score.pop(name)
        rows.append({**key, **score})
        for (session, symbol, date), frame in details.groupby(['session', 'symbol', 'date']):
            session_rows.append({**key, 'session': session, 'symbol': symbol, 'date': date,
                                 'nll': frame.nll.mean(), 'accuracy': (frame.label == frame.predicted_label).mean(), 'examples': len(frame)})

    for source in sources:
        seed = source['seed']
        for budget in plan['budgets']:
            indices = data[f'train_selection_seed{seed}'][:budget]
            y = data['train_y'][indices]
            prior = (np.bincount(y, minlength=3) + 1) / (len(y) + 3)
            logistic, mean, scale = fit_logistic(data['train_numeric'][indices], y)
            save_checkpoint(root / 'models' / f'logistic_n{budget}_seed{seed}.pt',
                            {'model': logistic.state_dict(), 'mean': torch.tensor(mean), 'scale': torch.tensor(scale), 'prior': torch.tensor(prior)})
            for split in ('val', 'test', 'heldout'):
                record('prior', seed, budget, split, np.tile(prior, (len(data[f'{split}_y']), 1)))
                with torch.no_grad():
                    p = logistic(torch.tensor((data[f'{split}_numeric'] - mean) / scale, dtype=torch.float64)).softmax(-1).numpy()
                record('logistic', seed, budget, split, p)
            for mode in ('scratch', 'pretrained'):
                folder = root / 'models' / f'{mode}_n{budget}_seed{seed}'
                best = load_checkpoint(folder / 'best.pt')
                if best['signature'] != job_signature(source, budget, mode, plan, meta):
                    raise ValueError('Best checkpoint mismatch')
                model = DirectionClassifier(source['cfg'])
                model.load_state_dict(best['model'])
                training_rows.append({**read_json(folder / 'training_summary.json'), 'best_step': best['step'],
                                      'checkpoint_sha256': file_hash(folder / 'best.pt')})
                for split in ('val', 'test', 'heldout'):
                    record(mode, seed, budget, split, predict(model, data[f'{split}_x']))
            print(f'Evaluated downstream seed={seed} labels={budget}', flush=True)
    pd.DataFrame(rows).to_csv(root / 'metrics.csv', index=False)
    pd.DataFrame(session_rows).to_csv(root / 'session_metrics.csv', index=False)
    pd.DataFrame(training_rows).to_csv(root / 'training_runs.csv', index=False)
    write_json(root / 'confusions.json', confusion_rows)
    write_json(root / 'evaluation_metadata.json', {'dataset_signature': meta['signature'], 'plan': plan,
               'sources': sources, 'code_sha256': file_hash(Path(__file__)), 'provenance': meta['provenance'],
               'warning': meta['warning'], 'selection': 'Validation NLL only; no task test tuning'})
    from .downstream_report import report
    report(root)


def run(path, action='run'):
    plan, sources, root, meta = prepare_task(path)
    torch.set_num_threads(plan['train']['threads'])
    with np.load(root / 'dataset.npz', allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    if action in ('run', 'train'):
        for source in sources:
            for budget in plan['budgets']:
                for mode in ('scratch', 'pretrained'):
                    train_one(source, budget, mode, plan, root, meta, data)
    if action in ('run', 'evaluate'):
        evaluate(plan, sources, root, meta, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'evaluate', 'run'])
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    run(args.plan, args.action)


if __name__ == '__main__':
    main()
