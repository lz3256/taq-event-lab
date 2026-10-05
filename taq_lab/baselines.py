"""Training-only first-order event transitions with validation-selected shrinkage."""
from __future__ import annotations

import numpy as np

from .tokenization import joint_encode


def transition_counts(data, bins):
    """Count eligible previous-event -> target-event pairs, never across sessions."""
    vocab = bins ** 3
    counts = np.zeros((vocab, vocab), dtype=np.int64)
    for array in data.arrays:
        previous = joint_encode(np.asarray(array[data.context - 1:-1:data.stride]), bins)
        targets = joint_encode(np.asarray(array[data.context::data.stride]), bins)
        if len(previous) != len(targets):
            raise ValueError('Transition slices are misaligned')
        counts += np.bincount(previous * vocab + targets, minlength=vocab ** 2).reshape(vocab, vocab)
    if not counts.sum():
        raise ValueError('No eligible training transitions')
    return counts


def shrinkage_probabilities(counts, prior, strength):
    """Dirichlet shrinkage toward training frequencies; unseen rows equal prior."""
    counts, prior = np.asarray(counts), np.asarray(prior)
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError('strength must be finite and positive')
    if counts.shape != (len(prior), len(prior)) or (counts < 0).any():
        raise ValueError('Invalid transition counts')
    if not np.isfinite(prior).all() or (prior <= 0).any() or not np.isclose(prior.sum(), 1):
        raise ValueError('Prior must be strictly positive and sum to one')
    return (counts + strength * prior[None, :]) / (counts.sum(1, keepdims=True) + strength)


def select_strength(counts, prior, previous, targets, candidates):
    """Caller supplies ONLY validation pairs. Freeze this choice before test scoring."""
    scores = []
    for strength in candidates:
        probabilities = shrinkage_probabilities(counts, prior, float(strength))
        value = float(-np.log(probabilities[previous, targets]).mean())
        scores.append({'strength': float(strength), 'val_nll': value})
    if not scores:
        raise ValueError('Empty strength grid')
    selected = min(scores, key=lambda row: (row['val_nll'], row['strength']))['strength']
    return selected, scores
