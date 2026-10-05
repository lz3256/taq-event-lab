from types import SimpleNamespace

import numpy as np
import pytest

from taq_lab.baselines import transition_counts, shrinkage_probabilities, select_strength


@pytest.mark.parametrize('stride, expected', [(1, {(0, 1): 2, (1, 0): 2, (1, 1): 1}),
                                             (2, {(0, 1): 2, (1, 1): 1})])
def test_markov_eligible_targets_and_session_boundaries(stride, expected):
    arrays = []
    for values in ([0, 1, 0, 1], [1, 1, 0]):
        x = np.zeros((len(values), 3), dtype=np.int16)
        x[:, 2] = values
        arrays.append(x)
    counts = transition_counts(SimpleNamespace(arrays=arrays, context=1, stride=stride), bins=2)
    reference = np.zeros((8, 8), dtype=np.int64)
    for pair, value in expected.items():
        reference[pair] = value
    np.testing.assert_array_equal(counts, reference)


def test_markov_unseen_context_backs_off_and_validation_selects():
    counts = np.array([[90, 10], [0, 0]])
    prior = np.array([.5, .5])
    p = shrinkage_probabilities(counts, prior, 10)
    np.testing.assert_allclose(p.sum(1), 1)
    np.testing.assert_array_equal(p[1], prior)
    np.testing.assert_allclose(p[0], [95 / 110, 15 / 110])
    # Validation contradicts the observed majority; stronger shrinkage should win.
    strength, scores = select_strength(counts, prior, np.array([0, 0]), np.array([1, 1]), [1, 100])
    assert strength == 100
    assert scores[1]['val_nll'] < scores[0]['val_nll']
    with pytest.raises(ValueError):
        shrinkage_probabilities(counts, prior, 0)
