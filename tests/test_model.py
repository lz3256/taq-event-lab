import math

import torch
import pytest

from taq_lab.model import EventTransformer


def test_uniform_predictions_have_identical_event_nll(cfg):
    events = torch.randint(8, (3, cfg["model"]["context_events"] + 1, 3))
    for kind in ("joint", "sequential"):
        m = EventTransformer(cfg, kind).eval()
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
        actual = m.event_nll(events)
        torch.testing.assert_close(actual, torch.full_like(actual, 3 * math.log(8)))


def test_causal_mask_prevents_future_token_leakage(cfg):
    m = EventTransformer(cfg, "sequential").eval()
    tokens = torch.randint(m.vocab, (2, m.max_length))
    altered = tokens.clone()
    altered[:, 7:] = (altered[:, 7:] + 1) % m.vocab
    with torch.no_grad():
        torch.testing.assert_close(m(tokens)[:, :7], m(altered)[:, :7], rtol=0, atol=0)


@pytest.mark.parametrize('order', [[0, 1, 2], [1, 0, 2], [1, 2, 0]])
def test_event_loss_matches_actual_autoregressive_conditionals(cfg, order):
    cfg["tokenizer"]["field_order"] = order
    m = EventTransformer(cfg, "sequential").eval()
    events = torch.randint(8, (2, m.context + 1, 3))
    prefix = m.encode(events[:, :-1])
    expected = torch.zeros(2)
    with torch.no_grad():
        for field in m.order:
            logits = m(prefix)[:, -1, field * 8:(field + 1) * 8]
            target = events[:, -1, field]
            expected += torch.nn.functional.cross_entropy(logits, target, reduction="none")
            prefix = torch.cat([prefix, (target + field * 8)[:, None]], 1)
        torch.testing.assert_close(m.event_nll(events), expected)


def test_generation_and_gradients(cfg):
    for kind in ("joint", "sequential"):
        m = EventTransformer(cfg, kind)
        events = torch.randint(8, (2, m.context + 1, 3))
        m.event_nll(events).mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        m.eval()
        pred = m.next_event(events[:, :-1], sample=True)
        assert pred.shape == (2, 3)
        assert ((pred >= 0) & (pred < 8)).all()


def test_can_overfit_tiny_pattern(cfg):
    torch.manual_seed(9)
    torch.set_num_threads(1)
    events = torch.zeros((4, cfg["model"]["context_events"] + 1, 3), dtype=torch.long)
    for kind in ("joint", "sequential"):
        m = EventTransformer(cfg, kind)
        opt = torch.optim.AdamW(m.parameters(), lr=.02)
        initial = m.eval().event_nll(events).mean().item()
        for _ in range(25):
            m.train()
            opt.zero_grad()
            loss = m.event_nll(events).mean()
            loss.backward()
            opt.step()
        final = m.eval().event_nll(events).mean().item()
        assert final < initial * .25, (kind, initial, final)
