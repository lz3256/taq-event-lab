from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, d))


class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class EventTransformer(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        if kind not in ("joint", "sequential"):
            raise ValueError(kind)
        m = cfg["model"]
        self.kind, self.bins = kind, cfg["tokenizer"]["bins"]
        self.order = tuple(cfg["tokenizer"]["field_order"])
        self.context = m["context_events"]
        self.vocab = self.bins ** 3 if kind == "joint" else 3 * self.bins
        self.max_length = self.context if kind == "joint" else 3 * self.context + 2
        self.embedding = nn.Embedding(self.vocab, m["d_model"])
        self.position = nn.Embedding(self.max_length, m["d_model"])
        self.blocks = nn.Sequential(*[Block(m["d_model"], m["heads"], m["dropout"]) for _ in range(m["layers"])])
        self.norm = nn.LayerNorm(m["d_model"])
        self.head = nn.Linear(m["d_model"], self.vocab, bias=False)
        self.apply(self._initialize)
        # Untied output embeddings; report total and backbone parameter counts separately.

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.shape[1] > self.max_length:
            raise ValueError("Input exceeds configured event context")
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embedding(tokens) + self.position(pos)[None]
        return self.head(self.norm(self.blocks(x)))

    def encode(self, events):
        if self.kind == "joint":
            return (events[..., 0] * self.bins + events[..., 1]) * self.bins + events[..., 2]
        order = torch.tensor(self.order, device=events.device)
        return (events[..., list(self.order)] + order * self.bins).flatten(1)

    def event_nll(self, events):
        """Nats per complete LAST event. History is identical across representations.

        Sequential loss conditions each target field on previous target fields only.
        Its three field NLLs are summed, never averaged.
        """
        if events.shape[1] != self.context + 1:
            raise ValueError("Expected context_events + 1 complete events")
        tokens = self.encode(events)
        if self.kind == "joint":
            logits = self(tokens[:, :-1])[:, -1].float()
            return F.cross_entropy(logits, tokens[:, -1], reduction="none")
        logits = self(tokens[:, :-1]).float()
        start = self.context * 3 - 1
        losses = []
        for j, field in enumerate(self.order):
            # Normalize only over valid tokens for this field, during training AND inference.
            valid = logits[:, start + j, field * self.bins:(field + 1) * self.bins]
            losses.append(F.cross_entropy(valid, events[:, -1, field], reduction="none"))
        return torch.stack(losses, dim=1).sum(1)

    def dense_event_nll(self, events):
        """B x C complete-event NLLs for events 1..C, conditioned causally.

        Event zero supplies initial context and is never a target. At target e,
        both representations see e preceding complete events; sequential fields
        additionally see only earlier fields of that target. Sum field losses
        within an event; the caller averages across events and batch.
        """
        if events.shape[1] != self.context + 1:
            raise ValueError("Expected context_events + 1 complete events")
        tokens = self.encode(events)
        logits = self(tokens[:, :-1]).float()
        if self.kind == "joint":
            return F.cross_entropy(logits.transpose(1, 2), tokens[:, 1:], reduction="none")
        losses = []
        for j, field in enumerate(self.order):
            # Prediction positions: 3*e-1+j for target event e=1..C.
            valid = logits[:, 2 + j::3, field * self.bins:(field + 1) * self.bins]
            losses.append(F.cross_entropy(valid.transpose(1, 2), events[:, 1:, field], reduction="none"))
        return torch.stack(losses, dim=0).sum(0)

    def training_nll(self, events, supervision="last_event"):
        if supervision == "last_event":
            return self.event_nll(events)
        if supervision == "dense":
            return self.dense_event_nll(events)
        raise ValueError(f"Unknown supervision: {supervision}")

    @torch.no_grad()
    def next_event(self, history, sample=False):
        """One event; sequential implementation recomputes prefixes (no KV cache)."""
        if history.shape[1] != self.context:
            raise ValueError("Exactly context_events history events required")
        tokens = self.encode(history)

        def choose(logits):
            return torch.multinomial(logits.float().softmax(-1), 1).squeeze(1) if sample else logits.argmax(-1)

        if self.kind == "joint":
            z = choose(self(tokens)[:, -1])
            return torch.stack([z // self.bins ** 2, z // self.bins % self.bins, z % self.bins], -1)
        result = torch.empty((len(history), 3), dtype=torch.long, device=history.device)
        for field in self.order:
            logits = self(tokens)[:, -1, field * self.bins:(field + 1) * self.bins]
            value = choose(logits)
            result[:, field] = value
            tokens = torch.cat([tokens, (value + field * self.bins)[:, None]], dim=1)
        return result

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "backbone": sum(p.numel() for p in self.blocks.parameters()) + sum(p.numel() for p in self.norm.parameters())}
