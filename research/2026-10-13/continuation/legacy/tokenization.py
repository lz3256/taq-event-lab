from __future__ import annotations

import numpy as np

FIELD_NAMES = ["log1p_interarrival_seconds", "log_return_bps", "log1p_size"]


class EventBinner:
    """Shared quantization, fitted only on training events; edge bins absorb tails."""

    def __init__(self, bins, edges=None, centers=None, train_min=None, train_max=None):
        self.bins = int(bins)
        self.edges = None if edges is None else np.asarray(edges, dtype=np.float64)
        self.centers = None if centers is None else np.asarray(centers, dtype=np.float64)
        self.train_min = train_min
        self.train_max = train_max

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Expected finite, nonempty (N,3) training features")
        self.edges = np.quantile(values, np.arange(1, self.bins) / self.bins, axis=0).T
        codes = self.transform(values)
        self.centers = np.zeros((3, self.bins))
        for f in range(3):
            for b in range(self.bins):
                subset = values[codes[:, f] == b, f]
                self.centers[f, b] = np.median(subset) if len(subset) else np.quantile(values[:, f], (b + .5) / self.bins)
        self.train_min, self.train_max = values.min(0).tolist(), values.max(0).tolist()
        return self

    def transform(self, values):
        if self.edges is None:
            raise ValueError("Binner has not been fitted")
        values = np.asarray(values)
        if values.shape[-1] != 3 or not np.isfinite(values).all():
            raise ValueError("Features must have three finite fields")
        return np.stack([np.searchsorted(self.edges[f], values[..., f], side="right") for f in range(3)], -1).astype(np.int16)

    def inverse(self, codes):
        codes = np.asarray(codes)
        return np.stack([self.centers[f, codes[..., f]] for f in range(3)], -1)

    def to_dict(self):
        return {"bins": self.bins, "edges": self.edges.tolist(), "centers": self.centers.tolist(),
                "train_min": self.train_min, "train_max": self.train_max}

    @classmethod
    def from_dict(cls, obj):
        return cls(**obj)


def joint_encode(codes, bins):
    c = np.asarray(codes, dtype=np.int64)
    return (c[..., 0] * bins + c[..., 1]) * bins + c[..., 2]


def joint_decode(tokens, bins):
    t = np.asarray(tokens, dtype=np.int64)
    return np.stack([t // (bins * bins), t // bins % bins, t % bins], -1)


def sequential_encode(codes, bins, order=(0, 1, 2)):
    c = np.asarray(codes, dtype=np.int64)
    return (c[..., list(order)] + np.asarray(order) * bins).reshape(*c.shape[:-2], -1)


def sequential_decode(tokens, bins, order=(0, 1, 2)):
    t = np.asarray(tokens, dtype=np.int64)
    if t.shape[-1] % 3:
        raise ValueError("Incomplete event")
    t = t.reshape(*t.shape[:-1], -1, 3)
    if not np.all(t // bins == np.asarray(order)):
        raise ValueError("Invalid field token range")
    result = np.empty_like(t)
    result[..., list(order)] = t % bins
    return result
