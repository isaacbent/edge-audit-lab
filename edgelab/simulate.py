"""Synthetic trader bench: traders whose true behaviour is known.

Units: each trade result is in % of account. Base risk = `risk` % per trade, outcome in
R-multiples = mu + noise, noise = Student-t(df) scaled to unit variance (fat tails).

Kinds
  skilled    : mu = +edge for all trades
  breakeven  : mu = 0 (pure luck)
  loser      : mu = -cost (typical retail trader)
  martingale : mu = -cost, size doubles after each loss (cap `mart_cap` x), resets on a win
  tilt       : mu = -cost; after a loss, with prob `tilt_p`, a burst of extra trades at mu = -tilt_edge
  decaying   : mu = +edge in the first half, -cost in the second half
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KINDS = ("skilled", "breakeven", "loser", "martingale", "tilt", "decaying")


@dataclass
class Trader:
    kind: str
    pnl: np.ndarray                 # % of account per trade
    size: np.ndarray                # risk multiple used per trade (1 = base)
    true_mu: np.ndarray             # true expected R per trade
    meta: dict = field(default_factory=dict)


def _noise(rng, n, df):
    x = rng.standard_t(df, n)
    return x / np.sqrt(df / (df - 2))


def make_trader(kind: str, n: int, rng, edge=0.15, cost=0.05, risk=1.0, df=4,
                mart_cap=8, tilt_p=0.5, tilt_burst=3, tilt_edge=0.2) -> Trader:
    if kind not in KINDS:
        raise ValueError(kind)
    eps = _noise(rng, n, df)
    if kind in ("skilled", "breakeven", "loser", "decaying"):
        mu = {"skilled": np.full(n, edge), "breakeven": np.zeros(n), "loser": np.full(n, -cost)}.get(kind)
        if kind == "decaying":
            mu = np.where(np.arange(n) < n // 2, edge, -cost)
        r = mu + eps
        size = np.ones(n)
        return Trader(kind, risk * r, size, mu.astype(float))
    if kind == "martingale":
        size, mu = np.ones(n), np.full(n, -cost)
        r = mu + eps
        s = 1.0
        for i in range(n):
            size[i] = s
            s = min(s * 2, mart_cap) if r[i] < 0 else 1.0
        return Trader(kind, risk * size * r, size, mu)
    # tilt: build the sequence trade by trade until n trades
    pnl, size, mu = [], [], []
    burst = 0
    while len(pnl) < n:
        m = -tilt_edge if burst > 0 else -cost
        sz = 1.5 if burst > 0 else 1.0
        x = m + _noise(rng, 1, df)[0]
        pnl.append(risk * sz * x); size.append(sz); mu.append(m)
        if burst > 0:
            burst -= 1
        elif x < 0 and rng.random() < tilt_p:
            burst = tilt_burst
    return Trader(kind, np.array(pnl), np.array(size), np.array(mu))


def population(mix: dict, n_traders: int, n_trades: int, seed: int, **kw) -> list:
    """mix = {kind: share}; shares are normalised. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    kinds = list(mix)
    w = np.array([mix[k] for k in kinds], dtype=float)
    counts = np.floor(w / w.sum() * n_traders).astype(int)
    counts[0] += n_traders - counts.sum()
    out = []
    for k, c in zip(kinds, counts):
        out += [make_trader(k, n_trades, rng, **kw) for _ in range(c)]
    return out


def challenge_pass(pnl, target=8.0, max_loss=10.0) -> bool:
    """Typical prop-firm rule: cumulative P&L reaches +target % before -max_loss % (static)."""
    eq = np.cumsum(pnl)
    hit_t = np.nonzero(eq >= target)[0]
    hit_l = np.nonzero(eq <= -max_loss)[0]
    if not hit_t.size:
        return False
    return bool(not hit_l.size or hit_t[0] < hit_l[0])
