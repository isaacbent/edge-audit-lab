"""Real account series for REAL-01 (Hyperliquid portfolio histories, OKX copy-trading daily P&L), written from the
data package README only, before any data file was opened.

Returns are deposit-free: Hyperliquid interval return (pnl_i - pnl_{i-1}) / accountValue_{i-1} between consecutive
history points (irregular spacing kept as interval length dt in days); OKX daily return from the cumulative ratio,
(1 + R_d) / (1 + R_{d-1}) - 1. The drift (mean return per day) of an account is estimated as if r_i ~ N(mu dt_i,
sigma^2 dt_i): mu = sum r / sum dt, sigma^2 = sum (r - mu dt)^2 / dt / (n - 1), se = sigma / sqrt(sum dt).
"""
from __future__ import annotations

import json

import numpy as np

DAY_MS = 86_400_000


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _num(x) -> float:
    return float(x)


def hl_intervals(portfolio, period: str = "allTime", min_value: float = 100.0) -> dict:
    """Interval returns from a Hyperliquid portfolio response (list of [period, data]). Intervals whose starting
    account value is below min_value USDC are dropped (returns on near-empty accounts are not meaningful)."""
    data = dict((p, d) for p, d in portfolio)[period]
    av = sorted(((int(t), _num(v)) for t, v in data["accountValueHistory"]), key=lambda x: x[0])
    pnl = dict((int(t), _num(v)) for t, v in data["pnlHistory"])
    t0, t1, r, dt = [], [], [], []
    for (ta, va), (tb, _) in zip(av[:-1], av[1:]):
        if tb <= ta or va < min_value or ta not in pnl or tb not in pnl:
            continue
        t0.append(ta); t1.append(tb); r.append((pnl[tb] - pnl[ta]) / va); dt.append((tb - ta) / DAY_MS)
    return {"t0": np.array(t0), "t1": np.array(t1), "r": np.array(r), "dt": np.array(dt)}


def okx_daily(pnl_rows) -> dict:
    """Daily returns from an OKX lead-trader P&L list ({beginTs, pnl, pnlRatio cumulative}, newest first). A top-level
    {"data": [...]} wrapper is accepted."""
    rows = pnl_rows["data"] if isinstance(pnl_rows, dict) and "data" in pnl_rows else pnl_rows
    pts = sorted((int(x["beginTs"]), _num(x["pnlRatio"])) for x in rows)
    t0, t1, r, dt = [], [], [], []
    for (ta, ra), (tb, rb) in zip(pts[:-1], pts[1:]):
        if tb <= ta or 1.0 + ra <= 0:
            continue
        t0.append(ta); t1.append(tb); r.append((1.0 + rb) / (1.0 + ra) - 1.0); dt.append((tb - ta) / DAY_MS)
    return {"t0": np.array(t0), "t1": np.array(t1), "r": np.array(r), "dt": np.array(dt)}


def split(series: dict, cut_ms: int):
    """Intervals ending at or before the cut (pre) and starting at or after it (post); intervals spanning the cut
    are dropped."""
    pre = series["t1"] <= cut_ms
    post = series["t0"] >= cut_ms
    pick = lambda m: {k: v[m] for k, v in series.items()}
    return pick(pre), pick(post)


def drift(r, dt) -> dict | None:
    r, dt = np.asarray(r, dtype=float), np.asarray(dt, dtype=float)
    if r.size < 2 or dt.sum() <= 0:
        return None
    mu = r.sum() / dt.sum()
    s2 = float(np.sum((r - mu * dt) ** 2 / dt) / (r.size - 1))
    return {"mu": float(mu), "sigma": float(np.sqrt(s2)), "se": float(np.sqrt(s2 / dt.sum())),
            "n": int(r.size), "days": float(dt.sum())}
