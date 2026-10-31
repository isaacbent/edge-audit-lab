"""Trade-level real data for REAL-04 / REAL-05 (Hyperliquid fills, Ostium orders), written from the data owner's
docs (docs/DEV_DATA.md, tools/hl3_analyze.py, tools/ostium_analyze.py on strategy-validator / funded-check) before
any data file was opened.

Hyperliquid trade = flat -> flat per coin (a flip closes one trade and opens the next), as in hl3_analyze.trades;
result = sum(closedPnl - fee); size = entry notional, sum of |sz| * px over the fills that increase |position|.
Ostium trade = one closing order (Close, StopLoss, TakeProfit, Liquidation, CloseDayTrade); result =
profitPercent / 1e8 * collateral / 1e6 (USDC, net); size = collateral / 1e6. Times in ms.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

EPS = 1e-9
OSTIUM_CLOSES = {"close", "stoploss", "takeprofit", "liquidation", "closedaytrade"}
OSTIUM_COL, OSTIUM_PP = 1e6, 1e8


def hl_trades(fills) -> list:
    """[(t_first_ms, t_last_ms, pnl, entry_notional)] in time order of the last fill."""
    pos, cur, out = defaultdict(float), {}, []
    for f in sorted(fills, key=lambda x: (int(x["time"]), str(x.get("tid", "")))):
        c, t = f["coin"], int(f["time"])
        sz = float(f["sz"]) * (1 if f["side"] == "B" else -1)
        px = float(f["px"])
        before = pos[c]
        after = before + sz
        if c not in cur:
            cur[c] = [t, t, 0.0, 0.0]
        tr = cur[c]
        tr[1] = t
        tr[2] += float(f.get("closedPnl") or 0) - float(f.get("fee") or 0)
        flipped = abs(before) > EPS and abs(after) > EPS and (before > 0) != (after > 0)
        if flipped:
            tr[3] += 0.0
        elif abs(after) > abs(before):
            tr[3] += abs(sz) * px
        if abs(after) <= EPS or flipped:
            out.append(tuple(tr))
            del cur[c]
            if flipped:
                cur[c] = [t, t, 0.0, abs(after) * px]
            after = 0.0 if abs(after) <= EPS else after
        pos[c] = after
    return sorted(out, key=lambda x: x[1])


def ostium_trades(orders) -> list:
    """[(t_ms, t_ms, pnl_usdc, collateral_usdc)] for closing orders, time order."""
    out = []
    for o in orders or []:
        if str(o.get("orderAction", "")).lower() not in OSTIUM_CLOSES:
            continue
        t = int(o.get("executedAt") or 0)
        if not t:
            continue
        col = float(o.get("collateral") or 0) / OSTIUM_COL
        pnl = float(o.get("profitPercent") or 0) / OSTIUM_PP * col
        tm = t * 1000 if t < 10 ** 12 else t
        out.append((tm, tm, pnl, col))
    return sorted(out, key=lambda x: x[1])


def r_scale(pnl) -> float | None:
    """Median loss of the given (pre-cut) results, or None with fewer than 5 losses."""
    x = np.asarray(pnl, dtype=float)
    losses = -x[x < 0]
    return float(np.median(losses)) if losses.size >= 5 else None


def tilt_counts(trades, window_ms: int = 3_600_000):
    """For each trade: number of later trades starting within window_ms after it ends; split by its result.
    Returns (counts after losses, counts after wins)."""
    t0 = np.array([x[0] for x in trades], dtype=np.int64)
    order = np.argsort(t0, kind="mergesort")
    t0s = t0[order]
    after_loss, after_win = [], []
    for _, t_end, pnl, _ in trades:
        n = int(np.searchsorted(t0s, t_end + window_ms, side="right") - np.searchsorted(t0s, t_end, side="right"))
        (after_loss if pnl < 0 else after_win if pnl > 0 else []).append(n)
    return np.array(after_loss, dtype=float), np.array(after_win, dtype=float)


def huber_drift(r, dt, k: float = 1.345, iters: int = 50) -> dict | None:
    """Robust drift per day: Huber M-estimate of the rates r_i / dt_i, weighted by dt_i, scale = weighted MAD;
    se from the Huber sandwich, using sum(dt) as effective exposure."""
    r, dt = np.asarray(r, dtype=float), np.asarray(dt, dtype=float)
    if r.size < 3 or dt.sum() <= 0:
        return None
    z = r / dt
    mu = float(np.average(z, weights=dt))
    dev = np.abs(z - np.median(z))
    s = 1.4826 * float(np.median(dev)) or float(np.std(z)) or 1e-12
    for _ in range(iters):
        u = (z - mu) / s
        w = dt * np.minimum(1.0, k / np.maximum(np.abs(u), 1e-12))
        new = float(np.sum(w * z) / np.sum(w))
        if abs(new - mu) < 1e-12 * max(1.0, abs(mu)):
            mu = new
            break
        mu = new
    u = (z - mu) / s
    psi = np.clip(u, -k, k)
    num = np.sum(dt * psi ** 2) / dt.sum()
    den = (np.sum(dt * (np.abs(u) <= k)) / dt.sum()) ** 2
    se = s * np.sqrt(num / max(den, 1e-12)) / np.sqrt(dt.sum()) * np.sqrt(np.mean(dt))
    return {"mu": mu, "se": float(se)}
