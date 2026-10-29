"""Next-month payouts of a funded book (PAYOUT-01). Rules at end of day (EOD): an account breaches when its equity
(cumulative P&L since funding, %) closes <= -max_loss or a day's P&L <= -daily_loss; at month end every account
still alive is paid split * max(0, that month's P&L).

forecast_payouts simulates next month from this month's day P&L by resampling days:
  - "independent": each account draws its own past days (ignores the common market)
  - "common_days": one past calendar day is drawn for the whole book and every account replays its own P&L of that
    day (keeps cross-account correlation without modelling it)
"""
from __future__ import annotations

import numpy as np


def month_payout(start_eq, alive, day_pnl, max_loss=10.0, daily_loss=5.0, split=0.8):
    """Payout of one month for every account (vectorised over leading axes). start_eq / alive: (..., accounts);
    day_pnl: (..., accounts, days). Returns (payouts summed over accounts, end equity, alive at end)."""
    eq = start_eq[..., None] + np.cumsum(day_pnl, axis=-1)
    breach = ((eq <= -max_loss) | (day_pnl <= -daily_loss)).any(axis=-1)
    alive_end = alive & ~breach
    pay = split * np.maximum(day_pnl.sum(axis=-1), 0.0) * alive_end
    return pay.sum(axis=-1), eq[..., -1], alive_end


def forecast_payouts(month1, method: str, sims: int = 500, seed: int = 0, **rules):
    """Simulated next-month total payouts (sims,) from month-1 day P&L (accounts x days)."""
    rng = np.random.default_rng(seed)
    a, d = month1.shape
    eq1 = np.cumsum(month1, axis=1)
    alive = ~(((eq1 <= -rules.get("max_loss", 10.0)) | (month1 <= -rules.get("daily_loss", 5.0))).any(axis=1))
    start = eq1[:, -1]
    if method == "common_days":
        idx = rng.integers(0, d, (sims, d))
        sim = np.transpose(month1[:, idx], (1, 0, 2))           # (sims, accounts, days)
    elif method == "independent":
        idx = rng.integers(0, d, (sims, a, d))
        sim = np.take_along_axis(np.broadcast_to(month1, (sims, a, d)), idx, axis=2)
    else:
        raise ValueError(method)
    total, _, _ = month_payout(np.broadcast_to(start, (sims, a)), np.broadcast_to(alive, (sims, a)), sim, **rules)
    return total
