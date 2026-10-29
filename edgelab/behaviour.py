"""Risk behaviour after losses, from a trader's own trade list (TILT-01).

size_after_loss_p: one-sided Welch t-test that log position size is larger right after a losing trade than right
after a winning one (martingale doubles after losses; tilt adds bigger trades after losses). Uses only what a firm
sees per trade: size and result. Traders with fewer than 3 trades after losses or after wins get p = 1.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


def size_after_loss_p(pnl, size) -> float:
    pnl = np.asarray(pnl, dtype=float)
    ls = np.log(np.maximum(np.asarray(size, dtype=float), 1e-12))
    prev_loss, prev_win = pnl[:-1] < 0, pnl[:-1] > 0
    a, b = ls[1:][prev_loss], ls[1:][prev_win]
    if a.size < 3 or b.size < 3:
        return 1.0
    if np.std(a) == 0 and np.std(b) == 0:
        return 0.0 if a.mean() > b.mean() else 1.0
    return float(_st.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue)


def size_ratio_after_loss(pnl, size) -> float:
    """Geometric mean size after a loss / after a win (descriptive effect size)."""
    pnl = np.asarray(pnl, dtype=float)
    ls = np.log(np.maximum(np.asarray(size, dtype=float), 1e-12))
    a, b = ls[1:][pnl[:-1] < 0], ls[1:][pnl[:-1] > 0]
    return float(np.exp(a.mean() - b.mean())) if a.size and b.size else 1.0
