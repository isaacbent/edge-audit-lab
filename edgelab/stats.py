"""Statistical pieces for Edge Audit.

Copied/adapted from the Research Engine (isaacbent-research-engine), not imported, so this
project stays independent:
  - sign_flip_p        <- tools/power_test._sign_flip_np (same test and semantics)
  - benjamini_hochberg <- tactical_search.bh
  - credibility_mean   new: Buhlmann credibility (actuarial), shrinks each trader to the population
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


def sign_flip_p(net, seed: int = 35, iters: int = 5000) -> float:
    """One-sided p for mean(net) > 0 under random sign flips, (ge+1)/(iters+1),
    refined to 200k draws when the tail is reached. Per-account audit test."""
    a = np.abs(np.asarray(net, dtype=float))
    obs = float(np.sum(net))
    rng = np.random.default_rng(seed)
    ge, done = 0, 0
    while done < iters:
        k = min(20000, iters - done)
        s = (np.where(rng.random((k, a.size)) < 0.5, 1.0, -1.0) * a).sum(axis=1)
        ge += int((s >= obs - 1e-15).sum())
        done += k
    if ge <= 2 and iters < 200000:
        return sign_flip_p(net, seed + 1, 200000)
    return (ge + 1) / (iters + 1)


def t_p(net) -> float:
    """One-sided Student-t p for mean(net) > 0 (fast population screening)."""
    x = np.asarray(net, dtype=float)
    if x.size < 3 or x.std(ddof=1) == 0:
        return 1.0
    t = x.mean() / (x.std(ddof=1) / np.sqrt(x.size))
    return float(_st.t.sf(t, x.size - 1))


def benjamini_hochberg(pvals, q: float = 0.10) -> np.ndarray:
    """Boolean array of discoveries at FDR level q (step-up)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    ok = np.nonzero(p[order] <= q * np.arange(1, m + 1) / m)[0]
    out = np.zeros(m, dtype=bool)
    if ok.size:
        out[order[: ok[-1] + 1]] = True
    return out


def credibility_mean(traders) -> dict:
    """Buhlmann credibility of each trader's mean trade result.

    mu_i = Z_i * xbar_i + (1 - Z_i) * m,  Z_i = n_i / (n_i + s2 / tau2)
    s2 = expected within-trader variance, tau2 = between-trader variance of true means
    (method of moments, floored at a tiny positive value).
    """
    xs = [np.asarray(t, dtype=float) for t in traders]
    n = np.array([x.size for x in xs], dtype=float)
    xbar = np.array([x.mean() for x in xs])
    s2_i = np.array([x.var(ddof=1) for x in xs])
    s2 = float(np.average(s2_i, weights=n - 1))
    m = float(np.average(xbar, weights=n))
    tau2 = float(np.var(xbar, ddof=1) - np.mean(s2 / n))
    tau2 = max(tau2, 1e-12)
    z = n / (n + s2 / tau2)
    return {"mean": z * xbar + (1 - z) * m, "z": z, "pop_mean": m, "s2": s2, "tau2": tau2}
