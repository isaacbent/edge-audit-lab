"""Graded skill scores per trader, and the usual prop-firm metrics they are compared with."""
from __future__ import annotations

import numpy as np
from scipy import stats as _st

from .stats import credibility_mean


def eb_prob(traders) -> dict:
    """Empirical-Bayes probability that a trader's true mean trade result is > 0.

    Prior on true means: Normal(m, tau2) estimated from the population (Buhlmann moments).
    Each trader's likelihood uses its own variance s2_i / n_i (martingale / tilt traders have
    larger variance, so their results are trusted less).
    """
    c = credibility_mean(traders)
    xs = [np.asarray(t, dtype=float) for t in traders]
    n = np.array([x.size for x in xs], dtype=float)
    xbar = np.array([x.mean() for x in xs])
    s2 = np.array([max(x.var(ddof=1), 1e-12) for x in xs])
    prec_data, prec_prior = n / s2, 1.0 / c["tau2"]
    post_var = 1.0 / (prec_data + prec_prior)
    post_mean = post_var * (prec_data * xbar + prec_prior * c["pop_mean"])
    return {"prob": _st.norm.sf(0.0, loc=post_mean, scale=np.sqrt(post_var)),
            "post_mean": post_mean, "post_sd": np.sqrt(post_var), "pop_mean": c["pop_mean"], "tau2": c["tau2"]}


def t_stat(x) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(x.size))) if sd > 0 else 0.0


def total_profit(x) -> float:
    return float(np.sum(x))


def win_rate(x) -> float:
    return float(np.mean(np.asarray(x) > 0))


def profit_factor(x) -> float:
    x = np.asarray(x, dtype=float)
    loss = -x[x < 0].sum()
    gain = x[x > 0].sum()
    if loss == 0:
        return 1e9 if gain > 0 else 1.0
    return float(gain / loss)


def auc(score, label) -> float:
    """Probability a random positive outranks a random negative (ties = 1/2)."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=bool)
    npos, nneg = y.sum(), (~y).sum()
    if not npos or not nneg:
        return float("nan")
    r = _st.rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def precision_at_k(score, label, k: int) -> float:
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=bool)
    if k <= 0 or s.size == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")[: min(k, s.size)]
    return float(y[order].mean())


def precision_at_k_ties(score, label, k: int) -> float:
    """precision_at_k with ties at the cut-off averaged (expected precision under a random tie-break).
    precision_at_k breaks ties by list order, which favours whichever kind the population lists first
    (skilled, in simulate.population*); kept as is so earlier studies reproduce their recorded numbers."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label, dtype=float)
    if k <= 0 or s.size == 0:
        return float("nan")
    k = min(k, s.size)
    cut = np.sort(s)[::-1][k - 1]
    above, tie = s > cut, s == cut
    return float((y[above].sum() + (k - above.sum()) * y[tie].mean()) / k)


def npmle_prob(traders, grid=None, iters=500, thr=0.0) -> dict:
    """P(true mean > thr) under a population prior estimated non-parametrically (Kiefer-Wolfowitz NPMLE on a
    grid, EM): xbar_i ~ N(mu_i, se_i^2), mu_i ~ G (discrete on the grid). No normal-shape assumption, so a
    population that is mostly zero/negative with a small positive group is represented as such.
    thr > 0 asks for an edge that matters (e.g. above costs) instead of any edge: breakeven traders sit at
    exactly 0 and leak probability above 0, not above thr."""
    xs = [np.asarray(t, dtype=float) for t in traders]
    xbar = np.array([x.mean() for x in xs])
    se = np.array([max(x.std(ddof=1), 1e-9) / np.sqrt(x.size) for x in xs])
    if grid is None:
        lo, hi = np.quantile(xbar, [0.001, 0.999])
        span = max(hi - lo, 1e-6)
        grid = np.unique(np.concatenate([np.linspace(lo - 0.1 * span, hi + 0.1 * span, 300), [0.0, thr]]))
    lik = _st.norm.pdf(xbar[:, None], loc=grid[None, :], scale=se[:, None]) + 1e-300
    g = np.full(grid.size, 1.0 / grid.size)
    for _ in range(iters):
        post = lik * g
        post /= post.sum(axis=1, keepdims=True)
        g = post.mean(axis=0)
    post = lik * g
    post /= post.sum(axis=1, keepdims=True)
    return {"prob": post[:, grid > thr].sum(axis=1), "post_mean": post @ grid, "grid": grid, "prior": g}
