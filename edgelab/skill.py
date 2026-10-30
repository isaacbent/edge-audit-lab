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


def npmle_prob(traders, grid=None, iters=500, thr=0.0, var="own", lik="normal", boot_b=200, seed=0) -> dict:
    """P(true mean > thr) under a population prior estimated non-parametrically (Kiefer-Wolfowitz NPMLE on a
    grid, EM): xbar_i ~ N(mu_i, se_i^2), mu_i ~ G (discrete on the grid). No normal-shape assumption, so a
    population that is mostly zero/negative with a small positive group is represented as such.
    thr > 0 asks for an edge that matters (e.g. above costs) instead of any edge: breakeven traders sit at
    exactly 0 and leak probability above 0, not above thr.
    var: "own" (se_i from the trader's own variance), "pooled" (one within-trader variance for everyone) or
    "moderated" (own variance shrunk toward the population's, more for short records: Smyth 2004; the Student-t
    likelihood then has d0 + n - 1 df).
    lik: "normal", "t" (Student-t, n-1 df, for the uncertainty of se_i) or "boot" (bootstrap-t density per
    trader, for skewed / fat-tailed trade results)."""
    xs = [np.asarray(t, dtype=float) for t in traders]
    n = np.array([x.size for x in xs], dtype=float)
    xbar = np.array([x.mean() for x in xs])
    s2 = np.array([x.var(ddof=1) for x in xs])
    df = n - 1
    if var == "pooled":
        s2 = np.full(s2.size, float(np.average(s2, weights=n - 1)))
    elif var == "moderated":
        d0, s02 = moderation_prior(s2, n - 1)
        s2, df = (d0 * s02 + (n - 1) * s2) / (d0 + n - 1), d0 + n - 1
    elif var != "own":
        raise ValueError(var)
    se = np.maximum(np.sqrt(s2), 1e-9) / np.sqrt(n)
    if grid is None:
        lo, hi = np.quantile(xbar, [0.001, 0.999])
        span = max(hi - lo, 1e-6)
        grid = np.unique(np.concatenate([np.linspace(lo - 0.1 * span, hi + 0.1 * span, 300), [0.0, thr]]))
    z = (xbar[:, None] - grid[None, :]) / se[:, None]
    if lik == "normal":
        lik_m = _st.norm.pdf(z) / se[:, None]
    elif lik == "t":
        lik_m = _st.t.pdf(z, df=np.broadcast_to(df, n.shape)[:, None]) / se[:, None]
    elif lik == "boot":
        lik_m = _boot_t_density(xs, z, boot_b, seed) / se[:, None]
    else:
        raise ValueError(lik)
    lik_m = lik_m + 1e-300
    g = np.full(grid.size, 1.0 / grid.size)
    for _ in range(iters):
        post = lik_m * g
        post /= post.sum(axis=1, keepdims=True)
        g = post.mean(axis=0)
    post = lik_m * g
    post /= post.sum(axis=1, keepdims=True)
    return {"prob": post[:, grid > thr].sum(axis=1), "post_mean": post @ grid, "grid": grid, "prior": g, "post": post}


def moderation_prior(s2, d):
    """Smyth (2004) empirical-Bayes prior for variances: s2_i ~ s0^2 * chi2(d0) / d0 a priori. Moments of log s2_i
    give d0 (trigamma inversion) and s0^2. d0 = inf (full pooling) when log-variances vary no more than sampling."""
    from scipy.special import digamma, polygamma
    z = np.log(np.maximum(np.asarray(s2, dtype=float), 1e-300))
    d = np.asarray(d, dtype=float)
    e = z - digamma(d / 2) + np.log(d / 2)
    excess = np.var(e, ddof=1) - np.mean(polygamma(1, d / 2))
    if excess <= 0:
        return np.inf, float(np.exp(e.mean()))
    lo, hi = 1e-6, 1e6                      # trigamma is decreasing: find x with trigamma(x) = excess
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if polygamma(1, mid) > excess else (lo, mid)
    d0 = 2 * np.sqrt(lo * hi)
    s02 = float(np.exp(e.mean() - digamma(d0 / 2) + np.log(d0 / 2)))
    return float(d0), s02


def _boot_t_density(xs, z, b, seed):
    """Per trader, density of the bootstrap t statistic (xbar* - xbar) / se* (Gaussian KDE, Silverman bandwidth),
    evaluated at the rows of z."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(z)
    for i, x in enumerate(xs):
        r = x[rng.integers(0, x.size, (b, x.size))]
        sd = np.maximum(r.std(axis=1, ddof=1), 1e-12)
        tb = (r.mean(axis=1) - x.mean()) / (sd / np.sqrt(x.size))
        h = max(1.06 * tb.std() * b ** -0.2, 1e-3)
        out[i] = _st.norm.pdf((z[i][:, None] - tb[None, :]) / h).mean(axis=1) / h
    return out


def isotonic_fit(score, label):
    """Pool-adjacent-violators fit of P(label | score), non-decreasing. Returns (x knots, y values)."""
    o = np.argsort(np.asarray(score, dtype=float), kind="mergesort")
    x = np.asarray(score, dtype=float)[o]
    blocks = []                     # [sum y, count, max x]
    for xi, yi in zip(x, np.asarray(label, dtype=float)[o]):
        blocks.append([yi, 1.0, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            y2, c2, x2 = blocks.pop()
            blocks[-1][0] += y2; blocks[-1][1] += c2; blocks[-1][2] = x2
    return np.array([bl[2] for bl in blocks]), np.array([bl[0] / bl[1] for bl in blocks])


def isotonic_apply(knots, score):
    """Step function from isotonic_fit: value of the first block whose max score is >= score."""
    kx, ky = knots
    idx = np.minimum(np.searchsorted(kx, np.asarray(score, dtype=float), side="left"), kx.size - 1)
    return ky[idx]


def npmle_summary(xbar, se, thr=0.0, grid=None, iters=500) -> dict:
    """npmle_prob from per-account summaries (estimate xbar_i, standard error se_i) instead of trade lists: for
    series that are not trade lists (daily or irregular account returns, REAL-01). Same grid and EM as npmle_prob."""
    xbar = np.asarray(xbar, dtype=float)
    se = np.maximum(np.asarray(se, dtype=float), 1e-12)
    if grid is None:
        lo, hi = np.quantile(xbar, [0.001, 0.999])
        span = max(hi - lo, 1e-9)
        grid = np.unique(np.concatenate([np.linspace(lo - 0.1 * span, hi + 0.1 * span, 300), [0.0, thr]]))
    lik = _st.norm.pdf((xbar[:, None] - grid[None, :]) / se[:, None]) / se[:, None] + 1e-300
    g = np.full(grid.size, 1.0 / grid.size)
    for _ in range(iters):
        post = lik * g
        post /= post.sum(axis=1, keepdims=True)
        g = post.mean(axis=0)
    post = lik * g
    post /= post.sum(axis=1, keepdims=True)
    return {"prob": post[:, grid > thr].sum(axis=1), "post_mean": post @ grid, "grid": grid, "prior": g, "post": post}


def predictive_positive(post, grid, se_future) -> np.ndarray:
    """P(the future sample mean > 0) for each account: posterior over the true mean (rows of post on grid) combined
    with the noise of a future estimate whose standard error is se_future."""
    se_f = np.maximum(np.asarray(se_future, dtype=float), 1e-12)
    return (post * _st.norm.cdf(np.asarray(grid)[None, :] / se_f[:, None])).sum(axis=1)
