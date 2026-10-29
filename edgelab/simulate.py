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


# ---------------------------------------------------------------- BENCH-2: heterogeneous trading styles
STYLE = {"win_rate": (0.30, 0.80), "size_sigma": 0.30, "gap_p": 0.02, "gap_mult": 3.0}


def _lognorm1(rng, n, sigma):
    """Lognormal multipliers with mean exactly 1."""
    return np.exp(sigma * rng.standard_normal(n) - sigma ** 2 / 2)


def style_outcomes(rng, n, mu, w, sigma=STYLE["size_sigma"], gap_p=STYLE["gap_p"], gap_mult=STYLE["gap_mult"]):
    """R-multiples for a trader who wins with probability w. Losses are -1R (stop) with lognormal noise and,
    with probability gap_p, a gap loss of gap_mult R. The win size b is set so that E[R] = mu exactly:
    w*b - (1-w)*(1 + gap_p*(gap_mult-1)) = mu. mu may be an array (per trade)."""
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (n,))
    el = 1 + gap_p * (gap_mult - 1)
    b = (mu + (1 - w) * el) / w
    win = rng.random(n) < w
    gap = rng.random(n) < gap_p
    loss = np.where(gap, gap_mult, 1.0) * _lognorm1(rng, n, sigma)
    return np.where(win, b * _lognorm1(rng, n, sigma), -loss)


def make_style_trader(kind: str, n: int, rng, edge=0.15, cost=0.05, risk=1.0,
                      mart_cap=8, tilt_p=0.5, tilt_burst=3, tilt_edge=0.2) -> Trader:
    """Same kinds as make_trader, but each trader has its own win rate (style), drawn independently of skill."""
    if kind not in KINDS:
        raise ValueError(kind)
    w = float(rng.uniform(*STYLE["win_rate"]))
    meta = {"win_rate": w}
    if kind in ("skilled", "breakeven", "loser", "decaying"):
        mu = {"skilled": np.full(n, edge), "breakeven": np.zeros(n), "loser": np.full(n, -cost)}.get(kind)
        if kind == "decaying":
            mu = np.where(np.arange(n) < n // 2, edge, -cost)
        r = style_outcomes(rng, n, mu, w)
        return Trader(kind, risk * r, np.ones(n), mu.astype(float), meta)
    if kind == "martingale":
        mu = np.full(n, -cost)
        r = style_outcomes(rng, n, mu, w)
        size, s = np.ones(n), 1.0
        for i in range(n):
            size[i] = s
            s = min(s * 2, mart_cap) if r[i] < 0 else 1.0
        return Trader(kind, risk * size * r, size, mu, meta)
    pnl, size, mu = [], [], []
    burst = 0
    while len(pnl) < n:
        m = -tilt_edge if burst > 0 else -cost
        sz = 1.5 if burst > 0 else 1.0
        x = float(style_outcomes(rng, 1, m, w)[0])
        pnl.append(risk * sz * x); size.append(sz); mu.append(m)
        if burst > 0:
            burst -= 1
        elif x < 0 and rng.random() < tilt_p:
            burst = tilt_burst
    return Trader(kind, np.array(pnl), np.array(size), np.array(mu), meta)


def population_v2(mix: dict, n_traders: int, n_trades: int, seed: int, **kw) -> list:
    rng = np.random.default_rng(seed)
    kinds = list(mix)
    w = np.array([mix[k] for k in kinds], dtype=float)
    counts = np.floor(w / w.sum() * n_traders).astype(int)
    counts[0] += n_traders - counts.sum()
    out = []
    for k, c in zip(kinds, counts):
        out += [make_style_trader(k, n_trades, rng, **kw) for _ in range(c)]
    return out


# ---------------------------------------------------------------- BENCH-3: continuous edges, variable trade counts
BENCH3 = {"edge_loc": -0.05, "edge_scale": 0.07, "edge_df": 5, "behaviour": {"plain": 0.80, "martingale": 0.10, "tilt": 0.10}}


def make_bench3_trader(rng, n, mu, behaviour, mart_cap=8, tilt_p=0.5, tilt_burst=3, tilt_edge=0.2, shock=None) -> Trader:
    """One trader with true edge `mu` (R per base-risk trade), own win-rate style (as BENCH-2) and a behaviour:
    plain, martingale (size doubles after a loss, cap mart_cap) or tilt (after a loss, with prob tilt_p, a burst of
    tilt_burst trades at -tilt_edge and 1.5x size). `shock` (length n, zero mean) is a common market move added to
    each trade's R. meta["edge"] = true expected P&L per trade in base-risk units, mean(size * true_mu): the
    quantity a skill score estimates; "skilled" is decided by comparing it with a threshold fixed in the prereg."""
    w = float(rng.uniform(*STYLE["win_rate"]))
    shock = np.zeros(n) if shock is None else np.asarray(shock, dtype=float)
    if behaviour == "tilt":
        pnl, size, tmu = [], [], []
        burst = 0
        while len(pnl) < n:
            m, sz = (-tilt_edge, 1.5) if burst > 0 else (mu, 1.0)
            x = float(style_outcomes(rng, 1, m, w)[0]) + shock[len(pnl)]
            pnl.append(sz * x); size.append(sz); tmu.append(m)
            if burst > 0:
                burst -= 1
            elif x < 0 and rng.random() < tilt_p:
                burst = tilt_burst
        pnl, size, tmu = np.array(pnl), np.array(size), np.array(tmu)
    else:
        r = style_outcomes(rng, n, mu, w) + shock
        size, tmu = np.ones(n), np.full(n, float(mu))
        if behaviour == "martingale":
            s = 1.0
            for i in range(n):
                size[i] = s
                s = min(s * 2, mart_cap) if r[i] < 0 else 1.0
        elif behaviour != "plain":
            raise ValueError(behaviour)
        pnl = size * r
    return Trader(behaviour, pnl, size, tmu, {"win_rate": w, "mu": float(mu), "edge": float(np.mean(size * tmu))})


def population_v3(n_traders: int, n_trades, seed: int, edge_loc=BENCH3["edge_loc"], edge_scale=BENCH3["edge_scale"],
                  edge_df=BENCH3["edge_df"], behaviour=None, corr_sd=0.0, horizon=None, risk=1.0) -> list:
    """BENCH-3. True edges are continuous: mu_i = edge_loc + edge_scale * Student-t(edge_df). n_trades is an int
    (same for everyone) or (lo, hi): each trader draws an integer count uniformly in [lo, hi]. corr_sd > 0 adds a
    shared market factor: trader i trades on consecutive days from a random start in [0, horizon), and every trade
    on day d gets the same N(0, corr_sd) move (off by default). Trader.kind is the behaviour; use meta["edge"]."""
    rng = np.random.default_rng(seed)
    beh = dict(BENCH3["behaviour"] if behaviour is None else behaviour)
    kinds = list(beh)
    w = np.array([beh[k] for k in kinds], dtype=float)
    counts = np.floor(w / w.sum() * n_traders).astype(int)
    counts[0] += n_traders - counts.sum()
    lo, hi = (n_trades, n_trades) if np.isscalar(n_trades) else n_trades
    ns = rng.integers(lo, hi + 1, n_traders)
    mus = edge_loc + edge_scale * rng.standard_t(edge_df, n_traders)
    horizon = int(hi) if horizon is None else int(horizon)
    days = rng.normal(0.0, corr_sd, horizon + int(hi)) if corr_sd > 0 else None
    starts = rng.integers(0, horizon, n_traders)
    out, i = [], 0
    for k, c in zip(kinds, counts):
        for _ in range(c):
            shock = None if days is None else days[starts[i]: starts[i] + ns[i]]
            t = make_bench3_trader(rng, int(ns[i]), mus[i], k, shock=shock)
            t.pnl = risk * t.pnl
            out.append(t)
            i += 1
    return out


# ---------------------------------------------------------------- BENCH-3D: funded traders whose edge may stop
def population_drift(n_traders: int, n_trades: int, seed: int, decay_share=0.5, tau_range=(150, 300), edge_min=0.10,
                     edge_loc=BENCH3["edge_loc"], edge_scale=BENCH3["edge_scale"], edge_df=BENCH3["edge_df"],
                     post_mu=-0.05, risk=1.0) -> list:
    """Monitored funded traders, all with a real edge to start: mu_pre = BENCH-3 edge distribution truncated to
    > edge_min. A decay_share of them switch to post_mu at trade tau (uniform integer in tau_range, 0-based index of
    the first post-change trade); the rest keep mu_pre. Own win-rate style (as BENCH-2/3), plain sizing.
    Trader.kind = "decaying" or "stable"; meta: mu_pre, tau (None for stable), win_rate."""
    rng = np.random.default_rng(seed)
    out = []
    n_decay = int(round(decay_share * n_traders))
    for i in range(n_traders):
        mu = -np.inf
        while mu <= edge_min:
            mu = edge_loc + edge_scale * rng.standard_t(edge_df)
        w = float(rng.uniform(*STYLE["win_rate"]))
        decays = i < n_decay
        tau = int(rng.integers(tau_range[0], tau_range[1] + 1)) if decays else None
        mus = np.full(n_trades, float(mu))
        if decays:
            mus[tau:] = post_mu
        out.append(Trader("decaying" if decays else "stable", risk * style_outcomes(rng, n_trades, mus, w),
                          np.ones(n_trades), mus, {"mu_pre": float(mu), "tau": tau, "win_rate": w}))
    return out


# ---------------------------------------------------------------- BENCH-3T: BENCH-3 with noisy position sizes
def population_tilt(n_traders: int, n_trades, seed: int, size_sigma=0.30, **kw) -> list:
    """BENCH-3 (population_v3, same arguments) where every trader's position size also varies for reasons unrelated
    to their results: size and pnl of each trade are multiplied by an independent mean-1 lognormal(size_sigma)
    factor (real traders change lot sizes). Without it plain traders would have size exactly 1 and any size change
    would give martingale / tilt away. meta['edge'] is unchanged (the factor has mean 1 and is independent)."""
    pop = population_v3(n_traders, n_trades, seed, **kw)
    rng = np.random.default_rng(seed + 2_000_000)
    for t in pop:
        f = _lognorm1(rng, t.pnl.size, size_sigma)
        t.size = t.size * f
        t.pnl = t.pnl * f
    return pop


# ---------------------------------------------------------------- BENCH-3S: funded accounts over calendar days
def account_paths(n_accounts: int, seed: int, horizon=90, rate_median=3.0, rate_sigma=0.5, max_loss=10.0,
                  daily_loss=5.0, size_sigma=0.30, risk=1.0, **kw) -> list:
    """Funded accounts followed day by day for `horizon` days, fully observed (no censoring here). Traders are BENCH-3
    (continuous edge, behaviour plain / martingale / tilt, own win-rate style) with noisy sizes (as BENCH-3T); each
    trades Poisson(rate) times a day, rate ~ lognormal(log rate_median, rate_sigma). An account breaches on the
    first trade where cumulative P&L <= -max_loss % (static) or the day's P&L <= -daily_loss %.
    Returns dicts: trader, day (day index of every trade, 0-based), breach_day (None if alive at horizon), rate."""
    rng = np.random.default_rng(seed)
    base = population_v3(n_accounts, 1, seed, **kw)          # draws edges / behaviours; trades regenerated below
    out = []
    for t in base:
        rate = float(rate_median * np.exp(rate_sigma * rng.standard_normal()))
        per_day = rng.poisson(rate, horizon)
        n = max(int(per_day.sum()), 1)
        tr = make_bench3_trader(rng, n, t.meta["mu"], t.kind)
        f = _lognorm1(rng, n, size_sigma)
        tr.size, tr.pnl = tr.size * f, risk * tr.pnl * f
        day = np.repeat(np.arange(horizon), per_day)[:n] if per_day.sum() else np.zeros(1, dtype=int)
        eq = np.cumsum(tr.pnl)
        day_start = np.concatenate([[0.0], eq[:-1]])
        start_of_day = {}
        for i, d in enumerate(day):
            start_of_day.setdefault(int(d), day_start[i])
        intraday = eq - np.array([start_of_day[int(d)] for d in day])
        bad = np.nonzero((eq <= -max_loss) | (intraday <= -daily_loss))[0]
        out.append({"trader": tr, "day": day, "breach_day": int(day[bad[0]]) if bad.size else None, "rate": rate})
    return out


def observe_accounts(accounts, seed: int, start_max=120, obs_end=150, dropout=0.005, horizon=90) -> list:
    """Censoring as a firm sees it: account i starts on calendar day U[0, start_max], observation stops at obs_end;
    traders also leave on their own (geometric, `dropout` per day). Returns (time, event) per account: time = days
    until breach or until censoring, event = breach seen."""
    rng = np.random.default_rng(seed + 3_000_000)
    out = []
    for a in accounts:
        start = int(rng.integers(0, start_max + 1))
        quit_day = int(rng.geometric(dropout)) if dropout > 0 else horizon + 1
        cens = min(obs_end - start, quit_day, horizon)
        b = a["breach_day"]
        out.append((b + 1, 1) if b is not None and b + 1 <= cens else (cens, 0))
    return out


# ---------------------------------------------------------------- BENCH-3P: a funded book, day by day, with a common market
def book_days(n_accounts: int, seed: int, days=42, corr_sd=0.2, rate_median=3.0, rate_sigma=0.5, size_sigma=0.30,
              risk=1.0, **kw):
    """A firm's funded book: BENCH-3 traders (continuous edge, plain / martingale / tilt, own style, noisy sizes) with
    Poisson trades per day, and a shared market move: every trade on calendar day d gets the same N(0, corr_sd) R
    shock (all accounts trade the same days). Returns (day P&L matrix accounts x days in %, kinds)."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, corr_sd, days) if corr_sd > 0 else np.zeros(days)
    base = population_v3(n_accounts, 1, seed, **kw)
    out = np.zeros((n_accounts, days))
    for i, t in enumerate(base):
        rate = float(rate_median * np.exp(rate_sigma * rng.standard_normal()))
        per_day = rng.poisson(rate, days)
        n = int(per_day.sum())
        if n == 0:
            continue
        day = np.repeat(np.arange(days), per_day)
        tr = make_bench3_trader(rng, n, t.meta["mu"], t.kind, shock=f[day])
        pnl = risk * tr.pnl * _lognorm1(rng, n, size_sigma)
        out[i] = np.bincount(day, weights=pnl, minlength=days)
    return out, [t.kind for t in base]
