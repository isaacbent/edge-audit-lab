"""EXPLORATORY (development data, not evidence): round-2 candidates approved by the user on 2026-11-03
(docs/FORMULAS.md): N1b market-adjusted alpha with shrunk beta, A5 random-effects payout intervals, A3 early-warning
signals. Budget: at most 2 logged attempts each. Development data only; the holdout is never read.

  python3 -m tools.explore_round2 n1b --curves DIR [DIR ...] [--log --note TEXT]
  python3 -m tools.explore_round2 a5 --curves DIR [DIR ...] [--log --note TEXT]
  python3 -m tools.explore_round2 a3 --hl3 DIR [--log --note TEXT]
How each is judged is fixed here before any run: N1b on the reversal cuts seen in L2 (2025-01, 2025-02, 2025-09,
2025-10, 2025-11) vs the other cuts; A5 by coverage of the 90 % interval month by month, per book; A3 by out-of-fold
breach AUC gain over the levels (volatility, lag-1 autocorrelation). Aggregates only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy import stats as _st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import realdata, skill  # noqa: E402
from edgelab.registry import Registry  # noqa: E402
from tools import explore_invented as ei  # noqa: E402

DAY = realdata.DAY_MS
REVERSAL_CUTS = {"2025-01", "2025-02", "2025-09", "2025-10", "2025-11"}


def load_curves(roots):
    out = []
    for root in roots:
        for a in realdata.load_json(os.path.join(root, "sample.json"))["sample"][1:]:
            p = os.path.join(root, "portfolio", f"{a}.json")
            if os.path.exists(p):
                try:
                    out.append(realdata.hl_intervals(realdata.load_json(p)))
                except (KeyError, ValueError, TypeError):
                    pass
    return out


def day_of(ms):
    return (np.asarray(ms) // DAY).astype(np.int64)


# ---------------------------------------------------------------- N1b
def market_index(series):
    """Daily market rate = median of r / dt over intervals of <= 3 days ending that day; cumulative sum by day."""
    t1 = np.concatenate([s["t1"] for s in series]); r = np.concatenate([s["r"] for s in series])
    dt = np.concatenate([s["dt"] for s in series])
    m = (dt > 0) & (dt <= 3)
    d, rate = day_of(t1[m]), r[m] / dt[m]
    o = np.argsort(d, kind="mergesort"); d, rate = d[o], rate[o]
    days, start = np.unique(d, return_index=True)
    med = np.array([np.median(x) for x in np.split(rate, start[1:])])
    d0, d1 = days.min(), days.max()
    full = np.zeros(d1 - d0 + 1)
    full[days - d0] = med
    return d0, np.concatenate([[0.0], np.cumsum(full)])


def market_move(cum, d0, t0, t1):
    a = np.clip(day_of(t0) - d0 + 1, 0, len(cum) - 1); b = np.clip(day_of(t1) - d0 + 1, 0, len(cum) - 1)
    return cum[b] - cum[a]


def fit_alpha_beta(r, dt, M):
    """WLS of r = a dt + b M with weights 1 / dt. Returns a, b, se_a, se_b, residual var."""
    w = 1.0 / dt
    X = np.column_stack([dt, M])
    A = X.T @ (X * w[:, None])
    try:
        inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    coef = inv @ (X.T @ (w * r))
    e = r - X @ coef
    s2 = float(np.sum(w * e ** 2) / max(len(r) - 2, 1))
    return coef[0], coef[1], np.sqrt(s2 * inv[0, 0]), np.sqrt(s2 * inv[1, 1]), s2


def run_n1b(roots):
    series = load_curves(roots)
    d0, cum = market_index(series)
    cuts = [int(datetime(2024 + (m // 12), m % 12 + 1, 1, tzinfo=timezone.utc).timestamp() * 1000) for m in range(6, 31)]
    per = []
    for c in cuts:
        a_, b_ = c - 182 * DAY, c + 92 * DAY
        rows = []
        for s in series:
            w = (s["t0"] >= a_) & (s["t1"] <= c)
            q = (s["t0"] >= c) & (s["t1"] <= b_)
            if w.sum() < 10 or s["dt"][w].sum() < 60 or q.sum() < 1 or s["dt"][q].sum() < 14:
                continue
            r, dt = s["r"][w], s["dt"][w]
            M = market_move(cum, d0, s["t0"][w], s["t1"][w])
            d = realdata.drift(r, dt)
            f = fit_alpha_beta(r, dt, M)
            if d is None or f is None or d["se"] <= 0 or not np.isfinite(f[3]) or f[3] <= 0:
                continue
            rows.append((d["mu"], d["se"], f[1], f[3], r, dt, M, s["r"][q].sum() / s["dt"][q].sum()))
        if len(rows) < 100:
            continue
        b = np.array([x[2] for x in rows]); seb = np.array([x[3] for x in rows])
        true_var = max(float(np.var(b) - np.mean(seb ** 2)), 1e-12)
        wgt = true_var / (true_var + seb ** 2)
        b_adj = wgt * b + (1 - wgt) * 1.0
        alpha, se_a = [], []
        for x, ba in zip(rows, b_adj):
            r, dt, M = x[4], x[5], x[6]
            a = float(np.sum(r - ba * M) / np.sum(dt))
            e = r - a * dt - ba * M
            s2 = float(np.sum(e ** 2 / dt) / max(len(r) - 1, 1))
            alpha.append(a); se_a.append(np.sqrt(s2 / np.sum(dt)))
        alpha, se_a = np.array(alpha), np.maximum(np.array(se_a), 1e-12)
        mu = np.array([x[0] for x in rows]); se = np.array([x[1] for x in rows])
        post = np.array([x[7] for x in rows]); y = post > np.median(post)
        t_raw = (mu - np.median(mu)) / se
        t_alpha = (alpha - np.median(alpha)) / se_a
        dec = lambda t: np.minimum((np.argsort(np.argsort(t, kind="mergesort")) * 10) // len(t), 9)
        dr, da = dec(t_raw), dec(t_alpha)
        label = datetime.fromtimestamp(c / 1000, timezone.utc).strftime("%Y-%m")
        per.append({"cut": label, "n": len(rows), "reversal": label in REVERSAL_CUTS,
                    "bottom_raw": round(float(y[dr == 0].mean()), 4), "bottom_alpha": round(float(y[da == 0].mean()), 4),
                    "n_bottom": int((dr == 0).sum()), "y_raw": int(y[dr == 0].sum()), "y_alpha": int(y[da == 0].sum()),
                    "median_beta_adj": round(float(np.median(b_adj)), 3), "shrink_weight": round(float(np.median(wgt)), 3),
                    "beta_bottom_raw_minus_rest": round(float(np.median(b_adj[dr == 0]) - np.median(b_adj[dr != 0])), 3)})
    pool = lambda rev, key: round(sum(p[key] for p in per if p["reversal"] == rev) / max(sum(p["n_bottom"] for p in per if p["reversal"] == rev), 1), 4)
    return {"cuts": per, "reversal_cuts": {"raw": pool(True, "y_raw"), "alpha": pool(True, "y_alpha")},
            "other_cuts": {"raw": pool(False, "y_raw"), "alpha": pool(False, "y_alpha")}}


# ---------------------------------------------------------------- A5
def run_a5(roots):
    out = {}
    for root in roots:
        series = load_curves([root])
        months = [(y, m) for y in range(2023, 2027) for m in range(1, 13) if (y, m) >= (2023, 7) and (y, m) <= (2026, 9)]
        edges = [int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000) for y, m in months]
        edges.append(int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp() * 1000))
        P, SE, N = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            pay = []
            for s in series:
                m = (s["t1"] > a) & (s["t1"] <= b)
                if m.any():
                    pay.append(0.8 * max(0.0, float(np.clip(s["r"][m].sum(), -0.5, 0.5))))
            pay = np.array(pay)
            P.append(pay.mean() if pay.size else np.nan); SE.append(pay.std(ddof=1) / np.sqrt(pay.size) if pay.size > 1 else np.nan)
            N.append(int(pay.size))
        P, SE = np.array(P), np.array(SE)
        rows = []
        for i in range(12, len(P)):
            hist, hse = P[i - 12:i], SE[i - 12:i]
            if np.isnan(hist).any() or np.isnan(P[i]) or N[i] < 100:
                continue
            centre = float(hist.mean())
            tau2 = max(0.0, float(np.var(hist, ddof=1) - np.mean(hse ** 2)))
            naive = 1.645 * SE[i - 1]
            re = 1.645 * np.sqrt(tau2 + SE[i - 1] ** 2)
            rows.append({"month": f"{months[i][0]}-{months[i][1]:02d}", "actual": round(float(P[i]), 5), "centre": round(centre, 5),
                         "in_naive": bool(abs(P[i] - centre) <= naive), "in_random_effects": bool(abs(P[i] - centre) <= re),
                         "half_naive": round(float(naive), 5), "half_re": round(float(re), 5)})
        out[os.path.basename(root.rstrip("/"))] = {
            "months": len(rows), "coverage_naive": round(float(np.mean([r["in_naive"] for r in rows])), 4) if rows else None,
            "coverage_random_effects": round(float(np.mean([r["in_random_effects"] for r in rows])), 4) if rows else None,
            "rows": rows}
    return out


# ---------------------------------------------------------------- A3
def ac1(x):
    x = x - x.mean()
    d = np.sum(x * x)
    return float(np.sum(x[1:] * x[:-1]) / d) if d > 0 else 0.0


def logistic_oof_auc(X, y, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.integers(0, folds, len(y))
    p = np.zeros(len(y))
    for f in range(folds):
        tr, te = k != f, k == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xt = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd]); Xe = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd])
        w = np.zeros(Xt.shape[1])
        for _ in range(50):
            q = 1 / (1 + np.exp(-Xt @ w))
            H = Xt.T @ (Xt * (q * (1 - q))[:, None]) + 1e-6 * np.eye(len(w))
            w += np.linalg.solve(H, Xt.T @ (y[tr] - q))
        p[te] = Xe @ w
    return round(ei.auc(p, y), 4)


def run_a3(hl3):
    accts, trunc = ei.hl3_accounts(hl3)
    rows = ei.breach_outcomes(accts)
    feats, y = [], []
    for _, f, br in rows:
        ends = range(30, 183, 10)
        a1, lv = [], []
        for e in ends:
            w = f[e - 30:e]
            if w.std() <= 0:
                a1.append(np.nan); lv.append(np.nan); continue
            a1.append(ac1(w)); lv.append(np.log(w.var()))
        a1, lv = np.array(a1), np.array(lv)
        ok = ~np.isnan(a1)
        if ok.sum() < 5:
            continue
        x = np.arange(len(a1))[ok]
        feats.append([np.log(f.std() + 1e-12), ac1(f), np.polyfit(x, a1[ok], 1)[0], np.polyfit(x, lv[ok], 1)[0]])
        y.append(br)
    X, y = np.array(feats), np.array(y, dtype=float)
    ews = _st.zscore(X[:, 2]) + _st.zscore(X[:, 3])
    return {"accounts": len(y), "truncated_excluded": trunc, "breach_rate": round(float(y.mean()), 4),
            "auc_ews_alone": round(ei.auc(ews, y), 4), "auc_slope_ac1": round(ei.auc(X[:, 2], y), 4),
            "auc_slope_logvar": round(ei.auc(X[:, 3], y), 4), "auc_vol_level": round(ei.auc(X[:, 0], y), 4),
            "oof_auc_levels": logistic_oof_auc(X[:, :2], y), "oof_auc_levels_plus_trends": logistic_oof_auc(X, y)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["n1b", "a5", "a3"])
    ap.add_argument("--curves", nargs="*")
    ap.add_argument("--hl3")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    res = run_n1b(a.curves) if a.what == "n1b" else run_a5(a.curves) if a.what == "a5" else run_a3(a.hl3)
    print(json.dumps(res, indent=1))
    if a.log:
        Registry(os.path.join(ROOT, "state", "registry.jsonl")).note(
            {"kind": "attempts", "study": f"explore {a.what.upper()} (round 2, development data, NOT evidence)",
             "note": a.note, "values": res})
    return 0


if __name__ == "__main__":
    sys.exit(main())
