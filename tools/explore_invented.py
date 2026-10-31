"""EXPLORATORY (development data, not evidence): the lab-invented candidates L1 tail debt, L2 regime gate and
L3 first-passage breach (docs/FORMULAS.md), approved for exploration by the user on 2026-10-31. Every run is logged
in the registry with --log. Development data only (docs/DEV_DATA.md on strategy-validator / funded-check); the holdout
is never read.

  python3 -m tools.explore_invented l1 --hl3 DIR [--log]
  python3 -m tools.explore_invented l2 --curves DIR [DIR ...] [--log]
  python3 -m tools.explore_invented l3 --hl3 DIR [--log]
Prop rules as the data owner's risk model (docs/PAYOUT_MODEL_EXPLORATORY.md): cut 2026-07-01, outcome July-October,
target +8 % -> payout and reset, trailing drawdown 10 % -> breach. Aggregates only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy import stats as _st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import realdata, realtrades, skill  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

DAY = realdata.DAY_MS
CUT = int(datetime.fromisoformat("2026-07-01T00:00:00+00:00").timestamp() * 1000)
END = int(datetime.fromisoformat("2026-10-30T00:00:00+00:00").timestamp() * 1000)
FILL_CAP = 10000


def hl3_accounts(root):
    """Per account: trades and account value at the cut (from the portfolio curve). Truncated accounts excluded."""
    sample = realdata.load_json(os.path.join(root, "sample.json"))["sample"][1:]
    out, trunc = [], 0
    for a in sample:
        pf, pp = os.path.join(root, "fills", f"{a}.json"), os.path.join(root, "portfolio", f"{a}.json")
        if not (os.path.exists(pf) and os.path.exists(pp)):
            continue
        fills = realdata.load_json(pf) or []
        if len(fills) >= FILL_CAP:
            trunc += 1
            continue
        try:
            data = dict((p, d) for p, d in realdata.load_json(pp))["allTime"]
            av = sorted((int(t), float(v)) for t, v in data["accountValueHistory"])
        except (KeyError, ValueError, TypeError):
            continue
        before = [v for t, v in av if t <= CUT]
        if not before:
            continue
        out.append({"trades": realtrades.hl_trades(fills), "base": before[-1]})
    return out, trunc


def daily(trades, a_ms, b_ms):
    """Realised P&L per calendar day in [a, b) (trade result booked on its last fill), zeros on idle days."""
    n = int((b_ms - a_ms) // DAY)
    v = np.zeros(n)
    for _, t1, pnl, _ in trades:
        if a_ms <= t1 < b_ms:
            v[int((t1 - a_ms) // DAY)] += pnl
    return v


def replay(r):
    """Prop rules on daily returns (fraction of the account): trailing 10 % drawdown -> breach; +8 % -> payout, reset."""
    eq = peak = start = 0.0
    paid = 0.0
    for x in r:
        eq += x
        peak = max(peak, eq)
        if eq - peak <= -0.10:
            return True, paid
        if eq - start >= 0.08:
            paid += 0.8 * (eq - start)
            eq = peak = start = 0.0
    return False, paid


def auc(s, y):
    return skill.auc(np.asarray(s, dtype=float), np.asarray(y, dtype=bool))


def deciles_gap(p, y):
    dec = np.minimum((np.argsort(np.argsort(p, kind="mergesort")) * 10) // len(p), 9)
    return [[round(float(p[dec == k].mean()), 3), round(float(y[dec == k].mean()), 3)] for k in range(10)]


def breach_outcomes(accts):
    """Accounts with base >= 1,000 USDC, >= 30 trades in the 182 days before the cut: (formation daily returns,
    outcome breach flag)."""
    rows = []
    for a in accts:
        if a["base"] < 1000:
            continue
        pre = [x for x in a["trades"] if CUT - 182 * DAY <= x[1] < CUT]
        if len(pre) < 30:
            continue
        f = daily(a["trades"], CUT - 182 * DAY, CUT) / a["base"]
        o = daily(a["trades"], CUT, END) / a["base"]
        f, o = np.clip(f, -0.5, 0.5), np.clip(o, -0.5, 0.5)
        rows.append((a, f, replay(o)[0]))
    return rows


def first_passage(mu, s, B=0.10, T=121.0):
    s = np.maximum(s, 1e-9)
    a = (-B - mu * T) / (s * np.sqrt(T))
    b = (-B + mu * T) / (s * np.sqrt(T))
    ex = np.exp(np.clip(-2 * mu * B / s ** 2, -50, 50))
    return np.clip(_st.norm.cdf(a) + ex * _st.norm.cdf(b), 0, 1)


def run_l3(root):
    accts, trunc = hl3_accounts(root)
    rows = breach_outcomes(accts)
    f = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    mu, sd = f.mean(axis=1), f.std(axis=1, ddof=1)
    se = sd / np.sqrt(f.shape[1])
    shr = skill.npmle_summary(mu, se, thr=0.0)["post_mean"]
    res = {"accounts": len(rows), "truncated_excluded": trunc, "breach_rate": round(float(y.mean()), 4)}
    for name, m in (("own_drift", mu), ("shrunk_drift", shr), ("zero_drift", np.zeros_like(mu))):
        p = first_passage(m, sd)
        res[name] = {"auc": round(auc(p, y), 4), "pred_mean": round(float(p.mean()), 4), "deciles": deciles_gap(p, y)}
    res["vol_only_auc"] = round(auc(sd, y), 4)
    return res


def run_l1(root):
    accts, trunc = hl3_accounts(root)
    rows = []
    for a in accts:
        pre = [x for x in a["trades"] if x[1] < CUT]
        post = [x for x in a["trades"] if CUT <= x[0] < END]
        R = realtrades.r_scale([x[2] for x in pre])
        if R is None or len(pre) < 50:
            continue
        r = np.array([x[2] for x in pre]) / R
        rows.append({"r": r, "post": np.array([x[2] for x in post]) / R, "wr": float(np.mean(r > 0)), "acct": a})
    wr = np.array([x["wr"] for x in rows])
    band = np.minimum((np.argsort(np.argsort(wr, kind="mergesort")) * 10) // len(wr), 9)
    lam, Ls = {}, {}
    for b in range(10):
        allr = np.concatenate([rows[i]["r"] for i in np.nonzero(band == b)[0]])
        big = allr[allr < -2]
        lam[b] = big.size / allr.size
        Ls[b] = float(-big.mean()) if big.size else 0.0
    out = {"accounts": len(rows), "truncated_excluded": trunc,
           "bands": [[round(float(np.median(wr[band == b])), 3), round(lam[b], 4), round(Ls[b], 2)] for b in range(10)]}
    xbar = np.array([x["r"].mean() for x in rows]); n = np.array([x["r"].size for x in rows])
    k = np.array([np.sum(x["r"] < -2) for x in rows])
    debt = np.maximum(0, np.array([lam[b] for b in band]) * n - k) * np.array([Ls[b] for b in band])
    adj = xbar - debt / n
    se = np.array([x["r"].std(ddof=1) / np.sqrt(x["r"].size) for x in rows])
    npost = np.array([x["post"].size for x in rows])
    ev = npost >= 30
    y = np.array([x["post"].mean() > 0.05 if x["post"].size else False for x in rows])
    for name, mean in (("plain", xbar), ("tail_debt", adj)):
        fit = skill.npmle_summary(mean, se, thr=0.05)
        sd = np.array([x["r"].std(ddof=1) for x in rows])
        pp = skill.predictive_positive(fit["post"], fit["grid"] - 0.05, sd / np.sqrt(np.maximum(npost, 1)))
        res = {}
        for lo, hi in ((50, 150), (150, 10 ** 9)):
            m = ev & (n >= lo) & (n < hi)
            f = m & (pp >= 0.7)
            res[f"n{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = {
                "evaluated": int(m.sum()), "flagged": int(f.sum()),
                "precision": round(float(y[f].mean()), 4) if f.any() else None,
                "auc": round(auc(pp[m], y[m]), 4) if 0 < y[m].sum() < m.sum() else None,
                "high_wr_share_of_flags": round(float(np.mean(wr[f] > np.quantile(wr, 0.8))), 3) if f.any() else None}
        out[name] = res
    # hidden-tail use: does the debt per trade predict breach beyond volatility?
    br = {id(r[0]): r for r in breach_outcomes([x["acct"] for x in rows])}
    idx = [i for i, x in enumerate(rows) if id(x["acct"]) in br]
    if idx:
        yb = np.array([br[id(rows[i]["acct"])][2] for i in idx])
        vol = np.array([br[id(rows[i]["acct"])][1].std(ddof=1) for i in idx])
        dpt = (debt / n)[idx]
        no_big = k[idx] == 0
        out["breach"] = {"accounts": len(idx), "breach_rate": round(float(yb.mean()), 4),
                         "auc_debt_per_trade": round(auc(dpt, yb), 4), "auc_vol": round(auc(vol, yb), 4),
                         "auc_vol_rank_plus_debt_rank": round(auc(_st.rankdata(vol) + _st.rankdata(dpt), yb), 4),
                         "no_big_loss_yet": {"n": int(no_big.sum()),
                                             "breach_rate": round(float(yb[no_big].mean()), 4) if no_big.any() else None,
                                             "auc_debt_within": round(auc(dpt[no_big], yb[no_big]), 4)
                                             if no_big.sum() > 20 and 0 < yb[no_big].sum() < no_big.sum() else None}}
    return out


def run_l2(roots):
    series = []
    for root in roots:
        sample = realdata.load_json(os.path.join(root, "sample.json"))["sample"][1:]
        for a in sample:
            p = os.path.join(root, "portfolio", f"{a}.json")
            if os.path.exists(p):
                try:
                    series.append(realdata.hl_intervals(realdata.load_json(p)))
                except (KeyError, ValueError, TypeError):
                    pass
    cuts = [int(datetime(2024 + (m // 12), m % 12 + 1, 1).timestamp() * 1000) for m in range(6, 31)]   # 2024-07 .. 2026-07
    per = []
    for c in cuts:
        a, h, b = c - 182 * DAY, c - 91 * DAY, c + 92 * DAY
        mu, se, m1, m2, post = [], [], [], [], []
        for s in series:
            w = (s["t0"] >= a) & (s["t1"] <= c)
            d = realdata.drift(s["r"][w], s["dt"][w])
            if d is None or d["n"] < 10 or d["days"] < 60 or d["se"] <= 0:
                continue
            q = (s["t0"] >= c) & (s["t1"] <= b)
            if q.sum() < 1 or s["dt"][q].sum() < 14:
                continue
            w1 = (s["t0"] >= a) & (s["t1"] <= h); w2 = (s["t0"] >= h) & (s["t1"] <= c)
            d1, d2 = realdata.drift(s["r"][w1], s["dt"][w1]), realdata.drift(s["r"][w2], s["dt"][w2])
            mu.append(d["mu"]); se.append(d["se"]); post.append(s["r"][q].sum() / s["dt"][q].sum())
            m1.append(d1["mu"] if d1 else np.nan); m2.append(d2["mu"] if d2 else np.nan)
        if len(mu) < 100:
            continue
        mu, se, post, m1, m2 = map(np.array, (mu, se, post, m1, m2))
        t = (mu - np.median(mu)) / se
        y = post > np.median(post)
        dec = np.minimum((np.argsort(np.argsort(t, kind="mergesort")) * 10) // len(t), 9)
        ok = ~np.isnan(m1) & ~np.isnan(m2)
        rho = float(_st.spearmanr(m1[ok], m2[ok]).correlation) if ok.sum() > 50 else np.nan
        per.append({"cut": datetime.utcfromtimestamp(c / 1000).strftime("%Y-%m"), "n": int(len(t)),
                    "rho_pre_halves": round(rho, 4), "bottom_share": round(float(y[dec == 0].mean()), 4),
                    "n_bottom": int((dec == 0).sum()), "y_bottom": int(y[dec == 0].sum())})
    rhos = np.array([p["rho_pre_halves"] for p in per]); bs = np.array([p["bottom_share"] for p in per])
    ok = ~np.isnan(rhos)
    gate = ok & (rhos > 0.1)
    pool = lambda m: round(sum(p["y_bottom"] for p, k in zip(per, m) if k) / max(sum(p["n_bottom"] for p, k in zip(per, m) if k), 1), 4)
    return {"cuts": per, "spearman_rho_vs_bottom_share": round(float(_st.spearmanr(rhos[ok], bs[ok]).correlation), 4),
            "pooled_bottom_share_all": pool(ok), "pooled_gate_on": pool(gate), "pooled_gate_off": pool(ok & ~gate),
            "cuts_gate_on": int(gate.sum()), "cuts_gate_off": int((ok & ~gate).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["l1", "l2", "l3"])
    ap.add_argument("--hl3")
    ap.add_argument("--curves", nargs="*")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    res = run_l1(a.hl3) if a.what == "l1" else run_l3(a.hl3) if a.what == "l3" else run_l2(a.curves)
    print(json.dumps(res, indent=1))
    if a.log:
        Registry(os.path.join(ROOT, "state", "registry.jsonl")).note(
            {"kind": "attempts", "study": f"explore {a.what.upper()} (development data, NOT evidence)", "note": a.note, "values": res})
    return 0


if __name__ == "__main__":
    sys.exit(main())
