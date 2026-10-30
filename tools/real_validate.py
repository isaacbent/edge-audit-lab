"""REAL-01: first test on real trader data (data package realdata_2026-10-30 from isaacbenta/strategy-validator).
Score each account on its returns before a cut date with the empirical-Bayes (NPMLE) skill model, then check against
what happened after the cut: is the predictive probability calibrated, and does it rank at least as well as the
t-stat? Protocol written from the package README only and pre-registered before any data file was opened.

Usage:
  python3 -m tools.real_validate --smoke                      # fake package with the README's structure
  python3 -m tools.real_validate --register
  python3 -m tools.real_validate --data /home/user/strategy-validator/unzipped
Outputs are aggregates only (no addresses, trader codes or individual series: package README).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import realdata, skill, study  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "REAL-01"
CUT = "2026-07-30T00:00:00+00:00"
SPEC = {
    "question": "On real accounts, is the NPMLE predictive probability of a positive post-cut drift calibrated, and does "
                "it rank post-cut outcomes at least as well as the pre-cut t-stat?",
    "data_package": {"repo": "isaacbenta/strategy-validator", "branch": "realdata-2026-10-30",
                     "commit": "5862985824d4f358fc4153407f2cc2db2733532a",
                     "zip": "realdata_2026-10-30.zip",
                     "zip_sha256": "0b851ea48bb28a2b887ed20e8f7154fef88a43576756e5437c57ef7a6c2240ff",
                     "integrity": "after registration: unzip, then sha256sum -c MANIFEST_hyperliquid.sha256 / "
                                  "MANIFEST_okx.sha256 (package README); any mismatch = stop, no analysis"},
    "opened_before_registration": "README.md only (plus the zip's sha256); no data file opened",
    "cut": CUT,
    "datasets": {
        "hyperliquid": {"role": "PRIMARY (random sample of the full leaderboard)",
                        "accounts": "sample.json 'sample' list, first address excluded (calibration account of the "
                                    "other study); portfolio/<address>.json, period 'allTime'",
                        "returns": "interval return (pnl_i - pnl_{i-1}) / accountValue_{i-1} between consecutive "
                                   "history points; intervals starting below 100 USDC account value dropped",
                        "eligible": "pre-cut: >= 10 intervals spanning >= 60 days; post-cut: >= 1 interval spanning "
                                    ">= 14 days in total"},
        "okx": {"role": "SECONDARY (current lead traders only: survivorship bias)",
                "accounts": "lead_traders.json list, first trader excluded (calibration account of the other study); "
                            "pnl/<uniqueCode>.json",
                "returns": "daily return (1 + R_d) / (1 + R_{d-1}) - 1 from the cumulative pnlRatio",
                "eligible": "pre-cut >= 60 days of returns; post-cut >= 30 days"}},
    "split": "intervals ending at or before the cut = pre; starting at or after it = post; spanning ones dropped",
    "estimator": "realdata.drift: r_i ~ N(mu dt_i, sigma^2 dt_i); mu = sum r / sum dt, se = sigma / sqrt(sum dt)",
    "model": "skill.npmle_summary(mu_pre, se_pre, thr=0) fitted per dataset on its eligible accounts; "
             "P_pred = skill.predictive_positive(post, grid, se_f) with se_f = sigma_pre / sqrt(post days)",
    "outcome": "y = post-cut drift mu_post > 0",
    "metrics": {"calibration": "deciles of P_pred: mean P_pred vs share y; ECE = mean over deciles of |gap|; ece_noise95 = "
                               "95th percentile of ECE when y is drawn from P_pred itself (perfect calibration, 2000 "
                               "simulations, rng seed 1); max decile gap; Brier vs base-rate Brier",
                "auc": "AUC of P_pred, of t_pre = mu_pre / se_pre and of mu_pre for y; paired bootstrap over "
                       "accounts (2000 resamples, rng seed 0) of AUC(P_pred) - AUC(t_pre), 95 % percentile interval",
                "top": "share y among accounts with P_pred >= 0.7", "counts": "accounts found / eligible / excluded"},
    "success": "hyperliquid, all of: (1) ECE of P_pred <= max(0.05, ece_noise95) AND (2) AUC(P_pred) - AUC(t_pre) "
               "lower 95 % bound >= -0.01",
    "criterion_note": "a max-decile-gap <= 0.10 rule was drafted and dropped before registration: with 800 accounts a "
                      "perfectly calibrated model exceeds it 29 % of the time (300: 79 %), and the eligible count is "
                      "unknown until the data is opened; ECE with a noise floor adapts to the sample size",
    "secondary": "okx: same metrics, no criterion; AUC(P_pred) > 0.5 reported for both",
    "deviations": "if the parsers do not match the real files, nothing is tuned: the study stops, the mismatch is "
                  "recorded, and a corrected parser runs as a NEW study id with this protocol unchanged",
    "notes": "units are account returns per day, not R per trade: this tests the method (calibrated empirical-Bayes "
             "skill probability), not the product thresholds (150 trades, 0.05 R); no results of the other study "
             "are read before this run",
}


def cut_ms() -> int:
    return int(datetime.fromisoformat(CUT).timestamp() * 1000)


def account_rows(series_iter, min_pre_n, min_pre_days, min_post_days, min_post_n=1):
    rows, counts = [], {"found": 0, "missing": 0, "unreadable": 0, "eligible": 0}
    c = cut_ms()
    for s in series_iter:
        if s == "missing":
            counts["missing"] += 1
            continue
        if s == "unreadable":
            counts["unreadable"] += 1
            continue
        counts["found"] += 1
        pre, post = realdata.split(s, c)
        dp, dq = realdata.drift(pre["r"], pre["dt"]), None
        if post["r"].size >= min_post_n and post["dt"].sum() >= min_post_days:
            dq = {"mu": float(post["r"].sum() / post["dt"].sum()), "days": float(post["dt"].sum())}
        if dp is None or dp["n"] < min_pre_n or dp["days"] < min_pre_days or dq is None or dp["se"] <= 0:
            continue
        counts["eligible"] += 1
        rows.append((dp["mu"], dp["se"], dp["sigma"], dq["days"], dq["mu"]))
    return rows, counts


def evaluate(rows, seed=0):
    a = np.array(rows, dtype=float)
    mu, se, sig, pdays, mpost = a.T
    y = mpost > 0
    fit = skill.npmle_summary(mu, se, thr=0.0)
    pp = skill.predictive_positive(fit["post"], fit["grid"], sig / np.sqrt(pdays))
    t = mu / se
    dec = np.minimum((np.argsort(np.argsort(pp, kind="mergesort")) * 10) // len(pp), 9)
    deciles = [[round(float(pp[dec == d].mean()), 4), round(float(y[dec == d].mean()), 4), int((dec == d).sum())] for d in range(10)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].all() or not y[i].any():
            continue
        diffs.append(skill.auc(pp[i], y[i]) - skill.auc(t[i], y[i]))
    auc_pp, auc_t = skill.auc(pp, y), skill.auc(t, y)
    ece = float(np.mean([abs(p - o) for p, o, _ in deciles]))
    rs = np.random.default_rng(1)
    sims = []
    for _ in range(2000):
        ys = rs.random(len(pp)) < pp
        sims.append(np.mean([abs(pp[dec == d].mean() - ys[dec == d].mean()) for d in range(10)]))
    return {"n": int(len(y)), "share_positive_post": round(float(y.mean()), 4),
            "deciles_pred_obs_n": deciles, "max_decile_gap": round(max(abs(p - o) for p, o, _ in deciles), 4),
            "ece": round(ece, 4), "ece_noise95": round(float(np.quantile(sims, 0.95)), 4),
            "brier": round(float(np.mean((pp - y) ** 2)), 4), "brier_base_rate": round(float(np.mean((y.mean() - y) ** 2)), 4),
            "auc_pred": round(auc_pp, 4), "auc_t_pre": round(auc_t, 4), "auc_mu_pre": round(skill.auc(mu, y), 4),
            "auc_pred_minus_t": {"mean": round(auc_pp - auc_t, 4),
                                 "ci95": [round(float(q), 4) for q in np.quantile(diffs, [0.025, 0.975])]},
            "top_p07": {"n": int((pp >= 0.7).sum()), "share_positive": round(float(y[pp >= 0.7].mean()), 4) if (pp >= 0.7).any() else None},
            "p_edge_mean": round(float(fit["prob"].mean()), 4)}


def _safe(fn, path):
    if not os.path.exists(path):
        return "missing"
    try:
        return fn(realdata.load_json(path))
    except (KeyError, ValueError, TypeError, IndexError):
        return "unreadable"


def load_hyperliquid(root):
    sample = realdata.load_json(os.path.join(root, "data", "hyperliquid", "sample.json"))["sample"][1:]
    pdir = os.path.join(root, "data", "hyperliquid", "portfolio")
    return (_safe(realdata.hl_intervals, os.path.join(pdir, f"{a}.json")) for a in sample)


def load_okx(root):
    leads = realdata.load_json(os.path.join(root, "data", "okx", "lead_traders.json"))
    leads = leads["data"] if isinstance(leads, dict) and "data" in leads else leads
    codes = [x["uniqueCode"] if isinstance(x, dict) else str(x) for x in leads][1:]
    pdir = os.path.join(root, "data", "okx", "pnl")
    return (_safe(realdata.okx_daily, os.path.join(pdir, f"{c}.json")) for c in codes)


def run(root, out=print):
    res = {}
    rows, counts = account_rows(load_hyperliquid(root), 10, 60.0, 14.0)
    res["hyperliquid"] = {"counts": counts, **(evaluate(rows) if len(rows) >= 20 else {"error": "fewer than 20 eligible"})}
    out(json.dumps({"hyperliquid": res["hyperliquid"]}))
    rows, counts = account_rows(load_okx(root), 60, 60.0, 30.0, min_post_n=30)
    res["okx"] = {"counts": counts, **(evaluate(rows) if len(rows) >= 20 else {"error": "fewer than 20 eligible"})}
    out(json.dumps({"okx": res["okx"]}))
    h = res["hyperliquid"]
    res["success"] = bool("error" not in h and h["ece"] <= max(0.05, h["ece_noise95"])
                          and h["auc_pred_minus_t"]["ci95"][0] >= -0.01)
    return res


def fake_package(root, n=300, seed=9970):
    """A tiny package with the README's structure (for --smoke and tests); no real data."""
    rng = np.random.default_rng(seed)
    c = cut_ms()
    hl, ok = os.path.join(root, "data", "hyperliquid"), os.path.join(root, "data", "okx")
    os.makedirs(os.path.join(hl, "portfolio")); os.makedirs(os.path.join(ok, "pnl"))
    addrs = [f"0x{i:040x}" for i in range(n)]
    json.dump({"universe": n, "sample": addrs}, open(os.path.join(hl, "sample.json"), "w"))
    for a in addrs:
        mu = rng.normal(-0.001, 0.003)
        ts = np.sort(c + rng.uniform(-200, 90, 80) * realdata.DAY_MS).astype(int)
        dt = np.diff(np.concatenate([[ts[0] - realdata.DAY_MS], ts])) / realdata.DAY_MS
        r = mu * dt + rng.normal(0, 0.02, ts.size) * np.sqrt(dt)
        av = 10000 * np.cumprod(1 + r); pnl = av - 10000                 # interval return = d pnl / previous value
        hist = lambda v: [[int(t), str(x)] for t, x in zip(ts, v)]
        json.dump([["allTime", {"accountValueHistory": hist(av), "pnlHistory": hist(pnl), "vlm": "0"}]],
                  open(os.path.join(hl, "portfolio", f"{a}.json"), "w"))
    codes = [f"C{i}" for i in range(n)]
    json.dump([{"uniqueCode": k} for k in codes], open(os.path.join(ok, "lead_traders.json"), "w"))
    for k in codes:
        mu = rng.normal(0.0, 0.002)
        days = c + np.arange(-270, 92) * realdata.DAY_MS
        R = np.cumprod(1 + mu + rng.normal(0, 0.01, days.size)) - 1
        json.dump([{"beginTs": str(int(t)), "pnl": "0", "pnlRatio": str(x)} for t, x in zip(days, R)][::-1],
                  open(os.path.join(ok, "pnl", f"{k}.json"), "w"))
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        with tempfile.TemporaryDirectory() as d:
            res = run(fake_package(d), out=lambda s: None)
        print(json.dumps({"smoke": "ok", "hl_counts": res["hyperliquid"]["counts"], "okx_counts": res["okx"]["counts"]}))
        return 0
    reg = Registry(args.registry)
    script = os.path.abspath(__file__)
    if args.register:
        dirty = study.uncommitted(script)
        if dirty:
            raise SystemExit(f"commit these first: {dirty}")
        print(reg.prereg(STUDY, SPEC, lock=study.code_lock(script))["payload"]["spec_hash"])
        return 0
    pre = reg.spec_of(STUDY)
    if pre["spec"] != json.loads(json.dumps(SPEC)):
        raise SystemExit("SPEC differs from the pre-registration: refusing to run")
    lock = study.check_lock(pre, script)
    if not args.data:
        raise SystemExit("--data (unzipped package root, MANIFESTs checked) is required")
    res = run(args.data)
    with atomic_open(os.path.join(ROOT, "state", f"{STUDY}_report.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    reg.result(STUDY, {"run_commit": lock["commit"], **res})
    print(json.dumps({"success": res["success"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
