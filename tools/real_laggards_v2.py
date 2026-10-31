"""REAL-06 (= REAL-04 with the HLS-04 download gate corrected): do accounts that lag their peers keep lagging? Confirmation on data the lab has not seen (HLS-02 primary;
HLS-03, HLS-04 and Ostium replications), four cuts, all accounts (no +8 % filter). Protocol approved by the user on
2026-10-31 and registered before any data file was opened.

Usage:
  python3 -m tools.real_laggards_v2 --smoke
  python3 -m tools.real_laggards_v2 --register
  python3 -m tools.real_laggards_v2 --data ROOT      # ROOT/{hl2,hl3,hl4,ostium}, each MANIFEST-checked
Outputs are aggregates only (no addresses or individual series).
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

from edgelab import realdata, realtrades, skill, study  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "REAL-06"
DAY = realdata.DAY_MS
CUTS = ["2024-10-30", "2025-04-30", "2025-10-30", "2026-04-30"]
REOPEN = "2026-07-23"
SPEC = {
    "question": "Among all accounts, do those with the lowest peer-relative t-stat over 6 months keep lagging the median "
                "of their peers over the next 3 months?",
    "data": {"source": "isaacbenta/strategy-validator, private branches (docs/DEV_DATA.md on funded-check commit "
                       "5b90ba69879fe84b2c837da303840cb4cbcb4757)",
             "hl2": {"role": "PRIMARY", "branches": "hl2-data, hl2-data-0..14", "content": "6,000 portfolios",
                     "manifest": "research/hyperliquid2/MANIFEST.sha256 (sha256 c856674bebcf0833cd529acc2ea813a146918c8dea5d2016e85d309590756ae2)"},
             "hl3": {"role": "replication", "branches": "hl3-data, hl3-data-0..14", "content": "9,000 portfolios",
                     "manifest": "research/hyperliquid3/MANIFEST.sha256 (sha256 8ed04fcbc8031571937a839a7c57e7853643869719b3a1109fab50344db9381b)"},
             "hl4": {"role": "replication (2,137 portfolios of accounts with >= 30 xyz fills before 2026-07-01: "
                             "selected by activity, not random)", "branches": "hl4-data, hl4-data-0..14",
                     "manifest": "research/hyperliquid4/MANIFEST.sha256 (sha256 c2b8abd991f0fd9e599b73218f894dd031ad17fbe0c7d351daabcb2d180d79a0)"},
             "ostium": {"role": "replication", "branches": "ostium-data, ostium2-data-0..14", "content": "25,419 traders, orders",
                        "manifest": "research/ostium/MANIFEST.sha256 (sha256 41583801033675cb0862a581e8c5711a5dd1228d7bac23ce784c783b039587e1)"},
             "overlap": "HLS-02/03/04 exclude the 3,000 HLS-01 addresses the lab saw (data owner, verified); holdout not touched"},
    "gate": {"rule": "docs/RESEARCH_PROTOCOL.md section 2, checked before any result; any failure = study NOT VALID, "
                     "no results computed",
             "G1_frame": "sample.json 'sample' has the documented size (hl2 6,000, hl3 9,000, hl4 9,000, ostium 25,419)",
             "G2_download": "files present for >= 99 % of sampled accounts (portfolio for hl2/hl3, orders for ostium); "
                            "hl4: >= 99 % of the 2,137 portfolio files listed in its MANIFEST",
             "G3_data": "share of sampled accounts with any history >= 10 %",
             "G5_units": "calibration account (first in each sample): printed derived quantities only, must be finite",
             "G6_periods": "every dataset has accounts with data before and after each cut it uses; Ostium halt recorded"},
    "ostium_halt": "start = beginning of the longest gap between consecutive orders platform-wide in 2026 (printed as "
                   "dates only); reopening 2026-07-23 (data owner); cuts whose post window overlaps [start, reopening] "
                   "are dropped for Ostium",
    "cuts": CUTS, "windows": "pre = 182 days before the cut; post = 92 days after",
    "accounts": "sample minus the first entry (calibration account)",
    "series": {"hyperliquid": "realdata.hl_intervals (allTime, intervals starting below 100 USDC dropped)",
               "ostium": "per-trade R: realtrades.ostium_trades results / median pre-window loss (>= 5 losses)"},
    "eligible": {"hyperliquid": "pre: >= 10 intervals spanning >= 60 days; post: >= 1 interval and >= 14 days",
                 "ostium": "pre: >= 10 trades spanning >= 60 days with an R scale; post: >= 5 trades over >= 14 days"},
    "score": "t_rel = (drift - median drift over the cut's eligible accounts) / se; hyperliquid drift = realdata.drift "
             "mu (per day); ostium drift = mean R per trade, se = sd / sqrt(n)",
    "outcome": "y = post drift > median post drift of the cut's evaluated accounts",
    "metric": "share y in the bottom decile of t_rel (deciles per cut among evaluated accounts), pooled over cuts; "
              "95 % interval by bootstrap over accounts (2000 resamples, rng seed 0; an account's rows from all cuts "
              "move together)",
    "success": "hl2: upper 95 % bound of the pooled bottom-decile share <= 0.30",
    "secondary": {"replications": "hl3, hl4, ostium and every cut alone, same metric and threshold (no verdict)",
                  "robust": "t_huber = realtrades.huber_drift mu / se (hyperliquid: rates r/dt weighted by dt; "
                            "ostium: per-trade R); pooled AUC of t_huber vs t_rel for y, paired account bootstrap",
                  "hypothesis_2_replication": "share y in the top decile of t_rel (seen in part by the project's "
                                              "benchmark: replication only, its confirmation belongs to the holdout)",
                  "follow_up": "pre-eligible accounts without an outcome, by pre decile of t_rel",
                  "counts": "eligible accounts per cut and dataset"},
    "known_biases": {"old_cuts": "samples come from today's leaderboard: dead accounts are included but traders who "
                                 "stopped long ago may be missing (2024-10 and 2025-04 cuts)",
                     "sparse_curves": "allTime points are sparser in the past: fewer eligible accounts in 2024",
                     "hl4": "activity-selected subset"},
    "privacy": "aggregates only in the repo; local copies deleted when the study is done",
    "rerun_of": {"study": "REAL-04",
                 "a_reason": "REAL-04 was declared NOT VALID by its own quality gate: G2 required portfolio files for 99 % of "
                             "HLS-04's 9,000 sampled accounts, but HLS-04 has 2,137 portfolios by design (documented subset). "
                             "Specification error of the gate, detected without computing any result",
                 "b_contamination": "REAL-05 (another metric: trade-level skill / behaviour, cut 2026-04-30) already showed "
                                    "outcomes of HLS-03, HLS-04 and Ostium accounts over the same post period as the "
                                    "2026-04-30 cut here; possible slight contamination, the laggard-persistence metric on "
                                    "curves has not been computed by the lab",
                 "c_unchanged": "everything else is REAL-04's protocol unchanged: data, cuts, windows, eligibility, score, "
                                "outcome, metric, threshold, secondaries; only G2 for hl4 differs",
                 "approved": "user, 2026-10-31"},
}
EXPECTED = {"hl2": 6000, "hl3": 9000, "hl4": 9000, "ostium": 25419}
PORTFOLIO_FILES = {"hl4": 2137}


def ms(day: str) -> int:
    return int(datetime.fromisoformat(day + "T00:00:00+00:00").timestamp() * 1000)


def sample_of(root):
    return realdata.load_json(os.path.join(root, "sample.json"))["sample"]


def load_series(ds, root, a):
    if ds == "ostium":
        p = os.path.join(root, "orders", f"{a}.json")
        return realtrades.ostium_trades(realdata.load_json(p)) if os.path.exists(p) else None
    p = os.path.join(root, "portfolio", f"{a}.json")
    if not os.path.exists(p):
        return None
    try:
        return realdata.hl_intervals(realdata.load_json(p))
    except (KeyError, ValueError, TypeError):
        return {"t0": np.array([]), "t1": np.array([]), "r": np.array([]), "dt": np.array([])}


def ostium_halt(root, ids) -> int | None:
    t = []
    for a in ids:
        p = os.path.join(root, "orders", f"{a}.json")
        if os.path.exists(p):
            t += [int(o.get("executedAt") or 0) for o in realdata.load_json(p) or [] if o.get("executedAt")]
    t = np.unique([x * 1000 if x < 10 ** 12 else x for x in t])
    t = t[(t >= ms("2026-01-01")) & (t <= ms(REOPEN) + DAY)]
    if t.size < 2:
        return None
    i = int(np.argmax(np.diff(t)))
    return int(t[i])


def gate(data_root, datasets):
    g, series, halt = {}, {}, None
    for ds in datasets:
        root = os.path.join(data_root, ds)
        sample = sample_of(root)
        ids = sample[1:]
        loaded = {a: load_series(ds, root, a) for a in ids}
        present = sum(v is not None for v in loaded.values())
        with_hist = sum(v is not None and (len(v) > 0 if isinstance(v, list) else v["r"].size > 0) for v in loaded.values())
        cal = load_series(ds, root, sample[0])
        if isinstance(cal, list):
            cal_info = {"trades": len(cal), "median_size_usdc": float(np.median([x[3] for x in cal])) if cal else None,
                        "sum_pnl_usdc": float(sum(x[2] for x in cal)) if cal else None}
        else:
            cal_info = {"intervals": int(cal["r"].size) if cal else 0,
                        "median_interval_days": float(np.median(cal["dt"])) if cal and cal["dt"].size else None}
        g[ds] = {"G1_frame": {"sample": len(sample), "expected": EXPECTED[ds], "ok": len(sample) == EXPECTED[ds]},
                 "G2_download": {"present": present, "of": PORTFOLIO_FILES.get(ds, len(ids)),
                                 "ok": present >= 0.99 * PORTFOLIO_FILES.get(ds, len(ids))},
                 "G3_data": {"with_history": with_hist, "ok": with_hist >= 0.10 * len(ids)},
                 "G5_units_calibration": {**cal_info, "ok": all(v is None or np.isfinite(v) for v in cal_info.values())}}
        series[ds] = loaded
        if ds == "ostium":
            halt = ostium_halt(root, ids)
            g[ds]["ostium_halt_start"] = None if halt is None else datetime.fromtimestamp(halt / 1000, timezone.utc).strftime("%Y-%m-%d")
    cuts = usable_cuts(datasets, halt)
    for ds in datasets:
        per = {}
        for c in cuts[ds]:
            cm = ms(c)
            before = after = 0
            for v in series[ds].values():
                if v is None:
                    continue
                tt = np.array([x[1] for x in v]) if isinstance(v, list) else v["t1"]
                before += bool((tt < cm).any()); after += bool((tt >= cm).any())
            per[c] = {"accounts_before": before, "accounts_after": after}
        g[ds]["G6_periods"] = {"per_cut": per, "ok": bool(cuts[ds]) and all(v["accounts_before"] > 0 and v["accounts_after"] > 0 for v in per.values())}
    ok = all(all(v["ok"] for k, v in g[ds].items() if isinstance(v, dict) and "ok" in v) for ds in datasets)
    return ok, g, series, cuts


def usable_cuts(datasets, halt):
    out = {}
    for ds in datasets:
        if ds != "ostium":
            out[ds] = list(CUTS)
        else:
            out[ds] = [c for c in CUTS if halt is None or not (ms(c) < ms(REOPEN) and ms(c) + 92 * DAY > halt)]
    return out


def rows_for(ds, series, cut):
    c = ms(cut)
    a, b = c - 182 * DAY, c + 92 * DAY
    pre_rows = []
    for acc, s in series.items():
        if s is None:
            continue
        if ds == "ostium":
            pre = [x for x in s if a <= x[1] < c]
            post = [x for x in s if c <= x[0] < b]
            R = realtrades.r_scale([x[2] for x in pre])
            if len(pre) < 10 or pre[-1][1] - pre[0][0] < 60 * DAY or R is None:
                continue
            rp = np.array([x[2] for x in pre]) / R
            d = {"mu": float(rp.mean()), "se": float(rp.std(ddof=1) / np.sqrt(rp.size))}
            h = realtrades.huber_drift(rp, np.ones(rp.size))
            out = None
            if len(post) >= 5 and post[-1][0] - post[0][0] >= 14 * DAY:
                out = float(np.mean([x[2] for x in post]) / R)
        else:
            w = {k: v[(s["t0"] >= a) & (s["t1"] <= c)] for k, v in s.items()}
            q = {k: v[(s["t0"] >= c) & (s["t1"] <= b)] for k, v in s.items()}
            d = realdata.drift(w["r"], w["dt"])
            if d is None or d["n"] < 10 or d["days"] < 60:
                continue
            h = realtrades.huber_drift(w["r"], w["dt"])
            out = float(q["r"].sum() / q["dt"].sum()) if q["r"].size >= 1 and q["dt"].sum() >= 14 else None
        if d["se"] <= 0 or h is None or h["se"] <= 0:
            continue
        pre_rows.append((acc, d["mu"], d["se"], h["mu"], h["se"], out))
    return pre_rows


def deciles(x):
    return np.minimum((np.argsort(np.argsort(x, kind="mergesort")) * 10) // len(x), 9)


def analyse(ds, series, cuts):
    allrows, per_cut, follow = [], {}, {}
    for cut in cuts:
        pr = rows_for(ds, series, cut)
        if len(pr) < 50:
            per_cut[cut] = {"eligible_pre": len(pr), "skipped": "fewer than 50 pre-eligible"}
            continue
        mu = np.array([r[1] for r in pr]); se = np.array([r[2] for r in pr])
        hm = np.array([r[3] for r in pr]); hs = np.array([r[4] for r in pr])
        t = (mu - np.median(mu)) / se
        th = (hm - np.median(hm)) / hs
        has = np.array([r[5] is not None for r in pr])
        dpre = deciles(t)
        follow[cut] = [round(float(1 - has[dpre == k].mean()), 4) for k in range(10)]
        if has.sum() < 50:
            per_cut[cut] = {"eligible_pre": len(pr), "with_outcome": int(has.sum()), "skipped": "fewer than 50 with outcome"}
            continue
        post = np.array([r[5] for r in pr if r[5] is not None])
        y = post > np.median(post)
        te, the = t[has], th[has]
        dec = deciles(te)
        acc = [r[0] for r, k in zip(pr, has) if k]
        per_cut[cut] = {"eligible_pre": len(pr), "with_outcome": int(has.sum()),
                        "bottom_share": round(float(y[dec == 0].mean()), 4), "top_share": round(float(y[dec == 9].mean()), 4)}
        allrows += [(a, int(d == 0), int(d == 9), float(yy), float(x1), float(x2)) for a, d, yy, x1, x2 in zip(acc, dec, y, te, the)]
    if not allrows:
        return {"per_cut": per_cut, "error": "no usable cut"}
    ids = sorted({r[0] for r in allrows})
    B = np.array([r[1] for r in allrows]); T = np.array([r[2] for r in allrows]); Y = np.array([r[3] for r in allrows])
    X1 = np.array([r[4] for r in allrows]); X2 = np.array([r[5] for r in allrows])
    pos = {a: [] for a in ids}
    for j, r in enumerate(allrows):
        pos[r[0]].append(j)
    pos = [np.array(pos[a]) for a in ids]
    rng = np.random.default_rng(0)
    bs_bottom, bs_top, bs_auc = [], [], []
    for _ in range(2000):
        pick = rng.integers(0, len(ids), len(ids))
        j = np.concatenate([pos[k] for k in pick])
        b, tt, yy = B[j], T[j], Y[j]
        if b.sum():
            bs_bottom.append(yy[b == 1].mean())
        if tt.sum():
            bs_top.append(yy[tt == 1].mean())
        if 0 < yy.sum() < yy.size:
            bs_auc.append(skill.auc(X2[j], yy == 1) - skill.auc(X1[j], yy == 1))
    q = lambda v: [round(float(x), 4) for x in np.quantile(v, [0.025, 0.975])]
    return {"per_cut": per_cut, "follow_up_lost_by_pre_decile": follow, "rows": len(allrows), "accounts": len(ids),
            "bottom_decile_share": {"share": round(float(Y[B == 1].mean()), 4), "n": int(B.sum()), "ci95": q(bs_bottom)},
            "top_decile_share": {"share": round(float(Y[T == 1].mean()), 4), "n": int(T.sum()), "ci95": q(bs_top)},
            "auc_t_rel": round(skill.auc(X1, Y == 1), 4), "auc_t_huber": round(skill.auc(X2, Y == 1), 4),
            "auc_huber_minus_t_ci95": q(bs_auc)}


def run(data_root, datasets=("hl2", "hl3", "hl4", "ostium"), out=print):
    ok, g, series, cuts = gate(data_root, datasets)
    out(json.dumps({"gate": g, "gate_ok": ok}))
    if not ok:
        return {"gate": g, "gate_ok": False, "valid": False, "success": False}
    res = {"gate": g, "gate_ok": True, "valid": True, "cuts_used": cuts}
    for ds in datasets:
        res[ds] = analyse(ds, series[ds], cuts[ds])
        out(json.dumps({ds: res[ds]}))
    h = res.get("hl2", {})
    res["success"] = bool("error" not in h and h["bottom_decile_share"]["ci95"][1] <= 0.30)
    return res


def fake_data(root, n=400, seed=9973, edge_sd=0.003):
    """Fake datasets with the documented structure (smoke / tests only): laggards persist by construction."""
    rng = np.random.default_rng(seed)
    t_a, t_b = ms("2024-01-01"), ms("2026-10-30")
    for ds in ("hl2", "hl3", "hl4", "ostium"):
        d = os.path.join(root, ds)
        os.makedirs(os.path.join(d, "orders" if ds == "ostium" else "portfolio"), exist_ok=True)
        ids = [f"{ds}_{i:05d}" for i in range(n)]
        json.dump({"sample": ids}, open(os.path.join(d, "sample.json"), "w"))
        for i in ids:
            mu = rng.normal(0, edge_sd)
            if ds == "ostium":
                ts = np.sort(rng.uniform(t_a, t_b, 300) / 1000).astype(int)
                ts = ts[(ts < ms("2026-05-10") / 1000) | (ts > ms(REOPEN) / 1000)]
                pp = (mu * 20 + rng.normal(0, 0.3, ts.size)) * 1e8
                orders = [{"orderAction": "Open", "executedAt": int(t) - 60, "collateral": "100000000"} for t in ts] + \
                         [{"orderAction": "Close", "executedAt": int(t), "collateral": "100000000", "profitPercent": str(int(p))} for t, p in zip(ts, pp)]
                json.dump(orders, open(os.path.join(d, "orders", f"{i}.json"), "w"))
            else:
                ts = np.sort(rng.uniform(t_a, t_b, 250)).astype(int)
                dt = np.diff(np.concatenate([[ts[0] - DAY], ts])) / DAY
                av = 10000 * np.cumprod(1 + mu * dt + rng.normal(0, 0.02, ts.size) * np.sqrt(dt)); pnl = av - 10000
                hh = lambda v: [[int(t), str(x)] for t, x in zip(ts, v)]
                json.dump([["allTime", {"accountValueHistory": hh(av), "pnlHistory": hh(pnl), "vlm": "0"}]],
                          open(os.path.join(d, "portfolio", f"{i}.json"), "w"))
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        global EXPECTED, PORTFOLIO_FILES
        EXPECTED = {k: 400 for k in EXPECTED}
        PORTFOLIO_FILES = {}
        with tempfile.TemporaryDirectory() as d:
            res = run(fake_data(d), out=lambda s: None)
        print(json.dumps({"smoke": "ok", "gate_ok": res["gate_ok"], "hl2_bottom": res.get("hl2", {}).get("bottom_decile_share"),
                          "ostium_cuts": res.get("cuts_used", {}).get("ostium")}))
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
        raise SystemExit("--data ROOT is required (ROOT/hl2, hl3, hl4, ostium, MANIFEST-checked)")
    res = run(args.data)
    with atomic_open(os.path.join(ROOT, "state", f"{STUDY}_report.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    reg.result(STUDY, {"run_commit": lock["commit"], **res})
    print(json.dumps({"valid": res["valid"], "success": res["success"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
