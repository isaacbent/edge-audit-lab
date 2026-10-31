"""REAL-05: the product on real trade-level data. A skill probability in R per trade from 150 trades, the SKILL-08
monitoring rule, martingale and tilt flags, and their combination, scored before a cut and checked after it. HLS-03
fills primary; Ostium orders and HLS-04 fills replications. Protocol approved by the user on 2026-10-31 and
registered before any data file was opened.

Usage:
  python3 -m tools.real_trades --smoke
  python3 -m tools.real_trades --register
  python3 -m tools.real_trades --data ROOT         # ROOT/{hl3,hl4,ostium}, each MANIFEST-checked
Outputs are aggregates only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
from scipy import stats as _st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import behaviour, realdata, realtrades, skill, stats, study  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "REAL-05"
DAY = realdata.DAY_MS
FILL_CAP = 10000
SPEC = {
    "question": "On real trades, (A) is the predictive skill probability from >= 150 trades precise and calibrated, "
                "(B) do martingale-flagged and (C) tilt-flagged accounts do worse afterwards, (D) how precise is the "
                "SKILL-08 monitoring rule, (E) does low skill + flagged behaviour mark the worst accounts?",
    "data": {"source": "isaacbenta/strategy-validator private branches (docs/DEV_DATA.md, funded-check commit "
                       "5b90ba69879fe84b2c837da303840cb4cbcb4757)",
             "hl3": {"role": "PRIMARY", "content": "9,000 fill files (<= 10,000 recent fills per account)",
                     "manifest": "research/hyperliquid3/MANIFEST.sha256 (sha256 8ed04fcbc8031571937a839a7c57e7853643869719b3a1109fab50344db9381b)",
                     "note": "accounts already used by the data owner for Edge Audit v1 (ROBUST / LUCK, outcome "
                             "July-October): the outcome period overlaps an analysis of another method"},
             "ostium": {"role": "replication", "content": "25,419 traders, orders",
                        "manifest": "research/ostium/MANIFEST.sha256 (sha256 41583801033675cb0862a581e8c5711a5dd1228d7bac23ce784c783b039587e1)"},
             "hl4": {"role": "replication", "content": "9,000 fill files from 2025-10-13",
                     "manifest": "research/hyperliquid4/MANIFEST.sha256 (sha256 c2b8abd991f0fd9e599b73218f894dd031ad17fbe0c7d351daabcb2d180d79a0)"}},
    "gate": {"rule": "docs/RESEARCH_PROTOCOL.md section 2, before any result; any failure = NOT VALID",
             "G1_frame": "sample size hl3 9,000, hl4 9,000, ostium 25,419",
             "G2_download": "fill / order files present for >= 99 % of sampled accounts",
             "G3_data": ">= 10 % of sampled accounts have at least one trade",
             "G4_truncation": "hyperliquid accounts with >= 10,000 fills are excluded from every analysis (counted, "
                              "with the median number of pre-cut trades they would have had)",
             "G5_units": "calibration account (first entry): derived trade count, median size and P&L printed only",
             "G6_periods": "accounts with trades before and after the cut exist; Ostium halt recorded"},
    "trades": "realtrades.hl_trades (flat -> flat per coin, net of fees, size = entry notional) / realtrades.ostium_trades "
              "(closing orders; P&L = profitPercent / 1e8 x collateral / 1e6; size = collateral / 1e6)",
    "cut": {"hl3": "2026-04-30, outcome to 2026-10-30", "hl4": "2026-04-30, outcome to 2026-10-30",
            "ostium": "halt start - 92 days, outcome to the halt start (halt start = beginning of the longest gap between "
                      "consecutive orders platform-wide in 2026, printed as a date only)"},
    "pre_post": "pre = trades ending before the cut; post = trades starting at or after the cut and before the outcome end",
    "r_scale": "median loss of the account's PRE trades (>= 5 losses), applied to pre and post trades",
    "A": {"population": "accounts with >= 150 pre trades and an R scale (prior fitted on them, no post condition)",
          "model": "skill.npmle_prob(pre R lists, thr=0.05); P_pred = skill.predictive_positive on the posterior shifted "
                   "by 0.05, se_f = pre sd / sqrt(n_post), i.e. P(post mean R > 0.05)",
          "evaluated": ">= 50 post trades", "flag": "P_pred >= 0.7", "outcome": "post mean R > 0.05",
          "criterion": "precision of the flag: lower 95 % bootstrap bound >= 0.60 (fewer than 20 flagged = not passed) "
                       "AND ECE of P_pred <= max(0.05, ece_noise95)"},
    "B": {"population": ">= 50 pre trades and an R scale", "flag": "behaviour.size_after_loss_p on pre trades (size = "
                                                                    "entry notional / collateral), Benjamini-Hochberg q 0.10",
          "outcome": "per-account post mean R, accounts with >= 20 post trades",
          "criterion": "median(flagged) - median(not flagged): upper 95 % bootstrap bound < 0 (fewer than 20 flagged "
                       "with outcome = not passed)"},
    "C": {"population": "as B", "test": "realtrades.tilt_counts (trades starting within 1 h after a trade ends), one-sided "
                                         "Welch t: more trades after losses than after wins; Benjamini-Hochberg q 0.10",
          "outcome": "as B", "criterion": "as B"},
    "D": {"rule": "SKILL-08: P(edge > 0.05 R) at 150, 175, ... pre trades with the A prior held fixed; flag at two "
                  "consecutive checkpoints >= 0.7", "outcome": "post mean R > 0.05 (>= 50 post trades)",
          "report": "precision with bootstrap interval, no criterion"},
    "E": {"groups": "P(edge > 0.05 R) < 0.2 AND (B or C flag) vs the rest of the A population with >= 20 post trades",
          "report": "median post mean R difference with bootstrap interval, no criterion"},
    "success": "hl3: A passes AND B passes (C, D, E and the replications are reported, no verdict)",
    "bootstrap": "2000 resamples of accounts, rng seed 0",
    "privacy": "aggregates only in the repo; local copies deleted when the study is done",
}
EXPECTED = {"hl3": 9000, "hl4": 9000, "ostium": 25419}
REOPEN = "2026-07-23"


def ms(day: str) -> int:
    return int(datetime.fromisoformat(day + "T00:00:00+00:00").timestamp() * 1000)


def load_trades(ds, root, a):
    """(trades or None if the file is missing, truncated flag)."""
    p = os.path.join(root, "orders" if ds == "ostium" else "fills", f"{a}.json")
    if not os.path.exists(p):
        return None, False
    raw = realdata.load_json(p) or []
    if ds == "ostium":
        return realtrades.ostium_trades(raw), False
    return realtrades.hl_trades(raw), len(raw) >= FILL_CAP


def ostium_halt(root, ids) -> int | None:
    t = []
    for a in ids:
        p = os.path.join(root, "orders", f"{a}.json")
        if os.path.exists(p):
            t += [int(o.get("executedAt") or 0) for o in realdata.load_json(p) or [] if o.get("executedAt")]
    t = np.unique([x * 1000 if x < 10 ** 12 else x for x in t])
    t = t[(t >= ms("2026-01-01")) & (t <= ms(REOPEN) + DAY)]
    return int(t[int(np.argmax(np.diff(t)))]) if t.size >= 2 else None


def windows(ds, halt):
    if ds == "ostium":
        return (halt - 92 * DAY, halt) if halt else (None, None)
    return ms("2026-04-30"), ms("2026-10-30")


def gate(data_root, datasets):
    g, data, win = {}, {}, {}
    for ds in datasets:
        root = os.path.join(data_root, ds)
        sample = realdata.load_json(os.path.join(root, "sample.json"))["sample"]
        loaded = {a: load_trades(ds, root, a) for a in sample[1:]}
        present = sum(v[0] is not None for v in loaded.values())
        any_tr = sum(bool(v[0]) for v in loaded.values())
        cal = load_trades(ds, root, sample[0])[0] or []
        halt = ostium_halt(root, sample[1:]) if ds == "ostium" else None
        c, e = windows(ds, halt)
        trunc = [v[0] for v in loaded.values() if v[1]]
        g[ds] = {"G1_frame": {"sample": len(sample), "expected": EXPECTED[ds], "ok": len(sample) == EXPECTED[ds]},
                 "G2_download": {"present": present, "of": len(loaded), "ok": present >= 0.99 * len(loaded)},
                 "G3_data": {"with_trades": any_tr, "ok": any_tr >= 0.10 * len(loaded)},
                 "G4_truncation": {"excluded": len(trunc), "ok": True,
                                   "median_pre_trades_lost": float(np.median([sum(x[1] < c for x in t) for t in trunc])) if trunc and c else None},
                 "G5_units_calibration": {"trades": len(cal), "median_size": float(np.median([x[3] for x in cal])) if cal else None,
                                          "sum_pnl": float(sum(x[2] for x in cal)) if cal else None, "ok": True},
                 "G6_periods": {"cut": None if c is None else datetime.fromtimestamp(c / 1000, timezone.utc).strftime("%Y-%m-%d"),
                                "end": None if e is None else datetime.fromtimestamp(e / 1000, timezone.utc).strftime("%Y-%m-%d"),
                                "accounts_before": sum(bool(v[0]) and any(x[1] < c for x in v[0]) for v in loaded.values()) if c else 0,
                                "accounts_after": sum(bool(v[0]) and any(c <= x[0] < e for x in v[0]) for v in loaded.values()) if c else 0}}
        g[ds]["G6_periods"]["ok"] = bool(c) and g[ds]["G6_periods"]["accounts_before"] > 0 and g[ds]["G6_periods"]["accounts_after"] > 0
        if ds == "ostium":
            g[ds]["ostium_halt_start"] = g[ds]["G6_periods"]["end"]
        data[ds] = {a: v[0] for a, v in loaded.items() if v[0] is not None and not v[1]}
        win[ds] = (c, e)
    ok = all(v["ok"] for ds in datasets for v in g[ds].values() if isinstance(v, dict) and "ok" in v)
    return ok, g, data, win


def boot_ci(fn, n, rng, reps=2000):
    vals = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        v = fn(i)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [round(float(q), 4) for q in np.quantile(vals, [0.025, 0.975])] if vals else None


def analyse(trades_by_acc, c, e):
    acc = []
    for a, tr in trades_by_acc.items():
        pre = [x for x in tr if x[1] < c]
        post = [x for x in tr if c <= x[0] < e]
        R = realtrades.r_scale([x[2] for x in pre])
        if R is None or len(pre) < 50:
            continue
        acc.append({"pre": pre, "R": R, "r_pre": np.array([x[2] for x in pre]) / R,
                    "r_post": np.array([x[2] for x in post]) / R, "size_pre": np.array([x[3] for x in pre])})
    out = {"accounts_50_pre": len(acc)}
    # B and C flags (>= 50 pre trades)
    pB = np.array([behaviour.size_after_loss_p(a["r_pre"], np.maximum(a["size_pre"], 1e-12)) for a in acc]) if acc else np.array([])
    pC = []
    for a in acc:
        al, aw = realtrades.tilt_counts(a["pre"])
        pC.append(float(_st.ttest_ind(al, aw, equal_var=False, alternative="greater").pvalue)
                  if al.size >= 3 and aw.size >= 3 and (al.std() > 0 or aw.std() > 0) else 1.0)
    pC = np.nan_to_num(np.array(pC), nan=1.0)
    fB = stats.benjamini_hochberg(pB, q=0.10) if acc else np.array([], dtype=bool)
    fC = stats.benjamini_hochberg(pC, q=0.10) if acc else np.array([], dtype=bool)
    rng = np.random.default_rng(0)
    post_mean = np.array([a["r_post"].mean() if a["r_post"].size >= 20 else np.nan for a in acc])

    def group_diff(flag, label):
        m = ~np.isnan(post_mean)
        f, v = flag[m], post_mean[m]
        res = {"flagged": int(flag.sum()), "flagged_with_outcome": int(f.sum()), "not_flagged_with_outcome": int((~f).sum())}
        if f.sum() < 20 or (~f).sum() < 20:
            return {**res, "passed": False, "note": "fewer than 20 in a group"}
        diff = float(np.median(v[f]) - np.median(v[~f]))
        ci = boot_ci(lambda i: (np.median(v[i][f[i]]) - np.median(v[i][~f[i]])) if f[i].any() and (~f[i]).any() else None, len(v), rng)
        return {**res, "median_flagged": round(float(np.median(v[f])), 4), "median_rest": round(float(np.median(v[~f])), 4),
                "diff": round(diff, 4), "ci95": ci, "passed": bool(ci and ci[1] < 0)}

    out["B_martingale"] = group_diff(fB, "B")
    out["C_tilt"] = group_diff(fC, "C")
    # A: >= 150 pre trades
    A = [i for i, a in enumerate(acc) if a["r_pre"].size >= 150]
    out["A_population"] = len(A)
    if len(A) >= 20:
        fit = skill.npmle_prob([acc[i]["r_pre"] for i in A], thr=0.05)
        grid, prior = fit["grid"], fit["prior"]
        ev = [k for k, i in enumerate(A) if acc[i]["r_post"].size >= 50]
        sd = np.array([acc[A[k]]["r_pre"].std(ddof=1) for k in ev])
        npost = np.array([acc[A[k]]["r_post"].size for k in ev])
        pp = skill.predictive_positive(fit["post"][ev], grid - 0.05, sd / np.sqrt(npost)) if ev else np.array([])
        y = np.array([acc[A[k]]["r_post"].mean() > 0.05 for k in ev])
        flag = pp >= 0.7
        resA = {"evaluated": len(ev), "flagged": int(flag.sum()), "share_outcome": round(float(y.mean()), 4) if ev else None}
        if ev:
            dec = np.minimum((np.argsort(np.argsort(pp, kind="mergesort")) * 10) // len(pp), 9)
            gaps = [abs(pp[dec == d].mean() - y[dec == d].mean()) for d in range(10) if (dec == d).any()]
            rs = np.random.default_rng(1)
            noise = [np.mean([abs(pp[dec == d].mean() - ys[dec == d].mean()) for d in range(10) if (dec == d).any()])
                     for ys in (rs.random(len(pp)) < pp for _ in range(2000))]
            resA.update(ece=round(float(np.mean(gaps)), 4), ece_noise95=round(float(np.quantile(noise, 0.95)), 4),
                        auc=round(skill.auc(pp, y), 4) if 0 < y.sum() < y.size else None)
            if flag.sum():
                yf = y[flag]
                resA.update(precision=round(float(yf.mean()), 4), precision_ci95=boot_ci(lambda i: yf[i].mean(), len(yf), rng),
                            mean_p_flagged=round(float(pp[flag].mean()), 4))
            resA["passed"] = bool(flag.sum() >= 20 and resA.get("precision_ci95") and resA["precision_ci95"][0] >= 0.60
                                  and resA["ece"] <= max(0.05, resA["ece_noise95"]))
        else:
            resA["passed"] = False
        out["A_skill"] = resA
        # D: SKILL-08 monitoring with the A prior fixed
        def p_at(r):
            xb, se = r.mean(), max(r.std(ddof=1), 1e-12) / np.sqrt(r.size)
            lik = _st.norm.pdf((xb - grid) / se) * prior
            return float(lik[grid > 0.05].sum() / max(lik.sum(), 1e-300))
        dflag, dy = [], []
        for k, i in enumerate(A):
            r = acc[i]["r_pre"]
            if acc[i]["r_post"].size < 50:
                continue
            ps = [p_at(r[:n]) for n in range(150, r.size + 1, 25)]
            dflag.append(any(a >= 0.7 and b >= 0.7 for a, b in zip(ps[:-1], ps[1:])))
            dy.append(acc[i]["r_post"].mean() > 0.05)
        dflag, dy = np.array(dflag, dtype=bool), np.array(dy, dtype=bool)
        out["D_monitoring"] = {"evaluated": int(dflag.size), "flagged": int(dflag.sum()),
                               **({"precision": round(float(dy[dflag].mean()), 4),
                                   "precision_ci95": boot_ci(lambda i: dy[dflag][i].mean(), int(dflag.sum()), rng)} if dflag.sum() else {})}
        # E
        pe = np.full(len(acc), np.nan)
        pe[A] = fit["prob"]
        low = (pe < 0.2) & (fB | fC)
        m = ~np.isnan(pe) & ~np.isnan(post_mean)
        if (low & m).sum() >= 10 and (~low & m).sum() >= 10:
            v, f = post_mean[m], low[m]
            out["E_combined"] = {"n_low_flagged": int(f.sum()), "n_rest": int((~f).sum()),
                                 "diff_median": round(float(np.median(v[f]) - np.median(v[~f])), 4),
                                 "ci95": boot_ci(lambda i: (np.median(v[i][f[i]]) - np.median(v[i][~f[i]])) if f[i].any() and (~f[i]).any() else None, len(v), rng)}
        else:
            out["E_combined"] = {"note": "fewer than 10 in a group", "n_low_flagged": int((low & m).sum())}
    else:
        out["A_skill"] = {"passed": False, "note": "fewer than 20 accounts with >= 150 pre trades"}
    return out


def run(data_root, datasets=("hl3", "ostium", "hl4"), out=print):
    ok, g, data, win = gate(data_root, datasets)
    out(json.dumps({"gate": g, "gate_ok": ok}))
    if not ok:
        return {"gate": g, "gate_ok": False, "valid": False, "success": False}
    res = {"gate": g, "gate_ok": True, "valid": True}
    for ds in datasets:
        res[ds] = analyse(data[ds], *win[ds])
        out(json.dumps({ds: res[ds]}))
    h = res["hl3"]
    res["success"] = bool(h.get("A_skill", {}).get("passed") and h.get("B_martingale", {}).get("passed"))
    return res


def fake_data(root, n=300, seed=9975, mart_every=10):
    """Fake fills / orders with the documented structure (smoke only); martingale accounts lose afterwards."""
    rng = np.random.default_rng(seed)
    t_a, t_b = ms("2025-10-14"), ms("2026-10-30")
    for ds in ("hl3", "hl4", "ostium"):
        d = os.path.join(root, ds)
        os.makedirs(os.path.join(d, "orders" if ds == "ostium" else "fills"), exist_ok=True)
        ids = [f"{ds}_{i:05d}" for i in range(n)]
        json.dump({"sample": ids}, open(os.path.join(d, "sample.json"), "w"))
        for k, i in enumerate(ids):
            mart = mart_every > 0 and k % mart_every == 0
            mu = -0.3 if mart else rng.normal(0, 0.15)
            ts = np.sort(rng.uniform(t_a, t_b, 400)).astype(int)
            ts = ts[(ts < ms("2026-05-10")) | (ts > ms(REOPEN))] if ds == "ostium" else ts
            res = mu + rng.normal(0, 1, ts.size)
            size = np.ones(ts.size)
            for j in range(1, ts.size):
                size[j] = min(size[j - 1] * 2, 8) if (mart and res[j - 1] < 0) else 1.0
            size *= np.exp(0.3 * rng.standard_normal(ts.size))
            if ds == "ostium":
                orders = []
                for t, r, s in zip(ts, res, size):
                    orders.append({"orderAction": "Open", "executedAt": int(t // 1000) - 30, "collateral": str(int(s * 1e8))})
                    orders.append({"orderAction": "Close", "executedAt": int(t // 1000), "collateral": str(int(s * 1e8)),
                                   "profitPercent": str(int(r * 0.01 * 1e8))})
                json.dump(orders, open(os.path.join(d, "orders", f"{i}.json"), "w"))
            else:
                fills = []
                for t, r, s in zip(ts, res, size):
                    fills.append({"time": int(t) - 1000, "coin": "BTC", "side": "B", "sz": str(s), "px": "100", "closedPnl": "0", "fee": "0"})
                    fills.append({"time": int(t), "coin": "BTC", "side": "A", "sz": str(s), "px": "100", "closedPnl": str(r * s), "fee": "0"})
                json.dump(fills, open(os.path.join(d, "fills", f"{i}.json"), "w"))
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        global EXPECTED
        EXPECTED = {k: 300 for k in EXPECTED}
        with tempfile.TemporaryDirectory() as d:
            res = run(fake_data(d), out=lambda s: None)
        h = res.get("hl3", {})
        print(json.dumps({"smoke": "ok", "gate_ok": res["gate_ok"], "A": h.get("A_skill"), "B": h.get("B_martingale"),
                          "D": h.get("D_monitoring"), "ostium_halt": res["gate"]["ostium"].get("ostium_halt_start")}))
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
        raise SystemExit("--data ROOT is required (ROOT/hl3, hl4, ostium, MANIFEST-checked)")
    res = run(args.data)
    with atomic_open(os.path.join(ROOT, "state", f"{STUDY}_report.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    reg.result(STUDY, {"run_commit": lock["commit"], **res})
    print(json.dumps({"valid": res["valid"], "success": res["success"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
