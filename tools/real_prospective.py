"""REAL-02 (prospective): will an account beat the median of its peers over the next 3 months? Predictions are made
on 2026-10-30 from data up to that day (package realdata_2026-10-30), frozen (sha256 in the registry), and evaluated
only when data up to 2027-01-30 exists. Follows REAL-01, which showed that 'positive returns' measures the market,
not the trader.

Usage:
  python3 -m tools.real_prospective --smoke
  python3 -m tools.real_prospective --register
  python3 -m tools.real_prospective --predict --data PKG                 # now: freeze the predictions (hash only)
  python3 -m tools.real_prospective --evaluate --data PKG --new-data NEWPKG   # after 2027-01-30, from the
                                                                            # registered commit (git checkout)
Outputs are aggregates only; per-account predictions are never written to the repo (package README: no addresses).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import realdata, skill, study  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "REAL-02"
WIN_START, CUT, END = "2026-04-30T00:00:00+00:00", "2026-10-30T00:00:00+00:00", "2027-01-30T00:00:00+00:00"
SPEC = {
    "question": "Prospective: is the NPMLE predictive probability that an account's drift beats the median of its peers "
                "over 2026-10-30 to 2027-01-30 calibrated, better than chance, and at least as good as the t-stat?",
    "prediction_data": {"package": "realdata_2026-10-30 (isaacbenta/strategy-validator, branch realdata-2026-10-30, "
                                   "commit 5862985824d4f358fc4153407f2cc2db2733532a)",
                        "zip_sha256": "0b851ea48bb28a2b887ed20e8f7154fef88a43576756e5437c57ef7a6c2240ff",
                        "note": "already opened for REAL-01; only data up to the cut 2026-10-30 is used here, and the "
                                "outcomes (after the cut) did not exist when this was registered"},
    "outcome_data": "a new package with the same structure (same Hyperliquid sample addresses, same OKX lead traders), "
                    "data up to at least 2027-01-30, integrity checked with its MANIFESTs; same parsers",
    "window": {"pre": [WIN_START, CUT], "post": [CUT, END]},
    "datasets": {"hyperliquid": {"role": "PRIMARY", "accounts": "sample.json 'sample', first address excluded",
                                 "returns": "realdata.hl_intervals (allTime, start value >= 100 USDC)",
                                 "eligible_pre": ">= 10 intervals spanning >= 60 days inside the pre window"},
                 "okx": {"role": "SECONDARY (current lead traders: survivorship bias)",
                         "accounts": "lead_traders.json, first trader excluded", "returns": "realdata.okx_daily",
                         "eligible_pre": ">= 60 days of returns inside the pre window"}},
    "relative_drift": "x_i = realdata.drift(pre-window intervals).mu minus the median of x over the dataset's eligible "
                      "accounts; se_i from the same fit",
    "model": "skill.npmle_summary(x, se, thr=0); P_beat = skill.predictive_positive(post, grid, sigma_i / sqrt(92))",
    "frozen_predictions": "--predict computes P_beat and the pre-window relative t-stat for every eligible account and "
                          "stores only the sha256 of the canonical JSON {id: [P_beat, t_rel]} in the registry",
    "outcome": "y_i = post-window drift of account i > median post-window drift of the evaluated accounts (accounts "
               "eligible pre and with >= 14 post-window days; Hyperliquid also >= 1 interval, OKX >= 30 daily returns)",
    "metrics": {"calibration": "ECE over deciles of P_beat and ece_noise95 (as REAL-01)",
                "auc": "AUC(P_beat), AUC(t_rel), paired bootstrap of AUC(P_beat) - AUC(t_rel) (2000, rng seed 0); "
                       "bootstrap interval of AUC(P_beat) - 0.5",
                "counts": "eligible pre, with outcome, lost to follow-up"},
    "success": "hyperliquid, all of: (1) ECE <= max(0.05, ece_noise95) AND (2) AUC(P_beat) - 0.5 lower 95 % bound > 0 "
               "AND (3) AUC(P_beat) - AUC(t_rel) lower 95 % bound >= -0.01",
    "evaluation_procedure": "check out the commit recorded at registration (code lock), recompute the predictions, "
                            "verify their sha256 against the registry, then compute outcomes; any mismatch = stop",
    "notes": "designed after REAL-01 (2026-07-30 cut) showed a market-wide drift; registered before any exploratory "
             "peer-relative look at the REAL-01 data; tests the method on account series, not the product thresholds",
}


def ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def window(series, a_ms, b_ms):
    m = (series["t0"] >= a_ms) & (series["t1"] <= b_ms)
    return {k: v[m] for k, v in series.items()}


def series_by_id(root):
    """{dataset: [(id, series or None)]} in README order, first entry excluded."""
    hl = os.path.join(root, "data", "hyperliquid")
    ids = realdata.load_json(os.path.join(hl, "sample.json"))["sample"][1:]
    out = {"hyperliquid": [], "okx": []}
    for a in ids:
        p = os.path.join(hl, "portfolio", f"{a}.json")
        out["hyperliquid"].append((a, realdata.hl_intervals(realdata.load_json(p)) if os.path.exists(p) else None))
    ok = os.path.join(root, "data", "okx")
    leads = realdata.load_json(os.path.join(ok, "lead_traders.json"))
    leads = leads["data"] if isinstance(leads, dict) and "data" in leads else leads
    for c in [x["uniqueCode"] if isinstance(x, dict) else str(x) for x in leads][1:]:
        p = os.path.join(ok, "pnl", f"{c}.json")
        out["okx"].append((c, realdata.okx_daily(realdata.load_json(p)) if os.path.exists(p) else None))
    return out


def predictions(root) -> dict:
    a, b = ms(WIN_START), ms(CUT)
    rules = {"hyperliquid": (10, 60.0), "okx": (60, 60.0)}
    out = {}
    for ds, items in series_by_id(root).items():
        ids, mu, se, sig = [], [], [], []
        for i, s in items:
            if s is None:
                continue
            w = window(s, a, b)
            d = realdata.drift(w["r"], w["dt"])
            if d is None or d["n"] < rules[ds][0] or d["days"] < rules[ds][1] or d["se"] <= 0:
                continue
            ids.append(i); mu.append(d["mu"]); se.append(d["se"]); sig.append(d["sigma"])
        if len(ids) < 20:
            out[ds] = {}
            continue
        mu, se, sig = map(np.array, (mu, se, sig))
        x = mu - np.median(mu)
        fit = skill.npmle_summary(x, se, thr=0.0)
        pb = skill.predictive_positive(fit["post"], fit["grid"], sig / np.sqrt(92.0))
        out[ds] = {i: [round(float(p), 6), round(float(t), 6)] for i, p, t in zip(ids, pb, x / se)}
    return out


def digest(pred: dict) -> str:
    return hashlib.sha256(json.dumps(pred, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summary(pred: dict) -> dict:
    out = {}
    for ds, d in pred.items():
        p = np.array([v[0] for v in d.values()]) if d else np.array([])
        out[ds] = {"n": int(p.size), **({"p_mean": round(float(p.mean()), 4), "p_q10": round(float(np.quantile(p, 0.1)), 4),
                                         "p_q90": round(float(np.quantile(p, 0.9)), 4), "share_p07": round(float(np.mean(p >= 0.7)), 4)}
                                        if p.size else {})}
    return out


def evaluate(pred: dict, new_root: str) -> dict:
    c, e = ms(CUT), ms(END)
    res = {}
    for ds, items in series_by_id(new_root).items():
        d = pred.get(ds, {})
        rows = []
        for i, s in items:
            if i not in d or s is None:
                continue
            w = window(s, c, e)
            need = 30 if ds == "okx" else 1
            if w["r"].size < need or w["dt"].sum() < 14:
                continue
            rows.append((d[i][0], d[i][1], float(w["r"].sum() / w["dt"].sum())))
        if len(rows) < 20:
            res[ds] = {"error": "fewer than 20 accounts with outcome", "eligible_pre": len(d)}
            continue
        p, t, post = map(np.array, zip(*rows))
        y = post > np.median(post)
        dec = np.minimum((np.argsort(np.argsort(p, kind="mergesort")) * 10) // len(p), 9)
        gaps = [abs(p[dec == k].mean() - y[dec == k].mean()) for k in range(10)]
        rs = np.random.default_rng(1)
        noise = [np.mean([abs(p[dec == k].mean() - ys[dec == k].mean()) for k in range(10)])
                 for ys in (rs.random(len(p)) < p for _ in range(2000))]
        rb = np.random.default_rng(0)
        da, dt_ = [], []
        for _ in range(2000):
            j = rb.integers(0, len(y), len(y))
            if y[j].all() or not y[j].any():
                continue
            a_p = skill.auc(p[j], y[j])
            da.append(a_p - 0.5); dt_.append(a_p - skill.auc(t[j], y[j]))
        res[ds] = {"eligible_pre": len(d), "with_outcome": len(rows), "ece": round(float(np.mean(gaps)), 4),
                   "ece_noise95": round(float(np.quantile(noise, 0.95)), 4), "max_decile_gap": round(float(max(gaps)), 4),
                   "auc": round(skill.auc(p, y), 4), "auc_t_rel": round(skill.auc(t, y), 4),
                   "auc_minus_half_ci95": [round(float(q), 4) for q in np.quantile(da, [0.025, 0.975])],
                   "auc_minus_t_ci95": [round(float(q), 4) for q in np.quantile(dt_, [0.025, 0.975])]}
    h = res.get("hyperliquid", {})
    res["success"] = bool("error" not in h and h and h["ece"] <= max(0.05, h["ece_noise95"])
                          and h["auc_minus_half_ci95"][0] > 0 and h["auc_minus_t_ci95"][0] >= -0.01)
    return res


def fake_package(root, n=300, seed=9971, end=END):
    """Fake package with the README's structure covering WIN_START..end (smoke / tests only)."""
    rng = np.random.default_rng(seed)
    hl, ok = os.path.join(root, "data", "hyperliquid"), os.path.join(root, "data", "okx")
    os.makedirs(os.path.join(hl, "portfolio"), exist_ok=True); os.makedirs(os.path.join(ok, "pnl"), exist_ok=True)
    addrs = [f"0x{i:040x}" for i in range(n)]
    json.dump({"universe": n, "sample": addrs}, open(os.path.join(hl, "sample.json"), "w"))
    t_a, t_b = ms(WIN_START) - 30 * realdata.DAY_MS, ms(end)
    market = 0.002
    for k, a in enumerate(addrs):
        mu = market + rng.normal(0, 0.003)
        ts = np.sort(rng.uniform(t_a, t_b, 150)).astype(int)
        dt = np.diff(np.concatenate([[ts[0] - realdata.DAY_MS], ts])) / realdata.DAY_MS
        av = 10000 * np.cumprod(1 + mu * dt + rng.normal(0, 0.02, ts.size) * np.sqrt(dt)); pnl = av - 10000
        h = lambda v: [[int(t), str(x)] for t, x in zip(ts, v)]
        json.dump([["allTime", {"accountValueHistory": h(av), "pnlHistory": h(pnl), "vlm": "0"}]],
                  open(os.path.join(hl, "portfolio", f"{a}.json"), "w"))
    codes = [f"C{i}" for i in range(n)]
    json.dump([{"uniqueCode": c} for c in codes], open(os.path.join(ok, "lead_traders.json"), "w"))
    days = np.arange(t_a, t_b, realdata.DAY_MS)
    for c in codes:
        mu = market + rng.normal(0, 0.002)
        R = np.cumprod(1 + mu + rng.normal(0, 0.01, days.size)) - 1
        json.dump([{"beginTs": str(int(t)), "pnl": "0", "pnlRatio": str(x)} for t, x in zip(days, R)][::-1],
                  open(os.path.join(ok, "pnl", f"{c}.json"), "w"))
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--new-data", default=None)
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        with tempfile.TemporaryDirectory() as d:
            fake_package(d)
            pred = predictions(d)
            res = evaluate(pred, d)
        print(json.dumps({"smoke": "ok", "pred": summary(pred), "hl_auc": res["hyperliquid"].get("auc")}))
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
        raise SystemExit("--data (the realdata_2026-10-30 package root) is required")
    pred = predictions(args.data)
    h = digest(pred)
    if args.predict:
        if any(r["kind"] == "note" and r["payload"].get("study") == STUDY and r["payload"].get("kind") == "frozen_predictions"
               for r in reg.records()):
            raise SystemExit("predictions already frozen for REAL-02")
        reg.note({"study": STUDY, "kind": "frozen_predictions", "sha256": h, "summary": summary(pred), "commit": lock["commit"]})
        print(json.dumps({"frozen_sha256": h, "summary": summary(pred)}))
        return 0
    if args.evaluate:
        frozen = [r["payload"]["sha256"] for r in reg.records() if r["kind"] == "note"
                  and r["payload"].get("study") == STUDY and r["payload"].get("kind") == "frozen_predictions"]
        if not frozen or frozen[0] != h:
            raise SystemExit("recomputed predictions do not match the frozen sha256: refusing to evaluate")
        if not args.new_data:
            raise SystemExit("--new-data (package with data to 2027-01-30, MANIFESTs checked) is required")
        res = evaluate(pred, args.new_data)
        with atomic_open(os.path.join(ROOT, "state", f"{STUDY}_report.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        reg.result(STUDY, {"run_commit": lock["commit"], "predictions_sha256": h, **res})
        print(json.dumps(res))
        return 0
    raise SystemExit("choose --predict or --evaluate")


if __name__ == "__main__":
    sys.exit(main())
