"""REAL-03 (prospective, second arm next to REAL-02): do accounts that lag their peers keep lagging, and does a
median-corrected probability calibrate? Same dates and data as REAL-02. Idea from the exploratory, peer-relative look
at the REAL-01 data (bottom decile by relative t-stat: 10 % beat the median afterwards); predictions frozen on
2026-10-30, before any outcome exists.

Usage:
  python3 -m tools.real_prospective_v2 --smoke
  python3 -m tools.real_prospective_v2 --register
  python3 -m tools.real_prospective_v2 --predict --data PKG
  python3 -m tools.real_prospective_v2 --evaluate --data PKG --new-data NEWPKG     # after 2027-01-30, from the
                                                                                   # registered commit
Outputs are aggregates only.
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
from scipy import stats as _st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import realdata, skill, study  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "REAL-03"
WIN_START, CUT, END = "2026-04-30T00:00:00+00:00", "2026-10-30T00:00:00+00:00", "2027-01-30T00:00:00+00:00"
SPEC = {
    "question": "Prospective: do the accounts with the lowest relative t-stat keep lagging the median of their peers over "
                "2026-10-30 to 2027-01-30? Secondary: calibration of a median-corrected NPMLE probability; OKX reversal",
    "data": "as REAL-02 (predictions from realdata_2026-10-30, zip sha256 0b851ea48bb28a2b887ed20e8f7154fef88a43576756e5437c57ef7a6c2240ff, "
            "data up to the cut only; outcomes from a new package with data to >= 2027-01-30, MANIFESTs checked)",
    "window": {"pre": [WIN_START, CUT], "post": [CUT, END]},
    "accounts_and_eligibility": "as REAL-02 (Hyperliquid sample minus first address, >= 10 intervals over >= 60 days in "
                                "the pre window; OKX leads minus first trader, >= 60 days)",
    "scores": {"t_rel": "(mu_i - median mu) / se_i from realdata.drift over the pre window",
               "p_corr": "NPMLE posterior of x_i = mu_i - median mu (skill.npmle_summary, thr 0); P_i(m) = sum_g post_ig "
                         "Phi((g - m) / se_f_i), se_f_i = sigma_i / sqrt(92); m = the value making mean_i P_i(m) = 0.5 "
                         "(median of the model's own predictive distribution; bisection); p_corr = P_i(m)"},
    "frozen_predictions": "sha256 of canonical JSON {dataset: {id: [p_corr, t_rel]}} stored in the registry by --predict",
    "outcome": "y_i = post-window drift > median post-window drift of the evaluated accounts (outcome rules as REAL-02)",
    "success": "hyperliquid: among the bottom decile of t_rel (evaluated accounts), share with y = 1 has an upper 95 % "
               "bootstrap bound (2000 resamples of accounts, rng seed 0) <= 0.30",
    "secondary": {"calibration": "ECE of p_corr vs ece_noise95 (as REAL-01)", "auc": "AUC of p_corr and of t_rel",
                  "top_decile": "share y among the top decile of t_rel (hyperliquid and okx); okx hypothesis from the "
                                "exploration: below 0.5 (reversal)", "bottom_decile_okx": "share y, bottom decile of t_rel"},
    "evaluation_procedure": "as REAL-02: registered commit, recompute predictions, verify sha256, then outcomes",
    "notes": "hypotheses come from an exploratory look at already-opened data (registry note 'REAL-02 exploratory'); "
             "only the prospective outcomes can confirm them",
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


def corrected_p(post, grid, se_f):
    def pm(m):
        return (post * _st.norm.cdf((np.asarray(grid)[None, :] - m) / se_f[:, None])).sum(axis=1)
    lo, hi = float(np.min(grid)) - 10 * se_f.max(), float(np.max(grid)) + 10 * se_f.max()
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if pm(mid).mean() > 0.5 else (lo, mid)
    return pm(0.5 * (lo + hi))


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
        p = corrected_p(fit["post"], fit["grid"], sig / np.sqrt(92.0))
        out[ds] = {i: [round(float(pi), 6), round(float(t), 6)] for i, pi, t in zip(ids, p, x / se)}
    return out


def digest(pred: dict) -> str:
    return hashlib.sha256(json.dumps(pred, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summary(pred: dict) -> dict:
    out = {}
    for ds, d in pred.items():
        p = np.array([v[0] for v in d.values()]) if d else np.array([])
        out[ds] = {"n": int(p.size), **({"p_mean": round(float(p.mean()), 4), "p_q10": round(float(np.quantile(p, 0.1)), 4),
                                         "p_q90": round(float(np.quantile(p, 0.9)), 4)} if p.size else {})}
    return out


def _share_ci(y, idx_mask, rng):
    yy = y[idx_mask]
    if yy.size == 0:
        return None
    bs = [yy[rng.integers(0, yy.size, yy.size)].mean() for _ in range(2000)]
    return {"share": round(float(yy.mean()), 4), "n": int(yy.size), "ci95": [round(float(q), 4) for q in np.quantile(bs, [0.025, 0.975])]}


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
            if w["r"].size < (30 if ds == "okx" else 1) or w["dt"].sum() < 14:
                continue
            rows.append((d[i][0], d[i][1], float(w["r"].sum() / w["dt"].sum())))
        if len(rows) < 20:
            res[ds] = {"error": "fewer than 20 accounts with outcome", "eligible_pre": len(d)}
            continue
        p, t, post = map(np.array, zip(*rows))
        y = (post > np.median(post)).astype(float)
        rng = np.random.default_rng(0)
        tdec = np.minimum((np.argsort(np.argsort(t, kind="mergesort")) * 10) // len(t), 9)
        pdec = np.minimum((np.argsort(np.argsort(p, kind="mergesort")) * 10) // len(p), 9)
        gaps = [abs(p[pdec == k].mean() - y[pdec == k].mean()) for k in range(10)]
        rs = np.random.default_rng(1)
        noise = [np.mean([abs(p[pdec == k].mean() - ys[pdec == k].mean()) for k in range(10)])
                 for ys in (rs.random(len(p)) < p for _ in range(2000))]
        res[ds] = {"eligible_pre": len(d), "with_outcome": len(rows),
                   "bottom_decile_t": _share_ci(y, tdec == 0, rng), "top_decile_t": _share_ci(y, tdec == 9, rng),
                   "ece": round(float(np.mean(gaps)), 4), "ece_noise95": round(float(np.quantile(noise, 0.95)), 4),
                   "auc_p": round(skill.auc(p, y), 4), "auc_t": round(skill.auc(t, y), 4)}
    h = res.get("hyperliquid", {})
    res["success"] = bool("error" not in h and h and h["bottom_decile_t"] and h["bottom_decile_t"]["ci95"][1] <= 0.30)
    return res


def fake_package(root, n=300, seed=9972, end=END):
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
        print(json.dumps({"smoke": "ok", "pred": summary(pred), "hl_bottom": res["hyperliquid"].get("bottom_decile_t")}))
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
            raise SystemExit("predictions already frozen for REAL-03")
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
