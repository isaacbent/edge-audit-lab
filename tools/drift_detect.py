"""DRIFT-01: can a firm detect, as trades arrive, that a funded trader has lost their edge (CUSUM vs EWMA), and at
what cost in false alarms? ROADMAP phase 3, step 9.

Usage:
  python3 -m tools.drift_detect --smoke
  python3 -m tools.drift_detect --register
  python3 -m tools.drift_detect
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import drift, simulate, study  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "DRIFT-01"
SPEC = {
    "question": "Detection rate and delay of an edge loss in funded traders (CUSUM vs EWMA on trade results), at a "
                "fixed false-alarm rate",
    "bench": "BENCH-3D: simulate.population_drift (all traders start with a real edge: BENCH-3 edge distribution "
             "truncated to > 0.10 R; decay_share 0.5 switch to -0.05 R at trade tau ~ U[150, 300]; own win-rate style)",
    "population": {"n_traders": 2000, "n_trades": 400, "decay_share": 0.5, "tau_range": [150, 300], "edge_min": 0.10,
                   "post_mu": -0.05, "risk_pct": 1.0},
    "warmup": 100,
    "monitoring": "trades 100-399 (300 trades); each trader standardised by the mean / sd of their first 100 trades",
    "detectors": {"CUSUM": {"k": 0.1, "why_k": "half of a 0.2 sd drop (edge 0.15-0.25 R to -0.05 R, trade sd about 1.2 R)"},
                  "EWMA": {"lam": 0.05}},
    "calibration": "thresholds (CUSUM h, EWMA L) set so that 10 % of in-control traders alarm within the 300 monitored "
                   "trades, on calibration seeds 8100-8109 with decay_share 0 (simulation-based, no evaluation data)",
    "calibration_seeds": list(range(8100, 8110)),
    "seeds": list(range(8000, 8020)),
    "metrics": {"false_alarm": "stable traders with any alarm (pooled over seeds, seed-bootstrap 95 % interval)",
                "early_alarm": "decaying traders alarmed before tau",
                "detect150": "decaying traders whose first alarm is in [tau, tau + 150)",
                "detect_any": "decaying traders whose first alarm is >= tau (before trade 400)",
                "delay": "first alarm - tau among detected decaying traders: median and quartiles",
                "by_edge": "detect150 by mu_pre band 0.10-0.15, 0.15-0.25, > 0.25"},
    "success": "CUSUM, all of: detect150 pooled ratio lower 95 % bound >= 0.50 AND false_alarm upper 95 % bound <= 0.15",
    "secondary": "EWMA same metrics; per-seed paired difference detect150(CUSUM) - detect150(EWMA)",
    "notes": "time unit = trades (at about 1 trade per day, 300 monitored trades = about 1.2 years); thresholds "
             "calibrated on the bench's in-control model, on real data they would come from each trader's warm-up",
}
BANDS = [[0.10, 0.15], [0.15, 0.25], [0.25, None]]


def population(spec, seed, decay_share=None):
    p = spec["population"]
    return simulate.population_drift(p["n_traders"], p["n_trades"], seed,
                                     decay_share=p["decay_share"] if decay_share is None else decay_share,
                                     tau_range=tuple(p["tau_range"]), edge_min=p["edge_min"], post_mu=p["post_mu"],
                                     risk=p["risk_pct"])


def detectors(spec):
    w, d = spec["warmup"], spec["detectors"]
    return {"CUSUM": lambda z, h: drift.cusum_down(z, d["CUSUM"]["k"], h, w),
            "EWMA": lambda z, L: drift.ewma_down(z, d["EWMA"]["lam"], L, w)}


def calibrate(spec):
    z = np.vstack([drift.standardise([t.pnl for t in population(spec, s, decay_share=0.0)], spec["warmup"])
                   for s in spec["calibration_seeds"]])
    fns = detectors(spec)
    return {"CUSUM": drift.calibrate(fns["CUSUM"], z, 0.10, 0.1, 100.0),
            "EWMA": drift.calibrate(fns["EWMA"], z, 0.10, 0.1, 20.0)}


def seed_counts(spec, seed, thresholds):
    pop = population(spec, seed)
    z = drift.standardise([t.pnl for t in pop], spec["warmup"])
    dec = np.array([t.kind == "decaying" for t in pop])
    tau = np.array([t.meta["tau"] if t.meta["tau"] is not None else -1 for t in pop])
    mu = np.array([t.meta["mu_pre"] for t in pop])
    out = {}
    for name, fn in detectors(spec).items():
        a = fn(z, thresholds[name])
        det = dec & (a >= tau) & (tau >= 0)
        d150 = det & (a < tau + 150)
        c = {"false_alarm": [int((~dec & (a >= 0)).sum()), int((~dec).sum())],
             "early_alarm": [int((dec & (a >= 0) & (a < tau)).sum()), int(dec.sum())],
             "detect150": [int(d150.sum()), int(dec.sum())], "detect_any": [int(det.sum()), int(dec.sum())],
             "delays": (a[det] - tau[det]).tolist()}
        for lo, hi in BANDS:
            m = dec & (mu > lo) & ((mu <= hi) if hi is not None else True)
            c[f"detect150_{lo}-{hi if hi is not None else 'inf'}"] = [int((d150 & m).sum()), int(m.sum())]
        out[name] = c
    return out


def run(spec, out=print):
    th = calibrate(spec)
    out(json.dumps({"thresholds": th}))
    rows = [seed_counts(spec, s, th) for s in spec["seeds"]]
    res = {"thresholds": th}
    for name in ("CUSUM", "EWMA"):
        r = {}
        for k in rows[0][name]:
            if k == "delays":
                dl = np.concatenate([row[name]["delays"] for row in rows]) if rows else np.array([])
                r["delay"] = ({"median": float(np.median(dl)), "q25": float(np.quantile(dl, 0.25)),
                               "q75": float(np.quantile(dl, 0.75)), "n": int(dl.size)} if dl.size else None)
            else:
                r[k] = study.pooled_ratio_ci([row[name][k][0] for row in rows], [row[name][k][1] for row in rows])
        res[name] = r
        out(json.dumps({name: r}))
    rate = lambda name, row: row[name]["detect150"][0] / max(row[name]["detect150"][1], 1)
    res["detect150_cusum_minus_ewma"] = study.paired_diff([rate("CUSUM", r) for r in rows], [rate("EWMA", r) for r in rows])
    c = res["CUSUM"]
    res["success"] = bool(c["detect150"]["ci95"][0] >= 0.50 and c["false_alarm"]["ci95"][1] <= 0.15)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        sm = study.smoke_spec(SPEC)
        sm["calibration_seeds"] = [9992, 9993]
        res = run(sm, out=lambda s: None)
        print(json.dumps({"smoke": "ok", "keys": sorted(res)}))
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
    res = run(SPEC)
    with atomic_open(os.path.join(ROOT, "state", f"{STUDY}_report.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    reg.result(STUDY, {"run_commit": lock["commit"], **res})
    print(json.dumps({"success": res["success"], "CUSUM": {k: res["CUSUM"][k] for k in ("detect150", "false_alarm", "delay")},
                      "EWMA": {k: res["EWMA"][k] for k in ("detect150", "false_alarm", "delay")},
                      "detect150_cusum_minus_ewma": res["detect150_cusum_minus_ewma"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
