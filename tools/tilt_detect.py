"""TILT-01: can a firm flag martingale and tilt traders from position size right after losses, with few false
flags among normal traders? ROADMAP phase 3, step 10.

Usage:
  python3 -m tools.tilt_detect --smoke
  python3 -m tools.tilt_detect --register
  python3 -m tools.tilt_detect
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import behaviour, simulate, skill, stats, study  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "TILT-01"
SPEC = {
    "question": "Recall for martingale and tilt traders, and false flags among normal traders, of a size-after-loss "
                "test with Benjamini-Hochberg across the population",
    "bench": "BENCH-3T: simulate.population_tilt = BENCH-3 (edge -0.05 + 0.07 * Student-t(5); plain 0.80, martingale "
             "0.10 (size doubles after a loss, cap 8x), tilt 0.10 (after a loss, prob 0.5, 3 trades at 1.5x size and "
             "-0.2 R)) with every trade's size and pnl multiplied by an independent mean-1 lognormal(0.30) factor",
    "population": {"n_traders": 2000, "edge_scale": 0.07, "size_sigma": 0.30, "risk_pct": 1.0},
    "n_trades": [50, 100, 200],
    "primary_n": 100,
    "seeds": list(range(10000, 10020)),
    "test": "behaviour.size_after_loss_p: one-sided Welch t, log size after a losing vs after a winning trade",
    "flag": "stats.benjamini_hochberg(p, q=0.10) across the population",
    "metrics": {"recall_martingale": "flagged martingale / martingale (pooled over seeds, seed-bootstrap 95 % interval)",
                "recall_tilt": "flagged tilt / tilt", "fdr": "flagged plain / flagged",
                "plain_flag_rate": "flagged plain / plain"},
    "success": "at primary_n, all of: recall_martingale lower bound >= 0.80 AND recall_tilt lower bound >= 0.50 AND "
               "fdr upper bound <= 0.10",
    "secondary": {"size_ratio": "median size-after-loss / size-after-win ratio by behaviour",
                  "skill_label_c": "top-5 % precision of the SKILL-06 score P(edge > 0.05 R) under label C (martingale "
                                   "/ tilt never skilled), before and after removing flagged traders from the ranking"},
    "notes": "no timestamps in the bench, so the Hawkes intensity planned in ROADMAP is not testable here; "
             "FORMULAS rule 3 (simplest piece first)",
}


def seed_counts(spec, n, seed):
    p = spec["population"]
    pop = simulate.population_tilt(p["n_traders"], n, seed, edge_scale=p["edge_scale"], size_sigma=p["size_sigma"],
                                   risk=p["risk_pct"])
    pv = np.array([behaviour.size_after_loss_p(t.pnl, t.size) for t in pop])
    flag = stats.benjamini_hochberg(pv, q=0.10)
    kind = np.array([t.kind for t in pop])
    c = {k: [int((flag & (kind == k)).sum()), int((kind == k).sum())] for k in ("martingale", "tilt", "plain")}
    c["fdr"] = [int((flag & (kind == "plain")).sum()), int(flag.sum())]
    c["ratios"] = {k: [behaviour.size_ratio_after_loss(t.pnl, t.size) for t in pop if t.kind == k]
                   for k in ("martingale", "tilt", "plain")}
    s0 = skill.npmle_prob([t.pnl for t in pop], thr=0.05)["prob"]
    y_c = np.array([t.meta["edge"] > 0.05 and t.kind == "plain" for t in pop])
    k5 = int(round(0.05 * len(pop)))
    c["skill_c_before"] = skill.precision_at_k_ties(s0, y_c, k5)
    keep = ~flag
    c["skill_c_after"] = skill.precision_at_k_ties(s0[keep], y_c[keep], k5)
    return c


def run(spec, out=print):
    res = {}
    for n in spec["n_trades"]:
        rows = [seed_counts(spec, n, s) for s in spec["seeds"]]
        ratio = lambda k: study.pooled_ratio_ci([r[k][0] for r in rows], [r[k][1] for r in rows])
        cell = {"recall_martingale": ratio("martingale"), "recall_tilt": ratio("tilt"), "fdr": ratio("fdr"),
                "plain_flag_rate": ratio("plain"),
                "size_ratio_median": {k: round(float(np.median(np.concatenate([r["ratios"][k] for r in rows]))), 3)
                                      for k in ("martingale", "tilt", "plain")},
                "skill_label_c_top5": {"before": study.mean_se([r["skill_c_before"] for r in rows]),
                                       "after": study.mean_se([r["skill_c_after"] for r in rows]),
                                       "after_minus_before": study.paired_diff([r["skill_c_after"] for r in rows],
                                                                               [r["skill_c_before"] for r in rows])}}
        res[str(n)] = cell
        out(json.dumps({"n_trades": n, **cell}))
    pc = res[str(spec["primary_n"])]
    success = bool(pc["recall_martingale"]["ci95"][0] >= 0.80 and pc["recall_tilt"]["ci95"][0] >= 0.50
                   and pc["fdr"]["ci95"] is not None and pc["fdr"]["ci95"][1] <= 0.10)
    return {"cells": res, "success": success}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        sm = study.smoke_spec(SPEC)
        sm["seeds"] = [9994, 9995]
        res = run(sm, out=lambda s: None)
        print(json.dumps({"smoke": "ok", "keys": sorted(res["cells"][str(SPEC["primary_n"])])}))
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
    print(json.dumps({"success": res["success"], "primary": res["cells"][str(SPEC["primary_n"])]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
