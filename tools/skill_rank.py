"""SKILL-02: does a graded skill probability rank traders better than the challenge rule and the
usual metrics (total profit, win rate, profit factor)?

Usage:
  python3 -m tools.skill_rank --register
  python3 -m tools.skill_rank
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import simulate, skill  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "SKILL-02"
SPEC = {
    "question": "Ranking quality of an empirical-Bayes skill probability vs the challenge rule and usual metrics",
    "population": {"n_traders": 2000,
                   "mix": {"skilled": 0.05, "breakeven": 0.25, "loser": 0.50, "martingale": 0.10, "tilt": 0.10},
                   "cost_R": 0.05, "risk_pct": 1.0, "t_df": 4},
    "grid": {"n_trades": [50, 100, 200], "edge_R": [0.10, 0.15, 0.25]},
    "seeds": list(range(2000, 2020)),
    "scores": {"S1_eb_prob": "P(true mean > 0) under a Normal(m, tau2) population prior, own variance per trader",
               "S2_t_stat": "mean / standard error",
               "S3_total_profit": "sum of trade results (what the challenge rewards)",
               "S4_win_rate": "share of winning trades",
               "S5_profit_factor": "gross profit / gross loss"},
    "label": "trader kind == skilled",
    "metrics": {"auc": "over all traders",
                "prec_top5pct": "precision of the top 5 % (100 traders) by score",
                "challenge_precision": "precision of challenge_pass(8 %, 10 %) (reference)",
                "passers_top20pct": "among challenge passers, precision of the top 20 % by score",
                "calibration_S1": "Brier score of S1 vs label, and mean S1 vs skilled share in S1 deciles"},
    "primary_cell": {"n_trades": 100, "edge_R": 0.15},
    "success": "primary cell, means over seeds: prec_top5pct(S1) >= 2 x challenge_precision AND "
               "auc(S1) >= auc of S3, S4 and S5",
    "notes": "new seeds (SKILL-01 used 1000-1019); calibration is reported, not part of the criterion",
}
SCORES = {"S2_t_stat": skill.t_stat, "S3_total_profit": skill.total_profit,
          "S4_win_rate": skill.win_rate, "S5_profit_factor": skill.profit_factor}


def run_cell(n_trades, edge, seed):
    p = SPEC["population"]
    pop = simulate.population(p["mix"], p["n_traders"], n_trades, seed, edge=edge, cost=p["cost_R"],
                              risk=p["risk_pct"], df=p["t_df"])
    y = np.array([t.kind == "skilled" for t in pop])
    passed = np.array([simulate.challenge_pass(t.pnl) for t in pop])
    sc = {"S1_eb_prob": skill.eb_prob([t.pnl for t in pop])["prob"]}
    for k, f in SCORES.items():
        sc[k] = np.array([f(t.pnl) for t in pop])
    k5 = int(round(0.05 * len(pop)))
    out = {"challenge_precision": float(y[passed].mean()) if passed.any() else None, "scores": {}}
    for k, s in sc.items():
        kp = max(1, int(round(0.2 * passed.sum())))
        out["scores"][k] = {"auc": skill.auc(s, y), "prec_top5pct": skill.precision_at_k(s, y, k5),
                            "passers_top20pct": skill.precision_at_k(s[passed], y[passed], kp) if passed.any() else None}
    s1 = sc["S1_eb_prob"]
    dec = np.minimum((np.argsort(np.argsort(s1, kind="mergesort")) * 10) // len(s1), 9)
    out["calibration_S1"] = {"brier": float(np.mean((s1 - y) ** 2)),
                             "deciles": [[float(s1[dec == d].mean()), float(y[dec == d].mean())] for d in range(10)]}
    return out


def _m(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    reg = Registry(args.registry)
    if args.register:
        print(reg.prereg(STUDY, SPEC)["payload"]["spec_hash"])
        return 0
    if reg.spec_of(STUDY)["spec"] != json.loads(json.dumps(SPEC)):
        raise SystemExit("SPEC differs from the pre-registration: refusing to run")
    cells = []
    for n in SPEC["grid"]["n_trades"]:
        for e in SPEC["grid"]["edge_R"]:
            runs = [run_cell(n, e, s) for s in SPEC["seeds"]]
            row = {"n_trades": n, "edge_R": e, "challenge_precision": _m([r["challenge_precision"] for r in runs]),
                   "scores": {k: {m: _m([r["scores"][k][m] for r in runs]) for m in ("auc", "prec_top5pct", "passers_top20pct")}
                              for k in runs[0]["scores"]},
                   "calibration_S1": {"brier": _m([r["calibration_S1"]["brier"] for r in runs]),
                                      "deciles": [[_m([r["calibration_S1"]["deciles"][d][j] for r in runs]) for j in (0, 1)]
                                                  for d in range(10)]}}
            cells.append(row)
            print(json.dumps(row))
    pc = next(c for c in cells if c["n_trades"] == 100 and c["edge_R"] == 0.15)
    s1 = pc["scores"]["S1_eb_prob"]
    success = bool(s1["prec_top5pct"] >= 2 * pc["challenge_precision"]
                   and all(s1["auc"] >= pc["scores"][k]["auc"] for k in ("S3_total_profit", "S4_win_rate", "S5_profit_factor")))
    res = {"cells": cells, "success": success,
           "primary": {"challenge_precision": pc["challenge_precision"], **{k: v for k, v in pc["scores"].items()}}}
    with atomic_open(os.path.join(ROOT, "state", f"{STUDY}_report.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    reg.result(STUDY, {"success": success, "primary": res["primary"]})
    print(json.dumps({"success": success, "primary": res["primary"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
