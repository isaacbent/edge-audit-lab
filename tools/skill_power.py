"""SKILL-01: can we separate skilled funded traders from lucky ones better than the challenge rule?

Usage:
  python3 -m tools.skill_power --register   # writes the pre-registration (once; never changed)
  python3 -m tools.skill_power              # runs the pre-registered spec, appends the result
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import simulate, stats  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "SKILL-01"
SPEC = {
    "question": "Precision/recall of skill flags vs the prop-firm challenge rule on a synthetic population",
    "population": {"n_traders": 2000,
                   "mix": {"skilled": 0.05, "breakeven": 0.25, "loser": 0.50, "martingale": 0.10, "tilt": 0.10},
                   "cost_R": 0.05, "risk_pct": 1.0, "t_df": 4},
    "grid": {"n_trades": [50, 100, 200], "edge_R": [0.10, 0.15, 0.25]},
    "seeds": list(range(1000, 1020)),
    "arms": {"A_firm": "challenge_pass(target 8 %, static max loss 10 %) over the n trades",
             "B_lab": "Benjamini-Hochberg q=0.10 on one-sided t p-values of per-trade P&L, across the population",
             "C_both": "A and B"},
    "placebo": "same population with skilled share 0 (moved to breakeven): count of B flags",
    "metrics": ["precision = skilled / flagged", "recall = flagged skilled / skilled", "flagged"],
    "primary_cell": {"n_trades": 100, "edge_R": 0.15},
    "success": "primary cell, mean over seeds: precision(C) >= 2 x precision(A) AND recall(C) >= 0.30",
    "secondary": "for each edge, smallest n_trades with mean recall(B) >= 0.5",
}


def run_cell(n_trades, edge, seed, placebo=False):
    p = SPEC["population"]
    mix = dict(p["mix"])
    if placebo:
        mix["breakeven"] += mix.pop("skilled")
    pop = simulate.population(mix, p["n_traders"], n_trades, seed, edge=edge, cost=p["cost_R"],
                              risk=p["risk_pct"], df=p["t_df"])
    skilled = np.array([t.kind == "skilled" for t in pop])
    a = np.array([simulate.challenge_pass(t.pnl) for t in pop])
    b = stats.benjamini_hochberg([stats.t_p(t.pnl) for t in pop], q=0.10)
    c = a & b
    out = {}
    for name, f in (("A_firm", a), ("B_lab", b), ("C_both", c)):
        nf = int(f.sum())
        out[name] = {"flagged": nf,
                     "precision": float((f & skilled).sum() / nf) if nf else None,
                     "recall": float((f & skilled).sum() / skilled.sum()) if skilled.sum() else None}
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    reg = Registry(args.registry)
    if args.register:
        print(json.dumps(reg.prereg(STUDY, SPEC)["payload"]["spec_hash"]))
        return 0
    pre = reg.spec_of(STUDY)
    if pre["spec"] != json.loads(json.dumps(SPEC)):
        raise SystemExit("SPEC differs from the pre-registration: refusing to run")
    cells = []
    for n in SPEC["grid"]["n_trades"]:
        for e in SPEC["grid"]["edge_R"]:
            runs = [run_cell(n, e, s) for s in SPEC["seeds"]]
            row = {"n_trades": n, "edge_R": e}
            for arm in ("A_firm", "B_lab", "C_both"):
                row[arm] = {m: _mean([r[arm][m] for r in runs]) for m in ("flagged", "precision", "recall")}
            cells.append(row)
            print(json.dumps(row))
    plac = [run_cell(100, 0.15, s, placebo=True)["B_lab"]["flagged"] for s in SPEC["seeds"]]
    pc = next(c for c in cells if c["n_trades"] == 100 and c["edge_R"] == 0.15)
    pa, pcc, rc = pc["A_firm"]["precision"], pc["C_both"]["precision"], pc["C_both"]["recall"]
    passed = bool(pa is not None and pcc is not None and pcc >= 2 * pa and rc >= 0.30)
    secondary = {}
    for e in SPEC["grid"]["edge_R"]:
        ok = [c["n_trades"] for c in cells if c["edge_R"] == e and (c["B_lab"]["recall"] or 0) >= 0.5]
        secondary[str(e)] = min(ok) if ok else None
    res = {"cells": cells, "placebo_B_flags_mean": _mean(plac), "placebo_B_flags_max": int(max(plac)),
           "primary": {"precision_A": pa, "precision_C": pcc, "recall_C": rc}, "success": passed,
           "secondary_min_n_for_recall50": secondary}
    with atomic_open(os.path.join(ROOT, "state", f"{STUDY}_report.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    reg.result(STUDY, {"success": passed, "primary": res["primary"],
                       "placebo_B_flags_mean": res["placebo_B_flags_mean"],
                       "secondary_min_n_for_recall50": secondary})
    print(json.dumps({"success": passed, "primary": res["primary"], "placebo": res["placebo_B_flags_mean"],
                      "secondary": secondary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
