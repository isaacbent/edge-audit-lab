"""SKILL-06: confirmation on BENCH-3 (continuous edges, variable trade counts) of the calibrated skill probability
found in SKILL-04 / SKILL-05: NPMLE, own variance, normal likelihood, P(true mean > cost). ROADMAP phase 1c.

Usage:
  python3 -m tools.skill_calib_b3 --smoke      # tiny run, no registry writes (must pass before registering)
  python3 -m tools.skill_calib_b3 --register   # pre-registration with code lock (committed code only)
  python3 -m tools.skill_calib_b3              # refuses to run unless SPEC, code and versions match the prereg
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import simulate, skill, study  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "SKILL-06"
SPEC = {
    "question": "Is the NPMLE probability of an edge above costs calibrated and as discriminating as the t-stat on "
                "BENCH-3 (continuous edges, variable trade counts)?",
    "bench": "BENCH-3: simulate.population_v3 (edge = -0.05 + edge_scale * Student-t(5); behaviour plain 0.80, "
             "martingale 0.10, tilt 0.10; own win-rate style; trades per trader uniform in n_range); population order "
             "shuffled per seed (rng seed + 1000000)",
    "population": {"n_traders": 2000, "risk_pct": 1.0, "edge_loc": -0.05, "edge_df": 5},
    "threshold_R": 0.05,
    "grid": {"n_range": [[30, 100], [50, 200], [100, 300]], "edge_scale": [0.05, 0.07, 0.10]},
    "primary_cell": {"n_range": [50, 200], "edge_scale": 0.07},
    "seeds": list(range(6000, 6020)),
    "score": "S0 = skill.npmle_prob(pnl lists, thr=threshold_R)['prob'] (own variance, normal likelihood)",
    "references": {"S2_t_stat": "mean / standard error", "S5_profit_factor": "gross profit / gross loss",
                   "challenge": "challenge_pass(8 %, 10 %)"},
    "label": "B (user decision): meta['edge'] (true expected P&L per trade, size included) > threshold_R",
    "success": "primary cell, all of: (1) max over deciles of |mean S0 - skilled share| <= 0.10, deciles of S0 over the "
               "traders of all 20 seeds pooled; (2) per-seed paired AUC(S0) - AUC(S2_t_stat): lower end of the 95 % "
               "t interval >= -0.01; (3) per-seed paired precision_at_k_ties(S0, k = 5 %) - 2 x challenge precision: "
               "lower end of the 95 % t interval > 0",
    "secondary": {"all_cells": "same metrics, no criterion",
                  "label_C": "precision of the top 5 % by S0 when martingale / tilt traders never count as skilled",
                  "martingale_top5": "share of martingale traders in the top 5 % by S0",
                  "corr": "primary cell with corr_sd = 0.2 (shared market factor), same metrics",
                  "bench2_control": "BENCH-2 primary cell (100 trades, edge 0.15, SKILL-03 mix, label kind == skilled), "
                                    "same metrics"},
    "stop_rule": "if the criterion fails: report probabilities only in coarse calibrated bands (low / medium / high), "
                 "no further iteration on this bench (ROADMAP)",
    "notes": "new seeds; score chosen from SKILL-05 (diagnostic) before any BENCH-3 data was looked at",
}
B2_MIX = {"skilled": 0.05, "breakeven": 0.25, "loser": 0.50, "martingale": 0.10, "tilt": 0.10}


def make_population(spec, cell, seed, corr_sd=0.0, bench2=False):
    p = spec["population"]
    if bench2:
        pop = simulate.population_v2(B2_MIX, p["n_traders"], 100, seed, edge=0.15, cost=spec["threshold_R"], risk=p["risk_pct"])
    else:
        pop = simulate.population_v3(p["n_traders"], tuple(cell["n_range"]), seed, edge_loc=p["edge_loc"],
                                     edge_scale=cell["edge_scale"], edge_df=p["edge_df"], corr_sd=corr_sd, risk=p["risk_pct"])
    pop = [pop[i] for i in np.random.default_rng(seed + 1_000_000).permutation(len(pop))]
    if bench2:
        y = np.array([t.kind == "skilled" for t in pop])
        return pop, y, y
    y = np.array([t.meta["edge"] > spec["threshold_R"] for t in pop])
    return pop, y, y & np.array([t.kind == "plain" for t in pop])


def run_seed(spec, cell, seed, **kw):
    pop, y, y_c = make_population(spec, cell, seed, **kw)
    xs = [t.pnl for t in pop]
    s0 = skill.npmle_prob(xs, thr=spec["threshold_R"])["prob"]
    t = np.array([skill.t_stat(x) for x in xs])
    pf = np.array([skill.profit_factor(x) for x in xs])
    passed = np.array([simulate.challenge_pass(x) for x in xs])
    k = int(round(0.05 * len(pop)))
    top = np.argsort(-s0, kind="mergesort")[:k]
    return {"s0": s0, "y": y, "auc_s0": skill.auc(s0, y), "auc_t": skill.auc(t, y), "auc_pf": skill.auc(pf, y),
            "prec_s0": skill.precision_at_k_ties(s0, y, k), "prec_t": skill.precision_at_k_ties(t, y, k),
            "challenge": float(y[passed].mean()) if passed.any() else 0.0,
            "prec_s0_label_c": skill.precision_at_k_ties(s0, y_c, k),
            "martingale_top5": float(np.mean([pop[i].kind == "martingale" for i in top]))}


def summarise(runs):
    s0 = np.concatenate([r["s0"] for r in runs])
    y = np.concatenate([r["y"] for r in runs])
    dec = np.minimum((np.argsort(np.argsort(s0, kind="mergesort")) * 10) // len(s0), 9)
    deciles = [[round(float(s0[dec == d].mean()), 4), round(float(y[dec == d].mean()), 4)] for d in range(10)]
    col = lambda k: [r[k] for r in runs]
    return {"deciles_pooled": deciles, "max_decile_gap": round(max(abs(a - b) for a, b in deciles), 4),
            "skilled_share": round(float(y.mean()), 4),
            **{k: study.mean_se(col(k)) for k in ("auc_s0", "auc_t", "auc_pf", "prec_s0", "prec_t", "challenge",
                                                  "prec_s0_label_c", "martingale_top5")},
            "auc_s0_minus_t": study.paired_diff(col("auc_s0"), col("auc_t")),
            "prec_s0_minus_2x_challenge": study.paired_diff(col("prec_s0"), [2 * c for c in col("challenge")])}


def passes(sm) -> bool:
    return bool(sm["max_decile_gap"] <= 0.10 and sm["auc_s0_minus_t"]["ci95"][0] >= -0.01
                and sm["prec_s0_minus_2x_challenge"]["ci95"][0] > 0)


def run(spec, out=print):
    cells = []
    for nr in spec["grid"]["n_range"]:
        for es in spec["grid"]["edge_scale"]:
            cell = {"n_range": nr, "edge_scale": es}
            sm = summarise([run_seed(spec, cell, s) for s in spec["seeds"]])
            cells.append({**cell, **sm})
            out(json.dumps(cells[-1]))
    pc = next(c for c in cells if c["n_range"] == spec["primary_cell"]["n_range"]
              and c["edge_scale"] == spec["primary_cell"]["edge_scale"])
    corr = summarise([run_seed(spec, spec["primary_cell"], s, corr_sd=0.2) for s in spec["seeds"]])
    b2 = summarise([run_seed(spec, None, s, bench2=True) for s in spec["seeds"]])
    out(json.dumps({"corr": corr}))
    out(json.dumps({"bench2_control": b2}))
    return {"cells": cells, "primary": pc, "corr": corr, "bench2_control": b2, "success": passes(pc)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        res = run(study.smoke_spec(SPEC), out=lambda s: None)
        print(json.dumps({"smoke": "ok", "success": res["success"], "max_decile_gap": res["primary"]["max_decile_gap"]}))
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
    keep = ("max_decile_gap", "deciles_pooled", "skilled_share", "auc_s0_minus_t", "prec_s0_minus_2x_challenge",
            "auc_s0", "auc_t", "prec_s0", "challenge", "prec_s0_label_c", "martingale_top5")
    reg.result(STUDY, {"success": res["success"], "run_commit": lock["commit"],
                       "primary": {k: res["primary"][k] for k in keep},
                       "corr": {k: res["corr"][k] for k in keep}, "bench2_control": {k: res["bench2_control"][k] for k in keep}})
    print(json.dumps({"success": res["success"], "primary": {k: res["primary"][k] for k in keep}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
