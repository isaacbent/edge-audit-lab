"""SKILL-05 (diagnostic, report only): where does the NPMLE skill probability's over-confidence come from?
Threshold (P(mean > 0) vs P(mean > cost)), per-trader vs pooled variance, normal vs Student-t vs bootstrap-t
likelihood, and isotonic recalibration, compared on BENCH-2. ROADMAP phase 1, step 1.

Usage:
  python3 -m tools.skill_diag --register
  nohup python3 -m tools.skill_diag > state/skill05.log 2>&1 &
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

STUDY = "SKILL-05"
FITS = {"normal_own": {"var": "own", "lik": "normal"}, "normal_pooled": {"var": "pooled", "lik": "normal"},
        "t_own": {"var": "own", "lik": "t"}, "boot_own": {"var": "own", "lik": "boot", "boot_b": 200}}
SPEC = {
    "question": "Diagnostic: which factor removes the NPMLE skill probability's over-confidence on BENCH-2 "
                "(threshold, variance, likelihood, isotonic recalibration)?",
    "type": "diagnostic, report only: no success criterion (ROADMAP phase 1 step 1)",
    "bench": "BENCH-2 (simulate.population_v2), population order shuffled per seed (rng seed + 1000000) so ties "
             "are not broken in favour of skilled traders",
    "population": {"n_traders": 2000,
                   "mix": {"skilled": 0.05, "breakeven": 0.25, "loser": 0.50, "martingale": 0.10, "tilt": 0.10},
                   "cost_R": 0.05, "risk_pct": 1.0},
    "grid": {"n_trades": [50, 100, 200], "edge_R": [0.10, 0.15, 0.25]},
    "seeds": list(range(5000, 5020)),
    "fits": {k: dict(v) for k, v in FITS.items()},
    "fit_notes": "one NPMLE per fit and population with the grid holding 0 and cost_R; P(mean > 0) and "
                 "P(mean > cost_R) read from the same posterior; bootstrap seed = population seed",
    "scores": "for each fit: <fit>@thr0 = P(mean > 0), <fit>@cost = P(mean > cost_R); "
              "iso(normal_own@thr0), iso(normal_own@cost) = isotonic (PAV) map fitted per cell on seeds 5000-5009 "
              "(pooled), applied to seeds 5010-5019; references S2_t_stat, S5_profit_factor",
    "label": "trader kind == skilled",
    "metrics": {"auc": "over all traders", "auc_minus_t": "auc - auc(S2_t_stat)",
                "prec_top5pct": "skill.precision_at_k_ties at k = 100",
                "challenge_precision": "precision of challenge_pass(8 %, 10 %)",
                "brier": "mean (prob - label)^2", "deciles": "mean prob vs skilled share in deciles of the score",
                "max_decile_gap": "max over deciles of |mean prob - skilled share|",
                "top_decile_gap": "mean prob - skilled share in the top decile (signed)"},
    "report": "all scores on eval seeds 5010-5019 (comparable with the isotonic scores); non-isotonic scores also "
              "on all 20 seeds",
    "attempts": "8 NPMLE variants + 2 isotonic = 10 attempts (docs/FORMULAS.md rule 1)",
    "notes": "the next study (with criterion, new seeds) is chosen from this report; isotonic needs labelled "
             "traders, which a firm only has after outcomes (paid / breached), and a map learned on a synthetic "
             "bench is bench-specific",
}
COST = SPEC["population"]["cost_R"]


def score_population(n_trades, edge, seed):
    p = SPEC["population"]
    pop = simulate.population_v2(p["mix"], p["n_traders"], n_trades, seed, edge=edge, cost=COST, risk=p["risk_pct"])
    pop = [pop[i] for i in np.random.default_rng(seed + 1_000_000).permutation(len(pop))]
    xs = [t.pnl for t in pop]
    y = np.array([t.kind == "skilled" for t in pop])
    passed = np.array([simulate.challenge_pass(x) for x in xs])
    sc = {}
    for name, kw in FITS.items():
        r = skill.npmle_prob(xs, thr=COST, seed=seed, **kw)
        sc[f"{name}@thr0"] = r["post"][:, r["grid"] > 0].sum(axis=1)
        sc[f"{name}@cost"] = r["prob"]
    sc["S2_t_stat"] = np.array([skill.t_stat(x) for x in xs])
    sc["S5_profit_factor"] = np.array([skill.profit_factor(x) for x in xs])
    return sc, y, float(y[passed].mean()) if passed.any() else None


def metrics(s, y, t_auc, prob=True):
    out = {"auc": skill.auc(s, y), "prec_top5pct": skill.precision_at_k_ties(s, y, int(round(0.05 * len(s))))}
    out["auc_minus_t"] = out["auc"] - t_auc
    if prob:
        dec = np.minimum((np.argsort(np.argsort(s, kind="mergesort")) * 10) // len(s), 9)
        d = [[float(s[dec == k].mean()), float(y[dec == k].mean())] for k in range(10)]
        out.update({"brier": float(np.mean((s - y) ** 2)), "deciles": d,
                    "max_decile_gap": max(abs(a - b) for a, b in d), "top_decile_gap": d[9][0] - d[9][1]})
    return out


def _avg(rows):
    """Mean over seeds of nested metric dicts (deciles averaged element-wise)."""
    out = {}
    for k in rows[0]:
        v = [r[k] for r in rows]
        out[k] = (np.round(np.mean(v, axis=0), 4).tolist() if k == "deciles" else round(float(np.mean(v)), 4))
    return out


def run_cell(n_trades, edge):
    seeds = SPEC["seeds"]
    data = {s: score_population(n_trades, edge, s) for s in seeds}
    fit_seeds, eval_seeds = seeds[:10], seeds[10:]
    iso = {}
    for base in ("normal_own@thr0", "normal_own@cost"):
        iso[base] = skill.isotonic_fit(np.concatenate([data[s][0][base] for s in fit_seeds]),
                                       np.concatenate([data[s][1] for s in fit_seeds]))
    per = {"all20": {}, "eval10": {}}
    for s in seeds:
        sc, y, _ = data[s]
        if s in eval_seeds:
            for base, knots in iso.items():
                sc[f"iso({base})"] = skill.isotonic_apply(knots, sc[base])
        t_auc = skill.auc(sc["S2_t_stat"], y)
        for k, v in sc.items():
            m = metrics(v, y, t_auc, prob=not k.startswith("S"))
            per["all20"].setdefault(k, []).append(m) if not k.startswith("iso") else None
            if s in eval_seeds:
                per["eval10"].setdefault(k, []).append(m)
    ch = {"all20": [data[s][2] for s in seeds], "eval10": [data[s][2] for s in eval_seeds]}
    return {"n_trades": n_trades, "edge_R": edge,
            **{part: {"challenge_precision": round(float(np.mean([c for c in ch[part] if c is not None])), 4),
                      "scores": {k: _avg(v) for k, v in per[part].items()}} for part in per}}


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
            row = run_cell(n, e)
            cells.append(row)
            print(json.dumps(row), flush=True)
    pc = next(c for c in cells if c["n_trades"] == 100 and c["edge_R"] == 0.15)
    summary = {k: {m: v[m] for m in ("auc", "auc_minus_t", "prec_top5pct", "max_decile_gap", "top_decile_gap", "brier")
                   if m in v} for k, v in pc["eval10"]["scores"].items()}
    with atomic_open(os.path.join(ROOT, "state", f"{STUDY}_report.json"), "w") as fh:
        json.dump({"cells": cells}, fh, indent=1)
    reg.result(STUDY, {"diagnostic": True, "primary_cell_eval10": summary,
                       "challenge_precision": pc["eval10"]["challenge_precision"]})
    print(json.dumps({"primary_cell_eval10": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
