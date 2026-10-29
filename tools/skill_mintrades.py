"""SKILL-07: how many trades until the calibrated skill probability (SKILL-06) can be trusted, by true edge size.
Output = the abstention table a firm sees ("with fewer than N trades we abstain"). ROADMAP phase 2.

Usage:
  python3 -m tools.skill_mintrades --smoke
  python3 -m tools.skill_mintrades --register
  nohup python3 -m tools.skill_mintrades > state/skill07.log 2>&1 &
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

STUDY = "SKILL-07"
SPEC = {
    "question": "How many trades until the rule P(edge > cost) >= 0.7 is trustworthy, and how recall grows with trades "
                "for each true-edge band (abstention table)",
    "bench": "BENCH-3: simulate.population_v3 with a fixed number of trades N per population (edge = -0.05 + 0.07 * "
             "Student-t(5), behaviour plain 0.80 / martingale 0.10 / tilt 0.10); population order shuffled per seed "
             "(rng seed + 1000000)",
    "population": {"n_traders": 2000, "risk_pct": 1.0, "edge_loc": -0.05, "edge_scale": 0.07, "edge_df": 5},
    "threshold_R": 0.05,
    "n_trades": [25, 50, 75, 100, 150, 200, 300],
    "seeds": list(range(7000, 7020)),
    "score": "P = skill.npmle_prob(pnl lists, thr=threshold_R)['prob'], fitted per population (the SKILL-06 score)",
    "rule": "flag as skilled when P >= 0.7",
    "label": "B: meta['edge'] (true expected P&L per trade, size included) > threshold_R",
    "edge_bands": [[0.05, 0.10], [0.10, 0.15], [0.15, None]],
    "success": "for every N >= 50: pooled precision of the rule (true flags / flags, summed over the 20 seeds) has the "
               "lower end of its 95 % seed-bootstrap interval (study.pooled_ratio_ci, 2000 resamples) >= 0.70",
    "main_output": "recall of the rule by N and edge band; N_min per band = smallest N with pooled recall >= 0.50 "
                   "('> 300' if none); flags and precision per N (N = 25 reported, outside the criterion)",
    "secondary": {"sequential": "N = 300 populations: P recomputed on the first c trades of every trader, c = 25, 50, "
                                "..., 300 (NPMLE refitted on the prefix population); decide 'skilled' at the first "
                                "P >= 0.7, 'dropped' at the first P <= 0.05; report mean trades to decision, undecided "
                                "share, precision, recall overall and per band, vs the fixed-N rule",
                  "label_C": "precision of the rule when martingale / tilt traders never count as skilled"},
    "failure_meaning": "the table is still published; below the first N where the criterion fails, the firm sees "
                       "'abstain' instead of a probability",
    "notes": "SPRT stays CANDIDATE in docs/FORMULAS.md (SKILL-08 only if the N_min are too long for a firm)",
}
FLAG, DROP = 0.7, 0.05


def make_population(spec, n, seed):
    p = spec["population"]
    pop = simulate.population_v3(p["n_traders"], n, seed, edge_loc=p["edge_loc"], edge_scale=p["edge_scale"],
                                 edge_df=p["edge_df"], risk=p["risk_pct"])
    return [pop[i] for i in np.random.default_rng(seed + 1_000_000).permutation(len(pop))]


def band_masks(spec, edge):
    out = {}
    for lo, hi in spec["edge_bands"]:
        out[f"{lo}-{hi if hi is not None else 'inf'}"] = (edge > lo) & ((edge <= hi) if hi is not None else True)
    return out


def counts(spec, flag, pop):
    edge = np.array([t.meta["edge"] for t in pop])
    y = edge > spec["threshold_R"]
    y_c = y & np.array([t.kind == "plain" for t in pop])
    c = {"flags": int(flag.sum()), "true_flags": int((flag & y).sum()), "true_flags_c": int((flag & y_c).sum()),
         "skilled": int(y.sum())}
    for name, m in band_masks(spec, edge).items():
        c[f"band_{name}"] = [int((flag & m).sum()), int(m.sum())]
    return c


def pool(spec, rows):
    col = lambda k: [r[k] for r in rows]
    out = {"precision": study.pooled_ratio_ci(col("true_flags"), col("flags")),
           "precision_label_c": study.pooled_ratio_ci(col("true_flags_c"), col("flags")),
           "recall": round(sum(col("true_flags")) / max(sum(col("skilled")), 1), 4),
           "flags_per_seed": study.mean_se(col("flags"))}
    for k in rows[0]:
        if k.startswith("band_"):
            hit, tot = sum(r[k][0] for r in rows), sum(r[k][1] for r in rows)
            out[f"recall_{k}"] = round(hit / tot, 4) if tot else None
    return out


def sequential_seed(spec, seed, checkpoints):
    pop = make_population(spec, max(checkpoints), seed)
    n = len(pop)
    decided_at, decision = np.zeros(n, dtype=int), np.zeros(n, dtype=int)   # decision: 1 skilled, -1 dropped
    for c in checkpoints:
        p = skill.npmle_prob([t.pnl[:c] for t in pop], thr=spec["threshold_R"])["prob"]
        new = decision == 0
        decision[new & (p >= FLAG)], decided_at[new & (p >= FLAG)] = 1, c
        new = decision == 0
        decision[new & (p <= DROP)], decided_at[new & (p <= DROP)] = -1, c
    row = counts(spec, decision == 1, pop)
    row["trades_to_decision"] = float(decided_at[decision != 0].mean()) if (decision != 0).any() else None
    row["undecided"] = float(np.mean(decision == 0))
    return row


def run(spec, out=print):
    fixed = {}
    for n in spec["n_trades"]:
        rows = []
        for s in spec["seeds"]:
            pop = make_population(spec, n, s)
            p = skill.npmle_prob([t.pnl for t in pop], thr=spec["threshold_R"])["prob"]
            rows.append(counts(spec, p >= FLAG, pop))
        fixed[n] = pool(spec, rows)
        out(json.dumps({"n_trades": n, **fixed[n]}))
    bands = [k for k in fixed[spec["n_trades"][0]] if k.startswith("recall_band_")]
    n_min = {b[len("recall_band_"):]: next((n for n in spec["n_trades"] if (fixed[n][b] or 0) >= 0.5), f"> {max(spec['n_trades'])}") for b in bands}
    checkpoints = list(range(25, max(spec["n_trades"]) + 1, 25))
    seq_rows = [sequential_seed(spec, s, checkpoints) for s in spec["seeds"]]
    seq = {**pool(spec, seq_rows),
           "trades_to_decision": study.mean_se([r["trades_to_decision"] for r in seq_rows]),
           "undecided": study.mean_se([r["undecided"] for r in seq_rows])}
    out(json.dumps({"sequential": seq}))
    crit = {n: fixed[n]["precision"]["ci95"] for n in spec["n_trades"] if n >= 50}
    success = bool(all(ci is not None and ci[0] >= 0.70 for ci in crit.values()))
    return {"fixed": {str(k): v for k, v in fixed.items()}, "n_min_recall50": n_min, "sequential": seq,
            "criterion_precision_ci": {str(k): v for k, v in crit.items()}, "success": success}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        sm = study.smoke_spec(SPEC)
        sm["n_trades"] = [25, 50, 100]
        res = run(sm, out=lambda s: None)
        print(json.dumps({"smoke": "ok", "keys": sorted(res), "n_min": res["n_min_recall50"]}))
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
    reg.result(STUDY, {"success": res["success"], "run_commit": lock["commit"], **{k: v for k, v in res.items() if k != "success"}})
    print(json.dumps({"success": res["success"], "n_min_recall50": res["n_min_recall50"],
                      "criterion_precision_ci": res["criterion_precision_ci"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
