"""SKILL-08: a monitoring rule for the skill probability that stays reliable when a firm re-checks it as trades
arrive (SKILL-07: deciding at the first P >= 0.7 dropped precision to 0.55). ROADMAP phase 2b.

Usage:
  python3 -m tools.skill_monitor --smoke
  python3 -m tools.skill_monitor --register
  python3 -m tools.skill_monitor
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

STUDY = "SKILL-08"
SPEC = {
    "question": "Precision and recall of monitoring rules that re-check P(edge > 0.05 R) every 25 trades from trade 150",
    "bench": "BENCH-3: simulate.population_v3 with 300 trades per trader (edge = -0.05 + 0.07 * Student-t(5), "
             "plain 0.80 / martingale 0.10 / tilt 0.10); population order shuffled per seed (rng seed + 1000000)",
    "population": {"n_traders": 2000, "n_trades": 300, "risk_pct": 1.0, "edge_loc": -0.05, "edge_scale": 0.07,
                   "edge_df": 5},
    "threshold_R": 0.05,
    "checkpoints": [150, 175, 200, 225, 250, 275, 300],
    "score": "P_c = skill.npmle_prob(first c trades of every trader, thr=threshold_R)['prob'], NPMLE refitted per "
             "checkpoint on the prefix population",
    "rules": {"R0_first07": "flag at the first checkpoint with P >= 0.7 (SKILL-07's sequential rule, from 150)",
              "R1_persist2": "flag at the first checkpoint where P >= 0.7 there AND at the previous checkpoint",
              "R2_first085": "flag at the first checkpoint with P >= 0.85",
              "fixed150": "P_150 >= 0.7 (one look)", "fixed300": "P_300 >= 0.7 (one look)"},
    "label": "B: meta['edge'] (true expected P&L per trade, size included) > threshold_R",
    "edge_bands": [[0.05, 0.10], [0.10, 0.15], [0.15, None]],
    "seeds": list(range(14000, 14020)),
    "metrics": {"precision": "true flags / flags, summed over seeds, seed-bootstrap 95 % interval",
                "recall": "true flags / skilled, overall and by edge band", "trades_to_flag": "mean checkpoint of flags"},
    "success": "R1_persist2: pooled precision lower 95 % bound >= 0.70",
    "secondary": "R0, R2, fixed150, fixed300 with the same metrics; recall of R1 vs fixed300",
    "attempts": "3 monitoring rules (R0 as reference, R1 tested, R2 alternative), docs/FORMULAS.md rule 1",
}
RULES = ("R0_first07", "R1_persist2", "R2_first085", "fixed150", "fixed300")


def make_population(spec, seed):
    p = spec["population"]
    pop = simulate.population_v3(p["n_traders"], p["n_trades"], seed, edge_loc=p["edge_loc"],
                                 edge_scale=p["edge_scale"], edge_df=p["edge_df"], risk=p["risk_pct"])
    return [pop[i] for i in np.random.default_rng(seed + 1_000_000).permutation(len(pop))]


def flags_for(P, cps):
    """P: (checkpoints, traders). Returns {rule: (flag bool array, checkpoint of the flag or 0)}."""
    k, n = P.shape
    out = {}
    for rule, test in (("R0_first07", P >= 0.7), ("R2_first085", P >= 0.85),
                       ("R1_persist2", np.vstack([np.zeros((1, n), dtype=bool), (P[1:] >= 0.7) & (P[:-1] >= 0.7)]))):
        hit = test.any(axis=0)
        first = np.where(hit, np.argmax(test, axis=0), -1)
        out[rule] = (hit, np.where(hit, np.array(cps)[np.maximum(first, 0)], 0))
    out["fixed150"] = (P[0] >= 0.7, np.where(P[0] >= 0.7, cps[0], 0))
    out["fixed300"] = (P[-1] >= 0.7, np.where(P[-1] >= 0.7, cps[-1], 0))
    return out


def seed_counts(spec, seed):
    pop = make_population(spec, seed)
    cps = spec["checkpoints"]
    P = np.vstack([skill.npmle_prob([t.pnl[:c] for t in pop], thr=spec["threshold_R"])["prob"] for c in cps])
    edge = np.array([t.meta["edge"] for t in pop])
    y = edge > spec["threshold_R"]
    out = {}
    for rule, (f, at) in flags_for(P, cps).items():
        c = {"flags": int(f.sum()), "true_flags": int((f & y).sum()), "skilled": int(y.sum()),
             "sum_at": float(at[f].sum())}
        for lo, hi in spec["edge_bands"]:
            m = (edge > lo) & ((edge <= hi) if hi is not None else True)
            c[f"band_{lo}-{hi if hi is not None else 'inf'}"] = [int((f & m).sum()), int(m.sum())]
        out[rule] = c
    return out


def run(spec, out=print):
    rows = [seed_counts(spec, s) for s in spec["seeds"]]
    res = {}
    for rule in RULES:
        col = lambda k: [r[rule][k] for r in rows]
        flags = sum(col("flags"))
        cell = {"precision": study.pooled_ratio_ci(col("true_flags"), col("flags")),
                "recall": round(sum(col("true_flags")) / max(sum(col("skilled")), 1), 4),
                "flags_per_seed": study.mean_se(col("flags")),
                "trades_to_flag": round(sum(col("sum_at")) / flags, 1) if flags else None}
        for k in rows[0][rule]:
            if k.startswith("band_"):
                hit, tot = sum(r[rule][k][0] for r in rows), sum(r[rule][k][1] for r in rows)
                cell[f"recall_{k}"] = round(hit / tot, 4) if tot else None
        res[rule] = cell
        out(json.dumps({rule: cell}))
    ci = res["R1_persist2"]["precision"]["ci95"]
    res["success"] = bool(ci is not None and ci[0] >= 0.70)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        sm = study.smoke_spec(SPEC)
        res = run(sm, out=lambda s: None)
        print(json.dumps({"smoke": "ok", "rules": sorted(k for k in res if k != "success")}))
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
    print(json.dumps({"success": res["success"], **{r: {k: res[r][k] for k in ("precision", "recall", "trades_to_flag")}
                                                    for r in RULES}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
