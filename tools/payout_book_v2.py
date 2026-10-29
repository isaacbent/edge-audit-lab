"""PAYOUT-02: PAYOUT-01 with the survivorship fix: common-day resampling where each account's days are re-centred on
its NPMLE posterior mean day P&L (prior fitted on all accounts). Same criterion as PAYOUT-01, new books.

Usage:
  python3 -m tools.payout_book_v2 --smoke
  python3 -m tools.payout_book_v2 --register
  python3 -m tools.payout_book
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import payout, simulate, study  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "PAYOUT-02"
SPEC = {
    "question": "Coverage of the central 90 % forecast interval for next month's total payouts of a funded book, "
                "with the survivorship fix (accounts re-centred on their calibrated edge estimate)",
    "bench": "BENCH-3P: simulate.book_days (BENCH-3 traders with noisy sizes, Poisson trades per day (median rate 3), "
             "42 trading days = 2 months of 21, shared daily market move N(0, corr_sd) R on every trade)",
    "n_accounts": 1000,
    "corr_sd": {"primary": 0.2, "control": 0.0},
    "books": list(range(13000, 13200)),
    "rules": "EOD: breach when equity since funding <= -10 % or a day's P&L <= -5 %; at month end alive accounts are "
             "paid 0.8 x max(0, month P&L) (edgelab.payout.month_payout)",
    "forecast": "from month 1 (days 0-20) only: payout.forecast_payouts with 500 simulated months, methods "
                "'common_days_shrunk' (tested) and 'common_days' (PAYOUT-01 reference); interval = 5 % and 95 % quantiles; simulation seed = book seed",
    "truth": "realised month-2 total payout of the same book (days 21-41)",
    "metrics": {"coverage": "share of books whose realised total falls inside the interval (Wilson 95 % interval)",
                "below / above": "misses on each side", "width": "median interval width / median realised total",
                "pit": "share of simulated totals <= realised (histogram in deciles)"},
    "success": "corr_sd 0.2: coverage(common_days_shrunk) between 0.85 and 0.95 (point estimate)",
    "secondary": "common_days (unfixed) as reference; both methods on the corr_sd 0 control",
    "notes": "method fixed after PAYOUT-01 and a logged exploration on dev books 9600-9639 (coverage 0.83 at corr 0.2, 1.0 at corr 0: the remaining miss is the month's market drift, not identifiable from one month); Hawkes activity clustering and an extreme-value tail (ROADMAP) need timestamps / real books; FORMULAS "
             "rule 3 (simplest piece first). One realised month per book, so coverage is measured across 200 books",
}


def wilson(k, n):
    if n == 0:
        return None
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(c - h), 4), round(float(c + h), 4)]


def book_result(spec, seed, corr_sd):
    m, _ = simulate.book_days(spec["n_accounts"], seed, days=42, corr_sd=corr_sd)
    m1, m2 = m[:, :21], m[:, 21:]
    eq1 = np.cumsum(m1, axis=1)
    alive = ~(((eq1 <= -10.0) | (m1 <= -5.0)).any(axis=1))
    truth, _, _ = payout.month_payout(eq1[:, -1], alive, m2)
    out = {"truth": float(truth)}
    for method in ("common_days_shrunk", "common_days"):
        sim = payout.forecast_payouts(m1, method, sims=500, seed=seed)
        lo, hi = np.quantile(sim, [0.05, 0.95])
        out[method] = {"lo": float(lo), "hi": float(hi), "pit": float(np.mean(sim <= truth))}
    return out


def summarise(rows):
    res = {}
    truth = np.array([r["truth"] for r in rows])
    for method in ("common_days_shrunk", "common_days"):
        lo = np.array([r[method]["lo"] for r in rows]); hi = np.array([r[method]["hi"] for r in rows])
        inside = (truth >= lo) & (truth <= hi)
        pit = np.array([r[method]["pit"] for r in rows])
        res[method] = {"coverage": round(float(inside.mean()), 4), "coverage_ci95": wilson(int(inside.sum()), len(rows)),
                       "below": round(float(np.mean(truth < lo)), 4), "above": round(float(np.mean(truth > hi)), 4),
                       "rel_width": round(float(np.median(hi - lo) / np.median(truth)), 4),
                       "pit_deciles": np.histogram(pit, bins=10, range=(0, 1))[0].tolist()}
    res["median_truth"] = round(float(np.median(truth)), 2)
    return res


def run(spec, out=print):
    res = {}
    for name, cs in spec["corr_sd"].items():
        res[name] = summarise([book_result(spec, s, cs) for s in spec["books"]])
        out(json.dumps({name: res[name]}))
    c = res["primary"]["common_days_shrunk"]["coverage"]
    res["success"] = bool(0.85 <= c <= 0.95)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        sm = json.loads(json.dumps(SPEC))
        sm.update({"n_accounts": 100, "books": [9983, 9984, 9985]})
        res = run(sm, out=lambda s: None)
        print(json.dumps({"smoke": "ok", "keys": sorted(res["primary"])}))
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
    print(json.dumps({"success": res["success"], **{k: v for k, v in res.items() if k != "success"}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
