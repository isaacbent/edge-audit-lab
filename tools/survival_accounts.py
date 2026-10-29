"""SURVIVAL-01: survival of funded accounts. Can a Cox model on what a firm sees in an account's first 10 days rank
which accounts will breach, better than the remaining loss buffer alone? Kaplan-Meier portfolio survival at 30 / 60
days as a check. ROADMAP phase 3, step 11.

Usage:
  python3 -m tools.survival_accounts --smoke
  python3 -m tools.survival_accounts --register
  python3 -m tools.survival_accounts
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import behaviour, simulate, study, survival  # noqa: E402
from edgelab.atomic_io import atomic_open  # noqa: E402
from edgelab.registry import Registry  # noqa: E402

STUDY = "SURVIVAL-01"
COVARIATES = ["buffer", "mean_pnl", "sd_pnl", "trades_per_day", "size_sd", "log_size_ratio_after_loss"]
SPEC = {
    "question": "Ranking of breach risk from an account's first 10 days (Cox) vs the remaining loss buffer alone; "
                "Kaplan-Meier portfolio survival at 30 / 60 days under censoring",
    "bench": "BENCH-3S: simulate.account_paths (BENCH-3 traders with noisy sizes, Poisson trades per day with "
             "lognormal rate (median 3, sigma 0.5), 90 days, breach at cumulative P&L <= -10 % or day P&L <= -5 %, "
             "1 % risk per trade) + simulate.observe_accounts (start U[0, 120], observation ends day 150, "
             "voluntary exit 0.5 % per day)",
    "n_accounts": 2000,
    "fit_seeds": list(range(11000, 11010)),
    "eval_seeds": list(range(11010, 11020)),
    "landmark_day": 10,
    "landmark_set": "accounts observed alive and uncensored after day 10 (time > 10); outcome = time - 10, event",
    "covariates": {"buffer": "10 + cumulative P&L at the end of day 9 (% left before the static max loss)",
                   "mean_pnl": "mean trade P&L, days 0-9", "sd_pnl": "sd of trade P&L, days 0-9",
                   "trades_per_day": "trades in days 0-9 / 10", "size_sd": "sd of log position size, days 0-9",
                   "log_size_ratio_after_loss": "log(behaviour.size_ratio_after_loss), days 0-9"},
    "model": "survival.cox_fit on standardised covariates (fit-seed means / sds), fitted once on the landmark sets of "
             "all fit seeds pooled; risk score = X @ beta on eval seeds",
    "baseline": "risk = -buffer",
    "metrics": {"c_cox": "Harrell's C per eval seed", "c_buffer": "Harrell's C of -buffer per eval seed",
                "km": "Kaplan-Meier S(30), S(60) from censored data vs true S from the full paths (all 20 seeds), "
                      "and the naive estimate that counts censored accounts as alive"},
    "success": "all of: mean C(Cox) over eval seeds, lower 95 % t bound >= 0.70 AND per-seed paired C(Cox) - C(buffer), "
               "lower 95 % bound > 0",
    "secondary": "KM - truth and naive - truth at 30 / 60 days (paired over seeds); Cox coefficients",
    "notes": "simulation-based truth (full uncensored paths); covariates use only trades a firm sees by day 10",
}


def landmark_rows(accounts, obs, day0):
    rows, times, events = [], [], []
    for a, (tm, ev) in zip(accounts, obs):
        if tm <= day0:
            continue
        tr, m = a["trader"], a["day"] < day0
        pnl, size = tr.pnl[m], tr.size[m]
        rows.append([10.0 + pnl.sum(), pnl.mean() if pnl.size else 0.0, pnl.std(ddof=1) if pnl.size > 1 else 0.0,
                     pnl.size / day0, np.log(size).std() if size.size > 1 else 0.0,
                     np.log(behaviour.size_ratio_after_loss(pnl, size)) if pnl.size > 2 else 0.0])
        times.append(tm - day0)
        events.append(ev)
    return np.array(rows), np.array(times, dtype=float), np.array(events, dtype=bool)


def seed_data(spec, seed):
    acc = simulate.account_paths(spec["n_accounts"], seed)
    obs = simulate.observe_accounts(acc, seed)
    return acc, obs


def run(spec, out=print):
    data = {s: seed_data(spec, s) for s in spec["fit_seeds"] + spec["eval_seeds"]}
    km = {d: {"km": [], "naive": [], "truth": []} for d in (30, 60)}
    for s, (acc, obs) in data.items():
        b = np.array([a["breach_day"] if a["breach_day"] is not None else 10 ** 6 for a in acc])
        tt, ee = np.array([o[0] for o in obs]), np.array([o[1] for o in obs], dtype=bool)
        for d in (30, 60):
            km[d]["truth"].append(float(np.mean(b + 1 > d)))
            km[d]["km"].append(survival.km_survival(tt, ee, d))
            km[d]["naive"].append(float(1 - np.mean(ee & (tt <= d))))
    km_res = {str(d): {"truth": study.mean_se(v["truth"]), "km_minus_truth": study.paired_diff(v["km"], v["truth"]),
                       "naive_minus_truth": study.paired_diff(v["naive"], v["truth"])} for d, v in km.items()}
    out(json.dumps({"km": km_res}))
    day0 = spec["landmark_day"]
    fit = [landmark_rows(*data[s], day0) for s in spec["fit_seeds"]]
    X = np.vstack([f[0] for f in fit])
    mu, sd = X.mean(axis=0), np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    beta = survival.cox_fit((X - mu) / sd, np.concatenate([f[1] for f in fit]), np.concatenate([f[2] for f in fit]))
    c_cox, c_buf, n_land = [], [], []
    for s in spec["eval_seeds"]:
        Xe, te, ee = landmark_rows(*data[s], day0)
        c_cox.append(survival.concordance(((Xe - mu) / sd) @ beta, te, ee))
        c_buf.append(survival.concordance(-Xe[:, 0], te, ee))
        n_land.append(int(len(te)))
    res = {"km": km_res, "cox_beta_standardised": dict(zip(COVARIATES, np.round(beta, 4).tolist())),
           "c_cox": study.mean_se(c_cox), "c_buffer": study.mean_se(c_buf),
           "c_cox_minus_buffer": study.paired_diff(c_cox, c_buf), "landmark_accounts_per_seed": study.mean_se(n_land)}
    t_lo = res["c_cox"]["mean"] - 2.262 * res["c_cox"]["se"]          # t(0.975, 9 df)
    res["c_cox_lower95"] = round(t_lo, 4)
    res["success"] = bool(t_lo >= 0.70 and res["c_cox_minus_buffer"]["ci95"][0] > 0)
    out(json.dumps({k: v for k, v in res.items() if k != "km"}))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--registry", default=os.path.join(ROOT, "state", "registry.jsonl"))
    args = ap.parse_args(argv)
    if args.smoke:
        sm = json.loads(json.dumps(SPEC))
        sm.update({"n_accounts": 300, "fit_seeds": [9996, 9997], "eval_seeds": [9998, 9999]})
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
    print(json.dumps({"success": res["success"], "c_cox": res["c_cox"], "c_buffer": res["c_buffer"],
                      "c_cox_minus_buffer": res["c_cox_minus_buffer"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
