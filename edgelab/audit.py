"""Frozen audit of a firm's funded book (phase 4). Only pieces that held up in the studies, with their thresholds
fixed in advance; nothing here is tuned on the firm's data.

Per account:
  - trade results in R = profit / the account's median loss (risk per trade is not in the export; on the benches
    losses are about -1 R). Accounts with fewer than 5 losses have no stable R scale and are not scored.
  - skill: P(edge > 0.05 R on top of costs) = skill.npmle_prob(..., thr=0.05), population prior fitted on the
    accounts in the file that have an R scale (SKILL-06 passed); shown only with >= 150 trades (SKILL-07: P >= 0.7
    trustworthy from 150 trades); needs at least 2 such accounts
  - martingale flag: behaviour.size_after_loss_p + Benjamini-Hochberg q 0.10 over accounts with >= 50 trades and a
    position size on every trade (TILT-01: recall 99.5 % at 50 trades, 99.6 % at 100, about 8 % false flags; tilt
    NOT reliably detected)
"""
from __future__ import annotations

import numpy as np

from . import behaviour, skill, stats

MIN_TRADES_SKILL = 150
MIN_TRADES_MARTINGALE = 50
MIN_LOSSES_R = 5
THRESHOLD_R = 0.05
FLAG_P = 0.7


def r_multiples(profit):
    """Results in units of the account's median loss, or None when it has fewer than MIN_LOSSES_R losses."""
    x = np.asarray(profit, dtype=float)
    losses = -x[x < 0]
    if losses.size < MIN_LOSSES_R:
        return None
    return x / float(np.median(losses))


def audit(trades: list) -> dict:
    """trades: importer.read_trades output (any number of accounts). Returns {accounts: [...], book: {...}}."""
    by = {}
    for t in trades:
        by.setdefault(t["account"], []).append(t)
    ids = sorted(by)
    n = np.array([len(by[a]) for a in ids])
    pnl = [r_multiples([t["profit"] for t in by[a]]) for a in ids]
    has_r = np.array([p is not None for p in pnl])
    prob = np.full(len(ids), np.nan)
    if has_r.sum() >= 2:
        idx = np.nonzero(has_r)[0]
        prob[idx] = skill.npmle_prob([pnl[i] for i in idx], thr=THRESHOLD_R)["prob"]
    sizes = [[t["volume"] for t in by[a]] for a in ids]
    full_size = np.array([all(v is not None for v in s) for s in sizes])
    has_volume = any(any(v is not None for v in s) for s in sizes)
    eligible = has_r & full_size & (n >= MIN_TRADES_MARTINGALE)
    flags, pvals = np.zeros(len(ids), dtype=bool), np.full(len(ids), np.nan)
    if eligible.any():
        e = np.nonzero(eligible)[0]
        pv = np.array([behaviour.size_after_loss_p(pnl[i], np.array(sizes[i], dtype=float)) for i in e])
        pvals[e] = pv
        flags[e] = stats.benjamini_hochberg(pv, q=0.10)
    rows = []
    for i, a in enumerate(ids):
        if not has_r[i]:
            verdict = f"abstain (fewer than {MIN_LOSSES_R} losses: no R scale)"
        elif has_r.sum() < 2:
            verdict = "abstain (needs >= 2 accounts with an R scale in the file)"
        elif n[i] < MIN_TRADES_SKILL:
            verdict = f"abstain (< {MIN_TRADES_SKILL} trades)"
        else:
            verdict = "skilled (P >= 0.7)" if prob[i] >= FLAG_P else ("unlikely" if prob[i] < 0.2 else "uncertain")
        shown = verdict[:7] != "abstain"
        if eligible[i]:
            mg = "FLAG" if flags[i] else "-"
        elif not has_volume:
            mg = "not checked (no volume column)"
        elif not full_size[i]:
            mg = "not checked (missing sizes)"
        elif not has_r[i]:
            mg = "not checked (no R scale)"
        else:
            mg = f"not checked (< {MIN_TRADES_MARTINGALE} trades)"
        rows.append({"account": a, "trades": int(n[i]),
                     "mean_R": None if pnl[i] is None else round(float(pnl[i].mean()), 3),
                     "skill_probability": round(float(prob[i]), 3) if shown else None, "skill_verdict": verdict,
                     "martingale": mg, "martingale_flag": bool(flags[i]) if eligible[i] else None,
                     "size_ratio_after_loss": round(behaviour.size_ratio_after_loss(pnl[i], np.array(sizes[i], dtype=float)), 3)
                     if eligible[i] else None,
                     "martingale_p": round(float(pvals[i]), 5) if eligible[i] else None})
    book = {"accounts": len(ids), "trades": int(n.sum()),
            "accounts_scored": sum(r["skill_probability"] is not None for r in rows),
            "accounts_abstain": sum(r["skill_probability"] is None for r in rows),
            "skilled": sum(r["skill_verdict"].startswith("skilled") for r in rows),
            "martingale_flagged": int(flags.sum()), "martingale_eligible": int(eligible.sum()),
            "volume_column": has_volume}
    return {"accounts": rows, "book": book,
            "frozen": {"threshold_R": THRESHOLD_R, "flag_p": FLAG_P, "min_trades_skill": MIN_TRADES_SKILL,
                       "min_trades_martingale": MIN_TRADES_MARTINGALE, "min_losses_r": MIN_LOSSES_R, "bh_q": 0.10}}
