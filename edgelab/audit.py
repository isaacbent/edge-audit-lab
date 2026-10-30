"""Frozen audit of a firm's funded book (phase 4). Only pieces that held up in the studies, with their thresholds
fixed in advance; nothing here is tuned on the firm's data.

Per account:
  - trade results in R: profit / account size, divided by the account's median loss size (R estimate: real risk per
    trade is not in the export; on the benches risk is 1 % and losses are about -1 R)
  - skill: P(edge > 0.05 R) = skill.npmle_prob(..., thr=0.05), population prior fitted on all accounts in the file
    (SKILL-06 passed); shown only with >= 150 trades, else "abstain" (SKILL-07: P >= 0.7 trustworthy from 150 trades)
  - martingale flag: behaviour.size_after_loss_p + Benjamini-Hochberg q 0.10 over accounts with >= 50 trades
    (TILT-01: martingale recall 99.6 %, 7.8 % false flags; tilt NOT reliably detected)
Book level: number of accounts, trades, share flagged, skill bands.
"""
from __future__ import annotations

import numpy as np

from . import behaviour, skill, stats

MIN_TRADES_SKILL = 150
MIN_TRADES_MARTINGALE = 50
THRESHOLD_R = 0.05
FLAG_P = 0.7


def r_multiples(profit, account_size: float) -> np.ndarray:
    pct = 100.0 * np.asarray(profit, dtype=float) / float(account_size)
    losses = -pct[pct < 0]
    r = np.median(losses) if losses.size >= 5 else (np.median(np.abs(pct)) if pct.size else 1.0)
    return pct / max(float(r), 1e-9)


def audit(trades: list, account_size: float) -> dict:
    """trades: importer.read_trades output (any number of accounts). Returns {accounts: [...], book: {...}}."""
    by = {}
    for t in trades:
        by.setdefault(t["account"], []).append(t)
    ids = sorted(by)
    pnl = [r_multiples([t["profit"] for t in by[a]], account_size) for a in ids]
    size = [np.array([t["volume"] for t in by[a]], dtype=float) for a in ids]
    n = np.array([p.size for p in pnl])
    prob = skill.npmle_prob(pnl, thr=THRESHOLD_R)["prob"] if len(ids) >= 2 and n.min() >= 3 else np.full(len(ids), np.nan)
    eligible = n >= MIN_TRADES_MARTINGALE
    flags = np.zeros(len(ids), dtype=bool)
    pvals = np.full(len(ids), np.nan)
    if eligible.any():
        pv = np.array([behaviour.size_after_loss_p(pnl[i], size[i]) for i in np.nonzero(eligible)[0]])
        pvals[eligible] = pv
        flags[eligible] = stats.benjamini_hochberg(pv, q=0.10)
    rows = []
    for i, a in enumerate(ids):
        if n[i] >= MIN_TRADES_SKILL and np.isfinite(prob[i]):
            verdict = "skilled (P >= 0.7)" if prob[i] >= FLAG_P else ("unlikely" if prob[i] < 0.2 else "uncertain")
            p_show = round(float(prob[i]), 3)
        else:
            verdict, p_show = f"abstain (< {MIN_TRADES_SKILL} trades)", None
        rows.append({"account": a, "trades": int(n[i]), "mean_R": round(float(pnl[i].mean()), 3),
                     "skill_probability": p_show, "skill_verdict": verdict,
                     "martingale_flag": bool(flags[i]) if eligible[i] else None,
                     "size_ratio_after_loss": round(behaviour.size_ratio_after_loss(pnl[i], size[i]), 3),
                     "martingale_p": None if not eligible[i] else round(float(pvals[i]), 5)})
    shown = [r for r in rows if r["skill_probability"] is not None]
    book = {"accounts": len(ids), "trades": int(n.sum()), "accounts_scored": len(shown),
            "accounts_abstain": len(rows) - len(shown),
            "skilled": sum(r["skill_verdict"].startswith("skilled") for r in rows),
            "martingale_flagged": int(flags.sum()), "martingale_eligible": int(eligible.sum())}
    return {"accounts": rows, "book": book,
            "frozen": {"threshold_R": THRESHOLD_R, "flag_p": FLAG_P, "min_trades_skill": MIN_TRADES_SKILL,
                       "min_trades_martingale": MIN_TRADES_MARTINGALE, "bh_q": 0.10}}
