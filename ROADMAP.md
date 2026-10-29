# Roadmap (as of 2026-10-29)

Goal: a skill score a prop firm can trust, i.e. a CALIBRATED probability that a funded trader has a real edge,
plus how many trades it takes to trust it. Ranking alone is not the differentiator: the plain t-stat already
ranks as well as anything tested (SKILL-03).

## Where we are
- SKILL-01/02/03 done (all FAILED their criteria, see README). Robust findings across all three:
  challenge rule picks only 12-19 % skilled traders; any serious score picks 2-4x better; win rate is useless once
  styles differ (AUC 0.50); the NPMLE probability is over-confident (top decile 0.49 predicted vs 0.27 actual).

## Phase 1: calibration (next)
1. SKILL-04 diagnosis (no criterion, report only, still pre-registered as a diagnostic):
   - where the over-confidence comes from: prior mass slightly above 0 assigned to breakeven traders (true mu = 0)?
     noisy per-trader standard errors with fat tails / martingale variance? normal likelihood for xbar at n = 50-100?
   - compare on BENCH-2: P(mu > 0) vs P(mu > delta) for a small practical threshold; pooled vs own variance;
     bootstrap or t likelihood instead of normal.
2. SKILL-05 (with criterion, decided BEFORE running): the best fix from SKILL-04 on new seeds. Suggested criterion:
   primary cell (100 trades, edge 0.15 R) |predicted - actual| <= 0.10 in every decile AND AUC within 0.01 of t-stat
   AND top-5 % precision >= 2x challenge.
   Stop rule: if SKILL-05 fails, report probabilities only in coarse bands (low / medium / high) that ARE calibrated,
   and say so; do not keep iterating on the same bench.

## Phase 2: minimum trades
3. SKILL-06: for each edge size, how many trades until the calibrated probability separates skilled from lucky
   (e.g. recall >= 50 % at precision >= 70 %). Output = the table a firm sees ("with fewer than N trades we abstain").

## Phase 3: other questions a firm asks (only after phase 1 passes)
4. DRIFT-01: detect the `decaying` kind (edge lost mid-period) with CUSUM / EWMA on trade results; false alarms per
   year vs detection delay.
5. TILT-01: detect `tilt` and `martingale` from size/frequency after losses (CUSUM, Hawkes intensity).
6. SURVIVAL-01: probability that a funded account is still alive at 30 / 60 days (Kaplan-Meier with censored
   accounts; Cox model with behaviour covariates: size changes, trade frequency, recent drawdown).
7. PAYOUT-01: distribution of next-month payouts for a whole funded book (Hawkes for clustered activity +
   extreme-value tail + simulation of correlated accounts), as a range with probabilities.
8. Optional RANK-01: Elo / Bradley-Terry ranking of traders with uncertainty, only if a firm asks for a ranking
   for capital allocation (SKILL-03 suggests a calibrated probability already covers most of this).

## Phase 4: real data (only when a firm shares trades)
9. Import an anonymised trade export (MT5 / cTrader / Match-Trader CSV), run the frozen calibrated score, and test
   it retrospectively: score at day 30 vs who was paid / breached later. This is the free 48-hour pilot offered in
   the outreach emails.

## Pieces still to copy from the Research Engine when needed
- Deflated Sharpe (`research_engine/robustness.dsr`): correct for the number of traders a firm evaluates.

## Formula catalogue
- docs/FORMULAS.md: all 26 catalogue formulas + 4 extra candidates, each marked PLANNED / CANDIDATE / EXCLUDED for
  trader analysis. Follow its rules: a CANDIDATE becomes a study only with a written case and the user's OK.

## Not now
- No product/platform integrations, no UI, no new features before a firm replies (user decision 2026-10-27).
- No price-prediction strategies here (that is the Research Engine's job).
