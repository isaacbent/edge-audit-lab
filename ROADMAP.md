# Roadmap (as of 2026-10-29)

Goal: a skill score a prop firm can trust, i.e. a CALIBRATED probability that a funded trader has a real edge,
plus how many trades it takes to trust it. Ranking alone is not the differentiator: the plain t-stat already
ranks as well as anything tested (SKILL-03).

## Where we are
- SKILL-01/02/03 FAILED their criteria (see README). Robust findings: the challenge rule picks only 12-19 % skilled
  traders; any serious score picks 2-4x better; win rate is useless once styles differ (AUC 0.50).
- SKILL-04 PASSED (BENCH-2): the NPMLE probability of an edge above costs, P(mu > cost), is calibrated (top decile
  0.285 predicted vs 0.279 actual). The over-confidence came from breakeven traders (mu = 0) leaking probability
  above 0. Caveat: SKILL-04 merged diagnosis and fix, and its criterion was SKILL-03's (weaker than the one below).
- SKILL-06 PASSED on BENCH-3 (calibrated and AUC above the t-stat in every cell; weak below ~50 trades).
- SKILL-05 (diagnostic, BENCH-2) done: the threshold P(mu > cost) is the fix; pooled variance, t and bootstrap
  likelihoods and isotonic add nothing (see README). Candidate for SKILL-06: NPMLE, own variance, normal, P(mu > cost).

## Phase 1a: infrastructure (next, before any new study)
Done 2026-10-29 (`edgelab/study.py`, CI, CLAUDE.md rules, post-hoc SE notes for SKILL-02/03):
1. Pre-registration lock: store the git commit and a hash of `edgelab/` and the study script (plus numpy / scipy
   versions) in the prereg; refuse to run if they differ or if those files have uncommitted changes (`state/` is
   not checked, the run writes there).
2. Uncertainty: report the standard error across seeds; write criteria on paired differences per seed
   (e.g. AUC(score) - AUC(t-stat) with a 95 % interval), not on bare means.
   Post-hoc note (verdicts unchanged, SKILL-02/03 stay FAILED): seed standard errors of SKILL-02/03, recomputed
   with their unchanged code and seeds.
3. Exploration seeds 9000-9999 only, every attempt logged in the registry; exploration never confirms anything.
4. `--smoke` mode in every study script: tiny population, no registry writes.
5. GitHub Actions running `python3 -m pytest -q` on every push.

## Phase 1b: BENCH-3
Done 2026-10-29 (`simulate.population_v3`); skilled = true expected P&L per trade (size included) > threshold
(user decision, option B); behaviour-adjusted label reported as secondary.
6. New bench functions (BENCH-1/2 untouched): continuous true edges (skilled = edge > cost, threshold fixed in the
   pre-registration), variable number of trades per trader; cross-trader correlation (shared market days) as an
   option, off by default.

## Phase 1c: calibration on BENCH-3
Done 2026-10-29: SKILL-06 PASSED (see README). Phase 1 complete; open limit: calibration with fewer than ~50 trades
(decile gap 0.10-0.13 at 30-100 trades), which is phase 2's question.
7. SKILL-06 (with criterion, new seeds): the best fix from SKILL-04/05 on BENCH-3, BENCH-2 reported as control.
   Suggested criterion (primary cell): |predicted - actual| <= 0.10 in every decile AND AUC(score) - AUC(t-stat)
   >= -0.01 (paired, 95 % interval) AND top-5 % precision >= 2x challenge.
   Stop rule: if SKILL-06 fails, report probabilities only in coarse bands (low / medium / high) that ARE
   calibrated, and say so; do not keep iterating on the same bench.

## Phase 2: minimum trades
Done 2026-10-29: SKILL-07 FAILED (see README). P >= 0.7 is trustworthy only from 150 trades (precision 0.77);
below that the score is over-confident in its top tail (flagged traders: P about 0.85, precision 0.31-0.69), driven
by high win-rate styles whose rare large losses are not yet visible. Recall is low (edge > 0.15 R: 45 % at 300 trades).
Decision (user, 2026-10-29): option A. Product rule = abstain below 150 trades, give the calibrated probability from
150 trades on. One fix attempt (moderated variance, Smyth 2004) was explored on dev seeds 9400-9409 and dropped
(precision at 100 trades 0.63 -> 0.68; moderation too weak, d0 about 3); not pre-registered. Phase 2 closed.
8. SKILL-07: for each edge size, how many trades until the calibrated probability separates skilled from lucky
   (e.g. recall >= 50 % at precision >= 70 %). Output = the table a firm sees ("with fewer than N trades we abstain").

## Phase 3: other questions a firm asks (next)
9. DRIFT-01: detect the `decaying` kind (edge lost mid-period) with CUSUM / EWMA on trade results; false alarms per
   year vs detection delay.
10. TILT-01: detect `tilt` and `martingale` from size/frequency after losses (CUSUM, Hawkes intensity).
11. SURVIVAL-01: probability that a funded account is still alive at 30 / 60 days (Kaplan-Meier with censored
   accounts; Cox model with behaviour covariates: size changes, trade frequency, recent drawdown).
12. PAYOUT-01: distribution of next-month payouts for a whole funded book (Hawkes for clustered activity +
   extreme-value tail + simulation of correlated accounts), as a range with probabilities.
13. Optional RANK-01: Elo / Bradley-Terry ranking of traders with uncertainty, only if a firm asks for a ranking
   for capital allocation (SKILL-03 suggests a calibrated probability already covers most of this).

## Phase 4: real data (only when a firm shares trades)
14. Import an anonymised trade export (MT5 / cTrader / Match-Trader CSV), run the frozen calibrated score, and test
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
