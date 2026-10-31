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

## Phase 2b: follow-ups approved 2026-10-30 (user), in this order
- Done 2026-10-30: SKILL-08 PASSED (see README): monitoring rule = abstain below 150 trades, then flag at two
  consecutive checkpoints with P >= 0.7 (precision 0.81).
  Was: SKILL-08: a monitoring rule that stays reliable when a firm re-checks the skill probability as trades arrive
  (SKILL-07: deciding at the first P >= 0.7 dropped precision to 0.55). Synthetic, new seeds.
- Done 2026-10-30: SURVIVAL-02 PASSED (see README): C 0.710 at day 20.
  Was: SURVIVAL-02: SURVIVAL-01 with a day-20 landmark and the skill probability and martingale test as covariates.

## Phase 3: other questions a firm asks (next)
9. Done 2026-10-29: DRIFT-01 FAILED (see README): CUSUM catches 31 % of edge losses within 150 trades at 10 % false
   alarms (57 % when the edge was > 0.25 R); EWMA worse. Bottleneck: noisy 100-trade baseline. Candidate follow-ups
   (need the user's OK): baseline from the calibrated skill score / NPMLE posterior instead of the raw warm-up mean,
   Bayesian online change-point (FORMULAS extra candidate).
   Original plan: DRIFT-01: detect the `decaying` kind (edge lost mid-period) with CUSUM / EWMA on trade results; false alarms per
   year vs detection delay.
10. Done 2026-10-29: TILT-01 FAILED on tilt only (see README): martingale flagged 99.6 % at 100 trades with 7.8 %
   false flags; tilt 26 % at 100 trades, 56 % at 200 (weak size signal). Martingale flag usable in the product now.
   Candidate follow-up (needs the user's OK): tilt from the outcome side (worse results right after losses) combined
   with size; Hawkes / frequency needs timestamps (real data).
   Original plan: TILT-01: detect `tilt` and `martingale` from size/frequency after losses (CUSUM, Hawkes intensity).
11. Done 2026-10-29: SURVIVAL-01 FAILED narrowly (see README): Cox on the first 10 days ranks breaches with C 0.700
   (needed lower bound >= 0.70), clearly better than the loss buffer alone (+0.076). Kaplan-Meier gives exact
   portfolio survival under censoring (usable now). Candidate follow-ups (need the user's OK): later landmark
   (day 20), skill probability and martingale flag as covariates.
   Original plan: SURVIVAL-01: probability that a funded account is still alive at 30 / 60 days (Kaplan-Meier with censored
   accounts; Cox model with behaviour covariates: size changes, trade frequency, recent drawdown).
12. Done 2026-10-29: PAYOUT-01 FAILED badly (see README): forecast intervals cover the realised payout in 23 % of
   books instead of 90 %, because replaying survivors' month-1 days carries their luck forward (survivorship bias).
   PAYOUT-02 (survivorship fix) FAILED too but removed the bias: coverage 0.26 -> 0.77, misses balanced; intervals
   too narrow with a shared market (month drift not identifiable from one month). Next step needs several months of
   a real book (estimate market-drift variance across months); no further iteration on the synthetic bench.
   Original plan: PAYOUT-01: distribution of next-month payouts for a whole funded book (Hawkes for clustered activity +
   extreme-value tail + simulation of correlated accounts), as a range with probabilities.
13. Optional RANK-01: Elo / Bradley-Terry ranking of traders with uncertainty, only if a firm asks for a ranking
   for capital allocation (SKILL-03 suggests a calibrated probability already covers most of this).

## Phase 4: real data (only when a firm shares trades)
REAL-01 (2026-10-30, public Hyperliquid / OKX account series, not firm trades) FAILED: probabilities near 0.5 and
miscalibrated against a market-wide positive drift; ranking slightly below the t-stat; OKX winners reverted.
Next (needs the user's OK): define outcomes relative to peers (excess over the cross-sectional median) and test on
data NOT yet seen (outcomes after 2026-10-30, or another untouched package): the 2026-10-30 package has now been
opened, so any further analysis of it is exploratory, not a confirmation.
REAL-02 (registered 2026-10-30, prospective): predictions of 'beats the median of its peers' over 2026-10-30 to
2027-01-30 frozen today (sha256 in the registry). TO DO after 2027-01-30: download a new package with the same
structure (same Hyperliquid sample, same OKX leads, data to >= 2027-01-30), check its MANIFESTs, check out the
REAL-02 registration commit, run `python3 -m tools.real_prospective --evaluate --data <2026-10-30 package>
--new-data <new package>`.
REAL-03 (registered 2026-10-30, prospective, same dates and data as REAL-02): do accounts with the lowest relative
t-stat keep lagging their peers (criterion: at most 30 % of the bottom decile beat the median), plus a median-corrected
probability; predictions frozen. Evaluate together with REAL-02: `python3 -m tools.real_prospective_v2 --evaluate ...`
from the REAL-03 registration commit.
REAL-04 (laggards, 2026-10-31) NOT VALID: gate error on HLS-04 portfolios (lab's protocol error, no result computed);
re-run as a new id with the corrected gate pending the user's OK. Done as REAL-06 (2026-10-31): FAILED the 0.30 threshold
but laggard persistence is real on all three Hyperliquid sets (bottom decile beats the median 35-38 % of the time vs
50 % by chance) and regime-dependent (absent at the 2025-10 cut). Winners do not persist. REAL-05 (trade-level product, 2026-10-31) FAILED:
the skill probability is rarely confident and miscalibrated on real trades; martingale / tilt flags point the right
way but are not significant.
Tooling done 2026-10-30: importer, frozen audit and report (`tools/audit_report.py`), pilot guide `docs/PILOT.md`.
Column names of real exports still to confirm on the first file.
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
