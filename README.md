# edge-audit-lab

Research lab for Edge Audit (GraniteStats): synthetic trader bench + statistical pieces to tell
funded traders with a real edge from lucky ones, detect deterioration, account survival and payout risk.

Independent from the Research Engine: the few utilities it needs are copied (`edgelab/stats.py`,
`edgelab/atomic_io.py`, simplified `edgelab/registry.py`), never imported.

Rules: every study is pre-registered in `state/registry.jsonl` (append-only, hash-chained) before it runs;
a study whose code differs from its pre-registration refuses to run; failed studies are kept, never re-tuned.

## Layout
- `edgelab/simulate.py`  synthetic traders: BENCH-1 (kinds), BENCH-2 (styles), BENCH-3 (continuous edges, variable n); challenge rule
- `edgelab/stats.py`     sign-flip and t p-values, Benjamini-Hochberg, Buhlmann credibility
- `edgelab/drift.py`     CUSUM / EWMA edge-loss detectors with warm-up standardisation and threshold calibration
- `edgelab/registry.py`  pre-registration / results log
- `edgelab/study.py`     code lock, seed standard errors / paired differences, --smoke helper
- `tools/skill_power.py`, `skill_rank.py`, `skill_rank_v2.py`, `skill_rank_v3.py`, `skill_diag.py`, `skill_calib_b3.py`, `skill_mintrades.py`  SKILL-01..07 studies; `tools/drift_detect.py` DRIFT-01
- `tests/`               `python3 -m pytest -q`

## Studies
| Study | Date | Result |
| --- | --- | --- |
| SKILL-01 | 2026-10-29 | FAILED the pre-registered criterion (recall 1.7 % < 30 % at 100 trades, edge 0.15 R). Challenge rule: only 12-19 % of passers are skilled; BH flags are 97-100 % precise but recall is low for small edges (38 % at 0.25 R / 100 trades, 80 % at 0.25 R / 200 trades). Placebo: 0 false flags. |
| SKILL-02 | 2026-10-29 | FAILED the pre-registered criterion: top-5 % precision of the skill probability 0.49 vs challenge 0.14 (passes 2x), but its AUC 0.913 < win rate 0.916. Every graded score beats the challenge rule 2-4x; win rate wins because the bench's noise is symmetric (all traders share one payoff shape), a bench limitation. Skill probability over-confident (top decile 0.71 predicted vs 0.33 actual). Review: over-confidence came from breakeven traders leaking probability above 0, resolved in SKILL-04; the win-rate top-5 % precision is inflated by about 0.03 by tie-breaking (notes in the registry). |
| SKILL-03 | 2026-10-29 | FAILED (BENCH-2, heterogeneous styles). Win rate collapses to AUC 0.50 (bench now realistic). NPMLE probability AUC 0.876 vs profit factor 0.879 and t-stat 0.882; top-5 % precision 0.37 vs challenge 0.14 (2.6x). Calibration improved but still over-confident: top decile 0.49 predicted vs 0.27 actual (normal prior: 0.58 vs 0.24). |
| SKILL-04 | 2026-10-29 | PASSED (BENCH-2). NPMLE probability of an edge above costs, P(mean > 0.05 R): top decile 0.285 predicted vs 0.279 actual (P(mean > 0): 0.48 vs 0.26; breakeven traders at exactly 0 leaked probability above 0). AUC 0.879 vs t-stat 0.871, profit factor 0.869; top-5 % precision 0.40 vs challenge 0.14 (2.9x). Top-decile gap <= 0.07 and highest AUC in all 9 cells. Fix found on dev seeds 9000-9005 before registration. |
| SKILL-05 | 2026-10-29 | DIAGNOSTIC (BENCH-2, no criterion). The threshold is what matters: NPMLE P(mean > 0) top decile +0.23 over-confident, P(mean > cost) +0.003 (eval seeds, primary cell). Pooled variance breaks calibration (gap 0.24 either threshold); Student-t likelihood = normal; bootstrap-t under-confident at cost (top decile -0.16); isotonic fixes P(mean > 0) (gap 0.06) but needs outcome labels and gains nothing over P(mean > cost). AUC of P(mean > cost) 0.884 vs t-stat 0.876. Outside the primary cell the seed-averaged decile gap reaches 0.08-0.11 at 50 and 200 trades. |
| SKILL-06 | 2026-10-29 | PASSED (BENCH-3: continuous edges, 50-200 trades per trader). P(edge > 0.05 R): max decile gap 0.047 (pooled seeds), slightly under-confident in deciles 5-9 (e.g. 0.09 predicted vs 0.14 actual); AUC 0.866 vs t-stat 0.860, paired diff +0.006 [0.003, 0.009]; top-5 % precision 0.56 vs challenge 0.26, diff vs 2x challenge [0.024, 0.064]. AUC above the t-stat in all 9 cells, with shared market factor and on BENCH-2. Limits: decile gap 0.10-0.13 with 30-100 trades; top-5 % precision below 2x challenge with 30-100 trades or a wide edge spread (0.10). Martingale traders: 7 % of the top 5 % (10 % of the population); precision 0.51 when they never count as skilled. |
| SKILL-07 | 2026-10-29 | FAILED (BENCH-3, fixed N per population). Rule P(edge > 0.05 R) >= 0.7: pooled precision 0.31 at 25 trades, 0.48 at 50, 0.62 at 75, 0.69 at 100 (CI [0.66, 0.73]), 0.77 at 150, 0.81 at 200, 0.84 at 300; criterion (lower CI >= 0.70 for every N >= 50) met only from 150 trades. Flagged traders get P about 0.85 on average: the decile calibration of SKILL-06 hid over-confidence in the extreme tail at small N. False positives are high win-rate styles (rare large losses not yet seen in a short record). Recall stays low: edge > 0.15 R detected in 45 % of cases at 300 trades, so no band reaches 50 % (N_min > 300). Sequential use (decide at first P >= 0.7) inflates false positives: precision 0.55. Abstention table: below 150 trades the firm sees 'abstain'. |
| DRIFT-01 | 2026-10-29 | FAILED (BENCH-3D: funded traders with edge > 0.10 R, half drop to -0.05 R at trade 150-300; 100-trade warm-up, 300 monitored trades, thresholds set for 10 % false alarms). CUSUM detects 31 % of edge losses within 150 trades (CI [0.30, 0.32], criterion >= 50 %) at 10.0 % false alarms; 37 % by trade 400; median delay 93 trades. By edge before the drop: 24 % (0.10-0.15 R), 36 % (0.15-0.25 R), 57 % (> 0.25 R). EWMA is worse: 17 % within 150 trades (paired diff CUSUM - EWMA +0.14 [0.13, 0.14]). Main cause: a 100-trade warm-up gives a noisy baseline, so the alarm threshold must be high (h = 28.9 for k = 0.1). Limit: traders whose drop comes after trade 250 have fewer than 150 trades left to be detected. |
