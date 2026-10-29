# edge-audit-lab

Research lab for Edge Audit (GraniteStats): synthetic trader bench + statistical pieces to tell
funded traders with a real edge from lucky ones, detect deterioration, account survival and payout risk.

Independent from the Research Engine: the few utilities it needs are copied (`edgelab/stats.py`,
`edgelab/atomic_io.py`, simplified `edgelab/registry.py`), never imported.

Rules: every study is pre-registered in `state/registry.jsonl` (append-only, hash-chained) before it runs;
a study whose code differs from its pre-registration refuses to run; failed studies are kept, never re-tuned.

## Layout
- `edgelab/simulate.py`  synthetic traders: skilled, breakeven, loser, martingale, tilt, decaying; challenge rule
- `edgelab/stats.py`     sign-flip and t p-values, Benjamini-Hochberg, Buhlmann credibility
- `edgelab/registry.py`  pre-registration / results log
- `tools/skill_power.py`, `skill_rank.py`, `skill_rank_v2.py`  SKILL-01/02/03 studies
- `tests/`               `python3 -m pytest -q`

## Studies
| Study | Date | Result |
| --- | --- | --- |
| SKILL-01 | 2026-10-29 | FAILED the pre-registered criterion (recall 1.7 % < 30 % at 100 trades, edge 0.15 R). Challenge rule: only 12-19 % of passers are skilled; BH flags are 97-100 % precise but recall is low for small edges (38 % at 0.25 R / 100 trades, 80 % at 0.25 R / 200 trades). Placebo: 0 false flags. |
| SKILL-02 | 2026-10-29 | FAILED the pre-registered criterion: top-5 % precision of the skill probability 0.49 vs challenge 0.14 (passes 2x), but its AUC 0.913 < win rate 0.916. Every graded score beats the challenge rule 2-4x; win rate wins because the bench's noise is symmetric (all traders share one payoff shape), a bench limitation. Skill probability over-confident (top decile 0.71 predicted vs 0.33 actual). |
| SKILL-03 | 2026-10-29 | FAILED (BENCH-2, heterogeneous styles). Win rate collapses to AUC 0.50 (bench now realistic). NPMLE probability AUC 0.876 vs profit factor 0.879 and t-stat 0.882; top-5 % precision 0.37 vs challenge 0.14 (2.6x). Calibration improved but still over-confident: top decile 0.49 predicted vs 0.27 actual (normal prior: 0.58 vs 0.24). |
