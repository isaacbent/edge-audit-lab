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
- `tools/skill_power.py` SKILL-01 study
- `tests/`               `python3 -m pytest -q`

## Studies
| Study | Date | Result |
| --- | --- | --- |
| SKILL-01 | 2026-10-29 | FAILED the pre-registered criterion (recall 1.7 % < 30 % at 100 trades, edge 0.15 R). Challenge rule: only 12-19 % of passers are skilled; BH flags are 97-100 % precise but recall is low for small edges (38 % at 0.25 R / 100 trades, 80 % at 0.25 R / 200 trades). Placebo: 0 false flags. |
