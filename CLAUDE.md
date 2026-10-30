# Instructions for Claude (edge-audit-lab)

Reply to the user in Spanish. Read README.md, ROADMAP.md and docs/FORMULAS.md before any work; ROADMAP.md says what comes next.

## What this is
Research lab for Edge Audit (GraniteStats' B2B service for prop firms): tell funded traders with a real edge
from lucky ones, before a firm scales their capital or pays out. Synthetic data only until a firm shares real trades.

## Hard rules
- Pre-register every study (`Registry.prereg`) BEFORE running it; the study script must refuse to run if its SPEC
  differs from the pre-registration. Never re-run a finished study with changed settings: a fix is a NEW study id.
- `state/registry.jsonl` is append-only and hash-chained: never edit or delete lines; `Registry.verify()` must be ok.
- Failed studies stay in the README table as FAILED, with their numbers. Do not tune on a failed study's seeds.
- Seeds already used: 1000-1019 (SKILL-01), 2000-2019 (SKILL-02), 3000-3019 (SKILL-03), 4000-4019 (SKILL-04),
  5000-5019 (SKILL-05), 6000-6019 (SKILL-06), 7000-7019 (SKILL-07), 8000-8019 + 8100-8109 calibration (DRIFT-01), 10000-10019 (TILT-01), 11000-11019 (SURVIVAL-01), 12000-12199 (PAYOUT-01 books), 13000-13199 (PAYOUT-02 books), 14000-14019 (SKILL-08), 15000-15019 (SURVIVAL-02),
  9000-9999 exploration only (used so far: 9000-9009, 9100-9119, 9200-9201, 9300-9304, 9400-9409, 9500, 9600-9639, 9700-9704, 9970 (REAL-01 fake package), 9980-9989 + 9990-9999 smoke).
  Use new ranges.
- Do not change BENCH-1 (`population`, `make_trader`) or BENCH-2 (`population_v2`, `make_style_trader`) in place:
  past studies must stay reproducible. A new bench = new functions (BENCH-3, ...).
- `python3 -m pytest -q` must pass before every commit.
- From SKILL-06 on: pre-register with `study.code_lock(__file__)` and call `study.check_lock` before running (refuses
  changed or uncommitted `edgelab/` / script, or other numpy / scipy / python versions); every study script has a
  `--smoke` mode (`study.smoke_spec`, no registry writes) that must run before registering.
- Criteria on paired per-seed differences with standard errors / 95 % intervals (`study.paired_diff`), not bare means.
- Exploration only on seeds 9000-9999, every attempt logged with `Registry.note`; exploration never confirms.
- Independent of the Research Engine (isaacbent/isaacbent-research-engine): copy small utilities, never import it,
  never modify that repo from here.
- Long runs (> 9 min): launch with nohup in the background and poll.
