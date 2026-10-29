# Instructions for Claude (edge-audit-lab)

Reply to the user in Spanish. Read README.md and ROADMAP.md before any work; ROADMAP.md says what comes next.

## What this is
Research lab for Edge Audit (GraniteStats' B2B service for prop firms): tell funded traders with a real edge
from lucky ones, before a firm scales their capital or pays out. Synthetic data only until a firm shares real trades.

## Hard rules
- Pre-register every study (`Registry.prereg`) BEFORE running it; the study script must refuse to run if its SPEC
  differs from the pre-registration. Never re-run a finished study with changed settings: a fix is a NEW study id.
- `state/registry.jsonl` is append-only and hash-chained: never edit or delete lines; `Registry.verify()` must be ok.
- Failed studies stay in the README table as FAILED, with their numbers. Do not tune on a failed study's seeds.
- Seeds already used: 1000-1019 (SKILL-01), 2000-2019 (SKILL-02), 3000-3019 (SKILL-03). Use new ranges.
- Do not change BENCH-1 (`population`, `make_trader`) or BENCH-2 (`population_v2`, `make_style_trader`) in place:
  past studies must stay reproducible. A new bench = new functions (BENCH-3, ...).
- `python3 -m pytest -q` must pass before every commit.
- Independent of the Research Engine (isaacbent/isaacbent-research-engine): copy small utilities, never import it,
  never modify that repo from here.
- Long runs (> 9 min): launch with nohup in the background and poll.
