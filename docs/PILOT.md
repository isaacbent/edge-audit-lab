# Pilot with a firm (ROADMAP phase 4)

## What to ask the firm
1. Closed-trade exports for a set of funded accounts (MT5 / cTrader / Match-Trader CSV, any format with close time,
   profit and volume per trade; account id per row or one file per account), anonymised.
2. The account size (or one per account) and the firm's rules (max loss, daily loss, payout split).
3. For the retrospective test: what happened to each account AFTER the export date (paid / breached / still active,
   with dates). Without outcomes we can only report; with them we can check the scores.

## Run
```
python3 -m tools.audit_report exports/*.csv --account-size 100000 --platform matchtrader --out reports/firm_x
```
Check the first lines of each file first: if a column is not found the error lists the header; map it with
`--map profit="Net USD" --map close_time="Closing Time (UTC)"`. Column presets were written without real files.

## What is frozen (do not change for a pilot)
- Skill: P(edge > 0.05 R) with the NPMLE population prior fitted on the firm's own accounts; shown from 150 trades,
  "abstain" below (SKILL-06 passed, SKILL-07: P >= 0.7 trustworthy from 150 trades).
- R per account = median loss (checked on dev seeds 9700-9704 with risk per trade U[0.5, 2] %: precision of P >= 0.7
  0.725 at 150 trades and 0.823 at 300, vs 0.699 / 0.807 with the true R).
- Martingale flag: size-after-loss test + Benjamini-Hochberg q 0.10, accounts with >= 50 trades (TILT-01).
- Not offered yet: tilt detection, edge-loss alerts (DRIFT-01 weak), payout ranges (PAYOUT-02: centred but too
  narrow), breach-risk ranking (SURVIVAL-01: C 0.70, needs the firm's own outcomes to refit).

## Retrospective test (the free 48-hour pilot)
Score every account on trades up to a cut-off date (e.g. day 30 of funding), freeze the report, then compare with
outcomes after the cut-off: share paid / breached by skill verdict (skilled / uncertain / unlikely / abstain) and by
martingale flag. Pre-register this comparison (Registry.prereg) before opening the outcome file: same rules as the
synthetic studies.
