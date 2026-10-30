"""Edge Audit pilot report: import a firm's closed-trade export(s), run the frozen audit (edgelab.audit), write a
Markdown report and a per-account CSV.

Usage:
  python3 -m tools.audit_report FILE [FILE ...] --account-size 100000 [--platform mt5|ctrader|matchtrader|generic]
         [--map profit=Net --map close_time="Closing Time"] [--out reports/firm_x]
MT5 exports one account per file: the account id is taken from the file name unless the file has an account column.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgelab import audit, importer  # noqa: E402

FIELDS = ["account", "trades", "mean_R", "skill_probability", "skill_verdict", "martingale_flag",
          "size_ratio_after_loss", "martingale_p"]


def markdown(res: dict, sources: list) -> str:
    b, fz = res["book"], res["frozen"]
    lines = ["# Edge Audit: pilot report", "",
             f"Sources: {', '.join(os.path.basename(s) for s in sources)}", "",
             "## Book", "",
             f"- Accounts: {b['accounts']} ({b['trades']} closed trades)",
             f"- Skill scored (>= {fz['min_trades_skill']} trades): {b['accounts_scored']}; abstain: {b['accounts_abstain']}",
             f"- Likely skilled (P(edge > {fz['threshold_R']} R) >= {fz['flag_p']}): {b['skilled']}",
             f"- Martingale-like sizing flagged: {b['martingale_flagged']} of {b['martingale_eligible']} accounts with "
             f">= {fz['min_trades_martingale']} trades", "",
             "## How to read it", "",
             "- Skill probability: chance that the account's real edge beats costs (0.05 R per trade), estimated against "
             "this book's own population. Validated on synthetic books only (SKILL-06/07); below 150 trades the record "
             "is too short and we abstain.",
             "- Martingale flag: position size rises after losses far more than after wins (validated: 99.6 % of "
             "martingale traders caught, about 8 % of flags are false). Tilt is NOT reliably detected yet.",
             "- Results are in R = the account's median loss, since risk per trade is not in the export.", "",
             "## Accounts", "",
             "| Account | Trades | Mean R | Skill P | Verdict | Martingale |", "| --- | --- | --- | --- | --- | --- |"]
    order = sorted(res["accounts"], key=lambda r: (r["skill_probability"] is None, -(r["skill_probability"] or 0)))
    for r in order:
        mg = "n/a (< 50 trades)" if r["martingale_flag"] is None else ("FLAG" if r["martingale_flag"] else "-")
        p = "-" if r["skill_probability"] is None else f"{r['skill_probability']:.2f}"
        lines.append(f"| {r['account']} | {r['trades']} | {r['mean_R']:+.3f} | {p} | {r['skill_verdict']} | {mg} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--account-size", type=float, required=True)
    ap.add_argument("--platform", default="generic", choices=sorted(importer.PRESETS))
    ap.add_argument("--map", action="append", default=[], help="field=column name in the file")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "audit"))
    args = ap.parse_args(argv)
    mapping = dict(m.split("=", 1) for m in args.map)
    trades = []
    for f in args.files:
        with open(f, encoding="utf-8-sig") as fh:
            trades += importer.read_trades(fh.read(), args.platform, mapping,
                                           account=os.path.splitext(os.path.basename(f))[0])
    res = audit.audit(trades, args.account_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write(markdown(res, args.files))
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(res["accounts"])
    print(json.dumps({"report": args.out + ".md", "csv": args.out + ".csv", **res["book"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
