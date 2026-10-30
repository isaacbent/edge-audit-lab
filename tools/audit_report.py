"""Edge Audit pilot report: import a firm's closed-trade export(s), run the frozen audit (edgelab.audit), write a
Markdown report and a per-account CSV.

Usage:
  python3 -m tools.audit_report FILE [FILE ...] [--platform mt5|ctrader|matchtrader|generic]
         [--map profit=Net --map close_time="Closing Time"] [--decimal auto|.|,] [--gross] [--time-format FMT]
         [--out reports/firm_x]
MT5 exports one account per file: the account id is taken from the file name unless the file has an account column.
Results are in R (each account's median loss), so the account size is not needed.
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

FIELDS = ["account", "trades", "mean_R", "skill_probability", "skill_verdict", "martingale", "martingale_flag",
          "size_ratio_after_loss", "martingale_p"]


def markdown(res: dict, sources: list, skipped: dict) -> str:
    b, fz = res["book"], res["frozen"]
    mg_line = (f"- Martingale-like sizing flagged: {b['martingale_flagged']} of {b['martingale_eligible']} accounts checked "
               f"(>= {fz['min_trades_martingale']} trades with a size on every trade)" if b["volume_column"]
               else "- Martingale check: not done (no volume column in the export)")
    lines = ["# Edge Audit: pilot report", "",
             f"Sources: {', '.join(os.path.basename(s) for s in sources)}",
             f"Rows skipped (not trades): {skipped}", "",
             "## Book", "",
             f"- Accounts: {b['accounts']} ({b['trades']} closed trades)",
             f"- Skill scored (>= {fz['min_trades_skill']} trades): {b['accounts_scored']}; abstain: {b['accounts_abstain']}",
             f"- Likely skilled (P(edge > {fz['threshold_R']} R) >= {fz['flag_p']}): {b['skilled']}", mg_line, "",
             "## How to read it", "",
             f"- Results are in R = the account's median loss (risk per trade is not in the export); accounts with fewer "
             f"than {fz['min_losses_r']} losses have no R scale and are not scored. Profit is net of commission and swap.",
             f"- Skill probability: chance that the account's real edge is above {fz['threshold_R']} R per trade on top "
             "of costs, estimated against this book's own population. Validated on synthetic books only (SKILL-06/07); "
             f"below {fz['min_trades_skill']} trades the record is too short and we abstain.",
             "- Martingale flag: position size rises after losses far more than after wins. On synthetic books it "
             "caught 99.5 % of martingale traders at 50 trades (99.6 % at 100), with about 8 % of flags false. "
             "Tilt is NOT reliably detected yet.", "",
             "## Accounts", "",
             "| Account | Trades | Mean R | Skill P | Verdict | Martingale |", "| --- | --- | --- | --- | --- | --- |"]
    order = sorted(res["accounts"], key=lambda r: (r["skill_probability"] is None, -(r["skill_probability"] or 0)))
    for r in order:
        p = "-" if r["skill_probability"] is None else f"{r['skill_probability']:.2f}"
        m = "-" if r["mean_R"] is None else f"{r['mean_R']:+.3f}"
        lines.append(f"| {r['account']} | {r['trades']} | {m} | {p} | {r['skill_verdict']} | {r['martingale']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--account-size", type=float, default=None, help="ignored (results are in R); kept for old commands")
    ap.add_argument("--platform", default="generic", choices=sorted(importer.PRESETS))
    ap.add_argument("--map", action="append", default=[], help="field=column name in the file")
    ap.add_argument("--decimal", default="auto", choices=["auto", ".", ","])
    ap.add_argument("--gross", action="store_true", help="profit column excludes commission / swap (add them)")
    ap.add_argument("--time-format", default=None, help="strptime format of close times, e.g. %%d.%%m.%%Y %%H:%%M")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "audit"))
    args = ap.parse_args(argv)
    mapping = dict(m.split("=", 1) for m in args.map)
    stems = [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    if len(set(stems)) < len(stems):
        raise SystemExit(f"two files share a name ({stems}); MT5 account ids come from file names: rename them")
    trades, skipped = [], {}
    for f, stem in zip(args.files, stems):
        with open(f, encoding="utf-8-sig") as fh:
            trades += importer.read_trades(fh.read(), args.platform, mapping, account=stem, decimal=args.decimal,
                                           net=False if args.gross else None, time_format=args.time_format)
        for k, v in importer.read_trades.skipped.items():
            skipped[k] = skipped.get(k, 0) + v
    res = audit.audit(trades)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write(markdown(res, args.files, skipped))
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(res["accounts"])
    print(json.dumps({"report": args.out + ".md", "csv": args.out + ".csv", "skipped": skipped, **res["book"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
