"""Import closed-trade exports into one table (phase 4: real data pilot).

Output: list of dicts {account, close_time (str, sortable), profit (account currency, commission and swap included),
volume (lots / units)}, one per closed trade, sorted by account then close time.

Platform exports differ by version and by how they were saved, so column names are matched loosely (case, spaces and
punctuation ignored) against the candidates below; anything else can be given with `mapping`. Column names MUST be
confirmed on the first real file of each firm (they could not be checked against real exports when this was
written). Duplicate header names (MT5 repeats Time / Price for open and close) are numbered: time, time_2, ...
"""
from __future__ import annotations

import csv
import io
import re

CANDIDATES = {
    "account": ["account", "login", "accountid", "accountnumber", "traderid", "trader"],
    "close_time": ["closetime", "closingtime", "closingtimeutc", "timeclose", "closedate", "time_2", "exittime"],
    "profit": ["profit", "netprofit", "net", "netusd", "pnl", "netpnl", "grossprofit"],
    "volume": ["volume", "closingquantity", "quantity", "lots", "size", "volumelots"],
    "commission": ["commission", "commissions", "fee", "fees"],
    "swap": ["swap", "swaps", "rollover"],
}
PRESETS = {"mt5": {"close_time": "time_2"}, "ctrader": {}, "matchtrader": {}, "generic": {}}
NET_PROFIT = {"net", "netusd", "netprofit", "netpnl"}           # already include commission / swap


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", name.strip().lower().replace(" ", ""))


def _num(x) -> float:
    s = str(x).strip().replace(" ", "").replace(" ", "")
    if s in ("", "-"):
        return 0.0
    if s.count(",") and not s.count("."):
        s = s.replace(",", ".")                                    # decimal comma
    return float(s.replace(",", ""))


def read_trades(text: str, platform: str = "generic", mapping: dict | None = None, account: str | None = None,
                delimiter: str | None = None) -> list:
    """Parse one export. `account` names the account when the file has no account column (MT5 exports one account
    per file). `mapping` overrides {field: column name as in the file}."""
    if platform not in PRESETS:
        raise ValueError(f"unknown platform {platform}; use one of {sorted(PRESETS)}")
    sample = text[:4096]
    delim = delimiter or (";" if sample.count(";") > sample.count(",") else ("\t" if "\t" in sample else ","))
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    header, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    seen, names = {}, []
    for h in header:
        n = _norm(h)
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    col = {}
    explicit = {k: _norm(v) for k, v in {**PRESETS[platform], **(mapping or {})}.items()}
    for field, cands in CANDIDATES.items():
        for c in ([explicit[field]] if field in explicit else []) + cands:
            if c in names:
                col[field] = names.index(c)
                break
    missing = [f for f in ("close_time", "profit") if f not in col] + ([] if "account" in col or account else ["account"])
    if missing:
        raise ValueError(f"columns not found: {missing}; header was {header}; pass mapping={{field: column}}")
    net = names[col["profit"]] in NET_PROFIT
    out = []
    for r in body:
        get = lambda f: r[col[f]] if f in col and col[f] < len(r) else ""
        try:
            profit = _num(get("profit"))
        except ValueError:
            continue                                               # summary / footer lines
        if not net:
            profit += _num(get("commission") or 0) + _num(get("swap") or 0)
        out.append({"account": str(get("account")).strip() if "account" in col else str(account),
                    "close_time": str(get("close_time")).strip(), "profit": profit,
                    "volume": abs(_num(get("volume"))) if "volume" in col and get("volume") else 1.0})
    return sorted(out, key=lambda t: (t["account"], t["close_time"]))
