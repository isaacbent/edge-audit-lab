"""Import closed-trade exports into one table (phase 4: real data pilot).

Output: list of dicts {account, close_time (datetime), profit (account currency, net of commission and swap),
volume (lots / units, or None when the cell is missing)}, one per closed trade, sorted by account then close time
(file order breaks ties).

Platform exports differ by version and by how they were saved, so column names are matched loosely (case, spaces and
punctuation ignored) against the candidates below; anything else can be given with `mapping`. Column names MUST be
confirmed on the first real file of each firm (they could not be checked against real exports when this was
written). Duplicate header names (MT5 repeats Time / Price for open and close) are numbered: time, time_2, ...

Rows are never dropped silently: totals / summary rows (empty close time, or account "total" / "summary"),
non-trade rows (a type / side column saying balance, deposit, credit, ...) and MT5 entry "in" deals are skipped and
counted in `read_trades.skipped`; any other row that cannot be read raises ValueError with its line number.
"""
from __future__ import annotations

import csv
import io
import math
import re
from datetime import datetime

CANDIDATES = {
    "account": ["account", "login", "accountid", "accountnumber", "traderid", "trader"],
    "close_time": ["closetime", "closingtime", "closingtimeutc", "timeclose", "closedate", "time_2", "exittime"],
    "profit": ["profit", "netprofit", "net", "netusd", "pnl", "netpnl", "netpl"],
    "volume": ["volume", "closingquantity", "quantity", "lots", "size", "volumelots"],
    "commission": ["commission", "commissions", "fee", "fees"],
    "swap": ["swap", "swaps", "rollover"],
    "type": ["type", "side", "openingdirection", "direction", "action"],
    "entry": ["entry"],
}
PRESETS = {"mt5": {"close_time": "time_2"}, "ctrader": {}, "matchtrader": {}, "generic": {}}
NET_PROFIT = {"net", "netusd", "netprofit", "netpnl", "netpl"}   # already include commission / swap
TRADE_TYPES = ("buy", "sell", "long", "short")
SUMMARY_ACCOUNTS = {"total", "totals", "sum", "summary", "subtotal"}
TIME_FORMATS = ["%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
                "%Y-%m-%d", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", name.strip().lower().replace(" ", ""))


def parse_number(x, decimal: str = "auto") -> float:
    """'(12.50)' and '-$5.00' are negative, currency text and the Unicode minus are handled. Separators: both present
    -> the last one is the decimal mark; a lone ',' is a thousands mark only in the d,ddd pattern (else decimal);
    a lone '.' is a decimal mark. decimal='.' or ',' forces the decimal mark."""
    s = str(x).strip().replace("−", "-").replace(" ", "").replace(" ", "")
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^0-9.,\-]", "", s.strip("()"))
    if s.count("-") > 1 or (s.count("-") == 1 and not s.startswith("-")):
        raise ValueError(f"not a number: {x!r}")
    neg = neg or s.startswith("-")
    s = s.lstrip("-")
    if not s or not re.search(r"\d", s):
        raise ValueError(f"not a number: {x!r}")
    if decimal == "auto":
        if "," in s and "." in s:
            decimal = "," if s.rfind(",") > s.rfind(".") else "."
        elif "," in s:
            decimal = "." if re.fullmatch(r"\d{1,3}(,\d{3})+", s) else ","
        else:
            decimal = "."
    thousands = "," if decimal == "." else "."
    s = s.replace(thousands, "").replace(decimal, ".")
    v = float(s)
    if not math.isfinite(v):
        raise ValueError(f"not a finite number: {x!r}")
    return -v if neg else v


def parse_times(values, fmt: str | None = None) -> list:
    """All values with one format: `fmt`, else the first of TIME_FORMATS that reads every value."""
    for f in ([fmt] if fmt else TIME_FORMATS):
        try:
            return [datetime.strptime(v.strip(), f) for v in values]
        except ValueError:
            continue
    bad = values[0] if values else ""
    raise ValueError(f"close times not in a known format (e.g. {bad!r}); pass time_format, e.g. '%d.%m.%Y %H:%M'")


def read_trades(text: str, platform: str = "generic", mapping: dict | None = None, account: str | None = None,
                delimiter: str | None = None, decimal: str = "auto", net: bool | None = None,
                time_format: str | None = None) -> list:
    """Parse one export. `account` names the account when the file has no account column (MT5 exports one account
    per file). `mapping` overrides {field: column name as in the file}. `net`: True if the profit column already
    includes commission / swap (default: True for net-named or explicitly mapped profit columns)."""
    if platform not in PRESETS:
        raise ValueError(f"unknown platform {platform}; use one of {sorted(PRESETS)}")
    sample = text[:4096]
    delim = delimiter or (";" if sample.count(";") > sample.count(",") else ("\t" if "\t" in sample else ","))
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    header = rows[0]
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
    if net is None:
        net = names[col["profit"]] in NET_PROFIT or "profit" in (mapping or {})
    skipped = {"summary": 0, "non_trade": 0, "entry_in": 0}
    parsed, times = [], []
    for line, r in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in r):
            continue
        get = lambda f: r[col[f]].strip() if f in col and col[f] < len(r) else ""
        acc = get("account") if "account" in col else str(account)
        if not get("close_time") or acc.lower() in SUMMARY_ACCOUNTS or ("account" in col and not acc):
            skipped["summary"] += 1
            continue
        if "type" in col and get("type") and not any(k in get("type").lower() for k in TRADE_TYPES):
            skipped["non_trade"] += 1
            continue
        if "entry" in col and get("entry") and "out" not in get("entry").lower():
            skipped["entry_in"] += 1
            continue
        try:
            profit = parse_number(get("profit"), decimal)
            if not net:
                for f in ("commission", "swap"):
                    if get(f):
                        profit += parse_number(get(f), decimal)
            vol = None
            if "volume" not in col:
                vol = 1.0
            elif get("volume"):
                v = abs(parse_number(get("volume"), decimal))
                vol = v if v > 0 else None
        except ValueError as e:
            raise ValueError(f"line {line}: {e}") from None
        parsed.append({"account": acc, "profit": profit, "volume": vol, "_line": line})
        times.append(get("close_time"))
    for t, ct in zip(parsed, parse_times(times, time_format)):
        t["close_time"] = ct
    parsed.sort(key=lambda t: (t["account"], t["close_time"], t["_line"]))
    for t in parsed:
        del t["_line"]
    read_trades.skipped = skipped
    return parsed
