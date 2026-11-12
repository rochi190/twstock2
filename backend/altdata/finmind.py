"""FinMind fetcher + readers (PLAN.md P1a, shared with P2 monthly revenue).  Research only.

Fetching goes through ``backend.data._http_get`` (called, never modified).  One request returns a dataset's whole
history for one stock; ``backfill`` stores one JSON file per (dataset, stock) under
``<cache_dir>/alt/finmind/<FinMind dataset name>/<code>.json`` and can be interrupted and resumed.

Politeness / secrets: requests are spaced >= 12 s apart without a token (>= 1 s with ``FINMIND_TOKEN``); the token is
read from the environment only and is scrubbed from every exception message.  The first HTTP 402/429 or non-success
``msg`` raises/returns :class:`RateLimited` -- there is no blind retry loop against a rate limit.

Field meanings below are taken from real responses sampled 2026-11-12 (``tests/fixtures/alt/finmind_*.json``):

* institutional: long format, one row per (date, name) with ``buy``/``sell`` in *shares*.  ``name`` is one of
  ``Foreign_Investor``, ``Foreign_Dealer_Self`` (only from 2017-12-18), ``Investment_Trust``, ``Dealer_self`` and
  ``Dealer_Hedging`` (from 2014-12-01 / the stock's listing), and the legacy ``Dealer`` (combined dealers, before the
  split; 3 rows per day instead of 4-5).
* margin: balances/flows in *lots* (張, x1000 -> shares).
* monthly revenue: ``date`` is the 1st of the month after the revenue month; ``create_time`` is '' on old rows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend import data as _data

BASE_URL = "https://api.finmindtrade.com/api/v4/data"
TOKEN_ENV = "FINMIND_TOKEN"
MIN_INTERVAL_NO_TOKEN = 12.0   # seconds between requests (free tier ~300 requests / hour)
MIN_INTERVAL_TOKEN = 1.0
FETCH_ATTEMPTS = 3             # transient (non rate-limit) failures per request
FETCH_BACKOFF = 5.0
MAX_CONSECUTIVE_FAILURES = 5   # backfill stops after this many failed items in a row (dead network)
LOT = 1000                     # margin figures are in lots (張); 1 lot = 1000 shares
POOL_FILE = "pool300.json"

ALIASES = {
    "inst": "TaiwanStockInstitutionalInvestorsBuySell",
    "margin": "TaiwanStockMarginPurchaseShortSale",
    "hold": "TaiwanStockShareholding",
    "per": "TaiwanStockPER",
    "rev": "TaiwanStockMonthRevenue",
}
DEFAULT_START = {
    "TaiwanStockInstitutionalInvestorsBuySell": "2012-05-01",
    "TaiwanStockMarginPurchaseShortSale": "2012-01-01",
    "TaiwanStockShareholding": "2012-01-01",
    "TaiwanStockPER": "2012-01-01",
    "TaiwanStockMonthRevenue": "2010-01-01",
}
_CODE_RE = re.compile(r"^[0-9A-Za-z]{1,10}$")


class RateLimited(RuntimeError):
    """FinMind refused the request: HTTP 402/429, or a response whose ``msg`` is not ``"success"``."""


# ----------------------------------------------------------------------------------------- small helpers
def _sleep(seconds: float) -> None:  # separate so tests can patch it out
    time.sleep(seconds)


def _now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def get_token() -> str:
    return (os.environ.get(TOKEN_ENV) or "").strip()


def min_interval(token: str | None = None) -> float:
    token = get_token() if token is None else token
    return MIN_INTERVAL_TOKEN if token else MIN_INTERVAL_NO_TOKEN


def resolve_dataset(name: str) -> str:
    """Alias (``inst``, ``margin``, ``hold``, ``per``, ``rev``) or full FinMind dataset name -> full name."""
    key = str(name).strip()
    if key in ALIASES:
        return ALIASES[key]
    if key in DEFAULT_START:
        return key
    raise ValueError(f"unknown FinMind dataset {name!r} (aliases: {', '.join(ALIASES)})")


def default_start(dataset: str) -> str:
    return DEFAULT_START[resolve_dataset(dataset)]


def build_url(dataset: str, stock_id: str, start_date: str, token: str | None = None) -> str:
    params = [("dataset", resolve_dataset(dataset)), ("data_id", str(stock_id)), ("start_date", str(start_date))]
    token = get_token() if token is None else token
    if token:
        params.append(("token", token))
    return BASE_URL + "?" + urllib.parse.urlencode(params)


def _scrub(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


def _http_status(exc: BaseException) -> int | None:
    """HTTP status behind a ``DataUnavailable`` raised by ``_http_get`` (it chains the ``HTTPError``)."""
    seen = 0
    cur: BaseException | None = exc
    while cur is not None and seen < 4:
        code = getattr(cur, "code", None)
        if isinstance(cur, urllib.error.HTTPError) and isinstance(code, int):
            return code
        cur = cur.__cause__
        seen += 1
    m = re.search(r"HTTP (\d{3}) for|HTTPError[^0-9]{0,20}(\d{3})|\bHTTP Error (\d{3})", str(exc))
    if m:
        return int(next(g for g in m.groups() if g))
    return None


def _check_path_code(code: str) -> str:
    code = str(code)
    if not _CODE_RE.match(code):
        raise ValueError(f"invalid stock code {code!r}")
    return code


def dataset_path(dataset: str, code: str) -> Path:
    return _data.cache_dir() / "alt" / "finmind" / resolve_dataset(dataset) / f"{_check_path_code(code)}.json"


# ----------------------------------------------------------------------------------------- fetching
def fetch_dataset(dataset: str, stock_id: str, start_date: str) -> list[dict]:
    """One FinMind request -> the ``data`` rows (possibly ``[]``).

    Raises :class:`RateLimited` on HTTP 402/429, a body status of 402/429, or ``msg != "success"``;
    ``backend.data.DataUnavailable`` when the network keeps failing (``FETCH_ATTEMPTS`` tries).
    """
    dataset = resolve_dataset(dataset)
    token = get_token()
    url = build_url(dataset, stock_id, start_date, token)
    last: Exception | None = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            body = _data._http_get(url, retries=1)
        except _data.DataUnavailable as exc:
            status = _http_status(exc)
            if status in (402, 429):
                raise RateLimited(f"FinMind HTTP {status} for {dataset}/{stock_id}") from None
            last = exc
        else:
            try:
                obj = json.loads(body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else body)
            except ValueError as exc:
                last = exc
            else:
                if not isinstance(obj, dict):
                    last = ValueError("unexpected FinMind payload (not an object)")
                else:
                    status = obj.get("status")
                    if status in (402, 429):
                        raise RateLimited(f"FinMind status {status}: {obj.get('msg')}")
                    if obj.get("msg") != "success":
                        raise RateLimited(f"FinMind msg={obj.get('msg')!r} status={status!r} for {dataset}/{stock_id}")
                    rows = obj.get("data")
                    if rows is None:
                        rows = []
                    if not isinstance(rows, list):
                        last = ValueError("unexpected FinMind payload ('data' is not a list)")
                    else:
                        return rows
        if attempt + 1 < FETCH_ATTEMPTS:
            _sleep(FETCH_BACKOFF * (attempt + 1))
    raise _data.DataUnavailable(_scrub(f"FinMind {dataset}/{stock_id} failed after {FETCH_ATTEMPTS} attempts: {last!r}",
                                       token))


# ----------------------------------------------------------------------------------------- storage
def load_dataset(dataset: str, code: str) -> dict | None:
    """The stored payload ``{fetched_at, dataset, stock_id, start_date, empty, data}`` or ``None``."""
    obj = _data.read_json_file(dataset_path(dataset, code))
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj
    return None


def _store(dataset: str, code: str, start_date: str, rows: list[dict]) -> Path:
    path = dataset_path(dataset, code)
    _data.atomic_write_json(path, {
        "fetched_at": _now_iso(),
        "dataset": dataset,
        "stock_id": code,
        "start_date": start_date,
        "empty": len(rows) == 0,
        "data": rows,
    })
    return path


class BackfillResult(dict):
    """Plain dict with attribute access: fetched, skipped, empty, requests, failed, rate_limited, message, remaining."""
    __getattr__ = dict.get


def backfill(datasets, codes, *, start_dates: dict | None = None, progress=None) -> BackfillResult:
    """Fetch every missing (dataset, code) file, dataset-major; resumable.

    Existing valid files are skipped (no request, no sleep).  Between two consecutive requests the function sleeps
    ``min_interval()`` seconds through the patchable ``_sleep``.  On :class:`RateLimited` it stops immediately and
    returns ``rate_limited=True`` with the untouched work in ``remaining`` (a list of ``(dataset, code)``).
    After ``MAX_CONSECUTIVE_FAILURES`` failures in a row (dead network) it also stops.
    """
    ds_list = [resolve_dataset(d) for d in datasets]
    code_list = [_check_path_code(c) for c in codes]
    starts = {resolve_dataset(k): v for k, v in (start_dates or {}).items()}
    say = progress or (lambda msg: None)
    todo: list[tuple[str, str]] = []
    res = BackfillResult(fetched=0, skipped=0, empty=0, requests=0, failed=[], rate_limited=False, message="",
                         remaining=[])
    for ds in ds_list:
        for code in code_list:
            if load_dataset(ds, code) is not None:
                res["skipped"] += 1
            else:
                todo.append((ds, code))
    interval = min_interval()
    consecutive_fail = 0
    for i, (ds, code) in enumerate(todo):
        if res["requests"] > 0:
            _sleep(interval)
        start = starts.get(ds) or DEFAULT_START[ds]
        res["requests"] += 1
        try:
            rows = fetch_dataset(ds, code, start)
        except RateLimited as exc:
            res["rate_limited"] = True
            res["message"] = str(exc)
            res["remaining"] = todo[i:]
            say(f"rate limited at {ds}/{code}: {exc}")
            return res
        except _data.DataUnavailable as exc:
            res["failed"].append((ds, code, str(exc)))
            consecutive_fail += 1
            say(f"[{i + 1}/{len(todo)}] {ds}/{code} FAILED: {exc}")
            if consecutive_fail >= MAX_CONSECUTIVE_FAILURES:
                res["message"] = f"{consecutive_fail} consecutive failures -- stopping"
                res["remaining"] = todo[i + 1:]
                return res
            continue
        consecutive_fail = 0
        _store(ds, code, start, rows)
        res["fetched"] += 1
        if not rows:
            res["empty"] += 1
        say(f"[{i + 1}/{len(todo)}] {ds}/{code} rows={len(rows)}")
    return res


# ----------------------------------------------------------------------------------------- parsers
def _num(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None   # 'nan' / 'inf' strings or JSON NaN must not leak into features


def parse_institutional(rows: list[dict]) -> dict[str, dict]:
    """Long format -> ``{date: {...}}`` in *shares* (net = buy - sell).  Keys:

    ``foreign_net`` (Foreign_Investor), ``foreign_dealer_self_net`` (None before 2017-12-18), ``trust_net``
    (Investment_Trust), ``dealer_self_net`` / ``dealer_hedge_net`` (None before the split, i.e. when only the legacy
    combined ``Dealer`` row exists) and ``dealer_net`` (legacy ``Dealer``, else Dealer_self + Dealer_Hedging).
    A name that is absent on a date is ``None``, never 0.  Exact duplicate rows are dropped; (date, name) rows that differ are summed.
    """
    names = {"Foreign_Investor": "foreign_net", "Foreign_Dealer_Self": "foreign_dealer_self_net",
             "Investment_Trust": "trust_net", "Dealer_self": "dealer_self_net",
             "Dealer_Hedging": "dealer_hedge_net", "Dealer": "_dealer_legacy"}
    out: dict[str, dict] = {}
    seen: set = set()
    for r in rows:
        key = names.get(r.get("name"))
        date = r.get("date")
        buy, sell = _num(r.get("buy")), _num(r.get("sell"))
        if key is None or not date or buy is None or sell is None:
            continue
        sig = (date, r.get("name"), buy, sell)
        if sig in seen:      # an exact repeat of the same row is a duplicate, not a second trade
            continue
        seen.add(sig)
        rec = out.setdefault(date, {k: None for k in names.values()})
        rec[key] = (rec[key] or 0.0) + (buy - sell)
    for rec in out.values():
        legacy = rec.pop("_dealer_legacy")
        if legacy is not None:
            rec["dealer_net"] = legacy
        elif rec["dealer_self_net"] is not None or rec["dealer_hedge_net"] is not None:
            rec["dealer_net"] = (rec["dealer_self_net"] or 0.0) + (rec["dealer_hedge_net"] or 0.0)
        else:
            rec["dealer_net"] = None
    return dict(sorted(out.items()))


def parse_margin(rows: list[dict]) -> dict[str, dict]:
    """``{date: {...}}`` in *shares* (FinMind reports lots; x1000).  Keys: margin_balance, margin_balance_prev,
    margin_buy, margin_sell, margin_cash_repay, short_balance, short_balance_prev, short_sell, short_buy,
    short_cash_repay, offset.  Missing/blank fields are None."""
    fields = {"margin_balance": "MarginPurchaseTodayBalance", "margin_balance_prev": "MarginPurchaseYesterdayBalance",
              "margin_buy": "MarginPurchaseBuy", "margin_sell": "MarginPurchaseSell",
              "margin_cash_repay": "MarginPurchaseCashRepayment",
              "short_balance": "ShortSaleTodayBalance", "short_balance_prev": "ShortSaleYesterdayBalance",
              "short_sell": "ShortSaleSell", "short_buy": "ShortSaleBuy",
              "short_cash_repay": "ShortSaleCashRepayment", "offset": "OffsetLoanAndShort"}
    out: dict[str, dict] = {}
    for r in rows:
        date = r.get("date")
        if not date:
            continue
        rec = {}
        for k, src in fields.items():
            v = _num(r.get(src))
            rec[k] = None if v is None else v * LOT
        out[date] = rec
    return dict(sorted(out.items()))


def parse_per(rows: list[dict]) -> dict[str, dict]:
    """``{date: {per, pbr, dividend_yield}}`` (None when blank; PER can be 0 for loss makers -- caller decides)."""
    out = {}
    for r in rows:
        if r.get("date"):
            out[r["date"]] = {"per": _num(r.get("PER")), "pbr": _num(r.get("PBR")),
                              "dividend_yield": _num(r.get("dividend_yield"))}
    return dict(sorted(out.items()))


def parse_shareholding(rows: list[dict]) -> dict[str, dict]:
    """``{date: {foreign_ratio (%), foreign_shares, issued_shares}}``."""
    out = {}
    for r in rows:
        if r.get("date"):
            out[r["date"]] = {"foreign_ratio": _num(r.get("ForeignInvestmentSharesRatio")),
                              "foreign_shares": _num(r.get("ForeignInvestmentShares")),
                              "issued_shares": _num(r.get("NumberOfSharesIssued"))}
    return dict(sorted(out.items()))


def parse_monthly_revenue(rows: list[dict]) -> list[dict]:
    """Rows sorted by (revenue_year, revenue_month): ``{revenue_year, revenue_month, revenue, date, create_time}``.

    ``create_time`` is None when FinMind left it '' (old rows) -- PLAN section 3 forbids using it for availability
    anyway.  Rows without a usable year/month/revenue are dropped; a repeated (year, month) keeps the last row.
    """
    best: dict[tuple[int, int], dict] = {}
    for r in rows:
        try:
            y, m = int(r["revenue_year"]), int(r["revenue_month"])
        except (KeyError, TypeError, ValueError):
            continue
        rev = _num(r.get("revenue"))
        if rev is None or not 1 <= m <= 12:
            continue
        best[(y, m)] = {"revenue_year": y, "revenue_month": m, "revenue": rev, "date": r.get("date"),
                        "create_time": r.get("create_time") or None}
    return [best[k] for k in sorted(best)]


# ----------------------------------------------------------------------------------------- CLI
def _pool_path() -> Path:
    return _data.cache_dir() / "research" / POOL_FILE


def load_universe_codes(size: int) -> list[str]:
    """First ``size`` codes of ``<cache>/research/pool300.json`` (built from the live names table if missing)."""
    path = _pool_path()
    pool = _data.read_json_file(path)
    if not (isinstance(pool, list) and pool):
        table = _data._names_table(network=True)
        if not table or not table.get("items"):
            raise _data.DataUnavailable("cannot build the universe: names table unavailable")
        pool = _data.build_universe(table["items"], size=max(int(size), 300))
        _data.atomic_write_json(path, pool)
    codes = [str(it["code"]) for it in pool if isinstance(it, dict) and it.get("code")]
    return codes[: int(size)]


def plan(datasets, codes) -> dict:
    ds_list = [resolve_dataset(d) for d in datasets]
    total = len(ds_list) * len(codes)
    todo = sum(1 for ds in ds_list for c in codes if load_dataset(ds, c) is None)
    interval = min_interval()
    return {"total": total, "existing": total - todo, "requests": todo, "interval": interval,
            "eta_seconds": max(0, todo - 1) * interval + todo * 1.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.altdata.finmind", description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", required=True, help="comma list of: " + ",".join(ALIASES) + " (or full names)")
    ap.add_argument("--universe", type=int, default=300, help="first N codes of data_cache/research/pool300.json")
    ap.add_argument("--codes", default="", help="comma list of codes; overrides --universe")
    ap.add_argument("--dry-run", action="store_true", help="only print the number of requests and the time estimate")
    args = ap.parse_args(argv)
    try:
        datasets = [resolve_dataset(d) for d in args.datasets.split(",") if d.strip()]
        if args.codes.strip():
            codes = [_check_path_code(c.strip()) for c in args.codes.split(",") if c.strip()]
        else:
            codes = load_universe_codes(args.universe)
    except (ValueError, _data.DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not datasets or not codes:
        print("error: nothing to do (no datasets or no codes)", file=sys.stderr)
        return 1
    p = plan(datasets, codes)
    print(f"datasets={len(datasets)} codes={len(codes)} files={p['total']} existing={p['existing']} "
          f"requests={p['requests']} interval={p['interval']:.0f}s token={'yes' if get_token() else 'no'} "
          f"eta~{p['eta_seconds'] / 3600:.2f}h")
    if args.dry_run:
        return 0
    res = backfill(datasets, codes, progress=lambda m: print(m, flush=True))
    print(f"done: fetched={res['fetched']} skipped={res['skipped']} empty={res['empty']} "
          f"failed={len(res['failed'])} requests={res['requests']}")
    if res["rate_limited"]:
        print(f"STOPPED: FinMind rate limit ({res['message']}). {len(res['remaining'])} files remain; "
              f"wait about an hour (or set {TOKEN_ENV}) and re-run the same command to resume.", file=sys.stderr)
        return 2
    if res["message"] or res["failed"]:
        print(f"INCOMPLETE: {res['message'] or str(len(res['failed'])) + ' file(s) failed'}; re-run to retry.",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
