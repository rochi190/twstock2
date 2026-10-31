"""Test doubles for the data layer (module B): a fake network, Yahoo/TWSE payload builders, a clock.

Nothing here touches the real network: ``FakeNet`` replaces ``backend.data._http_get``.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np

from backend import data

TAIPEI = timezone(timedelta(hours=8))


def bdays(start: date, n: int) -> list[date]:
    """``n`` consecutive Mon-Fri dates starting at (or after) ``start``."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def ts_of(d: date, hour_utc: int = 1) -> int:
    """Yahoo-style timestamp: 01:00 UTC == 09:00 Taipei of the same day."""
    return int(datetime(d.year, d.month, d.day, hour_utc, 0, tzinfo=timezone.utc).timestamp())


def yahoo_payload(dates, close, *, adj=None, open_=None, high=None, low=None, volume=None, divs=(), nulls=(),
                  meta=None, gmtoffset=28800, name="Fake Co", itype="EQUITY") -> bytes:
    """A Yahoo ``v8/finance/chart`` JSON body.  ``nulls`` = row indices whose OHLC are null; ``divs`` = [(date, amt)]."""
    n = len(dates)
    close = [float(x) for x in close]
    adj = [float(x) for x in (adj if adj is not None else close)]
    open_ = [float(x) for x in (open_ if open_ is not None else close)]
    high = [float(x) for x in (high if high is not None else [max(o, c) * 1.01 for o, c in zip(open_, close)])]
    low = [float(x) for x in (low if low is not None else [min(o, c) * 0.99 for o, c in zip(open_, close)])]
    volume = list(volume if volume is not None else [1_000_000 + 1000 * i for i in range(n)])
    cols = {"open": open_, "high": high, "low": low, "close": close}
    for i in nulls:
        for k in cols:
            cols[k] = list(cols[k])
            cols[k][i] = None
    adj = list(adj)
    for i in nulls:
        adj[i] = None
    m = {"currency": "TWD", "symbol": "X", "gmtoffset": gmtoffset, "exchangeTimezoneName": "Asia/Taipei",
         "instrumentType": itype, "longName": name}
    m.update(meta or {})
    result = {"meta": m, "timestamp": [ts_of(d) for d in dates],
              "indicators": {"quote": [dict(cols, volume=volume)], "adjclose": [{"adjclose": adj}]}}
    if divs:
        result["events"] = {"dividends": {str(ts_of(d)): {"amount": a, "date": ts_of(d)} for d, a in divs}}
    return json.dumps({"chart": {"result": [result], "error": None}}).encode()


def yahoo_error(code="Not Found") -> bytes:
    return json.dumps({"chart": {"result": None, "error": {"code": code, "description": "x"}}}).encode()


def random_walk(n: int, seed: int = 0, start: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.round(start * np.exp(np.cumsum(rng.normal(0, 0.01, n))), 2)


def twse_day_all(rows) -> bytes:
    """rows = [(code, name, trade_value, close)] -> STOCK_DAY_ALL JSON."""
    return json.dumps([{"Date": "1151021", "Code": c, "Name": nm, "TradeVolume": "1", "TradeValue": str(tv),
                        "OpeningPrice": "1", "HighestPrice": "1", "LowestPrice": "1", "ClosingPrice": str(cl),
                        "Change": "0", "Transaction": "1"} for c, nm, tv, cl in rows], ensure_ascii=False).encode()


def twse_company(rows) -> bytes:
    return json.dumps([{"公司代號": c, "公司簡稱": nm, "產業別": "24"} for c, nm in rows], ensure_ascii=False).encode()


def tpex_day(rows) -> bytes:
    return json.dumps([{"Date": "1151021", "SecuritiesCompanyCode": c, "CompanyName": nm, "Close": str(cl),
                        "TransactionAmount": str(tv)} for c, nm, tv, cl in rows], ensure_ascii=False).encode()


def holiday_json(year: int, rows) -> bytes:
    """rows = [(iso date, name)] -> TWSE holidaySchedule JSON endpoint body."""
    return json.dumps({"stat": "ok", "date": f"{year}0101", "title": "x", "fields": ["日期", "名稱", "說明"],
                       "data": [[d, nm, "說明"] for d, nm in rows], "queryYear": str(year),
                       "total": len(rows)}, ensure_ascii=False).encode()


HOLIDAYS_2026 = [
    ("2026-01-01", "中華民國開國紀念日"), ("2026-01-02", "國曆新年開始交易日"), ("2026-02-11", "農曆春節前最後交易日"),
    ("2026-02-12", "市場無交易，僅辦理結算交割作業"), ("2026-02-13", "市場無交易，僅辦理結算交割作業"),
    ("2026-02-15", "農曆除夕及春節"), ("2026-02-16", "農曆除夕及春節"), ("2026-02-17", "農曆除夕及春節"),
    ("2026-02-18", "農曆除夕及春節"), ("2026-02-19", "農曆除夕及春節"), ("2026-02-20", "農曆除夕及春節"),
    ("2026-02-23", "農曆春節後開始交易日"), ("2026-02-27", "和平紀念日"), ("2026-02-28", "和平紀念日"),
    ("2026-04-03", "兒童節及民族掃墓節"), ("2026-04-06", "兒童節及民族掃墓節"), ("2026-05-01", "勞動節"),
    ("2026-06-19", "端午節"), ("2026-09-25", "中秋節"), ("2026-09-28", "孔子誕辰紀念日/ 教師節"),
    ("2026-10-09", "國慶日"), ("2026-10-10", "國慶日"), ("2026-10-25", "臺灣光復暨金門古寧頭大捷紀念日"),
    ("2026-10-26", "臺灣光復暨金門古寧頭大捷紀念日"), ("2026-12-25", "行憲紀念日"),
]


class FakeNet:
    """Drop-in for ``data._http_get``: routes URL substrings to bytes / callables / exceptions and records calls.

    ``routes`` maps a substring (first match wins, in insertion order) to ``bytes``, an ``Exception`` instance
    (raised), or a callable ``f(url) -> bytes`` (may raise).  Unrouted URLs raise ``data.DataUnavailable``.
    """

    def __init__(self):
        self.routes: dict = {}
        self.calls: list[str] = []
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.active_chart = 0  # in-flight Yahoo chart requests (the ones DESIGN.md caps at 6)
        self.peak_chart = 0
        self.delay = 0.0

    def route(self, key: str, value) -> None:
        self.routes[key] = value

    def down(self, key: str = "/chart/", exc=None) -> None:
        """Make every URL containing ``key`` fail (the rule goes FIRST: routes are matched in insertion order)."""
        rest = {k: v for k, v in self.routes.items() if k != key}
        self.routes.clear()
        self.routes[key] = exc if exc is not None else data.DataUnavailable("network down")
        self.routes.update(rest)

    def __call__(self, url, *, timeout=None, retries=None, headers=None, semaphore=None):
        sem = semaphore
        if sem is not None:
            sem.acquire()
        try:
            chart = "/chart/" in url
            with self.lock:
                self.calls.append(url)
                self.active += 1
                self.peak = max(self.peak, self.active)
                if chart:
                    self.active_chart += 1
                    self.peak_chart = max(self.peak_chart, self.active_chart)
            try:
                if self.delay:
                    time.sleep(self.delay)
                for key, val in self.routes.items():
                    if key in url:
                        if isinstance(val, BaseException):
                            raise val
                        return val(url) if callable(val) else val
                raise data.DataUnavailable(f"no route for {url}")
            finally:
                with self.lock:
                    self.active -= 1
                    if chart:
                        self.active_chart -= 1
        finally:
            if sem is not None:
                sem.release()

    def count(self, substr: str) -> int:
        return sum(1 for u in self.calls if substr in u)

    def reset(self) -> None:
        self.calls.clear()
        self.peak = 0
        self.peak_chart = 0


class Clock:
    """Mutable fake for ``data._now`` / ``data._wall`` / ``data._mono`` (Taipei wall clock)."""

    def __init__(self, dt: datetime):
        self.dt = dt if dt.tzinfo else dt.replace(tzinfo=TAIPEI)
        self.mono = 1000.0

    def now(self):
        return self.dt

    def wall(self):
        return self.dt.timestamp()

    def monotonic(self):
        return self.mono

    def advance(self, **kw):
        self.dt += timedelta(**kw)
        self.mono += timedelta(**kw).total_seconds()


import shutil
import tempfile
import unittest
from unittest import mock


class DataCase(unittest.TestCase):
    """Base class: temp cache dir, fake network + clock, holiday schedule preloaded (no real I/O at all)."""

    START = datetime(2026, 10, 22, 16, 0, tzinfo=TAIPEI)  # Thursday, after the close
    HOLIDAYS = True

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="twq_data_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.net = FakeNet()
        self.clock = Clock(self.START)
        self.sleeps: list[float] = []
        patches = [mock.patch.object(data, "_http_get", self.net), mock.patch.object(data, "_now", self.clock.now),
                   mock.patch.object(data, "_wall", self.clock.wall),
                   mock.patch.object(data, "_mono", self.clock.monotonic),
                   mock.patch.object(data, "_sleep", self.sleeps.append)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        data.set_cache_dir(self.tmp)
        self.addCleanup(data.set_cache_dir, None)
        # no stray background threads in unit tests (they could outlive the patched network); tests of the
        # stale-while-revalidate behaviour switch it on explicitly
        p = mock.patch.object(data, "NAMES_BACKGROUND_REFRESH", False)
        p.start()
        self.addCleanup(p.stop)
        self.load_holidays()

    def load_holidays(self):
        """Preload the 2026 holiday schedule into the memory cache (2027: not published -> weekends only)."""
        if self.HOLIDAYS:
            from backend.data import parse_holiday_schedule
            payload = json.loads(holiday_json(2026, HOLIDAYS_2026))
            data._HOL_MEM[2026] = parse_holiday_schedule(payload, 2026)
            data._HOL_MEM[2027] = None

    def reset_memory(self):
        """Simulate a process restart: forget every memory cache (the files stay), but keep the fake calendar."""
        data.reset_memory_caches()
        self.load_holidays()

    # convenience -------------------------------------------------------------------------------
    def route_symbol(self, ysym: str, dates, close, **kw):
        """Route ``/chart/{ysym}`` to a fixed payload."""
        self.net.route(f"/chart/{ysym}?", yahoo_payload(dates, close, **kw))

    def route_names(self, twse=(), company=(), tpex=()):
        self.net.route("STOCK_DAY_ALL", twse_day_all(twse))
        self.net.route("t187ap03_L", twse_company(company))
        self.net.route("tpex_mainboard", tpex_day(tpex))
