"""Market-data layer: Yahoo prices, TWSE/TPEx names, trading calendar, liquid universe, disk cache.

Public API (DESIGN.md section 2)
--------------------------------
``get_price_data``, ``get_market_data``, ``get_universe``, ``search_instruments``, ``instrument_info``,
``last_trading_date_expected``, ``next_trading_day``, ``market_is_open``, ``completed`` plus a few helpers
(``normalize_symbol``, ``resolve_symbol``, ``prev_trading_day``, ``is_trading_day``, ``extra_closed_days``,
``placeholder_mask``, ``price_signature``, ``taipei_now``, ``popular_instruments``, ``warm_up``).

Only numpy + stdlib are needed (``urllib`` for HTTP).  Every network call goes through the module-level
function :func:`_http_get`, which is the single seam unit tests replace.

Operational environment switches (see README): ``TWQ_CACHE_DIR=<dir>`` (cache location, default ``data_cache/``),
``TWQ_OFFLINE=1`` makes every network call fail at once (caches keep
working: simulates a dead network), ``TWQ_YAHOO_CHART=<url template with {sym}>`` replaces the Yahoo endpoint (point it
at an unroutable host to simulate Yahoo being down), ``TWQ_HTTP_TIMEOUT=<seconds>`` shortens the per-request timeout.

Price source
------------
``https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=0&period2={now}&interval=1d&events=div,splits``
with ``User-Agent: Mozilla/5.0`` (never ``range=max``: Yahoo silently degrades it to monthly bars).  ``2330.TW`` is
tried first (unless the name table says the code is a TPEx one), then ``6488.TWO``; the suffix that worked is
remembered in the cache record.  Timestamps are shifted by the exchange ``gmtoffset`` (+8 h) to give the Taipei date.
Rows with a null/non-positive OHLC or adjusted close are dropped, duplicated dates keep the last row, and the
rows are sorted ascending.  ``close`` is split-adjusted but *not* dividend-adjusted (Yahoo semantics); ``adjclose``
carries the dividend adjustment and is the series used for modelling.

No-trade placeholder bars (audit P-01): Yahoo fills days on which the exchange was closed without a schedule entry
(typhoon days, 2026-10-23: 63 of 63 stocks) and trading halts with flat zero-volume rows (``open == high == low == close`` =
the previous close, volume 0).  They are not sessions: :func:`placeholder_mask` removes them while parsing (never for an index),
the cache format is version 3 and older records are cleaned on load (and re-downloaded in full on their next refresh).
Malformed answers (audit P-14): an answer without timestamps / quote block, with timestamps that are not epoch seconds, or a
first download without a single usable row is :class:`DataUnavailable` (503, retried later), never "unknown symbol" -- so a
transient fault does not make a valid code look non-existent for ``UNKNOWN_TTL``.

Refresh policy (``get_price_data``)
-----------------------------------
A cached record is refreshed when (a) it lacks bars up to :func:`last_trading_date_expected` (at most once per
``REFRESH_MIN_INTERVAL`` seconds), or (b) its last bar was fetched *before* that bar's session was final (14:00
Taipei) and the session is final now (a state transition, so no throttle), or (c) the caller passes
``refresh=True``.  Refreshes are *incremental* when possible: only the last ``INCREMENTAL_DAYS`` calendar days are
downloaded and merged, after verifying that the overlapping bars are unchanged (``close`` identical and
``adjclose/close`` unchanged within ``ADJ_RTOL``); a dividend or split changes that ratio/close and triggers a full
re-download, as does a record older than ``FULL_REFRESH_DAYS``.  After a failed refresh the cached record is
served with ``stale=True`` and no new attempt is made for ``FAIL_BACKOFF`` seconds.

History only moves forward (audit C-04): Yahoo intermittently serves an all-null row for the newest *completed* session.  The
parser drops it, and a refresh must not delete the valid final bar it replaced: :func:`_merge` (incremental) and
:func:`_keep_final_tail` (full download) put cached final bars that the new answer lacks back (adjusted close rescaled when a
dividend changed the adjustment; not possible across a split; a bar that was still intraday when cached is provisional and
is replaced as before).  ``fetched_at`` is the new time, so the 10-minute throttle still applies and the bar heals as soon as
Yahoo serves the row again.

Incomplete bar
--------------
A bar dated D is *final* once Taipei time passes 14:00 on D (the session ends 13:30).  ``PriceData.last_bar_complete``
is ``True`` iff the record was fetched after that moment; :func:`completed` drops a non-final last bar.

Calendar
--------
Trading days are Mon-Fri minus the TWSE holiday schedule (``holidaySchedule``; entries whose name says
"開始交易" / "最後交易" are *trading* days and ignored) minus the closures *learned from the data*: unannounced closures
cannot be predicted, but they are recognised afterwards (audit P-01): a weekday whose session is over (14:00) and for which
at least ``CLOSURE_MIN_SYMBOLS`` stocks serve a placeholder bar that make up at least ``CLOSURE_QUORUM`` of the stocks with any row
for it -- or one stock placeholder, no stock with a real bar and a null TAIEX row -- is not a trading day (``extra_closed_days``,
persisted in ``closures.json``, taken back as soon as a real bar for the day shows up).  This keeps ``asof``,
``last_trading_date_expected`` (the stale chip), ``next_trading_day`` and the refresh policy consistent after a closure.
When the schedule is unreachable only weekends (and learned closures) are excluded.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import random
import re
import socket
import ssl
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import OrderedDict
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config
from .types import PriceData

log = logging.getLogger("twquant.data")

# --------------------------------------------------------------------------------------- constants
TAIPEI = timezone(timedelta(hours=8), "Asia/Taipei")
SESSION_OPEN = dtime(9, 0)
SESSION_CLOSE = dtime(13, 30)
BAR_FINAL = dtime(14, 0)  # a daily bar is final from here on (session ended 13:30, Yahoo has settled)

USER_AGENT = "Mozilla/5.0"
YAHOO_CHART = os.environ.get("TWQ_YAHOO_CHART") or "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
TWSE_DAY_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_COMPANY = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_DAY = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
TWSE_HOLIDAY = "https://www.twse.com.tw/holidaySchedule/holidaySchedule?response=json&queryYear={year}"
TWSE_HOLIDAY_OPENAPI = "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule"

def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, ""))
        return v if v > 0 else default
    except ValueError:
        return default


HTTP_TIMEOUT = _env_float("TWQ_HTTP_TIMEOUT", 45.0)
HTTP_RETRIES = 4  # attempts per request (first try included)
BACKOFF_BASE = 0.8  # seconds; attempt k sleeps BACKOFF_BASE * 2**k (+ jitter), capped at BACKOFF_MAX
BACKOFF_MAX = 12.0
YAHOO_CONCURRENCY = 6
REFRESH_MIN_INTERVAL = 600.0  # at most one non-forced refresh per symbol per 10 minutes
FAIL_BACKOFF = 60.0  # after a failed refresh do not hit the network again for a minute
UNKNOWN_TTL = 600.0  # remember "no such symbol" for 10 minutes
INCREMENTAL_DAYS = 25  # calendar days re-downloaded by an incremental refresh
FULL_REFRESH_DAYS = 30  # force a full re-download of the whole history at least this often
ADJ_RTOL = 2e-6  # tolerance on adjclose/close when verifying that an incremental overlap is unchanged
NAMES_TTL = 24 * 3600.0
UNIVERSE_TTL = 7 * 24 * 3600.0
UNIVERSE_CACHE_N = 120  # ranked names persisted in universe.json (get_universe(n) slices it)
HOLIDAY_TTL = 3 * 24 * 3600.0
CACHE_VERSION = 3  # v3: no-trade placeholder bars are not stored (audit P-01); v2 records are migrated on load
LEGACY_CACHE_VERSIONS = (2,)
MAX_TIMESTAMP = 4_102_444_800  # 2100-01-01 UTC: a larger / negative "timestamp" is a unit error (milliseconds), not a date
EVIDENCE_WINDOW_DAYS = 40  # parse_yahoo_chart reports placeholder / null days of this many calendar days before the last bar
CLOSURE_WINDOW_DAYS = 21  # calendar days of closure evidence that are tallied / persisted (audit P-01)
CLOSURE_MIN_SYMBOLS = 3  # a closure needs this many stocks serving a placeholder for the day ...
CLOSURE_QUORUM = 0.8  # ... and at least this share of the stocks that have a bar for it

INDEX_SYMBOL = config.MARKET_SYMBOL  # '^TWII'
INDEX_NAME = "加權指數"
_INDEX_ALIASES = {"^TWII", "TWII", "^TAIEX", "TAIEX", "加權指數", "加權", "大盤", "台灣加權指數", "臺灣加權指數"}
_INDEX_ALIASES_UP = {a.upper() for a in _INDEX_ALIASES}
_CODE_RE = re.compile(r"^\d{4,6}[A-Z]?$")
_INSTRUMENT_RE = re.compile(r"^(\d{4}|00\d{3,4}[A-Z]?)$")  # 4-digit stocks and 00xxx(x)(A-Z) ETFs

# Curated Taiwan-50-style large caps (TWSE), most famous first.  Every code and name was verified against the
# live STOCK_DAY_ALL table (Oct 2026); at run time the live table overrides the names and drops codes that
# disappeared.  The first 12 double as the "popular" list of the search box.
FALLBACK_UNIVERSE = [
    ("2330", "台積電"), ("2317", "鴻海"), ("2454", "聯發科"), ("2308", "台達電"), ("2382", "廣達"),
    ("2412", "中華電"), ("2881", "富邦金"), ("2882", "國泰金"), ("2891", "中信金"), ("2886", "兆豐金"),
    ("1301", "台塑"), ("1303", "南亞"), ("2002", "中鋼"), ("3711", "日月光投控"), ("2603", "長榮"),
    ("2609", "陽明"), ("3034", "聯詠"), ("2379", "瑞昱"), ("2357", "華碩"), ("2303", "聯電"),
    ("2884", "玉山金"), ("2885", "元大金"), ("2892", "第一金"), ("2880", "華南金"), ("2883", "凱基金"),
    ("2887", "台新新光金"), ("5880", "合庫金"), ("2890", "永豐金"), ("2301", "光寶科"), ("2345", "智邦"),
    ("3008", "大立光"), ("3231", "緯創"), ("3045", "台灣大"), ("4904", "遠傳"), ("1216", "統一"),
    ("2207", "和泰車"), ("2327", "國巨*"), ("2395", "研華"), ("2408", "南亞科"), ("3037", "欣興"),
    ("3017", "奇鋐"), ("3661", "世芯-KY"), ("6669", "緯穎"), ("5871", "中租-KY"), ("5876", "上海商銀"),
    ("6505", "台塑化"), ("1326", "台化"), ("1101", "台泥"), ("2912", "統一超"), ("4938", "和碩"),
    ("2618", "長榮航"), ("2353", "宏碁"), ("3443", "創意"), ("3653", "健策"), ("2344", "華邦電"),
    ("2449", "京元電子"), ("2360", "致茂"), ("6415", "矽力*-KY"), ("2324", "仁寶"), ("2376", "技嘉"),
    ("2377", "微星"), ("2356", "英業達"), ("1590", "亞德客-KY"), ("2059", "川湖"),
]
# Well-known ETFs that must be searchable even if every name source is down.
FALLBACK_ETFS = [("0050", "元大台灣50"), ("0056", "元大高股息"), ("00878", "國泰永續高股息"),
                 ("006208", "富邦台50"), ("00919", "群益台灣精選高息"), ("00929", "復華台灣科技優息"),
                 ("00940", "元大台灣價值高息"), ("00713", "元大台灣高息低波")]
POPULAR_ETFS = ("0050", "0056", "00878")
POPULAR_UNIVERSE = 12  # names of the universe in the popular list
POPULAR_COUNT = POPULAR_UNIVERSE + len(POPULAR_ETFS)  # = 15: size of the empty-query list of the search box



# --------------------------------------------------------------------------------------- exceptions
class DataError(Exception):
    """Base class of data-layer failures."""


class BadSymbol(DataError, ValueError):
    """The text is not a valid Taiwan stock / ETF code (or index alias)."""


class UnknownSymbol(DataError, KeyError):
    """The code is well-formed but no data source knows it."""

    def __str__(self) -> str:  # KeyError would quote the message
        return str(self.args[0]) if self.args else "unknown symbol"


class DataUnavailable(DataError, RuntimeError):
    """The network / upstream service failed (after retries) and no usable cache exists."""


class NotFoundError(DataError):
    """HTTP 404 / Yahoo 'Not Found' for one URL (internal; mapped to UnknownSymbol by callers)."""


# --------------------------------------------------------------------------------------- time helpers
def _now() -> datetime:
    """Current time, Taipei zone.  Unit tests monkeypatch this function."""
    return datetime.now(TAIPEI)


def taipei_now(now=None) -> datetime:
    """Normalise ``now`` to an aware Taipei datetime (None -> current time; naive -> already Taipei local time)."""
    if now is None:
        return _now()
    if isinstance(now, datetime):
        if now.tzinfo is None:
            return now.replace(tzinfo=TAIPEI)
        return now.astimezone(TAIPEI)
    if isinstance(now, (int, float)):
        return datetime.fromtimestamp(float(now), TAIPEI)
    raise TypeError(f"unsupported 'now': {now!r}")


def final_time(d: date) -> datetime:
    """Moment from which the daily bar of date ``d`` is final (14:00 Taipei of that day)."""
    return datetime.combine(d, BAR_FINAL, TAIPEI)


def _iso(t: datetime) -> str:
    return t.astimezone(TAIPEI).isoformat(timespec="seconds")


def _parse_iso(s: str) -> datetime | None:
    try:
        t = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    return t.replace(tzinfo=TAIPEI) if t.tzinfo is None else t


# --------------------------------------------------------------------------------------- cache dir / io
_CACHE_DIR_OVERRIDE: Path | None = None


def cache_dir() -> Path:
    """Root of the on-disk cache: :func:`set_cache_dir` > env ``TWQ_CACHE_DIR`` > ``config.CACHE_DIR``."""
    if _CACHE_DIR_OVERRIDE is not None:
        return _CACHE_DIR_OVERRIDE
    env = os.environ.get("TWQ_CACHE_DIR", "").strip()
    return Path(env).expanduser() if env else Path(config.CACHE_DIR)


def set_cache_dir(path) -> None:
    """Redirect every cache (prices, names, universe, holidays, jobs) -- used by tests; also clears memory caches."""
    global _CACHE_DIR_OVERRIDE
    _CACHE_DIR_OVERRIDE = Path(path) if path is not None else None
    reset_memory_caches()


def reset_memory_caches() -> None:
    """Forget every in-memory cache (the disk caches are untouched)."""
    with _MEM_LOCK:
        _REC_MEM.clear()
        _FAIL.clear()
        _UNKNOWN.clear()
        _LAST_OK.clear()
        _CIRCUIT.update(fails=0, until=0.0)
        _HOL_MEM.clear()
        _HOL_FAIL.clear()
        _NAMES_MEM["table"] = None
        _NAMES_MEM["fail_until"] = 0.0
    with _CLOSE_LOCK:
        _CLOSE_MEM["tally"] = None


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` atomically (temp file in the same directory + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def read_json_file(path: Path):
    """Parsed JSON of ``path`` or ``None`` when missing / corrupt."""
    try:
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("corrupt cache file %s (%s) -- ignoring", path, exc)
        return None


REC_MEM_MAX = 160  # parsed price records kept in memory (LRU); a 25-year record is ~0.4 MB -> <= ~65 MB
BOOKKEEPING_MAX = 4096  # cap of the tiny per-symbol dicts below (a client could probe random codes forever)

_MEM_LOCK = threading.RLock()
_REC_MEM: "OrderedDict[str, tuple[tuple[int, int], dict]]" = OrderedDict()  # symbol -> ((mtime_ns, size), record)
_FAIL: dict[str, float] = {}  # symbol -> monotonic time of the last failed refresh
_UNKNOWN: dict[str, float] = {}  # symbol -> monotonic time we learnt it does not exist
_LAST_OK: dict[str, float] = {}  # symbol -> monotonic time of the last successful network refresh
_SYM_LOCKS: dict[str, threading.Lock] = {}


def _rec_mem_get(sym: str, sig: tuple[int, int]) -> dict | None:
    with _MEM_LOCK:
        hit = _REC_MEM.get(sym)
        if hit and hit[0] == sig:
            _REC_MEM.move_to_end(sym)
            return hit[1]
    return None


def _rec_mem_put(sym: str, sig: tuple[int, int], rec: dict) -> None:
    with _MEM_LOCK:
        _REC_MEM[sym] = (sig, rec)
        _REC_MEM.move_to_end(sym)
        while len(_REC_MEM) > REC_MEM_MAX:
            _REC_MEM.popitem(last=False)


def _cap(d: dict, limit: int = BOOKKEEPING_MAX) -> None:
    """Drop the oldest entries of an insertion-ordered bookkeeping dict (caller holds ``_MEM_LOCK``)."""
    while len(d) > limit:
        d.pop(next(iter(d)))


def _sym_lock(sym: str) -> threading.Lock:
    with _MEM_LOCK:
        lk = _SYM_LOCKS.get(sym)
        if lk is None:
            if len(_SYM_LOCKS) >= BOOKKEEPING_MAX:  # forget locks nobody holds right now
                for k in [k for k, v in _SYM_LOCKS.items() if not v.locked()][: BOOKKEEPING_MAX // 2]:
                    del _SYM_LOCKS[k]
            lk = _SYM_LOCKS[sym] = threading.Lock()
        return lk


# --------------------------------------------------------------------------------------- HTTP
_YAHOO_SEM = threading.BoundedSemaphore(YAHOO_CONCURRENCY)
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 522, 524}
_NET_ERRORS = (urllib.error.URLError, http.client.HTTPException, ConnectionError, socket.timeout, TimeoutError,
               ssl.SSLError, OSError, zlib.error, EOFError)


def _sleep(seconds: float) -> None:  # separate so tests can patch it out
    time.sleep(seconds)


def _wall() -> float:
    """Unix time (cache expiry); unit tests patch this and ``_mono`` to travel in time."""
    return time.time()


def _mono() -> float:
    """Monotonic clock (back-off / throttle bookkeeping)."""
    return time.monotonic()


def _http_get(url: str, *, timeout: float = HTTP_TIMEOUT, retries: int = HTTP_RETRIES,
              headers: dict | None = None, semaphore: threading.BoundedSemaphore | None = None) -> bytes:
    """GET ``url`` and return the (gunzipped) body.

    Retries with exponential backoff (+ jitter, honouring ``Retry-After``) on HTTP 429/5xx, connection errors,
    timeouts, truncated bodies (``IncompleteRead``) and corrupt gzip streams.  HTTP 404 raises
    :class:`NotFoundError` immediately; other 4xx raise :class:`DataUnavailable`.  When ``semaphore`` is given it
    bounds the number of simultaneous requests (only held while a request is in flight, not while backing off).
    """
    if os.environ.get("TWQ_OFFLINE", "") not in ("", "0"):  # simulated dead network (documented in the README)
        raise DataUnavailable(f"network disabled (TWQ_OFFLINE): {url}")
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "gzip"}
    hdrs.update(headers or {})
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        retry_after = 0.0
        try:
            if semaphore is not None:
                semaphore.acquire()
            try:
                req = urllib.request.Request(url, headers=hdrs)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read()
                    enc = (resp.headers.get("Content-Encoding") or "").lower()
                if enc == "gzip" or body[:2] == b"\x1f\x8b":
                    body = zlib.decompress(body, 16 + zlib.MAX_WBITS)
                return body
            finally:
                if semaphore is not None:
                    semaphore.release()
        except urllib.error.HTTPError as exc:
            exc.close()  # release the socket right away (the error object is also a response)
            if exc.code == 404:
                raise NotFoundError(url) from exc
            last = exc
            if exc.code not in _RETRY_STATUS:
                raise DataUnavailable(f"HTTP {exc.code} for {url}") from exc
            try:
                retry_after = min(float(exc.headers.get("Retry-After", "0") or 0), 30.0)
            except (TypeError, ValueError):
                retry_after = 0.0
        except _NET_ERRORS as exc:
            last = exc
        if attempt + 1 < max(1, retries):
            delay = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX) * (1.0 + 0.25 * random.random())
            _sleep(max(delay, retry_after))
    raise DataUnavailable(f"request failed after {max(1, retries)} attempts: {url} ({last!r})") from last


def _get_json(url: str, **kw):
    try:
        return json.loads(_http_get(url, **kw).decode("utf-8-sig"))
    except ValueError as exc:  # truncated / not JSON
        raise DataUnavailable(f"invalid JSON from {url}: {exc}") from exc


# --------------------------------------------------------------------------------------- symbols
def normalize_symbol(symbol) -> str:
    """Canonical symbol: '2330' / '00878' / '00400A' / '^TWII'.  Accepts '2330.TW', '6488.TWO', 'taiex', full-width digits.

    Raises :class:`BadSymbol` for anything else.
    """
    if symbol is None:
        raise BadSymbol("empty symbol")
    s = unicodedata.normalize("NFKC", str(symbol)).strip()
    u = s.upper()
    if u in _INDEX_ALIASES_UP:
        return INDEX_SYMBOL
    for suffix in (".TWO", ".TW"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            break
    if _CODE_RE.match(u):
        return u
    raise BadSymbol(f"invalid symbol: {symbol!r}")


def _fold(s: str) -> str:
    """Search folding: NFKC, upper-case, 臺->台 so '臺企銀' and '台企銀' match each other."""
    return unicodedata.normalize("NFKC", s).strip().upper().replace("臺", "台")


def kind_of(code: str) -> str:
    if code == INDEX_SYMBOL:
        return "index"
    return "etf" if code.startswith("00") else "stock"


# --------------------------------------------------------------------------------------- Yahoo parsing
def placeholder_mask(o, h, l, c, v, index: bool = False) -> np.ndarray:
    """No-trade placeholder rows of ascending bars (audit P-01): zero volume, ``open == high == low == close`` and a close
    equal to the previous row's close.

    Yahoo fills days on which the exchange was closed unannounced (typhoon, the 2026-10-23 closure) and trading halts
    with such rows; they are not observations (a zero return that never happened) and would give every stock a fake
    ``asof`` / ``chg_1d == 0``.  The first row is never a placeholder (no previous close), an index series (no volume)
    is exempt, and a run of placeholders is removed as a whole because each row repeats the close before it.
    """
    n = len(c)
    if index or n < 2:
        return np.zeros(n, dtype=bool)
    prev = np.r_[np.nan, c[:-1]]
    return (v == 0) & (o == h) & (h == l) & (l == c) & (c == prev)


def _is_index_meta(meta: dict) -> bool:
    return (str(meta.get("instrumentType") or "").upper() == "INDEX"
            or str(meta.get("symbol") or "").startswith(("^", "%5E")))


def parse_yahoo_chart(payload, allow_empty: bool = False) -> dict:
    """Parse a Yahoo ``v8/finance/chart`` response (bytes / str / dict) into a *bars* dict.

    Returns ``{"d": int64 days since 1970-01-01 (Taipei date), "o","h","l","c","adj","v": float64 arrays,
    "div": [(day, amount)], "meta": {...}, "ph_days"/"real_days"/"null_days": int64 arrays}`` with null / non-positive
    rows removed, duplicate dates collapsed (last wins), ascending order and ``high >= max(o,c)``, ``low <= min(o,c)``
    enforced.  No-trade placeholder rows (:func:`placeholder_mask`; never for an index) are dropped too (audit P-01); the
    three ``*_days`` arrays (placeholder / real / null-or-invalid rows of the last ``EVIDENCE_WINDOW_DAYS`` days) feed the
    closure learning of the calendar (:func:`is_trading_day`).  Raises :class:`NotFoundError` for Yahoo's "Not Found"
    and :class:`DataUnavailable` for other upstream errors and for malformed payloads: no timestamps (unless
    ``allow_empty``: an incremental window can legitimately be empty), no quote block, timestamps that are not epoch
    seconds (milliseconds) -- a transient malformed answer is never mistaken for an unknown symbol (audit P-14).
    """
    try:
        j = payload if isinstance(payload, dict) else json.loads(payload)
        chart = j["chart"]
    except (ValueError, KeyError, TypeError) as exc:
        raise DataUnavailable(f"malformed Yahoo response: {exc!r}") from exc
    err, res = chart.get("error"), chart.get("result")
    if err or not res:
        code = str((err or {}).get("code", ""))
        if code.lower() == "not found" or (not err and not res):
            raise NotFoundError("yahoo: not found")
        raise DataUnavailable(f"Yahoo error: {err}")
    r = res[0]
    meta = r.get("meta") or {}
    ts = r.get("timestamp") or []
    ind = r.get("indicators") or {}
    quote = (ind.get("quote") or [{}])[0] or {}
    adjl = ((ind.get("adjclose") or [{}])[0] or {}).get("adjclose")
    n = len(ts)
    meta_out = {"symbol": meta.get("symbol"), "name": meta.get("longName") or meta.get("shortName"),
                "type": meta.get("instrumentType"), "regular_market_time": meta.get("regularMarketTime")}
    if n == 0:
        if not allow_empty:
            raise DataUnavailable("malformed Yahoo response: no timestamps")
        z = np.zeros(0)
        return {"d": np.zeros(0, dtype=np.int64), "o": z, "h": z, "l": z, "c": z, "adj": z, "v": z, "div": [],
                "meta": meta_out, "ph_days": np.zeros(0, dtype=np.int64), "real_days": np.zeros(0, dtype=np.int64),
                "null_days": np.zeros(0, dtype=np.int64)}
    if not any(k in quote for k in ("open", "high", "low", "close")):
        raise DataUnavailable("malformed Yahoo response: no quote block")

    def col(seq, default=np.nan) -> np.ndarray:
        if seq is None:
            return np.full(n, default, dtype=float)
        if len(seq) != n:
            raise DataUnavailable("malformed Yahoo response: column length mismatch")
        return np.array([default if x is None else x for x in seq], dtype=float)

    try:
        offset = int(meta.get("gmtoffset") if meta.get("gmtoffset") is not None else 28800)
        tsa = np.asarray(ts, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataUnavailable(f"malformed Yahoo response: bad timestamps ({exc!r})") from exc
    if int(tsa.min()) < 0 or int(tsa.max()) > MAX_TIMESTAMP or abs(offset) > 86400:
        raise DataUnavailable("malformed Yahoo response: timestamps are not epoch seconds")
    days = (tsa + offset) // 86400
    o, h, l, c = (col(quote.get(k)) for k in ("open", "high", "low", "close"))
    adj = col(adjl) if adjl is not None else c.copy()
    v = col(quote.get("volume"), 0.0)
    v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
    ok_px = np.ones(n, dtype=bool)
    for a in (o, h, l, c):
        ok_px &= np.isfinite(a) & (a > 0)
    ok_adj = np.isfinite(adj) & (adj > 0)
    # Trailing bars (typically today's) whose adjclose is still null keep the last known adjclose/close ratio.
    good = np.flatnonzero(ok_px & ok_adj)
    if len(good) and good[-1] < n - 1:
        tail = np.arange(good[-1] + 1, n)
        fix = tail[ok_px[tail] & ~ok_adj[tail]]
        if len(fix):
            adj = adj.copy()
            adj[fix] = c[fix] * (adj[good[-1]] / c[good[-1]])
            ok_adj = np.isfinite(adj) & (adj > 0)
    ok = ok_px & ok_adj
    sel = np.flatnonzero(ok)
    # collapse duplicate dates (last occurrence wins), sort ascending
    order = sel[np.lexsort((sel, days[sel]))] if len(sel) else sel
    if len(order):
        dd = days[order]
        keep = np.r_[dd[1:] != dd[:-1], True]
        order = order[keep]
    null_days = np.unique(days[~ok])  # rows without a usable price (Yahoo's null row of a closed / not yet traded day)
    days, o, h, l, c, adj, v = (a[order] for a in (days, o, h, l, c, adj, v))
    # Yahoo serves float32 noise (106.9000015258789): prices have at most 2 decimals, split-adjusted history a few more
    o, h, l, c = (np.round(a, 4) for a in (o, h, l, c))
    adj = np.round(adj, 6)
    h = np.maximum(h, np.maximum(o, c))  # repair impossible bars rather than feed them to range estimators
    l = np.minimum(l, np.minimum(o, c))
    # no-trade placeholder rows (closure / halt filler) are not bars (audit P-01)
    ph = placeholder_mask(o, h, l, c, v, index=_is_index_meta(meta))
    null_days = np.setdiff1d(null_days, days)
    if len(days):
        lo_day = int(days[-1]) - EVIDENCE_WINDOW_DAYS
        recent = days >= lo_day
        ph_days, real_days = days[ph & recent], days[~ph & recent]
        null_days = null_days[null_days >= lo_day]
    else:
        ph_days = real_days = np.zeros(0, dtype=np.int64)
        null_days = null_days[:0]
    if ph.any():
        keep = ~ph
        days, o, h, l, c, adj, v = (a[keep] for a in (days, o, h, l, c, adj, v))
    divs = []
    for ev in ((r.get("events") or {}).get("dividends") or {}).values():
        try:
            dts = int(ev.get("date"))
            amt = float(ev.get("amount"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(amt) and amt > 0 and 0 <= dts <= MAX_TIMESTAMP:
            divs.append(((dts + offset) // 86400, amt))
    divs = sorted(set(divs))
    return {"d": days.astype(np.int64), "o": o, "h": h, "l": l, "c": c, "adj": adj, "v": v, "div": divs,
            "meta": meta_out, "ph_days": ph_days.astype(np.int64), "real_days": real_days.astype(np.int64),
            "null_days": null_days.astype(np.int64)}


# --------------------------------------------------------------------------------------- cache records
def _day_to_iso(d: int) -> str:
    return str(np.datetime64(int(d), "D"))


def _price_path(sym: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_-]", "_", sym)
    return cache_dir() / "prices" / f"{safe}.json"


def _bars_to_rec(sym: str, ysym: str, market: str, bars: dict, *, fetched_at: str, full_fetched_at: str) -> dict:
    meta = bars.get("meta") or {}
    return {"v": CACHE_VERSION, "symbol": sym, "yahoo_symbol": ysym, "market": market,
            "name_hint": meta.get("name") or "", "instrument_type": meta.get("type") or "",
            "fetched_at": fetched_at, "full_fetched_at": full_fetched_at,
            "d": bars["d"], "o": bars["o"], "h": bars["h"], "l": bars["l"], "c": bars["c"], "adj": bars["adj"],
            "vol": bars["v"], "div": [(int(d), float(a)) for d, a in bars["div"]]}


def _rec_to_json(rec: dict) -> dict:
    out = {k: rec[k] for k in ("v", "symbol", "yahoo_symbol", "market", "name_hint", "instrument_type",
                               "fetched_at", "full_fetched_at")}
    out["dates"] = (np.datetime64(0, "D") + rec["d"].astype("timedelta64[D]")).astype(str).tolist()
    for key, name in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("adj", "adjclose"),
                      ("vol", "volume")):
        out[name] = rec[key].tolist()
    out["dividends"] = [[_day_to_iso(d), a] for d, a in rec["div"]]
    return out


def _migrate_legacy(rec: dict) -> dict:
    """Bring a pre-v3 record up to date in memory: drop its no-trade placeholder bars (audit P-01: re-parsing is not
    possible offline, but ``placeholder_mask`` needs nothing more than the stored columns) and forget when it was last
    downloaded in full, so the next refresh re-downloads the whole history instead of merging into it."""
    ph = placeholder_mask(rec["o"], rec["h"], rec["l"], rec["c"], rec["vol"],
                          index=rec.get("symbol") == INDEX_SYMBOL or str(rec.get("instrument_type", "")).upper() == "INDEX")
    out = dict(rec)
    if ph.any():
        keep = ~ph
        for k in ("d", "o", "h", "l", "c", "adj", "vol"):
            out[k] = rec[k][keep]
    out["v"] = CACHE_VERSION
    out["full_fetched_at"] = ""
    return out


def _rec_from_json(j) -> dict | None:
    try:
        if not isinstance(j, dict) or j.get("v") not in (CACHE_VERSION, *LEGACY_CACHE_VERSIONS):
            return None
        d = np.array(j["dates"], dtype="datetime64[D]").astype(np.int64)
        arrs = {k: np.asarray(j[name], dtype=float) for k, name in (
            ("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("adj", "adjclose"), ("vol", "volume"))}
        if any(len(a) != len(d) for a in arrs.values()):
            return None
        if len(d) > 1 and not np.all(np.diff(d) > 0):
            return None
        rec = {k: j[k] for k in ("v", "symbol", "yahoo_symbol", "market", "name_hint", "instrument_type",
                                 "fetched_at", "full_fetched_at")}
        rec["d"] = d
        rec.update(arrs)
        rec["div"] = [(int(np.datetime64(x, "D").astype(np.int64)), float(a)) for x, a in j.get("dividends", [])]
        return rec if rec["v"] == CACHE_VERSION else _migrate_legacy(rec)
    except (KeyError, TypeError, ValueError):
        return None


def _load_rec(sym: str) -> dict | None:
    path = _price_path(sym)
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    sig = (st.st_mtime_ns, st.st_size)
    hit = _rec_mem_get(sym, sig)
    if hit is not None:
        return hit
    rec = _rec_from_json(read_json_file(path))
    if rec is None:
        log.warning("discarding unreadable price cache %s", path)
        return None
    _rec_mem_put(sym, sig, rec)
    return rec


def _save_rec(sym: str, rec: dict) -> None:
    path = _price_path(sym)
    atomic_write_json(path, _rec_to_json(rec))
    try:
        st = path.stat()
        _rec_mem_put(sym, (st.st_mtime_ns, st.st_size), rec)
    except OSError:
        with _MEM_LOCK:
            _REC_MEM.pop(sym, None)


def _last_date(rec: dict) -> date | None:
    if rec is None or len(rec["d"]) == 0:
        return None
    return date.fromordinal(int(rec["d"][-1]) + date(1970, 1, 1).toordinal())


def _bar_final_at_fetch(rec: dict) -> bool:
    """True iff the record was fetched after its last bar's session was final."""
    last, fetched = _last_date(rec), _parse_iso(rec.get("fetched_at", ""))
    if last is None or fetched is None:
        return False
    return fetched >= final_time(last)


# --------------------------------------------------------------------------------------- refresh logic
def _yahoo_url(ysym: str, period1: int, period2: int) -> str:
    return (f"{YAHOO_CHART.format(sym=urllib.parse.quote(ysym, safe='%'))}"
            f"?period1={period1}&period2={period2}&interval=1d&events=div%2Csplits")


CIRCUIT_THRESHOLD = 3  # consecutive failed Yahoo requests (each already retried) that open the circuit
CIRCUIT_OPEN_SECONDS = 30.0  # while open, Yahoo requests fail at once (stale caches are served without a 6 s wait each)
_CIRCUIT: dict = {"fails": 0, "until": 0.0}


def _circuit_check() -> None:
    with _MEM_LOCK:
        wait = _CIRCUIT["until"] - _mono()
    if wait > 0:
        raise DataUnavailable(f"Yahoo unreachable recently -- not retrying for another {wait:.0f}s")


def _circuit_record(ok: bool) -> None:
    with _MEM_LOCK:
        if ok:
            _CIRCUIT["fails"] = 0
            return
        _CIRCUIT["fails"] += 1
        if _CIRCUIT["fails"] >= CIRCUIT_THRESHOLD:
            _CIRCUIT["until"] = _mono() + CIRCUIT_OPEN_SECONDS
            _CIRCUIT["fails"] = 0
            log.warning("Yahoo failed %d times in a row -- pausing requests for %.0fs", CIRCUIT_THRESHOLD,
                        CIRCUIT_OPEN_SECONDS)


def _fetch_bars(ysym: str, period1: int, retries: int = HTTP_RETRIES, allow_empty: bool = False) -> dict:
    _circuit_check()
    url = _yahoo_url(ysym, max(int(period1), 0), int(_wall()))
    try:
        raw = _http_get(url, semaphore=_YAHOO_SEM, retries=retries)
    except NotFoundError:
        _circuit_record(True)  # Yahoo answered: the service is up, the symbol is unknown
        raise
    except DataUnavailable:
        _circuit_record(False)
        raise
    _circuit_record(True)
    return parse_yahoo_chart(raw, allow_empty=allow_empty)


def _candidate_yahoo_symbols(sym: str, rec: dict | None) -> list[tuple[str, str]]:
    """[(yahoo symbol, market)] in the order they should be tried."""
    if sym == INDEX_SYMBOL:
        return [("%5ETWII", "INDEX")]
    if rec and rec.get("yahoo_symbol"):
        first = rec["yahoo_symbol"]
        other = f"{sym}.TWO" if first.endswith(".TW") else f"{sym}.TW"
        return [(first, rec.get("market", "TWSE")), (other, "TWSE" if other.endswith(".TW") else "TPEx")]
    hint = None
    tbl = _names_table(network=False)
    if tbl is None:  # first ever request: load the official table once, its market column saves a wasted .TW probe
        try:
            tbl = _names_table(network=True)
        except Exception as exc:  # noqa: BLE001 - the hint is an optimisation only
            log.warning("name tables unavailable for the market hint: %s", exc)
    if tbl and sym in tbl["items"]:
        hint = tbl["items"][sym]["market"]
    order = [(f"{sym}.TWO", "TPEx"), (f"{sym}.TW", "TWSE")] if hint == "TPEx" else \
        [(f"{sym}.TW", "TWSE"), (f"{sym}.TWO", "TPEx")]
    return order


def _consistent_overlap(old: dict, new: dict) -> bool:
    """Is the freshly downloaded window ``new`` consistent with the cached record ``old`` on shared dates?

    Compares ``close`` (split adjustment) and ``adjclose/close`` (dividend adjustment) on every common date.  The
    last cached bar is exempt from the ``close`` comparison when it was still an intraday bar.
    """
    common, io, inw = np.intersect1d(old["d"], new["d"], return_indices=True)
    if len(common) < 3:
        return False
    # bars the new window should contain (old bars inside its date range) must not have vanished en masse
    lo = new["d"][0]
    in_window = int(np.count_nonzero(old["d"] >= lo))
    if in_window - len(common) > 2:
        return False
    ratio_old, ratio_new = old["adj"][io] / old["c"][io], new["adj"][inw] / new["c"][inw]
    if not np.all(np.abs(ratio_new / ratio_old - 1.0) <= ADJ_RTOL):
        return False
    skip_last = not _bar_final_at_fetch(old)
    cmp_mask = np.ones(len(common), dtype=bool)
    if skip_last:
        cmp_mask &= common != old["d"][-1]
    co, cn = old["c"][io][cmp_mask], new["c"][inw][cmp_mask]
    return bool(np.all(np.abs(cn / co - 1.0) <= 1e-9 + ADJ_RTOL))


def _merge(old: dict, new: dict) -> dict:
    """Replace everything in ``old`` from ``new``'s first date onward by ``new`` -- except cached FINAL bars that the new
    window lacks (audit C-04: Yahoo serves an all-null row now and then for a completed session; the parser drops it, and
    the bar must not disappear from the history with it, neither at the end nor as a hole in the middle).  A provisional
    last bar (cached while its session was still running) is replaced or dropped as before; ``_consistent_overlap`` has
    already limited the number of vanished bars and verified that the adjustment is unchanged, so the restored rows keep
    their stored ``adjclose``."""
    cut = int(np.searchsorted(old["d"], new["d"][0], side="left"))
    lost = ~np.isin(old["d"][cut:], new["d"])
    if lost.any() and not _bar_final_at_fetch(old):
        lost[-1] = False  # the intraday last bar is provisional
    out = dict(old)
    order = None
    out["d"] = np.concatenate([old["d"][:cut], new["d"], old["d"][cut:][lost]])
    if lost.any():
        order = np.argsort(out["d"], kind="stable")
        log.warning("%s: Yahoo served no row for %d completed bar(s) (last %s) -- keeping the cached bar(s)",
                    old.get("symbol"), int(lost.sum()), _day_to_iso(int(old["d"][cut:][lost][-1])))
        out["d"] = out["d"][order]
    for k_old, k_new in (("o", "o"), ("h", "h"), ("l", "l"), ("c", "c"), ("adj", "adj"), ("vol", "v")):
        col = np.concatenate([old[k_old][:cut], new[k_new], old[k_old][cut:][lost]])
        out[k_old] = col if order is None else col[order]
    first = int(new["d"][0])
    restored = {int(x) for x in old["d"][cut:][lost]}
    new_div_days = {int(d) for d, _ in new["div"]}
    out["div"] = sorted({(int(d), float(a)) for d, a in old["div"]
                         if d < first or (d in restored and d not in new_div_days)}
                        | {(int(d), float(a)) for d, a in new["div"]})
    return out


def _keep_final_tail(old: dict | None, new: dict) -> dict:
    """FULL-download counterpart of :func:`_merge`'s restore step (audit C-04): history only moves forward.

    Yahoo intermittently serves an all-null row for the most recent completed session; the parser drops it and the
    re-downloaded record would end a bar earlier than the cache did (``asof`` one day old with ``stale`` false, the
    recommendation job skipping the correct, newer names).  The cached bars after ``new``'s last date are appended
    again, unchanged -- ``adjclose`` rescaled by the change of ``adjclose / close`` at the shared last date (the dividend
    adjustment that is the very reason for a full re-download).  A bar that was still intraday when
    cached is NOT kept (it is provisional and its session is over by now), nor is anything when the close at the shared
    date differs (a split / rights adjustment: the old tail cannot be put on the new price scale).  The caller stamps
    ``fetched_at``, so the refresh throttle still applies and the bar heals as soon as Yahoo serves it.
    """
    if old is None or len(old["d"]) == 0 or len(new["d"]) == 0:
        return new
    last_new = int(new["d"][-1])
    cut = int(np.searchsorted(old["d"], last_new, side="right"))
    if cut >= len(old["d"]) or not _bar_final_at_fetch(old):
        return new
    scale = 1.0
    i = int(np.searchsorted(old["d"], last_new, side="left"))
    if i < len(old["d"]) and int(old["d"][i]) == last_new:
        if abs(float(new["c"][-1]) / float(old["c"][i]) - 1.0) > 1e-6:
            return new
        scale = (float(new["adj"][-1]) / float(new["c"][-1])) / (float(old["adj"][i]) / float(old["c"][i]))
    out = dict(new)
    out["d"] = np.concatenate([new["d"], old["d"][cut:]])
    for k in ("o", "h", "l", "c", "vol"):
        out[k] = np.concatenate([new[k], old[k][cut:]])
    out["adj"] = np.concatenate([new["adj"], old["adj"][cut:] * scale])
    out["div"] = sorted({(int(d), float(a)) for d, a in new["div"]} | {(int(d), float(a)) for d, a in old["div"] if d > last_new})
    log.warning("%s: Yahoo served no row for %d completed bar(s) after %s -- keeping the cached bar(s)",
                new.get("symbol") or old.get("symbol"), len(old["d"]) - cut, _day_to_iso(last_new))
    return out


def _refresh(sym: str, rec: dict | None, now: datetime) -> dict:
    """Download (incrementally when safe) and persist; returns the new record.  Raises UnknownSymbol/DataUnavailable."""
    started = _iso(now)
    cands = _candidate_yahoo_symbols(sym, rec)
    tries = 2 if rec is not None else HTTP_RETRIES  # a cache to fall back on: do not keep the user waiting ~6 s
    inc_ok = (rec is not None and len(rec["d"]) >= 60 and rec.get("v") == CACHE_VERSION)
    if inc_ok:
        full_at = _parse_iso(rec.get("full_fetched_at", ""))
        inc_ok = full_at is not None and (now - full_at) < timedelta(days=FULL_REFRESH_DAYS)
    last_err: Exception | None = None
    for ysym, market in cands:
        try:
            if inc_ok and ysym == rec.get("yahoo_symbol"):
                p1 = (int(rec["d"][-1]) - INCREMENTAL_DAYS) * 86400 - 8 * 3600
                try:
                    inc = _fetch_bars(ysym, p1, tries, allow_empty=True)
                except NotFoundError:
                    inc = None
                if inc is not None and len(inc["d"]) and _consistent_overlap(rec, inc):
                    _learn_closures(sym, inc)
                    new = _merge(rec, inc)
                    new["fetched_at"] = started
                    if inc["meta"].get("name"):
                        new["name_hint"] = inc["meta"]["name"]
                    _save_rec(sym, new)
                    return new
                log.info("%s: incremental window unusable (dividend/split/revision/empty) -> full re-download", sym)
            bars = _fetch_bars(ysym, 0, tries)
            _learn_closures(sym, bars)
            if len(bars["d"]) == 0:  # a well-formed answer without a single usable row is not "no such symbol" (audit P-14)
                raise DataUnavailable(f"{ysym}: Yahoo returned no usable rows")
            new = _bars_to_rec(sym, ysym, market, bars, fetched_at=started, full_fetched_at=started)
            new = _keep_final_tail(rec, new)
            _save_rec(sym, new)
            return new
        except NotFoundError as exc:
            last_err = exc
            continue
    raise UnknownSymbol(f"找不到代號 {sym}") from last_err


def _needs_refresh(sym: str, rec: dict | None, now: datetime, force: bool) -> bool:
    if rec is None or force:
        return True
    fail = _FAIL.get(sym)
    if fail is not None and _mono() - fail < FAIL_BACKOFF:
        return False
    last = _last_date(rec)
    if last is None:
        return True
    fetched = _parse_iso(rec.get("fetched_at", "")) or datetime.fromtimestamp(0, TAIPEI)
    age = (now - fetched).total_seconds()
    if not _bar_final_at_fetch(rec) and now >= final_time(last):
        return True  # the cached last bar was intraday and its session is final by now -> upgrade it
    exp = last_trading_date_expected(now)
    if last < exp:
        return age >= REFRESH_MIN_INTERVAL
    if not _bar_final_at_fetch(rec):
        return age >= REFRESH_MIN_INTERVAL  # intraday bar still running: refresh every ~10 min on demand
    return False


def _lookup_meta(sym: str, rec: dict) -> tuple[str, str, str]:
    """(name, market, kind) for a record, preferring the official Chinese name table."""
    if sym == INDEX_SYMBOL:
        return INDEX_NAME, "INDEX", "index"
    market = rec.get("market") or "TWSE"
    kind = kind_of(sym)
    name = ""
    tbl = _names_table(network=False)
    if tbl is None:  # very first run: load the official names once so the UI never shows an English fallback
        try:
            tbl = _names_table(network=True)
        except Exception as exc:  # noqa: BLE001 - names are cosmetic
            log.warning("name tables unavailable: %s", exc)
    item = tbl["items"].get(sym) if tbl else None
    if item:
        name, market = item["name"], item["market"]
    else:
        fb = dict(FALLBACK_UNIVERSE + FALLBACK_ETFS)
        name = fb.get(sym) or rec.get("name_hint") or sym
    return name, market, kind


def _to_price_data(sym: str, rec: dict, stale: bool) -> PriceData:
    name, market, kind = _lookup_meta(sym, rec)
    div = [(np.datetime64(int(d), "D"), float(a)) for d, a in rec["div"]]
    return PriceData(symbol=sym, name=name, market=market, kind=kind,
                     dates=(np.datetime64(0, "D") + rec["d"].astype("timedelta64[D]")).astype("datetime64[D]"),
                     open=rec["o"].copy(), high=rec["h"].copy(), low=rec["l"].copy(), close=rec["c"].copy(),
                     adjclose=rec["adj"].copy(), volume=rec["vol"].copy(), dividends=div,
                     last_bar_complete=_bar_final_at_fetch(rec), fetched_at=rec.get("fetched_at", ""), stale=stale)


def get_price_data(symbol: str, refresh: bool = False, allow_stale: bool = True) -> PriceData:
    """Daily bars of one stock / ETF / '^TWII' (cached, refreshed per the policy in the module docstring).

    Raises :class:`BadSymbol` (malformed), :class:`UnknownSymbol` (no data source knows it) or
    :class:`DataUnavailable` (network failure and no cache, or ``allow_stale=False``).
    """
    sym = normalize_symbol(symbol)
    with _sym_lock(sym):
        rec = _load_rec(sym)
        now = _now()
        if rec is None:
            unk = _UNKNOWN.get(sym)
            if unk is not None and not refresh and _mono() - unk < UNKNOWN_TTL:
                raise UnknownSymbol(f"找不到代號 {sym}")
        stale = False
        if _needs_refresh(sym, rec, now, refresh):
            try:
                rec = _refresh(sym, rec, now)
                with _MEM_LOCK:
                    _FAIL.pop(sym, None)
                    _UNKNOWN.pop(sym, None)
                    _LAST_OK[sym] = _mono()
                    _cap(_LAST_OK)
            except UnknownSymbol:
                if rec is None:
                    with _MEM_LOCK:
                        _UNKNOWN[sym] = _mono()
                        _cap(_UNKNOWN)
                    raise
                with _MEM_LOCK:
                    _FAIL[sym] = _mono()
                    _cap(_FAIL)
                stale = True
                log.warning("%s: upstream no longer knows the symbol -- serving cache", sym)
            except DataUnavailable as exc:
                with _MEM_LOCK:
                    _FAIL[sym] = _mono()
                    _cap(_FAIL)
                if rec is None or not allow_stale:
                    raise DataUnavailable(f"{sym}: {exc}") from exc
                stale = True
                log.warning("%s: refresh failed (%s) -- serving stale cache", sym, exc)
        else:
            stale = sym in _FAIL and rec is not None and (_last_date(rec) or date.min) < last_trading_date_expected(now)
        if rec is None or len(rec["d"]) == 0:
            raise UnknownSymbol(f"找不到代號 {sym}")
        return _to_price_data(sym, rec, stale)


def peek_price_data(symbol: str) -> PriceData | None:
    """PriceData from the on-disk cache only (never touches the network); ``None`` when not cached."""
    try:
        sym = normalize_symbol(symbol)
    except BadSymbol:
        return None
    rec = _load_rec(sym)
    return _to_price_data(sym, rec, False) if rec is not None and len(rec["d"]) else None


def get_market_data(refresh: bool = False) -> PriceData:
    """TAIEX (^TWII) daily bars; rows with null OHLC are dropped (about 140 historical ones)."""
    return get_price_data(INDEX_SYMBOL, refresh=refresh)


def completed(data: PriceData) -> PriceData:
    """``data`` without a still-running last bar (``data.slice(n-1)``); unchanged when the last bar is final."""
    if data.n and not data.last_bar_complete:
        out = data.slice(data.n - 1)
        out.last_bar_complete = True
        return out
    return data


def price_signature(data: PriceData) -> str:
    """Short hash identifying the exact numbers of ``data`` (cache key ingredient; changes on any revision of a date, an
    open / high / low / close / adjusted close or a volume: the models read all of them -- Parkinson range volatility, the
    volume z-score -- not only the closes)."""
    h = hashlib.sha1()
    h.update(f"{data.symbol}|{data.n}|{data.dates[-1] if data.n else ''}".encode())
    h.update(np.ascontiguousarray(data.dates.astype("datetime64[D]").astype(np.int64)).tobytes())
    for arr in (data.open, data.high, data.low, data.close, data.adjclose, data.volume):  # every model input (audit C-05)
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def avg_traded_value(data: PriceData, window: int = 20) -> float | None:
    """Mean of close x volume (TWD) over the last ``window`` bars; None when there is no volume information."""
    if data.n == 0:
        return None
    v = data.close[-window:] * data.volume[-window:]
    m = float(np.mean(v)) if len(v) else 0.0
    return m if np.isfinite(m) and m > 0 else None


# --------------------------------------------------------------------------------------- calendar
_HOL_MEM: dict[int, frozenset | None] = {}
_HOL_FAIL: dict[int, float] = {}  # year -> monotonic time of the last failed fetch


def _is_trading_marker(name: str, desc: str = "") -> bool:
    """TWSE lists '開始交易日' / '最後交易日' rows in the holiday schedule; those days ARE trading days."""
    if "無交易" in name:
        return False
    return ("開始交易" in name) or ("最後交易" in name)


def parse_holiday_schedule(payload, year: int) -> frozenset | None:
    """Closed weekdays of ``year`` from a TWSE holidaySchedule payload (JSON endpoint or OpenAPI), or None."""
    days: set[date] = set()
    try:
        if isinstance(payload, dict):
            if str(payload.get("stat", "")).lower() != "ok":
                return None
            for row in payload.get("data") or []:
                d = date.fromisoformat(str(row[0]).strip().replace("/", "-"))
                if _is_trading_marker(str(row[1]), str(row[2]) if len(row) > 2 else ""):
                    continue
                days.add(d)
        elif isinstance(payload, list):
            for row in payload:
                s = str(row.get("Date", "")).strip()
                if len(s) < 7 or not s.isdigit():
                    continue
                d = date(int(s[:-4]) + 1911, int(s[-4:-2]), int(s[-2:]))
                if _is_trading_marker(str(row.get("Name", "")), str(row.get("Description", ""))):
                    continue
                days.add(d)
        else:
            return None
    except (ValueError, IndexError, TypeError, KeyError):
        return None
    days = {d for d in days if d.year == year}
    return frozenset(days) if days else None


def _holiday_path(year: int) -> Path:
    return cache_dir() / f"holidays_{year}.json"


_HOL_FETCH_LOCK = threading.Lock()


def _read_holiday_cache(year: int):
    """(frozenset | None, is_fresh) from the on-disk copy."""
    disk = read_json_file(_holiday_path(year))
    if isinstance(disk, dict) and isinstance(disk.get("dates"), list):
        try:
            cached = frozenset(date.fromisoformat(s) for s in disk["dates"])
            age = _wall() - float(disk.get("fetched_ts", 0))
            ttl = HOLIDAY_TTL if year >= _now().year else 365 * 24 * 3600.0
            return cached, age < ttl
        except (ValueError, TypeError):
            pass
    return None, False


def holidays(year: int) -> frozenset | None:
    """Exchange-closed dates of ``year`` (weekends excluded from the set) or ``None`` if the schedule is unknown.

    Memory -> disk (``holidays_{year}.json``, refreshed every 3 days for the current/future years) -> TWSE.  After
    a failed download the stale copy (or None) is returned without touching the network again for 10 minutes.
    """
    with _MEM_LOCK:
        if year in _HOL_MEM:
            return _HOL_MEM[year]
    cached, fresh = _read_holiday_cache(year)
    if fresh:
        with _MEM_LOCK:
            _HOL_MEM[year] = cached
        return cached
    with _HOL_FETCH_LOCK:  # one downloader at a time; the others re-check the memo afterwards
        with _MEM_LOCK:
            if year in _HOL_MEM:
                return _HOL_MEM[year]
        fail = _HOL_FAIL.get(year)
        if fail is not None and _mono() - fail < 600.0:
            return cached  # recently failed -> do not block again; stale copy (or None) it is
        result = None
        for url in (TWSE_HOLIDAY.format(year=year), TWSE_HOLIDAY_OPENAPI):
            try:
                payload = _get_json(url, timeout=8.0, retries=2)
            except (DataUnavailable, NotFoundError) as exc:
                log.warning("holiday schedule %s unavailable: %s", url, exc)
                continue
            result = parse_holiday_schedule(payload, year)
            if result is not None:
                break
        if result is not None:
            try:
                atomic_write_json(_holiday_path(year), {"year": year, "fetched_ts": _wall(),
                                                        "dates": sorted(d.isoformat() for d in result)})
            except OSError as exc:
                log.warning("cannot persist holidays: %s", exc)
            with _MEM_LOCK:
                _HOL_MEM[year] = result
                _HOL_FAIL.pop(year, None)
            return result
        _HOL_FAIL[year] = _mono()
        return cached


# ---- unannounced closures, learned from Yahoo's placeholder bars (audit P-01)
_CLOSE_LOCK = threading.RLock()
_CLOSE_MEM: dict = {"tally": None}  # tally: {iso date: {"ph": set(symbols), "real": set(symbols), "idx_null": bool}}


def _closures_path() -> Path:
    return cache_dir() / "closures.json"


def _closure_tally() -> dict:
    """The evidence tally (memory, loaded once from ``closures.json``).  Caller holds ``_CLOSE_LOCK``."""
    tally = _CLOSE_MEM["tally"]
    if tally is None:
        tally = {}
        disk = read_json_file(_closures_path())
        for k, e in ((disk or {}).get("days") or {}).items() if isinstance(disk, dict) else ():
            try:
                date.fromisoformat(k)
                tally[k] = {"ph": {str(s) for s in e.get("ph", [])}, "real": {str(s) for s in e.get("real", [])},
                            "idx_null": bool(e.get("idx_null", False))}
            except (ValueError, TypeError, AttributeError):
                continue
        _CLOSE_MEM["tally"] = tally
    return tally


def _closed_days(tally: dict, now: datetime) -> frozenset:
    """Weekdays whose session is over (14:00 Taipei) and on which the stocks did not trade.

    Evidence: (a) at least ``CLOSURE_MIN_SYMBOLS`` stocks serve a flat zero-volume placeholder for the day and they are
    at least ``CLOSURE_QUORUM`` of the stocks that have any row for it (a halted or illiquid stock is a handful, a
    closure is nearly all); (b) a stock placeholder, no stock with a real bar, and the TAIEX served a null row.  A real
    bar for the day (any later fetch) takes it back out: the tally is evidence, not a verdict.
    """
    out = set()
    for k, e in tally.items():
        d = date.fromisoformat(k)
        if d.weekday() >= 5 or now < final_time(d):
            continue
        ph, real = len(e["ph"]), len(e["real"])
        if (ph >= CLOSURE_MIN_SYMBOLS and ph >= CLOSURE_QUORUM * (ph + real)) or (ph >= 1 and real == 0 and e["idx_null"]):
            out.add(d)
    return frozenset(out)


def extra_closed_days(now=None) -> frozenset:
    """Exchange-closed weekdays learned from the data (not on the TWSE schedule: typhoon days, 2026-10-23)."""
    t = taipei_now(now)
    with _CLOSE_LOCK:
        tally = _closure_tally()
        return _closed_days(tally, t) if tally else frozenset()


def _learn_closures(sym: str, bars: dict | None) -> None:
    """Record what a parsed Yahoo answer says about the last weeks (placeholder / real / null days) and persist it."""
    if not bars or "ph_days" not in bars:
        return
    try:
        index = sym == INDEX_SYMBOL
        today = _now().date()
        lo = today - timedelta(days=CLOSURE_WINDOW_DAYS)
        epoch = date(1970, 1, 1).toordinal()

        def iso(day) -> str | None:
            d = date.fromordinal(int(day) + epoch)
            return d.isoformat() if lo <= d <= today and d.weekday() < 5 else None

        with _CLOSE_LOCK:
            tally = _closure_tally()
            before = {k: (set(e["ph"]), set(e["real"]), e["idx_null"]) for k, e in tally.items()}

            def entry(k):
                return tally.setdefault(k, {"ph": set(), "real": set(), "idx_null": False})

            if index:
                for day in bars["real_days"]:
                    if (k := iso(day)) is not None:
                        entry(k)["idx_null"] = False
                for day in bars["null_days"]:
                    if (k := iso(day)) is not None:
                        entry(k)["idx_null"] = True
            else:
                for day in bars["ph_days"]:
                    if (k := iso(day)) is not None:
                        e = entry(k)
                        e["ph"].add(sym)
                        e["real"].discard(sym)
                for day in bars["real_days"]:
                    if (k := iso(day)) is not None:
                        e = entry(k)
                        e["real"].add(sym)
                        e["ph"].discard(sym)
            for k in [k for k in tally if date.fromisoformat(k) < lo]:
                del tally[k]
            after = {k: (e["ph"], e["real"], e["idx_null"]) for k, e in tally.items()}
            if after != before:
                atomic_write_json(_closures_path(), {"v": 1, "days": {
                    k: {"ph": sorted(e["ph"]), "real": sorted(e["real"]), "idx_null": e["idx_null"]}
                    for k, e in sorted(tally.items())}})
    except Exception as exc:  # noqa: BLE001 - closure learning is an optimisation of the calendar, never a failure
        log.warning("closure learning failed for %s: %s", sym, exc)


def is_trading_day(d: date) -> bool:
    """Mon-Fri, not on the TWSE holiday schedule (weekends only when the schedule is unreachable) and not a closure
    learned from Yahoo's placeholder bars (:func:`extra_closed_days`)."""
    if d.weekday() >= 5:
        return False
    hol = holidays(d.year)
    if hol and d in hol:
        return False
    return d not in extra_closed_days()


def prev_trading_day(d: date) -> date:
    """Latest trading day strictly before ``d``."""
    x = d - timedelta(days=1)
    for _ in range(40):
        if is_trading_day(x):
            return x
        x -= timedelta(days=1)
    return x


def next_trading_day(d: date) -> date:
    """First trading day strictly after ``d``."""
    x = d + timedelta(days=1)
    for _ in range(40):
        if is_trading_day(x):
            return x
        x += timedelta(days=1)
    return x


def last_trading_date_expected(now=None) -> date:
    """Date of the newest daily bar that can exist at ``now`` (Taipei): today once the 09:00 open has passed on a
    trading day, otherwise the previous trading day."""
    t = taipei_now(now)
    if is_trading_day(t.date()) and t.time() >= SESSION_OPEN:
        return t.date()
    return prev_trading_day(t.date())


def market_is_open(now=None) -> bool:
    """True on trading days between 09:00 and 13:30 Taipei time."""
    t = taipei_now(now)
    return is_trading_day(t.date()) and SESSION_OPEN <= t.time() < SESSION_CLOSE


# --------------------------------------------------------------------------------------- name tables
_NAMES_MEM: dict = {"table": None, "fail_until": 0.0}
_NAMES_LOCK = threading.RLock()


def _names_path() -> Path:
    return cache_dir() / "names.json"


def _num(x) -> float | None:
    try:
        v = float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def parse_name_tables(twse_day, twse_company, tpex_day) -> dict:
    """Merge the raw OpenAPI lists into ``{code: {code,name,market,kind,tv,close}}`` (stocks + ETFs only).

    ``tv`` is the day's traded value in TWD (TWSE ``TradeValue`` / TPEx ``TransactionAmount``).  Any argument may
    be None (source unavailable).
    """
    items: dict[str, dict] = {}
    for r in twse_day or []:
        code = str(r.get("Code", "")).strip()
        name = str(r.get("Name", "")).strip()
        if not _INSTRUMENT_RE.match(code) or not name:
            continue
        items[code] = {"code": code, "name": name, "market": "TWSE", "kind": kind_of(code),
                       "tv": _num(r.get("TradeValue")), "close": _num(r.get("ClosingPrice"))}
    for r in twse_company or []:
        code = str(r.get("公司代號", "")).strip()
        name = str(r.get("公司簡稱", "")).strip()
        if code in items or not _INSTRUMENT_RE.match(code) or not name:
            continue  # day table wins; this only adds listed names that did not trade that day
        items[code] = {"code": code, "name": name, "market": "TWSE", "kind": kind_of(code), "tv": None,
                       "close": None}
    for r in tpex_day or []:
        code = str(r.get("SecuritiesCompanyCode", "")).strip()
        name = str(r.get("CompanyName", "")).strip()
        if code in items or not _INSTRUMENT_RE.match(code) or not name:
            continue
        items[code] = {"code": code, "name": name, "market": "TPEx", "kind": kind_of(code),
                       "tv": _num(r.get("TransactionAmount")), "close": _num(r.get("Close"))}
    return items


def _fetch_name_tables() -> dict | None:
    """Download the three tables; None when none of them could be read."""
    twse_day = twse_co = tpex = None
    errs = []
    for label, url, big in (("twse_day", TWSE_DAY_ALL, False), ("twse_company", TWSE_COMPANY, False),
                            ("tpex", TPEX_DAY, True)):
        try:
            payload = _get_json(url, timeout=60.0 if big else 30.0, retries=3)
        except (DataUnavailable, NotFoundError) as exc:
            errs.append(f"{label}: {exc}")
            continue
        if not isinstance(payload, list):
            errs.append(f"{label}: unexpected payload")
            continue
        if label == "twse_day":
            twse_day = payload
        elif label == "twse_company":
            twse_co = payload
        else:
            tpex = payload
    if errs:
        log.warning("name tables partially unavailable: %s", "; ".join(errs))
    if twse_day is None and twse_co is None and tpex is None:
        return None
    return {"items": parse_name_tables(twse_day, twse_co, tpex), "sources": {
        "twse_day": twse_day is not None, "twse_company": twse_co is not None, "tpex": tpex is not None}}


def _names_fresh(tbl: dict | None) -> bool:
    if not tbl:
        return False
    exp = tbl.get("expires_ts")
    if exp is None:
        exp = float(tbl.get("fetched_ts", 0)) + NAMES_TTL
    return _wall() < float(exp)


_NAMES_FETCH_LOCK = threading.Lock()


NAMES_BACKGROUND_REFRESH = True  # serve an expired table at once and re-download in a daemon thread (tests: False)
_NAMES_BG: dict = {"thread": None}


def _spawn_names_refresh() -> None:
    """Start (at most one) background re-download of the name tables."""
    with _NAMES_LOCK:
        th = _NAMES_BG["thread"]
        if th is not None and th.is_alive():
            return

        def work():
            try:
                _names_table(network=True, refresh=True)
            except Exception as exc:  # noqa: BLE001 - background housekeeping must never raise
                log.warning("background name-table refresh failed: %s", exc)

        th = _NAMES_BG["thread"] = threading.Thread(target=work, name="names-refresh", daemon=True)
        th.start()


def _names_table(network: bool = True, refresh: bool = False, block: bool = False) -> dict | None:
    """``{"items": {code: item}, "fetched_ts", "expires_ts", "sources": {...}}`` or None when nothing is known.

    ``network=False`` only consults memory / disk (whatever age).  Otherwise an expired table (``NAMES_TTL`` = 1 day;
    30 minutes when a source was missing) or ``refresh`` triggers a re-download.  With nothing cached the caller waits
    for the download.  An expired table is returned immediately and refreshed in a background thread unless
    ``block`` (or ``refresh``) is set; if a download fails the old table is served as is.
    """
    with _NAMES_LOCK:
        tbl = _NAMES_MEM["table"]
        if tbl is None:
            disk = read_json_file(_names_path())
            if isinstance(disk, dict) and isinstance(disk.get("items"), dict) and disk["items"]:
                tbl = _NAMES_MEM["table"] = disk
    if not network:
        return tbl
    if _names_fresh(tbl) and not refresh:
        return tbl
    if _mono() < _NAMES_MEM["fail_until"] and not refresh:
        return tbl
    if tbl is not None and not refresh and not block and NAMES_BACKGROUND_REFRESH:
        _spawn_names_refresh()  # stale-while-revalidate: a search request must not wait for TWSE / TPEx
        return tbl
    if not _NAMES_FETCH_LOCK.acquire(blocking=(tbl is None)):
        return tbl  # somebody else is refreshing; serve what we have
    try:
        with _NAMES_LOCK:
            cur = _NAMES_MEM["table"]
        if cur is not tbl and _names_fresh(cur) and not refresh:
            return cur  # refreshed while we waited for the lock
        if _mono() < _NAMES_MEM["fail_until"] and not refresh:
            return cur  # the attempt we waited for just failed: do not repeat it N times in a row
        fetched = _fetch_name_tables()
        if fetched is None or not fetched["items"]:
            _NAMES_MEM["fail_until"] = _mono() + 300.0
            return cur
        if cur and not fetched["sources"].get("tpex"):
            # keep the OTC names we already know when only the (flaky) TPEx download failed
            for code, it in cur["items"].items():
                if it.get("market") == "TPEx" and code not in fetched["items"]:
                    fetched["items"][code] = it
        complete = all(fetched["sources"].values())
        now_ts = _wall()
        new = {"fetched_ts": now_ts, "fetched_at": _iso(_now()),
               "expires_ts": now_ts + (NAMES_TTL if complete else 1800.0), **fetched}
        try:
            atomic_write_json(_names_path(), new)
        except OSError as exc:
            log.warning("cannot persist names: %s", exc)
        with _NAMES_LOCK:
            _NAMES_MEM["table"] = new
        _NAMES_MEM["fail_until"] = 0.0
        return new
    finally:
        _NAMES_FETCH_LOCK.release()


def _fallback_items() -> dict:
    items = {}
    for code, name in FALLBACK_UNIVERSE:
        items[code] = {"code": code, "name": name, "market": "TWSE", "kind": "stock", "tv": None, "close": None}
    for code, name in FALLBACK_ETFS:
        items[code] = {"code": code, "name": name, "market": "TWSE", "kind": "etf", "tv": None, "close": None}
    return items


def _public(it: dict) -> dict:
    return {"code": it["code"], "name": it["name"], "market": it["market"], "kind": it["kind"]}


def instrument_info(symbol) -> dict:
    """``{code, name, market, kind}`` of a listed instrument; raises ``KeyError`` when unknown.

    '^TWII' (and aliases) give the index.  Consults the official name tables, then the price cache.
    """
    try:
        sym = normalize_symbol(symbol)
    except BadSymbol as exc:
        raise KeyError(str(symbol)) from exc
    if sym == INDEX_SYMBOL:
        return {"code": sym, "name": INDEX_NAME, "market": "INDEX", "kind": "index"}
    tbl = _names_table(network=True)
    items = tbl["items"] if tbl else _fallback_items()
    it = items.get(sym) or _fallback_items().get(sym)
    if it:
        return _public(it)
    rec = _load_rec(sym)
    if rec is not None and len(rec["d"]):
        name, market, kind = _lookup_meta(sym, rec)
        return {"code": sym, "name": name, "market": market, "kind": kind}
    raise KeyError(sym)


def resolve_symbol(text) -> str:
    """Like :func:`normalize_symbol` but also accepts an exact Chinese name ('台積電').  Raises BadSymbol."""
    try:
        return normalize_symbol(text)
    except BadSymbol:
        q = _fold(str(text or ""))
        if q:
            tbl = _names_table(network=True)
            for it in (tbl["items"] if tbl else _fallback_items()).values():
                if _fold(it["name"]) == q:
                    return it["code"]
        raise


def search_instruments(q: str, limit: int = 20) -> list[dict]:
    """Instruments whose code starts with ``q`` or whose Chinese name contains ``q`` (case/臺台-insensitive).

    Ranking: exact code, code prefix, name prefix, name substring; ties by yesterday's traded value (liquid first).
    An empty ``q`` returns :func:`popular_instruments` cut to ``limit`` (the HTTP endpoint raises ``limit`` to
    ``POPULAR_COUNT`` for it, so the documented list -- 12 universe names + 3 ETFs -- is never truncated).
    """
    limit = max(1, int(limit))
    qq = _fold(q or "")
    if not qq:
        return popular_instruments()[:limit]
    tbl = _names_table(network=True)
    items = tbl["items"] if tbl else _fallback_items()
    scored = []
    for it in items.values():
        code, name = it["code"].upper(), _fold(it["name"])
        if code == qq:
            rank = 0
        elif code.startswith(qq):
            rank = 1
        elif name.startswith(qq):
            rank = 2
        elif qq in name:
            rank = 3
        else:
            continue
        scored.append((rank, -(it.get("tv") or 0.0), it["code"], it))
    scored.sort(key=lambda t: t[:3])
    return [_public(t[3]) for t in scored[:limit]]


def popular_instruments() -> list[dict]:
    """First 12 names of the universe plus 0050 / 0056 / 00878 (the search box's empty-query list)."""
    out = [dict(i) for i in get_universe()[:POPULAR_UNIVERSE]]
    tbl = _names_table(network=False)
    items = tbl["items"] if tbl else {}
    fb = _fallback_items()
    for code in POPULAR_ETFS:
        it = items.get(code) or fb.get(code)
        if it:
            out.append(_public(it))
    return out


# --------------------------------------------------------------------------------------- universe
def _universe_path() -> Path:
    return cache_dir() / "universe.json"


def build_universe(items: dict | None, size: int = UNIVERSE_CACHE_N) -> list[dict]:
    """Ranked universe: curated large caps present in the live table first, then the highest traded-value 4-digit
    stocks (ETFs excluded).  ``items=None`` -> curated list only (offline)."""
    ranked: list[dict] = []
    seen = set()
    for code, fb_name in FALLBACK_UNIVERSE:
        it = items.get(code) if items else None
        if items and (it is None or it["kind"] != "stock"):
            continue  # code vanished (delisted / merged) -> drop; the top-up fills the slot
        ranked.append({"code": code, "name": it["name"] if it else fb_name,
                       "market": it["market"] if it else "TWSE", "kind": "stock"})
        seen.add(code)
    if items:
        extra = [it for it in items.values()
                 if it["kind"] == "stock" and re.fullmatch(r"\d{4}", it["code"]) and it["code"] not in seen
                 and it.get("tv") and it.get("close")]
        extra.sort(key=lambda it: (-it["tv"], it["code"]))
        for it in extra[: max(0, size - len(ranked))]:
            ranked.append(_public(it))
    return ranked[:size]


def get_universe(n: int = config.UNIVERSE_SIZE, refresh: bool = False) -> list[dict]:
    """The ``n`` most liquid names for the recommender (``[{code,name,market,kind}]``), stable for 7 days.

    Curated Taiwan-50-style large caps verified against the live TWSE table, topped up with the highest
    traded-value four-digit stocks.  ETFs are excluded.  Cached in ``data_cache/universe.json``; an offline build
    (curated list only) is retried after an hour and never replaces an older live ranking.
    """
    n = max(1, int(n))
    cached = read_json_file(_universe_path())
    if isinstance(cached, dict) and isinstance(cached.get("items"), list) and not refresh:
        age = _wall() - float(cached.get("created_ts", 0))
        ttl = UNIVERSE_TTL if cached.get("source") == "live" else 3600.0
        if age < ttl and len(cached["items"]) >= n:
            return [dict(i) for i in cached["items"][:n]]
    tbl = _names_table(network=True, block=True)
    live = bool(tbl and tbl.get("sources", {}).get("twse_day", True))
    items = build_universe(tbl["items"] if live else None, max(n, UNIVERSE_CACHE_N))
    if not live and isinstance(cached, dict) and cached.get("source") == "live" and len(cached.get("items", [])) >= n:
        return [dict(i) for i in cached["items"][:n]]  # keep the older live ranking rather than a degraded one
    try:
        atomic_write_json(_universe_path(), {"created_ts": _wall(), "created": _iso(_now()),
                                             "source": "live" if live else "fallback", "items": items})
    except OSError as exc:
        log.warning("cannot persist universe: %s", exc)
    return [dict(i) for i in items[:n]]


def warm_up() -> None:
    """Best-effort prefetch of the small reference data (holidays, names, universe).  Never raises."""
    for fn in (lambda: holidays(_now().year), lambda: holidays(_now().year + 1), get_universe):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - warm-up must never kill the server
            log.warning("warm-up step failed: %s", exc)
