"""News / announcement archiver (PLAN.md section 7, P4a).  Research data only: nothing here feeds the product engine.

Six sources are polled (endpoints measured in PLAN.md section 9) and every item is appended -- title, link, source and
times only, never the article body -- to ``<cache>/alt/news/YYYY-MM-DD.jsonl`` (the date is the *Taipei date of
``fetched_at``*)::

    {"id": sha1(source + link), "source", "title", "link", "published_at", "fetched_at", "codes": [...]}

``fetched_at`` is the only timestamp point-in-time code may trust for availability (PLAN section 3): a feed's own
publication time is stored for reference but is never evidence of when *we* could have known the item.  Items are
de-duplicated on ``id`` against every archived day-file.

Sources (``--sources``): ``twse`` / ``tpex`` (material announcements, company code given), ``yahoo`` (per-stock RSS for
the first 60 names of the research pool, code = the queried one), ``cnyes`` (JSON), ``cna`` (RSS) and ``google``
(RSS, keywords in ``news_queries.json``); the last three map titles to codes by matching company names of
``backend.data._names_table`` (names of 2 characters or fewer are never matched).

One failing source never fails the run.  Requests to the same host are spaced >= 3 s apart.

    python -m backend.altdata.news --once [--sources twse,tpex,yahoo,cnyes,cna,google] [--no-network]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from backend import data

log = logging.getLogger("twq.altdata.news")

TAIPEI = data.TAIPEI
SOURCES = ("twse", "tpex", "yahoo", "cnyes", "cna", "google")
MIN_INTERVAL = 3.0  # seconds between two requests to the same host
YAHOO_N = 60
MIN_NAME_LEN = 3  # company names with <= 2 characters are never matched against titles
MAX_TITLE = 500

URL_TWSE = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
URL_TPEX = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O"
URL_YAHOO = "https://tw.stock.yahoo.com/rss?s={code}"
URL_CNYES = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?limit=30"
URL_CNA = "https://feeds.feedburner.com/rsscna/finance"
URL_GOOGLE = "https://news.google.com/rss/search?q={q}+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
MOPS_PAGE = "https://mops.twse.com.tw/mops/web/t05sr01_1"  # the official announcement page (items have no URL of their own)
CNYES_ITEM = "https://news.cnyes.com/news/id/{id}"
QUERIES_FILE = Path(__file__).with_name("news_queries.json")
_CODE_RE = re.compile(r"^\d{4,6}[A-Z]?$")


# --------------------------------------------------------------------------------------- small helpers
def _now() -> datetime:
    """Current Taipei time (aware); tests inject their own clock."""
    return datetime.now(TAIPEI)


def _iso(t: datetime) -> str:
    return t.astimezone(TAIPEI).isoformat(timespec="seconds")


def clean_title(s) -> str:
    """Collapse all whitespace (the TPEx feed embeds ``\\r\\n``) and cap the length."""
    return re.sub(r"\s+", " ", str(s or "")).strip()[:MAX_TITLE]


def roc_date(s) -> date | None:
    """ROC date ``1151112`` -> ``2026-11-12`` (also ``991231``; 8 digits are taken as an AD date).  None if invalid."""
    digits = re.sub(r"\D", "", str(s if s is not None else ""))
    try:
        if len(digits) == 8:
            return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))
        if len(digits) in (6, 7):
            return date(int(digits[:-4]) + 1911, int(digits[-4:-2]), int(digits[-2:]))
    except ValueError:
        pass
    return None


def roc_datetime(date_s, time_s) -> str | None:
    """Announcement date + time -> Taipei ISO string.  ``發言時間`` is HHMMSS without the leading zero (``10405`` is
    01:04:05).  None when either part is invalid (no guessing)."""
    d = roc_date(date_s)
    digits = re.sub(r"\D", "", str(time_s if time_s is not None else ""))
    if d is None or not digits or len(digits) > 6:
        return None
    digits = digits.zfill(6)
    try:
        t = datetime(d.year, d.month, d.day, int(digits[:2]), int(digits[2:4]), int(digits[4:]), tzinfo=TAIPEI)
    except ValueError:
        return None
    return _iso(t)


def rfc822_to_iso(s) -> str | None:
    if not s:
        return None
    try:
        t = parsedate_to_datetime(str(s).strip())
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return _iso(t)


def epoch_to_iso(v) -> str | None:
    try:
        return _iso(datetime.fromtimestamp(float(v), tz=timezone.utc))
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def item_id(source: str, link: str) -> str:
    return hashlib.sha1((source + link).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------- HTTP (polite)
class Fetcher:
    """GET with a minimum spacing between two requests to the same host.  ``get`` defaults to
    ``backend.data._http_get`` (retries / back-off / ``TWQ_OFFLINE`` live there); tests inject fakes."""

    def __init__(self, get=None, *, min_interval: float = MIN_INTERVAL, sleep=time.sleep, clock=time.monotonic):
        self._get = get
        self.min_interval = float(min_interval)
        self._sleep = sleep
        self._clock = clock
        self._last: dict[str, float] = {}

    def get(self, url: str) -> bytes:
        host = urllib.parse.urlsplit(url).netloc.lower()
        last = self._last.get(host)
        if last is not None:
            wait = self.min_interval - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        fn = self._get or data._http_get
        try:
            return fn(url, retries=3, headers={"Accept": "*/*"}) if self._get is None else fn(url)
        finally:
            self._last[host] = self._clock()


def offline_fetcher() -> Fetcher:
    def dead(url, **kw):
        raise data.DataUnavailable(f"network disabled (--no-network): {url}")

    return Fetcher(dead, min_interval=0.0)


def _json_body(raw: bytes):
    return json.loads(raw.decode("utf-8-sig"))


# --------------------------------------------------------------------------------------- company-name matching
class NameMatcher:
    """Title -> company codes by plain substring match on company names (>= 3 characters, ordinary stocks only)."""

    def __init__(self, names: dict[str, str] | None = None):
        self._pairs: list[tuple[str, str]] = []
        for code, name in (names or {}).items():
            f = data._fold(str(name or ""))
            if len(f) >= MIN_NAME_LEN:
                self._pairs.append((f, str(code)))

    @classmethod
    def from_items(cls, items: dict | None) -> "NameMatcher":
        names = {}
        for code, it in (items or {}).items():
            if isinstance(it, dict) and it.get("kind", "stock") == "stock":
                names[it.get("code", code)] = it.get("name", "")
        return cls(names)

    def codes_for(self, title: str) -> list[str]:
        f = data._fold(title or "")
        if not f:
            return []
        return sorted({code for name, code in self._pairs if name in f})


def load_matcher(network: bool = True) -> NameMatcher:
    """Matcher from the cached official name table (disk first; downloaded only when nothing is cached)."""
    tbl = None
    try:
        tbl = data._names_table(network=False)
        if tbl is None and network:
            tbl = data._names_table(network=True, block=True)
    except Exception as exc:  # noqa: BLE001 - a missing name table only costs the code mapping
        log.warning("name table unavailable: %s", exc)
    if not tbl:
        log.warning("no company-name table: codes of RSS/JSON sources will be empty")
    return NameMatcher.from_items(tbl["items"] if tbl else {})


# --------------------------------------------------------------------------------------- parsers (pure)
def _row(source: str, title, link, published_at, codes) -> dict | None:
    title = clean_title(title)
    link = str(link or "").strip()
    if not title or not link:
        return None
    return {"source": source, "title": title, "link": link, "published_at": published_at,
            "codes": sorted({str(c) for c in codes if c})}


def parse_mops(rows, source: str) -> list[dict]:
    """TWSE (``t187ap04_L``) / TPEx (``mopsfin_t187ap04_O``) material announcements.

    Keys are stripped before lookup (the TWSE subject key is ``"主旨 "`` with a trailing space).  The code is
    ``公司代號`` (TWSE) or ``SecuritiesCompanyCode`` (TPEx).  The feeds carry no per-item URL, so the link is the
    official announcement page plus a stable fragment built from code / date / time / content hash."""
    out = []
    if not isinstance(rows, list):
        raise ValueError("expected a JSON list of announcements")
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = {str(k).strip(): v for k, v in r.items()}
        code = str(d.get("公司代號") or d.get("SecuritiesCompanyCode") or "").strip()
        subject = clean_title(d.get("主旨"))
        if not code or not subject:
            continue
        key = hashlib.sha1("|".join(str(d.get(k, "")) for k in ("主旨", "符合條款", "事實發生日")).encode("utf-8")).hexdigest()[:10]
        fragment = f"{code}-{str(d.get('發言日期', '')).strip()}-{str(d.get('發言時間', '')).strip()}-{key}"
        row = _row(source, subject, f"{MOPS_PAGE}#{fragment}",
                   roc_datetime(d.get("發言日期"), d.get("發言時間")), [code])
        if row:
            out.append(row)
    return out


def parse_twse(rows) -> list[dict]:
    return parse_mops(rows, "twse")


def parse_tpex(rows) -> list[dict]:
    return parse_mops(rows, "tpex")


def parse_rss(xml_bytes: bytes, source: str, matcher: NameMatcher | None = None, fixed_codes=None) -> list[dict]:
    """RSS 2.0 -> rows (title, link, pubDate only; ``description`` is deliberately ignored)."""
    root = ET.fromstring(xml_bytes)
    out = []
    for it in root.iter("item"):
        title = it.findtext("title")
        link = it.findtext("link")
        codes = list(fixed_codes or [])
        if matcher is not None:
            codes += matcher.codes_for(clean_title(title))
        row = _row(source, title, link, rfc822_to_iso(it.findtext("pubDate")), codes)
        if row:
            out.append(row)
    return out


def parse_cnyes(obj, matcher: NameMatcher | None = None) -> list[dict]:
    """Cnyes ``newslist`` JSON (``items.data[]``).  Codes = name matches in the title, plus the item's own ``stock``
    list when it holds plain stock codes."""
    items = obj["items"]["data"]
    out = []
    for d in items:
        if not isinstance(d, dict) or d.get("newsId") in (None, ""):
            continue
        title = clean_title(d.get("title"))
        codes = matcher.codes_for(title) if matcher is not None else []
        stock = d.get("stock")
        if isinstance(stock, list):
            codes += [str(s).strip() for s in stock if _CODE_RE.match(str(s).strip())]
        row = _row("cnyes", title, CNYES_ITEM.format(id=d["newsId"]), epoch_to_iso(d.get("publishAt")), codes)
        if row:
            out.append(row)
    return out


# --------------------------------------------------------------------------------------- fetchers (one per source)
def _stamp(rows: list[dict], now) -> list[dict]:
    fetched_at = _iso(now())
    for r in rows:
        r["fetched_at"] = fetched_at
    return rows


def fetch_twse(fetcher: Fetcher, now=_now, **_) -> list[dict]:
    return _stamp(parse_twse(_json_body(fetcher.get(URL_TWSE))), now)


def fetch_tpex(fetcher: Fetcher, now=_now, **_) -> list[dict]:
    return _stamp(parse_tpex(_json_body(fetcher.get(URL_TPEX))), now)


def fetch_cnyes(fetcher: Fetcher, now=_now, matcher: NameMatcher | None = None, **_) -> list[dict]:
    return _stamp(parse_cnyes(_json_body(fetcher.get(URL_CNYES)), matcher), now)


def fetch_cna(fetcher: Fetcher, now=_now, matcher: NameMatcher | None = None, **_) -> list[dict]:
    return _stamp(parse_rss(fetcher.get(URL_CNA), "cna", matcher), now)


def _many(label: str, keys, one) -> list[dict]:
    """Run ``one(key) -> rows`` per key; a failing key is logged and skipped, but all failing is an error."""
    rows, fails, last = [], 0, None
    keys = list(keys)
    for k in keys:
        try:
            rows += one(k)
        except Exception as exc:  # noqa: BLE001 - isolate one bad request
            fails += 1
            last = exc
            log.warning("%s %s failed: %s", label, k, exc)
    if keys and fails == len(keys):
        raise RuntimeError(f"all {fails} {label} requests failed (last: {last})")
    if fails:
        log.warning("%s: %d of %d requests failed", label, fails, len(keys))
    return rows


def fetch_yahoo(fetcher: Fetcher, now=_now, codes=None, **_) -> list[dict]:
    """One RSS request per code; each row's code is the queried one (and ``fetched_at`` is stamped per request)."""
    def one(code):
        return _stamp(parse_rss(fetcher.get(URL_YAHOO.format(code=code)), "yahoo", None, [code]), now)

    return _many("yahoo", codes if codes is not None else universe_codes(), one)


def fetch_google(fetcher: Fetcher, now=_now, matcher: NameMatcher | None = None, queries=None, **_) -> list[dict]:
    def one(q):
        url = URL_GOOGLE.format(q=urllib.parse.quote(q, safe=""))
        return _stamp(parse_rss(fetcher.get(url), "google", matcher), now)

    return _many("google", queries if queries is not None else load_queries(), one)


FETCHERS = {"twse": fetch_twse, "tpex": fetch_tpex, "yahoo": fetch_yahoo, "cnyes": fetch_cnyes, "cna": fetch_cna,
            "google": fetch_google}


def load_queries(path: Path | None = None) -> list[str]:
    obj = json.loads(Path(path or QUERIES_FILE).read_text(encoding="utf-8"))
    qs = obj.get("queries", []) if isinstance(obj, dict) else obj
    return [q.strip() for q in qs if isinstance(q, str) and q.strip()]


def universe_codes(n: int = YAHOO_N) -> list[str]:
    """First ``n`` codes of ``<cache>/research/pool300.json``; else ``backend.data.get_universe(n)``."""
    try:
        pool = data.read_json_file(data.cache_dir() / "research" / "pool300.json")
        codes = [str(it["code"]) for it in pool if isinstance(it, dict) and it.get("code")] if isinstance(pool, list) else []
        if codes:
            return codes[:n]
    except Exception as exc:  # noqa: BLE001
        log.warning("pool300.json unreadable: %s", exc)
    return [str(it["code"]) for it in data.get_universe(n)][:n]


# --------------------------------------------------------------------------------------- storage
@contextmanager
def _dir_lock(path: Path, timeout: float = 30.0, stale: float = 300.0):
    """Cross-process lock file (O_EXCL); a lock older than ``stale`` seconds is taken over."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except (FileExistsError, PermissionError):  # Windows reports a lock that is being deleted as PermissionError
            try:
                if time.time() - path.stat().st_mtime > stale:
                    path.unlink()
                    continue
            except OSError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"news archive lock busy: {path}")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            path.unlink()
        except OSError:
            pass


class NewsStore:
    """Append-only ``YYYY-MM-DD.jsonl`` files under ``<root>`` (default ``<cache>/alt/news``)."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else data.cache_dir() / "alt" / "news"
        self._id_cache: dict[Path, tuple[tuple[int, int], set[str]]] = {}

    def path_for(self, day: date) -> Path:
        return self.root / f"{day.isoformat()}.jsonl"

    def _file_ids(self, p: Path) -> set[str]:
        """Ids in one day-file, cached per (size, mtime) so a run does not re-parse unchanged files."""
        try:
            st = p.stat()
        except FileNotFoundError:
            return set()
        hit = self._id_cache.get(p)
        if hit is not None and hit[0] == (st.st_size, st.st_mtime_ns):
            return hit[1]
        ids: set[str] = set()
        with open(p, "rb") as fh:
            for line in fh:
                try:
                    ids.add(json.loads(line.decode("utf-8"))["id"])
                except (ValueError, KeyError, TypeError):
                    continue  # torn / foreign line: ignore
        self._id_cache[p] = ((st.st_size, st.st_mtime_ns), ids)
        return ids

    def _ids(self, day: date | None = None) -> set[str]:
        """Ids of EVERY archived day-file.  Feeds (Yahoo per-stock, Google) keep returning items for days, so a
        short look-back window would re-archive old items with a fresh ``fetched_at`` (``day`` is unused)."""
        ids: set[str] = set()
        for p in sorted(self.root.glob("????-??-??.jsonl")):
            ids |= self._file_ids(p)
        return ids

    def add(self, rows: list[dict], fetched_at: str | None = None) -> int:
        """Append the rows whose id is new; returns how many were written.  Rows missing ``fetched_at`` get
        ``fetched_at`` (default now).  Duplicate ids inside ``rows`` are merged (codes unioned) before writing."""
        stamp = fetched_at or _iso(_now())
        merged: dict[str, dict] = {}
        for r in rows:
            rec = {"id": item_id(r["source"], r["link"]), "source": r["source"], "title": r["title"],
                   "link": r["link"], "published_at": r.get("published_at"),
                   "fetched_at": r.get("fetched_at") or stamp, "codes": sorted(set(r.get("codes") or []))}
            old = merged.get(rec["id"])
            if old is None:
                merged[rec["id"]] = rec
            else:
                old["codes"] = sorted(set(old["codes"]) | set(rec["codes"]))
        by_day: dict[date, list[dict]] = {}
        for rec in merged.values():
            day = datetime.fromisoformat(rec["fetched_at"]).astimezone(TAIPEI).date()
            by_day.setdefault(day, []).append(rec)
        if not by_day:
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        added = 0
        with _dir_lock(self.root / ".lock"):
            known = self._ids()
            for day in sorted(by_day):
                fresh = [r for r in by_day[day] if r["id"] not in known]
                if not fresh:
                    continue
                blob = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in fresh).encode("utf-8")
                path = self.path_for(day)
                if path.exists() and path.stat().st_size > 0:
                    with open(path, "rb") as rd:
                        rd.seek(-1, os.SEEK_END)
                        if rd.read(1) != b"\n":  # an interrupted earlier write: start on a fresh line
                            blob = b"\n" + blob
                with open(path, "ab") as fh:
                    fh.write(blob)
                    fh.flush()
                    os.fsync(fh.fileno())
                added += len(fresh)
        return added

    def read_day(self, day: date) -> list[dict]:
        out = []
        try:
            with open(self.path_for(day), "rb") as fh:
                for line in fh:
                    try:
                        out.append(json.loads(line.decode("utf-8")))
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass
        return out


# --------------------------------------------------------------------------------------- orchestration
def run_once(sources=None, *, fetcher: Fetcher | None = None, store: NewsStore | None = None, now=_now,
             matcher: NameMatcher | None = None, codes=None, queries=None) -> dict:
    """Poll the chosen sources once.  Returns ``{"added": {src: n}, "fetched": {src: n}, "errors": {src: msg}}``.

    A source that raises is recorded in ``errors`` and the run continues; what each source returned is written right
    away (a later failure or an interrupt loses nothing)."""
    sel = list(sources) if sources else list(SOURCES)
    bad = [s for s in sel if s not in FETCHERS]
    if bad:
        raise ValueError(f"unknown source(s): {', '.join(bad)} (choose from {', '.join(SOURCES)})")
    fetcher = fetcher or Fetcher()
    store = store or NewsStore()
    res = {"added": {}, "fetched": {}, "errors": {}}
    for s in sel:
        try:
            if matcher is None and s in ("cnyes", "cna", "google"):
                matcher = load_matcher()
            rows = FETCHERS[s](fetcher, now=now, matcher=matcher, codes=codes, queries=queries)
            res["fetched"][s] = len(rows)
            res["added"][s] = store.add(rows)
        except Exception as exc:  # noqa: BLE001 - one source must never take the others down
            log.warning("source %s failed: %s", s, exc)
            res["errors"][s] = f"{type(exc).__name__}: {exc}"
            res["added"].setdefault(s, 0)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.altdata.news", description="Archive Taiwan market news headlines.")
    ap.add_argument("--once", action="store_true", help="poll every selected source once and exit")
    ap.add_argument("--sources", default=",".join(SOURCES), help="comma list of: " + ",".join(SOURCES))
    ap.add_argument("--no-network", action="store_true", help="tests only: every request fails")
    args = ap.parse_args(argv)
    if not args.once:
        ap.error("only --once is supported (scheduling is the user's decision)")
    sel = [s.strip() for s in args.sources.split(",") if s.strip()]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        if args.no_network:
            res = run_once(sel, fetcher=offline_fetcher(), matcher=load_matcher(network=False))
        else:
            res = run_once(sel)
    except ValueError as exc:
        ap.error(str(exc))
    for s in sel:
        err = res["errors"].get(s)
        print(f"{s}: +{res['added'].get(s, 0)} new (fetched {res['fetched'].get(s, 0)})" + (f"  FAILED: {err}" if err else ""))
    print(f"total new: {sum(res['added'].values())}")
    return 1 if res["errors"] and len(res["errors"]) == len(sel) else 0


if __name__ == "__main__":
    sys.exit(main())
