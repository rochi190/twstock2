"""US overnight market data for the research pipeline (PLAN.md section 7 P3a, section 3 and section 9).

Fetches daily bars of ^SOX, ^IXIC, ^GSPC, NVDA, TSM, ^VIX and TWD=X from Yahoo through the existing
``backend.data._fetch_bars`` (only *called*, never modified), caches them as JSON under
``<cache_dir>/alt/us/{name}.json`` and aligns them to the Taiwan trading calendar without look-ahead.

Date semantics (verified against the live ^SOX answer on 2026-11-12: the last, still-forming bar carried the
New York date of the moment of the request): ``parse_yahoo_chart`` converts timestamps with the *exchange's*
``gmtoffset``, so for US tickers/indices the date ``D`` is the **US Eastern calendar date** of the session.
``TWD=X`` (a currency) is stamped on the **London/UTC date**; its daily bar covers the UTC day ``D``.

Point-in-time rules implemented here (PLAN section 3):

* A bar of date ``D`` is *incomplete* (and dropped) while ``now_utc < D 21:30 UTC`` (``TWDUSD`` is stricter:
  ``D+1 00:00 UTC``, because a London-dated currency bar only ends at midnight UTC).
* A bar of date ``D`` may be used for the Taipei trading day ``S`` open iff ``D < S`` (strictly).

CLI: ``python -m backend.altdata.usmarket`` (about 7 requests, >= 1 s apart).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import data as _data

log = logging.getLogger("twq.altdata.usmarket")

# name -> (yahoo symbol, return basis, seconds after 00:00 UTC of date D at which the bar of D is final)
#   basis "adjclose": dividend-adjusted close (single stocks); "close": raw close (indices, currency)
_CLOSE_2130 = 21 * 3600 + 30 * 60
SERIES: dict[str, dict] = {
    "SOX": {"ysym": "^SOX", "basis": "close", "final_after_s": _CLOSE_2130},
    "IXIC": {"ysym": "^IXIC", "basis": "close", "final_after_s": _CLOSE_2130},
    "GSPC": {"ysym": "^GSPC", "basis": "close", "final_after_s": _CLOSE_2130},
    "NVDA": {"ysym": "NVDA", "basis": "adjclose", "final_after_s": _CLOSE_2130},
    "TSM": {"ysym": "TSM", "basis": "adjclose", "final_after_s": _CLOSE_2130},
    # VIX: genuine one-day spikes that reverse are real data -> never despiked
    "VIX": {"ysym": "^VIX", "basis": "close", "final_after_s": _CLOSE_2130, "despike": False},
    # TWD per 1 USD (a rise = TWD weakens). London-dated bar: final only at the end of the UTC day.
    "TWDUSD": {"ysym": "TWD=X", "basis": "close", "final_after_s": 24 * 3600},
}
MIN_REQUEST_GAP_S = 1.0
SCHEMA_VERSION = 1


# ------------------------------------------------------------------------------------------------ paths / time
def us_cache_dir(cache_dir: str | os.PathLike | None = None) -> Path:
    """``<cache root>/alt/us`` (cache root: ``TWQ_CACHE_DIR`` / ``config.CACHE_DIR`` unless ``cache_dir`` is given)."""
    root = Path(cache_dir) if cache_dir is not None else _data.cache_dir()
    return root / "alt" / "us"


def series_path(name: str, cache_dir: str | os.PathLike | None = None) -> Path:
    if name not in SERIES:
        raise KeyError(f"unknown US series {name!r}; known: {sorted(SERIES)}")
    return us_cache_dir(cache_dir) / f"{name}.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc64(now) -> np.datetime64:
    """datetime (aware or naive-as-UTC) / ISO string / datetime64 -> ``datetime64[s]`` in UTC."""
    if isinstance(now, np.datetime64):
        return now.astype("datetime64[s]")
    if isinstance(now, str):
        s = now.strip().replace("Z", "+00:00")
        now = datetime.fromisoformat(s)
    if isinstance(now, datetime):
        if now.tzinfo is not None:
            now = now.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(now.replace(microsecond=0), "s")
    raise TypeError(f"cannot interpret {now!r} as a UTC time")


# ------------------------------------------------------------------------------------------------ incomplete bars
def complete_mask(dates, now_utc, final_after_s: int = _CLOSE_2130) -> np.ndarray:
    """Boolean mask of bars that are final at ``now_utc``: ``now_utc >= D 00:00 UTC + final_after_s``.

    The boundary instant itself counts as complete (a bar is incomplete only while ``now_utc < D 21:30 UTC``).
    """
    d = np.asarray(dates).astype("datetime64[D]")
    boundary = d.astype("datetime64[s]") + np.timedelta64(int(final_after_s), "s")
    return boundary <= _to_utc64(now_utc)


def drop_incomplete(dates, values, now_utc, final_after_s: int = _CLOSE_2130):
    """Drop the bars that are not final at ``now_utc``.  ``values`` may be one array or a list of arrays."""
    keep = complete_mask(dates, now_utc, final_after_s)
    d = np.asarray(dates).astype("datetime64[D]")[keep]
    if isinstance(values, (list, tuple)):
        return d, [np.asarray(v)[keep] for v in values]
    return d, np.asarray(values)[keep]


# ------------------------------------------------------------------------------------------------ cache I/O
def _bars_to_payload(name: str, bars: dict, now: datetime) -> dict:
    spec = SERIES[name]
    dates = np.datetime64(0, "D") + np.asarray(bars["d"], dtype=np.int64).astype("timedelta64[D]")
    keep = complete_mask(dates, now, spec["final_after_s"])
    dropped = int((~keep).sum())
    cols = {"open": "o", "high": "h", "low": "l", "close": "c", "adjclose": "adj", "volume": "v"}
    out = {
        "v": SCHEMA_VERSION, "name": name, "yahoo_symbol": spec["ysym"], "basis": spec["basis"],
        "final_after_s": spec["final_after_s"],
        "date_semantics": ("London/UTC date of the daily bar" if name == "TWDUSD"
                           else "US Eastern calendar date of the session"),
        "fetched_at": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dropped_incomplete": dropped,
        "dates": dates[keep].astype(str).tolist(),
    }
    for key, src in cols.items():
        out[key] = np.asarray(bars[src], dtype=float)[keep].tolist()
    return out


def save_series(name: str, payload: dict, cache_dir: str | os.PathLike | None = None) -> Path:
    path = series_path(name, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_series(name: str, cache_dir: str | os.PathLike | None = None) -> dict:
    path = series_path(name, cache_dir)
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"no cached US series {name!r} at {path}; run python -m backend.altdata.usmarket") from exc
    if not isinstance(rec, dict) or "dates" not in rec or "fetched_at" not in rec:
        raise ValueError(f"malformed cache file {path}")
    return rec


# ------------------------------------------------------------------------------------------------ fetching
def fetch_all(names=None, *, now=None, fetcher=None, sleep=time.sleep,
              cache_dir: str | os.PathLike | None = None, min_gap_s: float = MIN_REQUEST_GAP_S) -> dict:
    """Download every series (>= ``min_gap_s`` seconds between requests) and write the cache.

    ``now`` (UTC datetime, injectable for tests) decides which bars are final; ``fetcher(ysym, period1)`` defaults
    to ``backend.data._fetch_bars``.  One failing series does not stop the others.  Returns
    ``{name: {"ok": bool, "n": int, "first": str, "last": str, "dropped_incomplete": int, "error": str}}``.
    """
    names = list(names) if names is not None else list(SERIES)
    fetcher = fetcher or _data._fetch_bars
    now_dt = _utc_now() if now is None else _to_utc_datetime(now)
    report: dict = {}
    for i, name in enumerate(names):
        if name not in SERIES:
            raise KeyError(f"unknown US series {name!r}; known: {sorted(SERIES)}")
        if i > 0 and min_gap_s > 0:
            sleep(min_gap_s)
        try:
            bars = fetcher(SERIES[name]["ysym"], 0)
            payload = _bars_to_payload(name, bars, now_dt)
            if not payload["dates"]:
                raise ValueError("no complete bars")
            save_series(name, payload, cache_dir)
            report[name] = {"ok": True, "n": len(payload["dates"]), "first": payload["dates"][0],
                            "last": payload["dates"][-1], "dropped_incomplete": payload["dropped_incomplete"]}
        except Exception as exc:  # noqa: BLE001 - report per series, keep going
            log.warning("US series %s failed: %s", name, exc)
            report[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return report


def _to_utc_datetime(now) -> datetime:
    if isinstance(now, datetime):
        return now.astimezone(timezone.utc) if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    t = _to_utc64(now).astype(datetime)
    return t.replace(tzinfo=timezone.utc)


# ------------------------------------------------------------------------------------------------ returns
SPIKE_JUMP = 0.25      # |ln jump| of a bar versus its predecessor ...
SPIKE_REVERT = 0.10    # ... that is undone by the next bar (|ln(P[k+1]/P[k-1])| < this) = a bad print, not a move


def spike_mask(px) -> np.ndarray:
    """One-bar price glitches: the bar jumped by more than ``SPIKE_JUMP`` (log) from the previous bar and the *next*
    bar is back within ``SPIKE_REVERT`` of the previous one.  Yahoo's ``TWD=X`` has such prints (e.g. a close of
    1.80 between 29.25 and 30.11 on 2011-10-24 and again on 2014-12-31).  The last bar is never flagged.
    """
    px = np.asarray(px, dtype=float)
    out = np.zeros(len(px), dtype=bool)
    if len(px) < 3:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        prev, cur, nxt = px[:-2], px[1:-1], px[2:]
        jump = np.abs(np.log(cur / prev))
        back = np.abs(np.log(nxt / prev))
        out[1:-1] = (jump > SPIKE_JUMP) & (back < SPIKE_REVERT)
    return out


def daily_log_returns(name: str, *, now=None, cache_dir: str | os.PathLike | None = None,
                      despike: bool | None = None):
    """``(dates, ret)`` of the cached series: ``ret[k] = ln(P[k] / P[k-1])`` stamped with the date ``D`` of bar k.

    With ``despike`` (default: on, except for VIX whose reversing spikes are genuine) one-bar bad prints
    (:func:`spike_mask`) are removed first and the return over the gap is bridged (``ln(P[k+1]/P[k-1])`` stamped ``D[k+1]``).  Looking one bar ahead here only repairs a bad print;
    it never changes a genuine move.

    ``P`` is the adjusted close for single stocks (NVDA, TSM) and the plain close for indices and the currency.
    Bars that were not final at ``now`` are dropped again at read time (default ``now`` = the file's
    ``fetched_at``, so a stale cache can never contain a bar that was unfinished when it was written).  A return
    whose either price is missing / non-positive is NaN.  The first bar has no return and is omitted.
    """
    rec = load_series(name, cache_dir)
    basis = rec.get("basis") or SERIES[name]["basis"]
    final_after = int(rec.get("final_after_s", SERIES[name]["final_after_s"]))
    dates = np.asarray(rec["dates"], dtype="datetime64[D]")
    px = np.array([np.nan if x is None else x for x in rec[basis]], dtype=float)
    if len(px) != len(dates):
        raise ValueError(f"{name}: dates / {basis} length mismatch")
    dates, px = drop_incomplete(dates, px, rec["fetched_at"] if now is None else now, final_after)
    if len(dates) > 1 and not np.all(dates[1:] > dates[:-1]):
        raise ValueError(f"{name}: dates are not strictly ascending")
    if SERIES[name].get("despike", True) if despike is None else despike:
        bad = spike_mask(px)
        dates, px = dates[~bad], px[~bad]
    if len(px) < 2:
        return dates[:0], np.zeros(0)
    ok = np.isfinite(px) & (px > 0)
    safe = np.where(ok, px, 1.0)
    ret = np.log(safe[1:] / safe[:-1])
    ret[~(ok[1:] & ok[:-1])] = np.nan
    return dates[1:], ret


# ------------------------------------------------------------------------------------------------ alignment
def align_us_to_tw(us_dates, us_ret, tw_cal, *, empty=np.nan) -> np.ndarray:
    """For each Taipei trading day ``S`` return the US log return that is new at the ``S`` open.

    Only bars with US date ``D < S`` (strictly) are usable.  The value for ``S_j`` is the *sum* of the log returns
    of the bars with ``S_{j-1} <= D < S_j`` -- i.e. every bar that was not yet usable at the previous trading
    day's open -- so a Taiwan holiday stretch (several US sessions between two trading days) yields the
    cumulative return.  NaN (or ``empty``) when that window holds no bar (US holiday, series not started) or any
    NaN return; NaN for the first calendar day (no previous trading day to bound the window).

    Both date arrays are ``datetime64[D]``.  ``us_dates`` is sorted if needed (duplicates raise ValueError),
    ``tw_cal`` must be strictly ascending.  Fully vectorised (searchsorted + cumulative sums).
    """
    ud = np.asarray(us_dates).astype("datetime64[D]")
    ur = np.asarray(us_ret, dtype=float)
    tw = np.asarray(tw_cal).astype("datetime64[D]")
    if ud.shape != ur.shape or ud.ndim != 1 or tw.ndim != 1:
        raise ValueError("us_dates / us_ret must be 1-D arrays of equal length; tw_cal 1-D")
    if len(ud) > 1 and not np.all(ud[1:] > ud[:-1]):
        order = np.argsort(ud, kind="stable")
        ud, ur = ud[order], ur[order]
        if not np.all(ud[1:] > ud[:-1]):
            raise ValueError("us_dates contains duplicates")
    if len(tw) > 1 and not np.all(tw[1:] > tw[:-1]):
        raise ValueError("tw_cal must be strictly ascending")
    out = np.full(len(tw), np.nan, dtype=float)
    if len(tw) == 0:
        return out
    hi = np.searchsorted(ud, tw, side="left")  # number of bars with D < S
    bad = ~np.isfinite(ur)
    cs = np.r_[0.0, np.cumsum(np.where(bad, 0.0, ur))]
    cb = np.r_[0, np.cumsum(bad.astype(np.int64))]
    lo = np.r_[0, hi[:-1]]
    val = cs[hi] - cs[lo]
    n_bars = hi - lo
    ok = (n_bars > 0) & ((cb[hi] - cb[lo]) == 0)
    ok[0] = False
    out[ok] = val[ok]
    if not (isinstance(empty, float) and np.isnan(empty)):
        fill = (n_bars == 0)
        fill[0] = False
        out[fill] = empty
    return out


def aligned_overnight(name: str, tw_cal, *, now=None, cache_dir: str | os.PathLike | None = None) -> np.ndarray:
    """Convenience: ``align_us_to_tw(*daily_log_returns(name), tw_cal)``."""
    d, r = daily_log_returns(name, now=now, cache_dir=cache_dir)
    return align_us_to_tw(d, r, tw_cal)


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch US overnight series (SOX, IXIC, GSPC, NVDA, TSM, VIX, TWD=X).")
    ap.add_argument("--names", nargs="*", default=None, help=f"subset of {sorted(SERIES)}")
    ap.add_argument("--cache-dir", default=None, help="cache root (default: TWQ_CACHE_DIR or data_cache)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rep = fetch_all(args.names, cache_dir=args.cache_dir)
    bad = 0
    for name, r in rep.items():
        if r["ok"]:
            print(f"{name:7s} {r['first']} .. {r['last']}  n={r['n']}  dropped_incomplete={r['dropped_incomplete']}")
        else:
            bad += 1
            print(f"{name:7s} FAILED {r['error']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
