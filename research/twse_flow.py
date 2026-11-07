"""TWSE T86 (三大法人買賣日報) download, parse and cache -- research only, not used by the app.

One request per trading day: ``https://www.twse.com.tw/rwd/zh/fund/T86?date=YYYYMMDD&selectType=ALLBUT0999&response=json``.
Parsed days are cached as small JSON files under ``<cache>/t86/YYYYMMDD.json`` (``{"status": "ok"|"closed", "rows": {code:
[foreign, trust, dealer, total]}}``), so a re-run never downloads a day twice.  The TWSE site blocks clients that ask too fast,
so requests are spaced by ``MIN_INTERVAL`` seconds (the first full download of five years takes about an hour).

Field names changed over the years (2017-12: 外資 split into 外陸資 / 外資自營商), so columns are found by name, not position.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from backend import data as D

T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86?date={ymd}&selectType=ALLBUT0999&response=json"
MIN_INTERVAL = 3.5          # seconds between two requests (TWSE throttles faster clients)
FLOW_KEYS = ("foreign", "trust", "dealer", "total")


class FlowUnavailable(RuntimeError):
    """The T86 page could not be fetched or parsed (network, throttling, unexpected format)."""


def _num(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).replace(",", "").strip()
    if t in ("", "--", "-"):
        return 0.0
    return float(t)


def _find(fields: list, *, must: tuple, avoid: tuple = ()) -> int | None:
    """Index of the first field containing every string of ``must`` and none of ``avoid``."""
    for i, f in enumerate(fields):
        f = str(f)
        if all(m in f for m in must) and not any(a in f for a in avoid):
            return i
    return None


def parse_t86(payload) -> dict:
    """``{"status": "ok", "rows": {code: [foreign, trust, dealer, total]}}`` (net shares) or ``{"status": "closed"}``.

    Raises :class:`FlowUnavailable` for anything else (an HTML block page, a changed format, ...).
    """
    if not isinstance(payload, dict):
        raise FlowUnavailable("T86: not a JSON object")
    stat = str(payload.get("stat", ""))
    if stat != "OK":
        if "沒有符合條件" in stat or "查詢日期" in stat:
            return {"status": "closed", "rows": {}}
        raise FlowUnavailable(f"T86: unexpected stat {stat!r}")
    fields = payload.get("fields") or []
    rows = payload.get("data") or []
    i_code = _find(fields, must=("證券代號",))
    i_for = _find(fields, must=("外陸資", "買賣超")) if _find(fields, must=("外陸資", "買賣超")) is not None \
        else _find(fields, must=("外資", "買賣超"), avoid=("自營商",))
    i_tru = _find(fields, must=("投信", "買賣超"))
    i_tot = _find(fields, must=("三大法人", "買賣超"))
    i_dea = next((i for i, f in enumerate(fields) if str(f).strip() == "自營商買賣超股數"), None)
    if i_dea is None:
        i_dea = _find(fields, must=("自營商", "買賣超"), avoid=("外資", "外陸資", "自行", "避險"))
    if None in (i_code, i_for, i_tru, i_tot):
        raise FlowUnavailable(f"T86: missing columns in {fields!r}")
    out = {}
    for r in rows:
        try:
            code = str(r[i_code]).strip()
            vals = [_num(r[i_for]), _num(r[i_tru]), _num(r[i_dea]) if i_dea is not None else 0.0, _num(r[i_tot])]
        except (IndexError, ValueError, TypeError):
            continue
        if code:
            out[code] = vals
    if not out:
        raise FlowUnavailable("T86: stat OK but no rows")
    return {"status": "ok", "rows": out}


def _cache_path(d: date) -> Path:
    return D.cache_dir() / "t86" / f"{d:%Y%m%d}.json"


_last_request = [0.0]


def fetch_day(d: date, *, network: bool = True) -> dict | None:
    """Parsed T86 of day ``d`` (cached).  ``None`` when not cached and ``network`` is False."""
    p = _cache_path(d)
    cached = D.read_json_file(p)
    if isinstance(cached, dict) and cached.get("status") in ("ok", "closed"):
        return cached
    if not network:
        return None
    wait = MIN_INTERVAL - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()
    try:
        body = D._http_get(T86_URL.format(ymd=f"{d:%Y%m%d}"), retries=3)
        payload = json.loads(body.decode("utf-8-sig"))
    except (D.DataError, ValueError) as exc:
        raise FlowUnavailable(f"T86 {d}: {exc}") from exc
    rec = parse_t86(payload)
    if rec["status"] == "closed" and d >= date.today():
        return rec                     # today's page may simply not be published yet: do not cache "closed"
    p.parent.mkdir(parents=True, exist_ok=True)
    D.atomic_write_json(p, rec)
    return rec


def download_range(start: date, end: date, *, progress=print) -> dict:
    """Fetch every weekday in [start, end] (cached days are skipped).  Returns ``{date: parsed}``."""
    out = {}
    d = start
    n_net = 0
    while d <= end:
        if d.weekday() < 5:
            had = _cache_path(d).exists()
            out[d] = fetch_day(d)
            if not had:
                n_net += 1
                if progress and n_net % 20 == 0:
                    progress(f"T86 {d} ({n_net} downloaded)")
        d += timedelta(days=1)
    return out


def flow_matrices(days: dict, codes: list, cal: np.ndarray) -> dict:
    """(T, S) net-share matrices ``{key: array}`` on calendar ``cal`` (datetime64[D]).

    A trading day whose T86 page is missing or 'closed' gives NaN for every stock (unknown); a stock absent from an 'ok'
    page had no institutional trade that day and gets 0.
    """
    T, S = len(cal), len(codes)
    out = {k: np.full((T, S), np.nan) for k in FLOW_KEYS}
    col = {c: j for j, c in enumerate(codes)}
    for t, d64 in enumerate(cal):
        rec = days.get(d64.astype(object))
        if not isinstance(rec, dict) or rec.get("status") != "ok":
            continue
        for k in FLOW_KEYS:
            out[k][t, :] = 0.0
        for code, vals in rec["rows"].items():
            j = col.get(code)
            if j is None:
                continue
            for k, v in zip(FLOW_KEYS, vals):
                out[k][t, j] = v
    return out
