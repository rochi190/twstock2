"""Monthly-revenue features R1-R4 (PLAN.md section 5 "月營收", section 3 revenue row, section 7 P2).

Data: FinMind ``TaiwanStockMonthRevenue`` (``backend.altdata.finmind``).  Field semantics were checked against the real
fixture ``tests/fixtures/alt/finmind_rev_2330.json`` and the downloaded files: ``revenue_year`` / ``revenue_month`` are the
revenue month itself; ``date`` is the 1st of the FOLLOWING month (e.g. ``date=2026-11-01`` is the October-2026 revenue).
``create_time`` is never used (it is '' on old rows; PLAN section 3).

Availability rule (PLAN section 3, deliberately conservative)
-------------------------------------------------------------
Revenue of month ``M`` is publicly due by the 10th of month ``M+1``.  It is usable for the decision of trading day ``t`` iff
``t >= revenue_available_day(M)``, where that day is the **2nd trading day strictly after the 10th of M+1** (the 10th itself
is never counted, even if it is a trading day; if the 10th is a weekend / holiday the count simply starts with the next
trading days).  The rule depends on the calendar only, so every stock gets the same availability day for the same month.
If the calendar does not extend far enough (fewer than two trading days after the 10th) the function returns ``None`` -- the
future is never extrapolated.  At the START of the calendar the rule is evaluated on the calendar's own days: if the calendar
begins after the 10th, the result can only be LATER than the true day (conservative).

Features (always "the latest month that is available on day t"; before the next month's availability day the previous
month's value is carried forward; stocks without a file, or without enough history, are NaN)
  R1 ``rev_yoy``        rev[M] / rev[M-12] - 1, clipped to [-0.9, 3]; NaN if rev[M-12] is missing or <= 0
  R2 ``rev_yoy_accel``  R1[M] - mean(R1[M-1], R1[M-2], R1[M-3])  (all four are the clipped YoY; NaN if any is missing)
  R3 ``rev_3m_yoy``     (rev[M]+rev[M-1]+rev[M-2]) / (rev[M-12]+rev[M-13]+rev[M-14]) - 1  (NaN if any of the six is
                        missing or the denominator <= 0; not clipped -- the plan clips R1 only)
  R4 ``rev_high12``     1.0 if rev[M] is >= every revenue of M-11..M else 0.0  (NaN if one of the 12 months is missing)

Every feature value belongs to a revenue month and uses revenue rows of months ``<= M`` only; it is switched on at the
availability day of ``M`` and no price is used, so poisoning anything dated after ``t`` cannot change row ``t``.
"""
from __future__ import annotations

import datetime as _dt
import logging

import numpy as np

from backend.altdata import finmind as FM

log = logging.getLogger(__name__)

NAMES = ("rev_yoy", "rev_yoy_accel", "rev_3m_yoy", "rev_high12")
IDS = {"rev_yoy": "R1", "rev_yoy_accel": "R2", "rev_3m_yoy": "R3", "rev_high12": "R4"}
EXPECTED_SIGN = {n: +1 for n in NAMES}         # PLAN section 5: every revenue feature is expected to be positive
YOY_CLIP = (-0.9, 3.0)
PUBLISH_DAY = 10                                # revenue is due by the 10th of the following month
LAG_TRADING_DAYS = 2                            # 2nd trading day strictly after the 10th


# ============================================================================= availability
def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def revenue_available_day(year: int, month: int, cal):
    """First trading day (``numpy.datetime64[D]``) on which the revenue of ``year``-``month`` may be used, or ``None``.

    ``cal``: ascending trading calendar (``datetime64[D]``, e.g. ``panel.cal``).  See the module docstring for the rule.
    Returns ``None`` when ``cal`` has fewer than ``LAG_TRADING_DAYS`` trading days after the 10th of the next month.
    """
    if not 1 <= int(month) <= 12:
        raise ValueError(f"month must be 1..12, got {month!r}")
    cal = np.asarray(cal).astype("datetime64[D]")
    ny, nm = _next_month(int(year), int(month))
    due = np.datetime64(_dt.date(ny, nm, PUBLISH_DAY), "D")
    first_after = int(np.searchsorted(cal, due, side="right"))       # index of the first trading day strictly after the 10th
    idx = first_after + LAG_TRADING_DAYS - 1
    return cal[idx] if idx < len(cal) else None


def availability_row(year: int, month: int, cal):
    """Row index in ``cal`` of :func:`revenue_available_day` (or ``None``)."""
    cal = np.asarray(cal).astype("datetime64[D]")
    d = revenue_available_day(year, month, cal)
    return None if d is None else int(np.searchsorted(cal, d))


# ============================================================================= per-month features
def _clip_yoy(cur: float, prev) -> float:
    if prev is None or not np.isfinite(prev) or prev <= 0 or not np.isfinite(cur):
        return np.nan
    return float(np.clip(cur / prev - 1.0, YOY_CLIP[0], YOY_CLIP[1]))


def monthly_feature_table(rows) -> dict:
    """``{(year, month): (R1, R2, R3, R4)}`` from parsed revenue rows (``finmind.parse_monthly_revenue`` output).

    Each entry uses only the rows of months ``<= (year, month)``; missing months inside a window make the dependent features NaN.
    """
    rev = {(int(r["revenue_year"]), int(r["revenue_month"])): float(r["revenue"]) for r in rows}

    def back(y: int, m: int, k: int) -> tuple[int, int]:
        n = y * 12 + (m - 1) - k
        return n // 12, n % 12 + 1

    def yoy(y: int, m: int) -> float:
        cur, prev = rev.get((y, m)), rev.get((y - 1, m))
        return np.nan if cur is None else _clip_yoy(cur, prev)

    out = {}
    for (y, m), cur in rev.items():
        r1 = yoy(y, m)
        prior = [yoy(*back(y, m, k)) for k in (1, 2, 3)]
        r2 = r1 - float(np.mean(prior)) if np.isfinite(r1) and all(np.isfinite(prior)) else np.nan
        cm = [rev.get(back(y, m, k)) for k in (0, 1, 2)]
        pm = [rev.get(back(y, m, k)) for k in (12, 13, 14)]
        if all(v is not None for v in cm + pm) and sum(pm) > 0:
            r3 = sum(cm) / sum(pm) - 1.0
        else:
            r3 = np.nan
        win = [rev.get(back(y, m, k)) for k in range(12)]
        r4 = (1.0 if cur >= max(win) else 0.0) if all(v is not None for v in win) else np.nan
        out[(y, m)] = (r1, r2, r3, r4)
    return out


def stock_features(rows, cal) -> np.ndarray:
    """``(T, 4)`` R1..R4 of one stock on ``cal``: the latest month available on each day, NaN before the first one.

    Months whose availability day is beyond the calendar end (``revenue_available_day`` is None) are never switched on.
    """
    cal = np.asarray(cal).astype("datetime64[D]")
    T = len(cal)
    out = np.full((T, 4), np.nan)
    if not rows or T == 0:
        return out
    table = monthly_feature_table(rows)
    months = sorted(table)
    start_rows, vals = [], []
    for (y, m) in months:
        a = availability_row(y, m, cal)
        if a is not None:
            start_rows.append(a)
            vals.append(table[(y, m)])
    if not start_rows:
        return out
    start_rows = np.asarray(start_rows)
    vals = np.asarray(vals, dtype=float)
    # availability rows are non-decreasing in the month (the rule is monotone), so the latest available month on day t is
    # the last one whose start row is <= t
    k = np.searchsorted(start_rows, np.arange(T), side="right") - 1
    ok = k >= 0
    out[ok] = vals[k[ok]]
    return out


# ============================================================================= panel level
def load_rows(code: str):
    """Parsed revenue rows of ``code`` from the FinMind cache, or ``None`` when its file is missing (-> NaN features)."""
    obj = FM.load_dataset("rev", code)
    if obj is None:
        return None
    return FM.parse_monthly_revenue(obj["data"])


def revenue_features(codes, cal, loader=load_rows):
    """``({name: (T, S)}, coverage)`` for ``codes`` on ``cal``.

    ``loader(code)`` -> parsed rows or ``None``.  ``coverage``: ``{'n_codes', 'n_with_file', 'codes_with_file'}``.
    """
    cal = np.asarray(cal).astype("datetime64[D]")
    feats = {n: np.full((len(cal), len(codes)), np.nan) for n in NAMES}
    have = []
    for j, c in enumerate(codes):
        rows = loader(str(c))
        if rows is None:
            continue
        have.append(str(c))
        f = stock_features(rows, cal)
        for i, n in enumerate(NAMES):
            feats[n][:, j] = f[:, i]
    return feats, {"n_codes": len(codes), "n_with_file": len(have), "codes_with_file": have}
