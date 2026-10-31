"""Shared data containers.  Plain dataclasses + numpy so every module (and subprocess pickling) agrees."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import QLEVELS


@dataclass
class PriceData:
    """Daily bars of one instrument.  Rows are ascending, unique dates, only valid OHLC rows."""

    symbol: str  # '2330', '0050', '^TWII'
    name: str  # '台積電'
    market: str  # 'TWSE' | 'TPEx' | 'INDEX'
    kind: str  # 'stock' | 'etf' | 'index'
    dates: np.ndarray  # datetime64[D], shape (n,)
    open: np.ndarray  # float64 raw (as traded)
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    adjclose: np.ndarray  # dividend/split adjusted close (total-return proxy)
    volume: np.ndarray  # shares
    dividends: list = field(default_factory=list)  # [(np.datetime64[D], cash_per_share)]
    last_bar_complete: bool = True  # False if the final bar is today's still-running session
    fetched_at: str = ""  # ISO timestamp of the network fetch
    stale: bool = False  # True if the network failed and an old cache was served

    @property
    def n(self) -> int:
        return len(self.dates)

    def __len__(self) -> int:
        return self.n

    @property
    def adj_factor(self) -> np.ndarray:
        return self.adjclose / self.close

    @property
    def adj_open(self) -> np.ndarray:
        return self.open * self.adj_factor

    @property
    def adj_high(self) -> np.ndarray:
        return self.high * self.adj_factor

    @property
    def adj_low(self) -> np.ndarray:
        return self.low * self.adj_factor

    def logret(self) -> np.ndarray:
        """r[i] = ln(adjclose[i] / adjclose[i-1]); r[0] = nan."""
        r = np.full(self.n, np.nan)
        r[1:] = np.diff(np.log(self.adjclose))
        return r

    def slice(self, stop: int) -> "PriceData":
        """Rows [:stop] — used by no-look-ahead tests."""
        return PriceData(self.symbol, self.name, self.market, self.kind, self.dates[:stop].copy(),
                         self.open[:stop].copy(), self.high[:stop].copy(), self.low[:stop].copy(),
                         self.close[:stop].copy(), self.adjclose[:stop].copy(), self.volume[:stop].copy(),
                         [d for d in self.dividends if d[0] <= self.dates[stop - 1]],
                         self.last_bar_complete, self.fetched_at, self.stale)


@dataclass
class HorizonForecasts:
    """Out-of-sample forecasts for one horizon.  Row i = forecast ORIGIN = close of dates[i].
    Everything at row i may only use bars 0..i.  NaN where no forecast exists."""

    h: int
    p_up: np.ndarray  # (n,) calibrated P(R > 0)
    p_up_raw: np.ndarray  # (n,) before calibration
    mu: np.ndarray  # (n,) E[R], R = ln(adjC[i+h] / adjC[i])
    sigma: np.ndarray  # (n,) std of R
    q: np.ndarray  # (n, len(QLEVELS)) predictive quantiles of R
    pit: np.ndarray  # (n,) predictive CDF evaluated at realized R (NaN until realized) — evaluation only
    realized: np.ndarray  # (n,) realized R (NaN for the last h rows) — evaluation only
    comp_p_up: dict = field(default_factory=dict)  # component name -> (n,)
    comp_mu: dict = field(default_factory=dict)  # component name -> (n,)
    weights: dict = field(default_factory=dict)  # component name -> (n,) ensemble weight used at row i


@dataclass
class WalkForward:
    symbol: str
    dates: np.ndarray  # (n,) datetime64[D]
    horizons: dict  # int h -> HorizonForecasts
    first_idx: int  # first row that has a forecast
    refit_every: int
    min_train: int
    model_names: list
    regime_prob: np.ndarray  # (n, K) filtered (NOT smoothed) regime probabilities
    regime_labels: list  # K strings, e.g. ['低波動盤整', '多頭', '高波動空頭']
    vol1d: np.ndarray  # (n,) conditional std of next-bar log return
    config: dict = field(default_factory=dict)
    qlevels: tuple = QLEVELS
