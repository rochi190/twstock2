"""Shared constants for the whole app.  Single source of truth: import, never duplicate.

Conventions
-----------
* Returns are decimal fractions (0.012 == +1.2 %).  "log return" means ln(P1/P0) of the
  dividend-adjusted close.  API payloads expose *simple* returns (exp(R) - 1).
* A "bar" is one TWSE/TPEx trading day.  Horizons are counted in bars.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data_cache"
FRONTEND_DIR = ROOT / "frontend"

# ----------------------------------------------------------------------------- forecasting
HORIZONS = (1, 5, 20)  # next trading day / one week / one month
QLEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)  # WalkForward quantile grid
FAN_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)  # fan-chart bands in the live forecast
FAN_DAYS = 20
MIN_BARS_FORECAST = 250  # fewer bars than this -> refuse to forecast (error code insufficient_history)
MIN_TRAIN = 500  # walk-forward: bars of history before the first out-of-sample forecast
REFIT_EVERY = 21  # walk-forward: re-estimate model parameters every N bars (absolute schedule)
TEST_BARS_DEFAULT = 1250  # ~5 years of out-of-sample forecasts for the default backtest
DAILY_LIMIT = 0.10  # TWSE/TPEx +-10 % daily price limit (simple return)
TRADING_DAYS_PER_YEAR = 245  # Taiwan has ~245 sessions a year

# ----------------------------------------------------------------------------- Taiwan trading costs
COMMISSION_RATE = 0.001425  # statutory max, charged on BOTH sides
COMMISSION_DISCOUNT = 0.6  # typical online-broker discount (user adjustable, 1.0 = no discount)
TAX_STOCK = 0.003  # securities transaction tax, sell side, stocks
TAX_ETF = 0.001  # sell side, ETFs
SLIPPAGE_BPS = 5.0  # per side, in basis points


def roundtrip_cost(kind: str = "stock", commission_discount: float = COMMISSION_DISCOUNT,
                   slippage_bps: float = SLIPPAGE_BPS) -> float:
    """Fractional cost of one buy + one sell (commission both sides + sell tax + slippage both sides)."""
    tax = TAX_ETF if kind == "etf" else TAX_STOCK
    return 2.0 * COMMISSION_RATE * commission_discount + tax + 2.0 * slippage_bps / 1e4


# ----------------------------------------------------------------------------- recommendations
MARKET_SYMBOL = "^TWII"  # TAIEX, used as market factor
UNIVERSE_SIZE = 60  # stocks scored for the daily recommendation list
MIN_AVG_VALUE_TWD = 1.0e8  # liquidity filter: 20d average traded value >= NT$100M
BUY_P_UP = 0.55  # default thresholds on calibrated P(up)
AVOID_P_UP = 0.45
KELLY_FRACTION = 0.25  # fractional Kelly
MAX_WEIGHT = 0.15  # cap per name in suggested portfolio

# ----------------------------------------------------------------------------- server
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
