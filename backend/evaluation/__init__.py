"""Module C: evaluation, backtest and recommendation logic (pure numpy + scipy).

Submodules
----------
metrics      statistical building blocks (direction, probabilistic, return-forecast, strategy statistics)
strategy     causal position rule + execution simulator + buy-and-hold benchmark
report       build_backtest_report / validate_backtest_report (DESIGN.md section 6.4)
recommend    annotate_forecast / build_recommendations (section 6.3)
universe_bt  evaluate_universe (section 6.5)

Nothing in this package imports the quant engine: it only consumes ``types.PriceData`` and
``types.WalkForward``.
"""

from .recommend import annotate_forecast, build_recommendations  # noqa: E402,F401
from .report import build_backtest_report, validate_backtest_report  # noqa: E402,F401
from .universe_bt import evaluate_universe, validate_universe_report  # noqa: E402,F401

__all__ = ["build_backtest_report", "validate_backtest_report", "annotate_forecast", "build_recommendations",
           "evaluate_universe", "validate_universe_report"]
