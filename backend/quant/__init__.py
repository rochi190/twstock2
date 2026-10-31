"""Quant core (module A): calibrated predictive distributions of cumulative log returns.

Public entry points live in :mod:`backend.quant.engine` (``walk_forward`` and ``forecast_latest``).
The remaining modules are the mathematical building blocks:

* ``garch``       GJR-GARCH(1,1) with standardised Student-t innovations (MLE, filter, h-step term structure)
* ``hmm``         Gaussian hidden Markov model (EM with scaling, filtered probabilities, h-step mixture)
* ``dlm``         discount-factor dynamic linear model with Normal-Inverse-Gamma variance learning
* ``features``    causal predictors (every feature at row i uses bars <= i only)
* ``ensemble``    predictive-distribution primitives, exact linear opinion pool, dynamic model averaging
* ``calibration`` online Bayesian logistic (Platt) recalibration of P(R > 0)
* ``mc``          filtered-historical-simulation Monte Carlo (fan chart, VaR / CVaR, drawdown)
* ``engine``      walk-forward and live forecasting on top of the above
"""

from .engine import forecast_latest, walk_forward  # noqa: E402,F401  (public entry points)

__all__ = ["walk_forward", "forecast_latest"]
