"""Walk-forward and live forecasting engine (one code path).

Goal: a calibrated predictive *distribution* of the cumulative log return ``R = ln(adjC[i+h] / adjC[i])`` for every
horizon ``h`` from four mathematically distinct components combined adaptively:

``garch``  GJR-GARCH(1,1)-t term structure: exact h-step variance, kurtosis-matched two-piece Student-t whose skewness is
           the model-implied leverage skewness damped by 0.6 (this makes P(R>0) unbiased vs Monte Carlo of the model;
           matching the skewness exactly over-shifts the median by about one percentage point at h = 20)
``hmm``    3-regime Gaussian HMM on (return, log range-vol): filtered-probability mixture, regime drift and volatility
``dlm``    discount-factor Bayesian regression of the vol-standardised return on causal features, GARCH scale
``naive``  random walk with shrunk long-run drift and slowly varying EWMA volatility (anchors the ensemble)

Combination: dynamic model averaging (exponentially discounted, 1/h-tempered log predictive likelihood of *matured*
outcomes) -> exact linear opinion pool -> online Bayesian logistic recalibration of ``P(R > 0)``.  The one-day
predictive is truncated to the +-10 % price limit.

No look-ahead, by construction
------------------------------
* Refits happen at the rows ``r = k * refit_every`` (an **absolute grid**: it does not depend on ``len(data)`` or on
  ``start_idx``).  Each refit is a deterministic cold fit on a window that ends at row ``r``; the parameters in force
  at origin ``i`` are those of the latest grid row ``r <= i``.
* Everything sequential is either refit-free (features, EWMA, DLM: single causal pass from row 0) or uses a *bounded*
  window of matured outcomes (DMA window ``dma_window``, calibration window ``cal_window``).  The forecast at origin
  ``i`` is therefore a pure function of ``data[:i+1]`` (and of the market bars dated <= date_i).
* ``walk_forward`` pre-rolls ``L_state = dma_window + cal_window`` rows before the first reported origin so that the
  combination state at every reported row is fully populated.  Consequently
  ``walk_forward(data).row(i) == walk_forward(data.slice(i+1)).last_row == forecast_latest(data.slice(i+1))`` for every
  ``i`` (unit-tested), and a live forecast only has to replay ``L_state`` rows (about 35 refits).

Provenance of the hyper-parameters (no data snooping)
-----------------------------------------------------
Every value in ``DEFAULT_CFG`` and in ``features`` / ``dlm`` / ``calibration`` is fixed a priori from theory or standard
defaults (priors, windows, discount factors).  Exactly four numbers were set with the help of numbers, and none used the
evaluation windows (the final 1250 bars of 2330 / 0050, i.e. dates >= 2021-08-27); the last two were added by the post-
integration audit and use development rows only (rows before the final 1250 bars of every stock):

* ``dma_eta = 0.25`` (learning rate of the model averaging, a generalised-Bayes "safe Bayes" tempering) was chosen from
  {1, 0.5, 0.25, 0} by the pooled log score on a dev period of 2330 (2012-01 .. 2021-08-26) and 0050 (2015-01 ..
  2021-08-26).  Equal weights scored best, so the learning rate is small; it still shifts weight to a component with a
  persistent advantage (tested with planted signals).
* ``garch_skew_damping = 0.6`` was calibrated by Monte Carlo of the fitted GJR dynamics *themselves* (no market data).
* ``weekend_var_uplift = 0.16`` (audit M-04): on 178,271 development one-day forecasts of 36 stocks the squared standardised
  error is 1.19 times larger when the next bar follows a weekend (stock-bootstrap interval 1.15-1.23, date-clustered bootstrap
  1.13-1.26 -- all stocks of a date resampled together, so market-wide shock days cannot fake the effect --; 1.16 in the
  audit's own fit on 16 stocks; theory: French & Roll 1986), 0.16 is the round value inside all intervals.
* ``hmm.TREND_THRESHOLD = 0.10`` (audit M-06): the 95th percentile of the calm regime's ``mu / sigma`` over 3,820 development
  refits (see ``hmm.regime_labels``).

Experiments tried on the dev period and rejected: an exponentially weighted variance-targeting level (no gain),
DMA forgetting 0.995 (worse), extra shrinkage of the weights (no gain over the learning rate), shrinking the HMM regime
means to the window mean instead of the shrunk drift (scored slightly better on 2330's dev window only because that
survivor trended strongly; rejected because it would extrapolate trailing trends).

Conventions: ``HorizonForecasts.mu`` / ``sigma`` are moments of the *untruncated* pool (the +-10 % limit only changes
quantiles, ``p_up`` and PIT of the one-day predictive); ``vol1d`` is the GARCH conditional std of the next log return (the
volatility STATE: before the weekend uplift, which only widens the one-day predictive of Friday origins);
``weights`` are the ensemble weights actually used at that row; ``comp_p_up`` / ``comp_mu`` are the component's own
P(R>0) and E[R] (log).  The HMM is refit cold at absolute anchor rows (every ``hmm_anchor_every`` refits) and warm-started
from the latest anchor in between, which keeps every fit a deterministic function of the data up to its row.

Real-data hygiene: models see a cleaned return path (``features.clean_returns``); the ``realized`` and ``pit`` arrays of
the walk-forward use the raw adjusted-close returns so that they match an independent evaluation exactly.

Audit fixes that change what the outputs MEAN (all of them keep the no-look-ahead guarantees)
----------------------------------------------------------------------------------------------
* Unchanged closes (audit C-03 / M-01).  The pooled predictive is continuous, a realised close on the tick grid is unchanged
  with a probability of several percent.  The recalibrator is trained on ``1[R > 0]`` of the rows where the price MOVED only
  (``_Ctx.up`` is NaN for ``R == 0``), so ``p_up`` is ``P(R > 0 | R != 0)``, the quantity the continuous ensemble predicts
  and the one DESIGN section 4 scores.  The "down" probability of the cards is therefore not ``1 - p_up`` for names with
  many flat closes (a flat close is neither).
* No-trade placeholder bars (audit C-09).  Zero-volume flat bars (halts, Yahoo's filler rows on exchange holidays and
  before listings) are not observations and are removed before any model sees the data (``features.placeholder_mask``);
  a placeholder row of the walk-forward repeats the forecast of the last real bar.  ``realized`` still follows the rows
  of the original data (what an independent evaluation recomputes).
* Randomised PIT (audit M-05).  ``HorizonForecasts.pit`` is the randomised probability integral transform of the rounded
  terminal close (``_randomised_pit``), exactly uniform for a calibrated forecaster, instead of the point value ``F(R)`` that
  piles up at ``F(0)`` when a close is unchanged.  The uniform variate is a pure function of (symbol, origin date, horizon).
* Calendar time (audit M-04).  The one-day predictive is ``1 + weekend_var_uplift`` wider in variance before a weekend
  (Friday origins), see ``features.next_bar_after_weekend``.  Holidays are not modelled (no causal historical calendar).
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from .. import config
from ..types import HorizonForecasts, PriceData, WalkForward
from . import calibration as Cal
from . import dlm as D
from . import ensemble as E
from . import features as F
from . import garch as G
from . import hmm as Hm
from . import mc as MC

MODEL_NAMES = ["garch", "hmm", "dlm", "naive"]
MODEL_LABELS = {"garch": "GJR-GARCH-t", "hmm": "隱馬可夫regime", "dlm": "動態線性模型(貝氏)", "naive": "隨機漫步基準"}
LOG_LIMIT = (math.log(1.0 - config.DAILY_LIMIT), math.log(1.0 + config.DAILY_LIMIT))
ENGINE_VERSION = "1.1"  # 1.1: audit fixes (unchanged closes carry no calibration target, placeholder bars, randomised PIT, ...)
REGIME_CONFIDENT = 0.6  # filtered probability of the most likely regime from which the live note states it plainly
# the event whose probability ``p_up`` is: P(R > 0 | R != 0).  A flat close (R == 0) is neither up nor down; a scorer must leave
# those rows out (DESIGN section 4) -- scoring them as "not up" compares a conditional probability with an unconditional event.
P_UP_EVENT = "up_given_move"

DEFAULT_CFG = {
    "garch_window": 2500,  # rows used to fit the GARCH (about 10 years)
    "garch_skew_damping": 0.6,  # share of the model-implied skewness fed to the two-piece t (Monte-Carlo calibrated)
    "garch_ew_halflife": None,  # half-life (rows) of the exponentially weighted variance-targeting level (None = equal)
    "hmm_window": 1000,  # rows used to fit the HMM (about 4 years)
    "hmm_states": 3,
    "hmm_mean_kappa": 500.0,  # pseudo-observations pulling each regime's mean return towards the shrunk drift
    "hmm_mean_to_drift": True,  # shrink regime means towards the shrunk drift (True) or the window mean (False)
    "hmm_anchor_every": 12,  # cold HMM fit every this many refits (absolute rows); refits in between warm-start from it
    "dma_alpha": 0.99,  # forgetting factor of the dynamic model averaging
    "dma_window": 400,  # hard window (origins) of matured scores used for the weights
    "dma_eta": 0.25,  # learning rate on the (1/h-tempered) log scores; chosen on pre-2021-08 dev data only (see report)
    "dma_shrink": 0.0,  # weights are shrunk towards equal weights: w <- (1 - s) w + s / M
    "cal_window": 300,  # hard window (origins) used by the recalibrator
    "cal_lambda": 0.995,  # forgetting inside the calibration window
    "cal_sigma_a": 0.12,  # prior sd of the logit intercept
    "cal_sigma_b": 0.35,  # prior sd of the logit slope (prior mean 1)
    "dlm_delta": 0.998,  # discount factor of the regression coefficients
    "dlm_beta_v": 0.998,  # discount of the learned noise variance
    "dlm_tau": 0.025,  # prior sd of the standardised-feature coefficients
    "dlm_tau_intercept": 0.03,
    "naive_nu": 6.0,  # degrees of freedom of the naive one-day Student-t
    "weekend_var_uplift": 0.16,  # relative extra variance of the one-day predictive when the next bar follows a weekend
    "min_fit_bars": 120,  # earliest refit uses at least this many rows
    "mc_paths": 5000,  # Monte-Carlo paths for the live fan chart / risk
    "mc_pool": 1500,  # standardised residuals pooled for the filtered historical simulation
}

DAYS_LABEL = {1: "明日", 5: "未來5日", 20: "未來20日"}


# ----------------------------------------------------------------------------- configuration helpers
def _merge_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULT_CFG)
    if cfg:
        unknown = set(cfg) - set(DEFAULT_CFG)
        if unknown:
            raise ValueError(f"unknown cfg keys: {sorted(unknown)}")
        out.update(cfg)
    return out


def _check_horizons(horizons, cfg: dict) -> tuple:
    hs = tuple(sorted({int(h) for h in horizons}))
    if not hs or hs[0] < 1:
        raise ValueError("horizons must be positive integers")
    if hs[-1] > min(cfg["dma_window"], cfg["cal_window"]):
        raise ValueError("horizon larger than the state windows")
    return hs


def _grid_first(refit_every: int, min_fit_bars: int) -> int:
    """First absolute refit row with at least ``min_fit_bars`` rows of history."""
    return refit_every * int(math.ceil((min_fit_bars - 1) / refit_every))


def _state_window(cfg: dict) -> int:
    return int(cfg["dma_window"] + cfg["cal_window"])


# ----------------------------------------------------------------------------- data hygiene / discrete outcomes
def _drop_placeholders(data):
    """``(real, keep)``: ``data`` without its no-trade placeholder bars, and the boolean row mask (``None`` = none dropped).

    The models only ever see real trading sessions (``features.placeholder_mask``).  ``walk_forward`` maps its outputs
    back onto the rows of the ORIGINAL data: a placeholder row repeats the forecast of the last real bar (nothing new was
    observed).  The mask is a prefix function of the data, so truncation invariance is preserved.
    """
    ph = F.placeholder_mask(data)
    if not ph.any():
        return data, None
    k = ~ph
    real = PriceData(data.symbol, data.name, data.market, data.kind, data.dates[k], data.open[k], data.high[k],
                     data.low[k], data.close[k], data.adjclose[k], data.volume[k], list(data.dividends),
                     data.last_bar_complete, data.fetched_at, data.stale)
    return real, k


def tick_size(price, kind: str = "stock") -> np.ndarray:
    """Minimum price increment of the TWSE / TPEx tick table at ``price`` (vectorised; 0 for an index).

    Stocks: < 10: 0.01, < 50: 0.05, < 100: 0.1, < 500: 0.5, < 1000: 1, otherwise 5.  ETFs: < 50: 0.01, otherwise 0.05
    (checked on the cached 0050 / 00400A closes).  Yahoo's ``close`` of older rows is split-adjusted and no longer sits on the
    grid of its own price level, which only makes the cell width below approximate (the bands are roughly relative).
    """
    p = np.asarray(price, dtype=float)
    if kind == "index":
        return np.zeros_like(p)
    if kind == "etf":
        return np.where(p < 50.0, 0.01, 0.05)
    return np.select([p < 10.0, p < 50.0, p < 100.0, p < 500.0, p < 1000.0], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def _row_uniform(symbol: str, dates, h: int) -> np.ndarray:
    """Deterministic U(0,1) variate per (symbol, origin date, horizon): splitmix64 finaliser of an integer key.

    A pure function of the row's own date, so the randomised PIT of row ``i`` is identical whether or not later rows exist
    (a seeded random *stream* would shift with the length of the data and break truncation invariance).
    """
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    key = (np.uint64(zlib.crc32(symbol.encode("utf-8"))) << np.uint64(32)) + (days + 4_000_000).astype(np.uint64) * np.uint64(512) \
        + np.uint64(h)
    with np.errstate(over="ignore"):
        x = key + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0


def _randomised_pit(pool, sel, realized, term_close, kind: str, u: np.ndarray) -> np.ndarray:
    """Randomised probability integral transform of rounded outcomes (Czado, Gneiting & Held 2009).

    The pooled predictive is continuous but the realised price is rounded to the tick grid: with ``F`` the predictive CDF
    of the latent return, the observation says only that the latent return lies in ``[R + ln(1 - t/2c), R + ln(1 + t/2c)]``
    (``t`` = tick at the terminal close ``c``).  ``PIT = F(lo) + U (F(hi) - F(lo))`` with ``U ~ U(0,1)`` is exactly
    uniform under a calibrated forecaster; the plain ``F(R)`` has a point mass at ``F(0)`` whenever a close is unchanged
    (6.8 % of the 1-day outcomes, 25 % for low-priced names) and made the PIT-KS test reject for a perfect forecaster.

    ``sel`` selects, for each output row, the pool row (``None`` = identity); ``realized`` / ``term_close`` / ``u`` are per
    output row; NaN ``realized`` -> NaN.  For the one-day pool the CDF saturates at the +-10 % limits.
    """
    realized = np.asarray(realized, float)
    out = np.full(len(realized), np.nan)
    fin = np.isfinite(realized)
    if not fin.any():
        return out
    if sel is not None:
        pool = E.Pool([p.subset(sel) for p in pool.prims], pool.w[sel], pool.trunc)
    c = np.asarray(term_close, float)
    half = np.where(np.isfinite(c) & (c > 0.0), 0.5 * tick_size(c, kind) / np.where(c > 0.0, c, 1.0), 0.0)
    half = np.minimum(half, 0.5)
    x = np.where(fin, realized, 0.0)
    f_lo = pool.cdf(x + np.log1p(-half))
    f_hi = pool.cdf(x + np.log1p(half))
    pit = np.clip(f_lo + np.asarray(u, float) * (f_hi - f_lo), 0.0, 1.0)
    return np.where(fin, pit, np.nan)


# ----------------------------------------------------------------------------- context (refit-free, causal)
class _Ctx:
    """Everything that does not depend on the refit schedule (all of it causal)."""

    def __init__(self, data, market, hs: tuple, cfg: dict):
        adj = np.asarray(data.adjclose, dtype=float)
        if len(adj) < 2 or not (np.all(np.isfinite(adj)) and np.all(adj > 0)):
            raise ValueError("bad_price_data: adjclose must be finite and positive")
        self.data, self.cfg, self.hs = data, cfg, hs
        self.n = n = len(adj)
        self.fs = fs = F.build_features(data, market)
        lp = np.log(adj)
        H = len(hs)
        self.Rraw = np.full((H, n), np.nan)  # realised raw log returns (evaluation)
        self.Rc = np.full((H, n), np.nan)  # realised cleaned log returns (learning)
        for a, h in enumerate(hs):
            if n > h:
                self.Rraw[a, : n - h] = lp[h:] - lp[:-h]
                self.Rc[a, : n - h] = fs.cs[h:] - fs.cs[:-h]
        # Direction target of the recalibrator.  The pooled predictive is continuous (no mass at 0) whereas a realised
        # price on the tick grid is unchanged with a probability of several percent (6.8 % of the 1-day outcomes on average,
        # 25 % for a 2412-type low-priced name).  Counting those ties as "not up" teaches the calibrator P(R > 0) with the
        # flat mass lumped into "down", which matches neither the continuous ensemble nor the evaluation (DESIGN section 4
        # drops R == 0).  Ties (and cleaned-away data artefacts, whose cleaned return is 0) therefore carry NO target: the
        # recalibrator learns P(up | the price moved) = P(R > 0) of the continuous predictive.
        no_dir = (self.Rraw == 0.0) | (self.Rc == 0.0)
        self.up = np.where(np.isfinite(self.Rc) & ~no_dir, (self.Rc > 0.0).astype(float), np.nan)
        Xd = np.column_stack([fs.X, np.ones(n)])
        Y = D.standardized_targets(fs.cs, fs.drift, fs.sigma_ref, hs)
        sd = D.default_prior_sd(Xd.shape[1], cfg["dlm_tau"], cfg["dlm_tau_intercept"])
        self.dlm = D.run_dlm(Xd, Y, hs, cfg["dlm_delta"], cfg["dlm_beta_v"], sd)
        self.Yh = np.column_stack([fs.rc, fs.logvol])  # HMM observations (return, log range vol)
        # Calendar time (audit M-04): variance multiplier of the NEXT bar of every origin.  Only the weekend is known with
        # certainty at the origin (a Friday is followed by the first bar after a weekend); see features.next_bar_after_weekend.
        self.m1 = 1.0 + float(cfg["weekend_var_uplift"]) * F.next_bar_after_weekend(data.dates)


# ----------------------------------------------------------------------------- refit blocks
def _fit_garch(ret: np.ndarray, mean: float, halflife: float | None = None) -> G.GarchParams:
    try:
        p = G.fit_gjr_garch_t(ret, mean=mean, target_var=G.ew_variance(ret - mean, halflife))
        if np.isfinite(p.loglik) and p.omega > 0 and p.persistence < 1:
            return p
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    v = max(float(np.mean((ret - mean) ** 2)), 1e-10)  # deterministic fallback: textbook equity parameters
    return G.GarchParams(omega=(1.0 - 0.96) * v, alpha=0.03, gamma=0.06, beta=0.90, nu=7.0, mean=float(mean),
                         n_obs=len(ret), converged=False)


def _fit_hmm(Y: np.ndarray, K: int, init: Hm.HMMParams | None = None, kappa: float = 250.0,
             prior_ret_mean: float | None = None) -> Hm.HMMParams:
    try:
        pm = None
        if prior_ret_mean is not None:  # return mean shrinks to the given drift, the other dimensions to the window mean
            pm = np.array([prior_ret_mean] + [float(np.mean(Y[:, j])) for j in range(1, Y.shape[1])])
        ks = np.full(Y.shape[1], 5.0)
        ks[0] = kappa
        p = Hm.fit_hmm(Y, K=K, init=init, prior_strength=ks, prior_mean=pm)
        if np.all(np.isfinite(p.A)) and np.all(np.isfinite(p.mu)) and np.all(np.isfinite(p.var)):
            return p
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    D_ = Y.shape[1]  # deterministic fallback: K identical states (the HMM then reduces to a single Gaussian)
    mu = np.tile(Y.mean(axis=0), (K, 1))
    var = np.tile(np.maximum(Y.var(axis=0), 1e-10), (K, 1))
    A = np.full((K, K), 1.0 / K)
    return Hm.HMMParams(A=A, mu=mu, var=var, lo=np.full(D_, -np.inf), hi=np.full(D_, np.inf),
                        alpha_last=np.full(K, 1.0 / K), converged=False, n_obs=len(Y))


class _Components:
    """Per-origin component inputs for the origin range [o_lo, o_hi] (arrays indexed by ``origin - o_lo``)."""

    def __init__(self, o_lo: int, o_hi: int, hs: tuple, K: int):
        m = o_hi - o_lo + 1
        H = len(hs)
        self.o_lo, self.o_hi, self.m, self.hs = o_lo, o_hi, m, hs
        self.u1 = np.full(m, np.nan)  # GARCH next-day variance
        self.gvar = np.full((H, m), np.nan)
        self.gkurt = np.full((H, m), np.nan)
        self.gskew = np.full((H, m), np.nan)
        self.pi = np.full((m, K), np.nan)  # filtered regime probabilities
        self.hm_mean = np.full((H, m, K), np.nan)
        self.hm_sd = np.full((H, m, K), np.nan)
        self.blocks = []  # (refit row, first origin, last origin, GarchParams, HMMParams, garch window start)


def _compute_components(ctx: _Ctx, o_lo: int, o_hi: int, refit_every: int, progress=None, done0: int = 0,
                        total: int | None = None) -> _Components:
    cfg, fs, hs = ctx.cfg, ctx.fs, ctx.hs
    K = int(cfg["hmm_states"])
    hmax = max(hs)
    comp = _Components(o_lo, o_hi, hs, K)
    g0 = _grid_first(refit_every, int(cfg["min_fit_bars"]))
    anchor_rows = int(cfg["hmm_anchor_every"]) * refit_every
    anchors: dict = {}  # anchor row -> cold-fitted HMM (a pure function of the data up to that row)

    def hmm_window(r: int) -> np.ndarray:
        return ctx.Yh[max(1, r - int(cfg["hmm_window"]) + 1) : r + 1]

    kap = float(cfg["hmm_mean_kappa"])
    to_drift = bool(cfg["hmm_mean_to_drift"])

    def hmm_fit1(r: int, init=None) -> Hm.HMMParams:
        return _fit_hmm(hmm_window(r), K, init=init, kappa=kap,
                        prior_ret_mean=float(fs.drift[r]) if to_drift else None)

    def hmm_fit(r: int) -> Hm.HMMParams:
        a = max(g0, (r // anchor_rows) * anchor_rows)  # latest absolute anchor row <= r
        if a == r:
            return hmm_fit1(r)
        if a not in anchors:
            anchors[a] = hmm_fit1(a)
        return hmm_fit1(r, init=anchors[a])

    rows = list(range((o_lo // refit_every) * refit_every, (o_hi // refit_every) * refit_every + 1, refit_every))
    total = (len(rows) + 1) if total is None else total
    for bi, r in enumerate(rows):
        i0, i1 = max(o_lo, r), min(o_hi, r + refit_every - 1)
        org = np.arange(i0, i1 + 1)
        pos = org - o_lo
        # --- GJR-GARCH-t: cold fit on the window ending at r, filter continued through the block
        w0 = max(1, r - int(cfg["garch_window"]) + 1)
        gp = _fit_garch(fs.rc[w0 : r + 1], float(fs.drift[r]), cfg["garch_ew_halflife"])
        eps = fs.rc[w0 : i1 + 1] - gp.mean
        s2 = G.variance_filter(eps, gp.omega, gp.alpha, gp.gamma, gp.beta, gp.uncond_var)
        u1 = s2[org - w0 + 1]  # variance of row i+1 given information up to row i
        comp.u1[pos] = u1
        for a, h in enumerate(hs):
            v, k, s = G.hstep_moments(u1, gp, h)
            comp.gvar[a, pos], comp.gkurt[a, pos], comp.gskew[a, pos] = v, k, s
        # --- HMM: cold fit on the window ending at r, filtered probabilities continued through the block
        v0 = max(1, r - int(cfg["hmm_window"]) + 1)
        hp = hmm_fit(r)
        if i1 > r:
            pis = np.vstack([hp.alpha_last[None, :], Hm.filter_probs(hp, ctx.Yh[r + 1 : i1 + 1], hp.alpha_last)])
        else:
            pis = hp.alpha_last[None, :]
        comp.pi[pos] = pis[org - r]
        mean_h, var_h = Hm.hstep_state_moments(hp, hmax)
        for a, h in enumerate(hs):
            comp.hm_mean[a, pos, :] = mean_h[h][None, :]
            comp.hm_sd[a, pos, :] = np.sqrt(var_h[h])[None, :]
        comp.blocks.append((r, i0, i1, gp, hp, w0, v0))
        if progress is not None:
            progress(done0 + bi + 1, total)
    return comp


# ----------------------------------------------------------------------------- combination
def _horizon_pool(ctx: _Ctx, comp: _Components, a: int):
    """Component predictives (as primitives) for horizon index ``a`` on the whole origin range."""
    cfg, fs = ctx.cfg, ctx.fs
    h = ctx.hs[a]
    org = np.arange(comp.o_lo, comp.o_hi + 1)
    mean_base = h * fs.drift[org]
    # calendar time (audit M-04): the one-day predictive is wider before a weekend; for h >= 5 every window holds exactly one
    # weekend bar, so the effect is a constant that merges into the level (and the development rows show no gain from it)
    cal = ctx.m1[org] if h == 1 else np.ones(len(org))
    gvar, gk, gs = comp.gvar[a] * cal, comp.gkurt[a], comp.gskew[a]
    damp = float(cfg["garch_skew_damping"])
    garch = E.SkewT.from_moments(mean_base, gvar, gk, damp * gs)
    sref = fs.sigma_ref[org]
    tilt = sref * math.sqrt(h) * ctx.dlm["f"][a][org]
    pu = sref * sref * h * ctx.dlm["pu"][a][org]
    vtot = gvar + pu
    ratio = gvar / vtot
    dlm = E.SkewT.from_moments(mean_base + tilt, vtot, gk * ratio ** 2, damp * gs * ratio ** 1.5)
    nu_h = 4.0 + h * (float(cfg["naive_nu"]) - 4.0)  # kurtosis of a sum of h iid shocks falls like 1/h
    naive = E.SkewT(mean_base, np.sqrt(h * fs.v97[org] * cal), nu_h, 1.0)
    K = comp.pi.shape[1]
    hmm_prims = [E.Normal(comp.hm_mean[a][:, k], comp.hm_sd[a][:, k] * np.sqrt(cal)) for k in range(K)]
    return garch, hmm_prims, dlm, naive


def _combine(ctx: _Ctx, comp: _Components, out_lo: int, keep: bool = False) -> list:
    """Ensemble for every horizon; reports origins >= ``out_lo`` (earlier origins only feed the combination state)."""
    cfg = ctx.cfg
    org = np.arange(comp.o_lo, comp.o_hi + 1)
    sel = org >= out_lo
    qlev = np.asarray(config.QLEVELS, float)
    out = []
    zeros_all = np.zeros(comp.m)
    for a, h in enumerate(ctx.hs):
        garch, hmm_prims, dlm, naive = _horizon_pool(ctx, comp, a)
        pi = comp.pi
        # --- matured log predictive scores (cleaned realised return), tempered DMA weights
        rc = ctx.Rc[a][org]
        ok = np.isfinite(rc)
        x = np.where(ok, rc, 0.0)
        l_h = E.mixture_logpdf(x, [pi[:, k] for k in range(pi.shape[1])], hmm_prims)
        sc = np.stack([garch.logpdf(x), l_h, dlm.logpdf(x), naive.logpdf(x)])
        sc = np.where(ok[None, :], sc, np.nan)
        u = np.full_like(sc, np.nan)
        if comp.m > h:
            u[:, h:] = sc[:, : comp.m - h]  # score of origin t-h becomes known at row t
        wd = E.dma_weights(u, float(cfg["dma_alpha"]), int(cfg["dma_window"]) - h + 1,
                           temper=float(cfg["dma_eta"]) / h)
        if cfg["dma_shrink"]:
            wd = (1.0 - float(cfg["dma_shrink"])) * wd + float(cfg["dma_shrink"]) / wd.shape[1]
        wmat = np.column_stack([wd[:, 0], wd[:, 2], wd[:, 3]] + [wd[:, 1] * pi[:, k] for k in range(pi.shape[1])])
        prims = [garch, dlm, naive] + hmm_prims
        trunc = LOG_LIMIT if h == 1 else None
        # --- raw probability on every origin (needed by the recalibrator), calibrated probability
        pool_all = E.Pool(prims, wmat, trunc)
        p_raw = np.clip(1.0 - pool_all.cdf(zeros_all), 1e-6, 1.0 - 1e-6)
        p_cal, _, _ = Cal.recalibrate(p_raw, ctx.up[a][org], h, window=int(cfg["cal_window"]),
                                      lam=float(cfg["cal_lambda"]), sigma_a=float(cfg["cal_sigma_a"]),
                                      sigma_b=float(cfg["cal_sigma_b"]))
        # --- reported rows
        prims_o = [p.subset(sel) for p in prims]
        w_o = wmat[sel]
        pool_o = E.Pool(prims_o, w_o, trunc)
        mo = int(sel.sum())
        q = pool_o.ppf(np.tile(qlev, (mo, 1)))
        g_o, d_o, n_o = prims_o[0], prims_o[1], prims_o[2]
        hm_o = prims_o[3:]
        pi_o = pi[sel]
        wdo = wd[sel]
        comp_p = {"garch": 1.0 - g_o.cdf(np.zeros(mo)), "dlm": 1.0 - d_o.cdf(np.zeros(mo)),
                  "naive": 1.0 - n_o.cdf(np.zeros(mo)),
                  "hmm": sum(pi_o[:, k] * (1.0 - hm_o[k].cdf(np.zeros(mo))) for k in range(len(hm_o)))}
        comp_m = {"garch": g_o.mean, "dlm": d_o.mean, "naive": n_o.mean,
                  "hmm": sum(pi_o[:, k] * hm_o[k].mean for k in range(len(hm_o)))}
        res = {"h": h, "p_up_raw": p_raw[sel], "p_up": p_cal[sel], "mu": pool_o.mean,
               "sigma": np.sqrt(pool_o.var), "q": q, "pool": pool_o,  # the PIT is computed by the caller (_randomised_pit)
               "comp_p_up": {k: np.clip(v, 1e-6, 1 - 1e-6) for k, v in comp_p.items()}, "comp_mu": comp_m,
               "weights": {name: wdo[:, i] for i, name in enumerate(MODEL_NAMES)}}
        if keep:
            res["prims"] = prims_o
            res["pi"] = pi_o
            res["wmat"] = w_o
        out.append(res)
    return out


# ----------------------------------------------------------------------------- walk-forward
def walk_forward(data, market, horizons=config.HORIZONS, min_train: int = config.MIN_TRAIN,
                 refit_every: int = config.REFIT_EVERY, start_idx: int | None = None, progress=None,
                 cfg: dict | None = None) -> WalkForward:
    """Strictly out-of-sample forecasts for every origin ``i >= start_idx`` (default ``min_train``).

    Row ``i`` of every output uses bars ``0..i`` only (and the market bars dated <= ``date_i``).  Parameters are
    re-estimated at the absolute rows ``k * refit_every``; the first reported origin uses the parameters of the latest
    grid row <= ``start_idx`` (fitted on all rows up to that row).  See the module docstring for the guarantees.
    """
    cfg = _merge_cfg(cfg)
    hs = _check_horizons(horizons, cfg)
    refit_every = int(refit_every)
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    full = data
    data, keep = _drop_placeholders(full)  # the models see real trading sessions only; outputs are mapped back to ``full``
    ctx = _Ctx(data, market, hs, cfg)
    n = ctx.n
    g0 = _grid_first(refit_every, int(cfg["min_fit_bars"]))
    if start_idx is None:
        s = max(int(min_train), 0)  # bars of (real) history
    else:  # row of ``full``; in the modelled series: the first real bar at or after it
        s = int(start_idx) if keep is None else int(keep[: max(int(start_idx), 0)].sum())
    first_c = max(s, g0)  # first reported origin, as a row of the modelled (placeholder-free) series
    if first_c > n - 1:
        raise ValueError("insufficient_history")
    o_lo = max(g0, first_c - _state_window(cfg))
    n_blocks = (n - 1) // refit_every - o_lo // refit_every + 1
    total = n_blocks + 1
    comp = _compute_components(ctx, o_lo, n - 1, refit_every, progress, 0, total)
    res = _combine(ctx, comp, out_lo=first_c)
    if progress is not None:
        progress(total, total)
    K = comp.pi.shape[1]
    sl = slice(first_c - o_lo, None)
    # --- map the modelled rows back onto the rows of the original data: a placeholder row repeats the forecast of the
    # last real bar (the information set is identical); without placeholders ``sel`` is the identity
    n_out = len(full.adjclose)
    if keep is None:
        first_idx, sel = first_c, None
    else:
        first_idx = int(np.flatnonzero(keep)[first_c])
        sel = (np.cumsum(keep) - 1)[first_idx:] - first_c  # reported-row position of the last real bar <= each output row
    rows = slice(first_idx, n_out)

    def put(arr, vals):
        arr[rows] = vals if sel is None else vals[sel]

    lp_full = np.log(np.asarray(full.adjclose, float)) if keep is not None else None
    horizons_out = {}
    for a, h in enumerate(hs):
        r = res[a]
        hf = dict(p_up=np.full(n_out, np.nan), p_up_raw=np.full(n_out, np.nan), mu=np.full(n_out, np.nan),
                  sigma=np.full(n_out, np.nan), q=np.full((n_out, len(config.QLEVELS)), np.nan), pit=np.full(n_out, np.nan))
        for key in ("p_up", "p_up_raw", "mu", "sigma", "q"):
            put(hf[key], r[key])
        cpu, cmu, wts = {}, {}, {}
        for name in MODEL_NAMES:
            cpu[name], cmu[name], wts[name] = np.full(n_out, np.nan), np.full(n_out, np.nan), np.full(n_out, np.nan)
            put(cpu[name], r["comp_p_up"][name])
            put(cmu[name], r["comp_mu"][name])
            put(wts[name], r["weights"][name])
        # realised raw returns on the rows of the ORIGINAL data (what an independent evaluation recomputes)
        if keep is None:
            realized = ctx.Rraw[a].copy()
        else:
            realized = np.full(n_out, np.nan)
            if n_out > h:
                realized[: n_out - h] = lp_full[h:] - lp_full[:-h]
        term = np.full(n_out, np.nan)  # terminal raw close of every origin (its tick cell randomises the PIT)
        if n_out > h:
            term[: n_out - h] = np.asarray(full.close, float)[h:]
        hf["pit"][rows] = _randomised_pit(r["pool"], sel, realized[rows], term[rows], full.kind,
                                          _row_uniform(full.symbol, np.asarray(full.dates)[rows], h))
        horizons_out[h] = HorizonForecasts(h=h, p_up=hf["p_up"], p_up_raw=hf["p_up_raw"], mu=hf["mu"],
                                           sigma=hf["sigma"], q=hf["q"], pit=hf["pit"], realized=realized,
                                           comp_p_up=cpu, comp_mu=cmu, weights=wts)
    regime_prob = np.full((n_out, K), np.nan)
    put(regime_prob, comp.pi[sl])
    vol1d = np.full(n_out, np.nan)
    put(vol1d, np.sqrt(comp.u1[sl]))
    last_hp = comp.blocks[-1][4]
    config_out = {"engine": "TW quant core", "version": ENGINE_VERSION, "cfg": dict(cfg),
                  "start_idx": int(start_idx) if start_idx is not None else max(int(min_train), 0),
                  "state_window": _state_window(cfg), "first_refit_row": int(g0), "horizons": list(hs),
                  "refit_rule": "refit at rows k*refit_every (absolute grid) on all rows up to the refit row (capped "
                                "windows); GARCH cold, HMM cold at anchors and warm-started from them in between; "
                                "parameters in force = latest grid row <= origin (rows counted after dropping the "
                                "no-trade placeholder bars)",
                  "model_labels": dict(MODEL_LABELS), "n_refits": len(comp.blocks),
                  "placeholder_bars_dropped": int(n_out - n),
                  "p_up_event": P_UP_EVENT}
    return WalkForward(symbol=full.symbol, dates=np.asarray(full.dates).copy(), horizons=horizons_out,
                       first_idx=int(first_idx), refit_every=refit_every, min_train=int(min_train),
                       model_names=list(MODEL_NAMES), regime_prob=regime_prob,
                       regime_labels=Hm.regime_labels(last_hp), vol1d=vol1d, config=config_out)


# ----------------------------------------------------------------------------- live forecast
def _simple_moments(ppf, m: int = 400):
    """(E[e^R] - 1, std(e^R - 1)) from a quantile function evaluated on a midpoint grid (trimmed at 0.125 %)."""
    ug = (np.arange(m) + 0.5) / m
    x = np.asarray(ppf(ug[None, :]))[0]
    s = np.expm1(x)
    return float(s.mean()), float(s.std())


def _fmt_level(v: float) -> str:
    return f"{v:g}"


def forecast_latest(data, market, horizons=config.HORIZONS, cfg: dict | None = None,
                    refit_every: int = config.REFIT_EVERY) -> dict:
    """The "forecast core" JSON object (DESIGN.md section 6.1) for the last bar of ``data``.

    Identical to the last row of ``walk_forward`` on the same data (same code path) plus the live-only extras: fan
    chart, VaR / CVaR / drawdown (filtered historical simulation aligned to the exact ensemble quantiles), regime and
    volatility state, drivers.  Raises ``ValueError("insufficient_history")`` below ``config.MIN_BARS_FORECAST`` bars.
    """
    cfg = _merge_cfg(cfg)
    hs = _check_horizons(horizons, cfg)
    full = data
    n_bars = len(full.adjclose)  # reported ``n_bars``: rows of the data as given
    data, keep = _drop_placeholders(full)  # the models see real trading sessions only (trailing placeholders: no new info)
    if len(data.adjclose) < config.MIN_BARS_FORECAST:
        raise ValueError("insufficient_history")
    refit_every = int(refit_every)
    ctx = _Ctx(data, market, hs, cfg)
    n = ctx.n
    i = n - 1
    g0 = _grid_first(refit_every, int(cfg["min_fit_bars"]))
    if i < g0:
        raise ValueError("insufficient_history")
    o_lo = max(g0, i - _state_window(cfg))
    comp = _compute_components(ctx, o_lo, i, refit_every)
    res = _combine(ctx, comp, out_lo=i, keep=True)
    fs = ctx.fs
    last = comp.m - 1
    r_row, i0, i1, gp, hp, w0, v0 = comp.blocks[-1]
    last_close = float(full.close[-1])
    adj = np.asarray(full.adjclose, float)
    ug_levels = np.asarray(config.FAN_LEVELS, float)
    qlev = np.asarray(config.QLEVELS, float)
    fan_cols = [int(np.argmin(np.abs(qlev - lv))) for lv in ug_levels]
    exact = all(abs(qlev[c] - lv) < 1e-12 for c, lv in zip(fan_cols, ug_levels))
    horizons_json, models_w, anchors = {}, {}, {}
    for a, h in enumerate(hs):
        r = res[a]
        pool = r["pool"]
        qlog = r["q"][0] if exact else pool.ppf(ug_levels[None, :])[0]
        qfan = qlog[fan_cols] if exact else qlog
        simple_q = np.expm1(qfan)
        median_ret = float(np.expm1(r["q"][0][int(np.argmin(np.abs(qlev - 0.5)))]))
        e_ret, sd_simple = _simple_moments(pool.ppf)
        prims = r["prims"]
        garch, dlm, naive = prims[0], prims[1], prims[2]
        hm_pool = E.Pool(prims[3:], r["pi"])
        comps = {}
        for name, ppf in (("garch", garch.ppf), ("hmm", hm_pool.ppf), ("dlm", dlm.ppf), ("naive", naive.ppf)):
            ce, _ = _simple_moments(ppf)
            comps[name] = {"p_up": float(r["comp_p_up"][name][0]), "mu": ce}
        horizons_json[str(h)] = {
            "h": h, "days_label": DAYS_LABEL.get(h, f"未來{h}日"), "p_up": float(r["p_up"][0]),
            "p_up_raw": float(r["p_up_raw"][0]), "exp_return": e_ret, "median_return": median_ret, "sigma": sd_simple,
            "quantiles": {_fmt_level(lv): float(v) for lv, v in zip(ug_levels, simple_q)},
            "price_quantiles": {_fmt_level(lv): float(last_close * (1.0 + v)) for lv, v in zip(ug_levels, simple_q)},
            "components": comps, "mu_log": float(r["mu"][0]), "sigma_log": float(r["sigma"][0])}
        models_w[str(h)] = {name: float(r["weights"][name][0]) for name in MODEL_NAMES}
        if h <= config.FAN_DAYS:
            anchors[h] = np.asarray(qfan, float)
    # --- filtered historical simulation, aligned to the exact ensemble quantiles at the horizons
    pool_n = int(cfg["mc_pool"])
    s2_all = G.variance_filter(fs.rc[w0 : i + 1] - gp.mean, gp.omega, gp.alpha, gp.gamma, gp.beta, gp.uncond_var)
    sig_t = np.sqrt(s2_all[:-1])
    z_all = (fs.rc[w0 : i + 1] - gp.mean) / sig_t
    pi_hist = Hm.filter_probs(hp, ctx.Yh[v0 : i + 1])  # filtered (never smoothed) regime probabilities of the pool rows
    take = min(pool_n, len(z_all), len(pi_hist))  # both series end at row i; pool the rows where both exist
    z_pool = z_all[-take:]
    pool_w = pi_hist[-take:]
    seed = (int(np.datetime64(data.dates[-1], "D").astype(np.int64)) * 7919 + n) % (2 ** 32)
    rng = np.random.default_rng(seed)
    cum = MC.fhs_simulate(gp, float(comp.u1[last]), z_pool, pool_w, hp.A, comp.pi[last], float(fs.drift[i]),
                          int(cfg["mc_paths"]), config.FAN_DAYS, rng, LOG_LIMIT)
    cum_adj, _ = MC.align_to_anchors(cum, ug_levels, anchors, LOG_LIMIT)
    fan, qmat = MC.fan_from_paths(cum_adj, ug_levels, last_close)
    for h, qv in anchors.items():  # exact ensemble quantiles on the horizon days
        for li in range(len(ug_levels)):
            qmat[li, h - 1] = qv[li]
    qmat = np.maximum.accumulate(qmat, axis=0)
    ret = np.expm1(qmat)
    fan["returns"] = ret.tolist()
    fan["prices"] = (last_close * (1.0 + ret)).tolist()
    risk = MC.risk_metrics(cum_adj, 1, config.FAN_DAYS)
    # --- regime / volatility
    labels = Hm.regime_labels(hp)
    probs = comp.pi[last]
    cur = labels[int(np.argmax(probs))]
    u1_now = float(comp.u1[last])
    sig_d = math.sqrt(u1_now)
    hist = sig_t[-min(len(sig_t), int(cfg["garch_window"])):]
    pct = float(np.mean(hist <= sig_d))
    level = "波動偏高" if pct >= 0.8 else ("波動偏低" if pct <= 0.2 else "波動處於中段")
    sig_a = sig_d * math.sqrt(config.TRADING_DAYS_PER_YEAR)
    pmax = float(probs.max())
    # the filtered regime probabilities move quickly (fitted persistence is only ~0.75-0.8): say so when no regime dominates
    state_txt = (f"模型判定目前最可能處於「{cur}」（機率 {pmax * 100:.0f}%）。" if pmax >= REGIME_CONFIDENT else
                 f"模型對目前所處狀態判斷不明確（機率最高者為「{cur}」，僅 {pmax * 100:.0f}%，狀態常在相鄰交易日間切換）。")
    note = (f"預期明日波動約 {sig_d * 100:.2f}%（年化 {sig_a * 100:.0f}%），落在近 {len(hist)} 個交易日的第 {pct * 100:.0f} 百分位（{level}）；"
            + state_txt)
    # --- drivers: contribution of each standardised feature to the DLM mean (log-return units)
    dh = 5 if 5 in hs else hs[0]
    drivers = []
    for j, key in enumerate(fs.keys):
        z = float(fs.X[i, j])
        raw = fs.raw[i, j]
        by_h = {}
        for b, hb in enumerate(hs):
            by_h[str(hb)] = float(ctx.dlm["m"][b, i, j] * z * fs.sigma_ref[i] * math.sqrt(hb))
        drivers.append({"name": F.FEATURE_NAMES[key], "key": key, "value": None if not np.isfinite(raw) else float(raw),
                        "unit": F.FEATURE_UNITS[key], "z": z, "contribution": by_h[str(dh)],
                        "contribution_by_h": by_h})
    drivers.sort(key=lambda d: -abs(d["contribution"]))
    out = {
        "status": "ready", "symbol": full.symbol, "asof": str(np.datetime64(full.dates[-1], "D")),
        # last_close is the raw quote; the 1-day change uses the adjusted close (a dividend drop is not a "change")
        "last_close": last_close, "chg_1d": float(adj[-1] / adj[-2] - 1.0), "n_bars": int(n_bars),
        "horizons": horizons_json, "fan": fan,
        "regime": {"labels": labels, "probs": [float(v) for v in probs], "current": cur,
                   "confident": bool(pmax >= REGIME_CONFIDENT),  # False: no regime dominates, do not present `current` as the state
                   "transition": [[float(v) for v in row] for row in hp.A]},
        "vol": {"sigma_daily": sig_d, "sigma_annual": sig_a, "percentile": pct, "regime_note": note},
        "risk": {k: float(v) for k, v in risk.items()},
        "models": {"weights": models_w, "names": dict(MODEL_LABELS)},
        "drivers": drivers,
        "engine": {"version": ENGINE_VERSION, "refit_every": refit_every,
                   "last_refit_row": int(r_row) if keep is None else int(np.flatnonzero(keep)[r_row]),
                   "state_window": _state_window(cfg), "has_market": bool(fs.has_market),
                   "placeholder_bars_dropped": int(n_bars - n), "p_up_event": P_UP_EVENT,
                   "mc": {"paths": int(cfg["mc_paths"]), "pool": int(take), "regime_mixing": bool(hp.K > 1)}},
    }
    return out
