"""Single-stock walk-forward backtest report (DESIGN.md 6.4) and its schema validator.

``build_backtest_report`` consumes ``PriceData`` + ``WalkForward`` only.  Realised returns are recomputed from the
adjusted closes and compared with ``HorizonForecasts.realized`` (a mismatch means an alignment bug and raises
``ValueError``).

Which event is scored (audit C-02)
----------------------------------
A close on the tick grid is unchanged (``R == 0``) on a few percent of the days (up to 25 % at h = 1 for low-priced names), and
"up" is a different event depending on how such a day is read.  The scorer must evaluate the event the forecaster actually
forecasts, so it follows the declaration ``WalkForward.config["p_up_event"]`` (see :func:`p_up_event`):

* ``"up_given_move"`` -- ``p_up = P(R > 0 | R != 0)`` (the quant engine: its calibrator learns the direction of MOVING closes
  and the continuous predictive it recalibrates has no mass at 0).  A flat close is neither up nor down, so the probabilistic
  block leaves those rows out, exactly like the direction block.  Scoring them as "not up" would compare a conditional
  probability with an unconditional event and make ``p_up`` look biased high by ``1 / (1 - tie rate)``.
* undeclared or ``"up"`` -- ``p_up = P(R > 0)`` (the ``types.HorizonForecasts`` contract): a flat close counts as *not up*,
  so the probabilistic block is scored on EVERY evaluated forecast with ``y = 1[R > 0]``.  Dropping the flat days would then
  score ``P(up | not flat)`` and make ``p_up`` look biased low.

The direction block (accuracy, baselines, binomial tests, rolling accuracy) always excludes and counts the flat days
(``n_tie_excluded``): "does the price go up or down?" is only defined when it moves.  ``probabilistic.p_up_event`` / ``.n`` state
the event and the number of rows that were scored.

Brier / log-loss references (DESIGN.md 9(a))
--------------------------------------------
The headline ``brier_baseline`` / ``brier_skill`` / ``log_loss_baseline`` / ``log_loss_skill`` use the CAUSAL reference: the
expanding-window frequency of the scored event known at each forecast origin (matured outcomes only, Laplace smoothed, flat
outcomes counted exactly as the scored event counts them).  The ORACLE reference (the up-rate of the verification sample,
i.e. hindsight; the best possible constant forecast) is reported separately in the ``*_oracle`` keys: a real forecaster scores
<= 0 against it almost by construction.  On the 62 cached real stocks (60 universe names + 0050 + 6488, n ~ 1250 each, data to
2026-10-23, final quant engine) the mean Brier skill is -0.19 / -0.58 / -2.33 % against the oracle reference (negative for
87 / 98 / 97 % of the stocks) but +0.00 / -0.02 / +0.00 % against the causal one at h = 1 / 5 / 20 (negative for 50 / 63 / 61 %):
the causal reference removes the hindsight mean-bias term and what is left is zero skill (plus the noise of the
expanding-window constant, next sentence).  The verdict's Brier test is *not* run against the causal reference alone: a noisy
expanding-window base rate is a worse constant than the truth, so any forecaster that merely knows the base rate would "beat"
it (one-sided Diebold-Mariano rejections at 5 % for a zero-information forecaster that knows P(up): 11 - 15 % / 24 - 27 % /
20 - 26 % at h = 1 / 5 / 20, tests/test_eval_report.py::TestBrierReferences).  The leg uses the larger (= less favourable) of
the one-sided p-values against the causal and the oracle reference, so the model has to beat both (0 % rejections in the same
simulation).

Verdict (conservative, rule-based; see :func:`assess_evidence`)
---------------------------------------------------------------
Four one-sided predictive tests ("is the model better than a causal, free baseline?"):

* ``direction``  exact binomial test of the call accuracy against the best of the baselines always-up / trailing-h-return
  sign / expanding majority class,
* ``brier``      Diebold-Mariano (HAC) on squared probability errors vs the base rate (causal AND oracle, see above),
* ``ic``         Spearman rank correlation of the forecast mean with the realised return (n_eff = n/h),
* ``mse``        Diebold-Mariano on squared return errors vs the better of {zero forecast, expanding mean}.

Thresholds (:data:`THRESH_STOCK`; the universe backtest uses the stricter :data:`THRESH_UNIVERSE`):

``no_edge``       Bonferroni-adjusted minimum p-value >= 1 %, or fewer than two tests individually below 5 %
                  (a lone significant test is treated as an artefact), or too few observations
                  (< 150 scored forecasts or < 30 effective observations),
``weak_edge``     adjusted min p < 1 % and >= 2 tests below 5 %,
``moderate_edge`` additionally >= 2 tests individually p < 1 %,
``strong_edge``   additionally >= 3 tests p < 0.1 %, n >= 500, and the net-of-cost strategy beats buy-and-hold
                  (PSR of the strategy against the buy-and-hold Sharpe >= 95 % and bootstrap p < 5 %).
A model that merely reproduces the base rate therefore lands on ``no_edge``.

Adverse-evidence veto (audit C-08): the IC and MSE legs are both functions of the forecast mean, the direction and Brier legs
of ``p_up``.  When the probability legs say the model is WORSE than the free baseline (direction or Brier one-sided p > 0.95,
or both > 0.5) the level is capped at ``weak_edge`` and the verdict says so.  The cases that triggered the audit are stocks
whose tick size is about half a daily sigma: the close bounces between bid and ask (lag-1 autocorrelation of returns about
-0.15), reversal features forecast the bounce (significant IC and MSE) while the direction calls lose to "always up".  When the
probability legs are merely silent (not adverse) and only the mean legs are significant, there is no veto but the text says
so and adds :data:`BOUNCE_HINT` (real h = 1 data after the quant tie fix: 2412 / 3045 / 4904, IC(model mean) 0.09 - 0.12 against
IC(-r_t) 0.13 - 0.16, i.e. the model learnt the one-day reversal).

Measured behaviour (synthetic worlds with known answers, see tests/test_eval_report.py): over 1800 runs of no-skill
models (constant base-rate model, over-confident noise model; h = 1, 5, 20) one run left ``no_edge`` (about 0.06 %).  That
figure is for synthetic *zero-drift, no-persistence* worlds.  On 62 real return series with independent persistent AR(1) fake
forecasters (rho 0.9 - 0.99) or a constant positive drift the single-stock false-edge rate was about 0.25 - 0.5 % per report
(``moderate`` or higher about 0.06 - 0.13 %), still below the 1 % family-wise target.  Reasons: the IC and MSE legs share the
forecast mean and are positively dependent (Spearman correlation of their p-values 0.3 - 0.7), so "two tests agree" is weaker
than two independent confirmations, and the MSE leg is partly a test for a positive drift.  Planted real edges are found
reliably when they are large (IC ~ 0.2: 100 % at h = 1 and 5) but the rule has little power for small effects and long
horizons: at h = 20 an IC of 0.33 is found in only about 25 - 50 % of runs (see :func:`metrics.min_detectable_ic` and the
disclosure sentence of the ``no_edge`` verdict: "no edge detected" is not "no edge").

Spearman IC p-value.  The single-stock IC p-value is the textbook one with n_eff = n/h (identical to scipy for h = 1).  For
forecasts that are smoothed functions of PAST returns (drift, HMM, DLM -- every real forecaster here) the finite-sample null
of the IC against returns of the same window is centred NEGATIVE (about -0.04 / -0.09 / -0.18 at h = 1 / 5 / 20, simulated),
so the displayed two-sided p-value is uncalibrated (2.4 - 7 times too liberal in the two-sided tail; for independent persistent
signals it is ~1.7 times too liberal).  The one-sided positive p-value used by the verdict is therefore conservative and the
text marks the two-sided p-value as 未校準 (informational only).
"""
from __future__ import annotations

import json
import math

import numpy as np

from .. import config
from ..types import HorizonForecasts, PriceData, WalkForward
from . import metrics as M
from . import strategy as S

LEVELS = ("no_edge", "weak_edge", "moderate_edge", "strong_edge")
# significance thresholds of the verdict: fw = family-wise (Bonferroni) level for any edge claim, confirm = per-test level
# that counts as a confirmation for 'moderate', strong = per-test level for 'strong', agree = at least two different
# tests must be individually below this for any edge claim
THRESH_STOCK = {"fw": 0.01, "confirm": 0.01, "strong": 0.001, "agree": 0.05}
# universe-level tests use HAC standard errors of persistent daily series whose far tails are heavier than the t
# reference (measured in no-skill simulations), hence stricter thresholds
THRESH_UNIVERSE = {"fw": 0.003, "confirm": 0.003, "strong": 0.0003, "agree": 0.05}
ALPHA_FW = THRESH_STOCK["fw"]
ALPHA_CONFIRM = THRESH_STOCK["confirm"]
ALPHA_STRONG = THRESH_STOCK["strong"]
ALPHA_AGREE = THRESH_STOCK["agree"]
MIN_N_VERDICT = 150      # scored forecasts needed before any edge may be claimed
MIN_NEFF_VERDICT = 30    # effective (non-overlapping) observations needed
MIN_N_STRONG = 500
PSR_TRADABLE = 0.95
ROLLING_WINDOW = 250
BASELINE_LABELS = {"always_up": "一直猜漲", "prev_sign": "沿用前期漲跌方向", "train_majority": "訓練期多數方向"}
TEST_LABELS = {"direction": "方向準確率", "brier": "Brier機率預測", "ic": "排序相關IC", "mse": "報酬預測誤差"}


# ============================================================================= alignment guards
def hf_for(wf: WalkForward, h: int) -> HorizonForecasts:
    """``wf.horizons[h]`` tolerating int/str keys."""
    for key in (h, int(h), str(h)):
        if key in wf.horizons:
            return wf.horizons[key]
    raise ValueError(f"horizon {h} not available in the walk-forward (have {sorted(map(str, wf.horizons))})")


def align_data(data: PriceData, wf: WalkForward) -> PriceData:
    """``data`` truncated to the walk-forward's length; raises ``ValueError`` when the dates disagree."""
    wd = np.asarray(wf.dates).astype("datetime64[D]")
    m = len(wd)
    if m > data.n or not np.array_equal(wd, data.dates[:m]):
        raise ValueError("alignment error: walk-forward dates do not match the price data dates")
    return data if m == data.n else data.slice(m)


def check_realized_alignment(data: PriceData, hf: HorizonForecasts, tol: float = 1e-8) -> dict:
    """Compare ``hf.realized`` with ``ln(adjC[i+h]/adjC[i])`` recomputed from ``data``.

    Returns ``{"ok", "max_abs_diff", "n_checked", "n_unexpected"}``; ``n_unexpected`` counts rows where the
    walk-forward claims a realised value the data cannot provide (or the reverse for finite forecasts).
    """
    R = M.realized_log_returns(data.adjclose, hf.h)
    given = np.asarray(hf.realized, dtype=float)
    if len(given) != len(R):
        return {"ok": False, "max_abs_diff": float("inf"), "n_checked": 0, "n_unexpected": abs(len(given) - len(R))}
    both = np.isfinite(R) & np.isfinite(given)
    diff = float(np.max(np.abs(R[both] - given[both]))) if both.any() else 0.0
    unexpected = int(np.sum(np.isfinite(given) & ~np.isfinite(R)))
    return {"ok": diff <= tol and unexpected == 0, "max_abs_diff": diff, "n_checked": int(both.sum()),
            "n_unexpected": unexpected}


P_UP_EVENTS = ("up", "up_given_move")


def p_up_event(wf) -> str:
    """The event whose probability ``HorizonForecasts.p_up`` is, as declared in ``wf.config["p_up_event"]`` (module docstring).

    ``"up_given_move"``: ``P(R > 0 | R != 0)``, flat outcomes are not part of the event and are not scored.  ``"up"`` (also what
    an undeclared / unknown value means, i.e. the ``types.py`` contract ``P(R > 0)``): a flat outcome counts as *not up*.
    """
    cfg = getattr(wf, "config", None)
    ev = cfg.get("p_up_event") if isinstance(cfg, dict) else None
    return ev if ev in P_UP_EVENTS else "up"


# ============================================================================= data-quality guard
DATA_ALERT_MOVE = 0.25   # |close-to-close return| above this is impossible under the +-10 % limit -> bad data


def data_alerts(data: PriceData, first_idx: int, limit: int = 3) -> list:
    """Days after ``first_idx`` whose adjusted close-to-close move exceeds :data:`DATA_ALERT_MOVE`.

    Taiwan prices are limited to +-10 % per day, so such a move almost always means an unadjusted stock split / capital
    reduction (seen in real Yahoo data: 0050 on 2014-01-02, -75 %).  Returns ``[(date, return), ...]`` (largest first,
    at most ``limit``) -- the evaluation cannot repair the data, it only warns.
    """
    cc = S.leg_returns(data)[2]
    lo = max(int(first_idx) + 1, 1)
    idx = np.flatnonzero(np.abs(cc[lo:]) > DATA_ALERT_MOVE) + lo
    idx = idx[np.argsort(-np.abs(cc[idx]))][:limit]
    return [(M.date_str(data.dates[i]), float(cc[i])) for i in idx]


def data_alert_text(alerts: list, who: str = "") -> str:
    items = "、".join(f"{d} {r * 100:+.1f}%" for d, r in alerts)
    return (f"資料品質警示：{who}回測期間出現單日報酬超過 ±{DATA_ALERT_MOVE * 100:.0f}% 的異常資料（{items}），"
            f"已超出 ±10% 漲跌幅限制，很可能是股票分割或減資等未調整的資料錯誤；此回測結果不可靠，請先檢查資料來源。")


# ============================================================================= analysis core
def analyse(data: PriceData, wf: WalkForward, h: int, *, threshold: float = config.BUY_P_UP,
            exec_mode: str = "next_open", commission_discount: float = config.COMMISSION_DISCOUNT,
            slippage_bps: float = config.SLIPPAGE_BPS, sizing: str = "threshold", n_trials: int = 1,
            n_boot: int = 1000, with_strategy: bool = True) -> dict:
    """Compute every block of the report for horizon ``h``; returns the blocks plus ``evidence`` for the verdict."""
    hf = hf_for(wf, h)
    h = int(hf.h)
    data = align_data(data, wf)
    chk = check_realized_alignment(data, hf)
    if not chk["ok"]:
        raise ValueError(f"alignment error: realised returns differ from the price data "
                         f"(max |diff| = {chk['max_abs_diff']:.3g}, unexpected rows = {chk['n_unexpected']})")
    n = data.n
    for name in ("p_up", "mu", "sigma", "pit", "realized"):
        if len(getattr(hf, name)) != n:
            raise ValueError(f"alignment error: HorizonForecasts.{name} has {len(getattr(hf, name))} rows, "
                             f"price data has {n}")
    q_arr = np.asarray(hf.q, dtype=float)
    if q_arr.ndim != 2 or q_arr.shape != (n, len(wf.qlevels)):
        raise ValueError(f"alignment error: HorizonForecasts.q has shape {q_arr.shape}, expected "
                         f"({n}, {len(wf.qlevels)})")
    R = M.realized_log_returns(data.adjclose, h)
    p_all = np.clip(np.asarray(hf.p_up, dtype=float), 0.0, 1.0)
    mu_all, sg_all = np.asarray(hf.mu, dtype=float), np.asarray(hf.sigma, dtype=float)
    first = int(wf.first_idx)
    have_fc = np.isfinite(p_all) & np.isfinite(mu_all) & np.isfinite(sg_all) & (np.arange(n) >= first)
    E = np.flatnonzero(have_fc & np.isfinite(R))                      # evaluated forecasts
    n_pending = int(np.sum(have_fc & ~np.isfinite(R)))
    tie = R[E] == 0.0
    Sx = E[~tie]                                                        # scored (direction defined)
    out = {"h": h, "n_pending": n_pending, "alignment": chk, "first_idx": first, "n_rows": n,
           "data_alerts": data_alerts(data, first)}
    out["period"] = {"start": M.date_str(data.dates[E[0]]) if len(E) else None,
                     "end": M.date_str(data.dates[E[-1]]) if len(E) else None,
                     "n_forecasts": int(len(E)), "n_scored": int(len(Sx)), "n_tie_excluded": int(tie.sum())}
    # ---- causal baselines (independent of the walk-forward) ----
    clim = M.expanding_up_rate(R, h)                          # P(up | not flat): reference of the direction block
    clim_up = M.expanding_up_rate(R, h, ties="not_up")        # P(R > 0): reference of the probabilistic block
    prev_pred, prev_valid = M.trailing_sign_prediction(data.adjclose, h)
    up = R[Sx] > 0
    p = p_all[Sx]
    baselines = {"always_up": (np.ones(len(Sx), dtype=bool), np.ones(len(Sx), dtype=bool)),
                 "prev_sign": (prev_pred[Sx], prev_valid[Sx]),
                 "train_majority": (clim[Sx] >= 0.5, np.ones(len(Sx), dtype=bool))}
    out["direction"] = M.direction_summary(p, up, baselines, h=h)
    # ---- probabilistic block: score the event the forecaster declares (module docstring): P(R > 0 | R != 0) on the moving rows
    # only, or P(R > 0) on EVERY evaluated row (flat = not up).  References: CAUSAL expanding base rate of the same event
    # (headline, DESIGN.md 9(a)) and ORACLE test-sample base rate (reported separately, a hindsight bound that no online
    # forecaster can reach)
    event = p_up_event(wf)
    cond_move = event == "up_given_move"
    PR = Sx if cond_move else E                                         # rows on which p_up is scored
    pE, yE = p_all[PR], (R[PR] > 0).astype(float)
    ref_causal = (clim if cond_move else clim_up)[PR]
    ref_oracle = np.full(len(PR), float(yE.mean()) if len(PR) else 0.5)
    prob = M.probabilistic_summary(pE, yE, ref_causal)
    orc = M.probabilistic_summary(pE, yE, ref_oracle)

    def _skill(loss, ref):
        return (1.0 - loss / ref) if (loss is not None and ref is not None and ref > 0) else None

    prob.update(log_loss_skill=_skill(prob["log_loss"], prob["log_loss_baseline"]),
                brier_baseline_oracle=orc["brier_baseline"], brier_skill_oracle=orc["brier_skill"],
                log_loss_baseline_oracle=orc["log_loss_baseline"],
                log_loss_skill_oracle=_skill(orc["log_loss"], orc["log_loss_baseline"]),
                ece_noise_floor=(M.ece_noise_floor(len(PR), h, len(prob["calibration"])) if prob["calibration"] else None),
                p_up_event=event, n=int(len(PR)))
    out["probabilistic"] = prob
    out["p_up_event"] = event
    # ---- return-forecast skill ----
    Re, mu, sg = R[E], mu_all[E], sg_all[E]
    hist = M.expanding_mean_return(R, h)[E]
    hist = np.where(np.isfinite(hist), hist, 0.0)
    sse_zero, sse_hist = float(np.sum(Re ** 2)), float(np.sum((Re - hist) ** 2))
    naive, naive_kind = (np.zeros(len(E)), "zero") if sse_zero <= sse_hist else (hist, "hist_mean")
    ic, ic_p, _ = M.spearman_ic(mu, Re, h)
    dm_stat, dm_p = M.dm_test((Re - naive) ** 2, (Re - mu) ** 2, h)
    cov = M.coverage_summary(Re, np.asarray(hf.q, dtype=float)[E] if len(E) else np.zeros((0, len(wf.qlevels))),
                             wf.qlevels, h)
    pit = np.asarray(hf.pit, dtype=float)[E]
    q_e = np.asarray(hf.q, dtype=float)[E] if len(E) else np.zeros((0, len(wf.qlevels)))
    out["returns"] = {
        "ic": M.finite_or_none(ic), "ic_pvalue": M.finite_or_none(ic_p),
        "oos_r2": M.finite_or_none(M.oos_r2(Re, mu, naive)),
        "rmse_model": M.finite_or_none(math.sqrt(float(np.mean((Re - mu) ** 2)))) if len(E) else None,
        "rmse_naive": M.finite_or_none(math.sqrt(float(np.mean((Re - naive) ** 2)))) if len(E) else None,
        "dm_stat": M.finite_or_none(dm_stat), "dm_pvalue": M.finite_or_none(dm_p),
        "coverage": {k: M.finite_or_none(v) for k, v in cov["coverage"].items()},
        "kupiec_pvalue": M.finite_or_none(cov["kupiec_pvalue"]),
        "pit_hist": [M.finite_or_none(v) for v in M.pit_histogram(pit)],
        "pit_ks_pvalue": M.finite_or_none(M.pit_ks_pvalue(pit, h)),
        "pinball_loss": M.finite_or_none(M.pinball_loss(Re, q_e, wf.qlevels)) if len(E) else None}
    out["naive_kind"] = naive_kind
    # ---- rolling accuracy vs base rate ----
    m = len(Sx)
    window = ROLLING_WINDOW if m >= 400 else (max(30, m // 3) if m >= 90 else 0)
    roll = {"window": int(window), "dates": [], "accuracy": [], "base_rate": []}
    if window and window <= m:
        correct = ((p >= 0.5) == up).astype(float)
        acc_r, base_r = M.rolling_mean(correct, window), M.rolling_mean(up.astype(float), window)
        ix = M.downsample_indices(len(acc_r))
        roll.update(dates=[M.date_str(d) for d in data.dates[Sx][window - 1:][ix]],
                    accuracy=np.round(acc_r[ix], 5).tolist(), base_rate=np.round(base_r[ix], 5).tolist())
    out["rolling"] = roll
    # ---- strategy ----
    res = (S.run_strategy(data, hf, first, kind=data.kind, threshold=threshold, exec_mode=exec_mode,
                          commission_discount=commission_discount, slippage_bps=slippage_bps, sizing=sizing,
                          n_trials=n_trials, n_boot=n_boot) if with_strategy else None)
    out["strategy_result"] = res
    # ---- evidence for the verdict ----
    ev = {"n_scored": int(m), "n_eff": M.effective_n(m, h) if m else 0, "p": {}, "effect": {}}
    d = out["direction"]
    if d["p_value_vs_best_baseline"] is not None:
        ev["p"]["direction"] = d["p_value_vs_best_baseline"]
        ev["effect"]["direction"] = (d["accuracy"] - d["baselines"][d["best_baseline"]]) if d["best_baseline"] else None
    if len(PR) >= 10:
        # Brier leg: the model must beat BOTH the causal and the oracle base rate (the causal reference alone is a noisy
        # constant that any base-rate-aware forecaster beats; see the module docstring) -> the less favourable p-value
        sq_model = (pE - yE) ** 2
        dm_c = M.dm_test((ref_causal - yE) ** 2, sq_model, h)
        dm_o = M.dm_test((ref_oracle - yE) ** 2, sq_model, h)
        p_c, p_o = M.p_one_sided_positive(*dm_c), M.p_one_sided_positive(*dm_o)
        legs = [(pv, dm) for pv, dm in ((p_c, dm_c), (p_o, dm_o)) if math.isfinite(pv)]
        if legs:
            pv, dm = max(legs, key=lambda t: t[0])
            ev["p"]["brier"] = pv
            ev["brier_dm"] = dm
        ev["p_brier_causal"], ev["p_brier_oracle"] = p_c, p_o
        ev["effect"]["brier"] = out["probabilistic"]["brier_skill"]
    if math.isfinite(ic_p):
        ev["p"]["ic"] = M.p_one_sided_positive(ic, ic_p)
        ev["effect"]["ic"] = ic
    if math.isfinite(dm_p):
        ev["p"]["mse"] = M.p_one_sided_positive(dm_stat, dm_p)
        ev["effect"]["mse"] = out["returns"]["oos_r2"]
    ev["p"] = {k: v for k, v in ev["p"].items() if v is not None and math.isfinite(v)}
    if res is not None:
        ev["psr_vs_bh"], ev["diff_p"] = res.psr_vs_bh, res.diff_p
    out["evidence"] = ev
    return out


# ============================================================================= verdict logic
ADVERSE_P_HARD = 0.95   # one-sided "model better" p above this = the model is significantly WORSE than the free baseline
ADVERSE_P_SOFT = 0.50   # ... both probability legs above this = neither supports the model
PROB_LEGS = ("direction", "brier")


def adverse_legs(ps: dict) -> list:
    """Probability-based legs (``direction``, ``brier``) that contradict an edge claim (audit C-08).

    ``ps`` holds one-sided "model beats the baseline" p-values.  A leg is adverse when its p-value exceeds
    :data:`ADVERSE_P_HARD` (the model is significantly worse than the baseline); when both legs are available and both exceed
    :data:`ADVERSE_P_SOFT` neither probability test supports the model and both are listed.
    """
    have = [k for k in PROB_LEGS if k in ps]
    hard = [k for k in have if ps[k] > ADVERSE_P_HARD]
    if hard:
        return hard
    if len(have) == len(PROB_LEGS) and all(ps[k] > ADVERSE_P_SOFT for k in have):
        return list(have)
    return []


def assess_evidence(ev: dict, thr: dict | None = None) -> dict:
    """Map the evidence dict to ``{"level", "p_fw", "n_tests", "n_confirm", "n_strong", "n_agree", "tradable", "reason",
    "conflict"}``.

    ``thr`` = threshold set (:data:`THRESH_STOCK` by default, :data:`THRESH_UNIVERSE` for the universe backtest).
    ``conflict`` lists the probability legs that contradict the edge claim (:func:`adverse_legs`); when it is non-empty the
    level is capped at ``weak_edge`` (``reason = "conflicting_evidence"``): IC and MSE are both functions of the forecast mean,
    so two significant tests alone must not outvote direction / Brier evidence that the model is worse than the baseline.
    """
    thr = thr or THRESH_STOCK
    ps = {k: v for k, v in ev["p"].items() if v is not None and math.isfinite(v)}
    res = {"level": "no_edge", "p_fw": None, "n_tests": len(ps), "n_confirm": 0, "n_strong": 0, "n_agree": 0,
           "tradable": False, "reason": "ok", "conflict": []}
    psr, dp = ev.get("psr_vs_bh"), ev.get("diff_p")
    res["tradable"] = bool(psr is not None and dp is not None and psr >= PSR_TRADABLE and dp < 0.05)
    if not ps:
        res["reason"] = "no_tests"
        return res
    res["p_fw"] = min(1.0, len(ps) * min(ps.values()))
    res["n_confirm"] = sum(v < thr["confirm"] for v in ps.values())
    res["n_strong"] = sum(v < thr["strong"] for v in ps.values())
    res["n_agree"] = sum(v < thr["agree"] for v in ps.values())
    if ev["n_scored"] < MIN_N_VERDICT or ev["n_eff"] < MIN_NEFF_VERDICT:
        res["reason"] = "small_sample"
        return res
    if res["p_fw"] >= thr["fw"] or res["n_agree"] < 2:
        res["reason"] = "not_significant" if res["p_fw"] >= thr["fw"] else "not_consistent"
        return res
    level = "weak_edge"
    if res["n_confirm"] >= 2:
        level = "moderate_edge"
        if res["n_strong"] >= 3 and ev["n_scored"] >= MIN_N_STRONG and res["tradable"]:
            level = "strong_edge"
    res["conflict"] = adverse_legs(ps)
    if res["conflict"]:
        level = "weak_edge"
        res["reason"] = "conflicting_evidence"
    res["level"] = level
    return res


# ============================================================================= Chinese text helpers
def pct(x, nd: int = 1, signed: bool = False) -> str:
    """Fraction -> '53.6%' ('—' for None/NaN)."""
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x * 100:+.{nd}f}%" if signed else f"{x * 100:.{nd}f}%"


def num(x, nd: int = 2, signed: bool = False) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def pval(p) -> str:
    """'p<0.001' / 'p=0.004' / 'p=0.31'."""
    if p is None or not math.isfinite(p):
        return "p=—"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def horizon_label(h: int) -> str:
    return {1: "明日", 5: "未來5日", 20: "未來20日"}.get(int(h), f"未來{int(h)}日")


def exec_label(mode: str) -> str:
    return "隔日開盤價成交" if mode == "next_open" else "當日收盤價成交(較樂觀)"


LEVEL_TITLES = {
    "no_edge": "沒有統計上顯著的預測優勢",
    "weak_edge": "有微弱的預測訊號，但證據有限",
    "moderate_edge": "有一定的預測優勢，但需持續檢驗",
    "strong_edge": "歷史樣本外呈現顯著的預測優勢",
}
DISCLAIM = "過去的回測結果不代表未來，也不構成投資建議。"


def _strategy_bullet(sec: dict, exec_mode: str) -> str:
    st, bm = sec["stats"], sec["benchmark"]
    if not st.get("n_trades"):
        return (f"策略（{sec['rule']}）在回測期間沒有出現符合條件的買進訊號（扣除交易成本後預期報酬幾乎不會大於 0），"
                f"全程空手；買進持有的總報酬 {pct(bm['total_return'])}、夏普 {num(bm['sharpe'])}、"
                f"最大回撤 {pct(bm['max_dd'])}。")
    lo, hi = sec["sharpe_diff_ci95"]
    return (f"策略（{sec['rule']}；{exec_label(exec_mode)}，已扣成本）：總報酬 {pct(st['total_return'])}、"
            f"年化 {pct(st['cagr'])}、夏普 {num(st['sharpe'])}、最大回撤 {pct(st['max_dd'])}，共交易 {st['n_trades']} 次、"
            f"平均曝險 {pct(st['exposure'], 0)}；買進持有：總報酬 {pct(bm['total_return'])}、夏普 {num(bm['sharpe'])}、"
            f"最大回撤 {pct(bm['max_dd'])}。策略夏普減買進持有夏普的 95% 信賴區間為 [{num(lo)}, {num(hi)}]。")


def power_sentence(ev: dict, h: int) -> str:
    """Disclosure appended to every ``no_edge`` verdict: what the test could and could not have detected (audit S-04)."""
    x = M.min_detectable_ic(h, ev["n_scored"])
    if not math.isfinite(x):
        return ""
    return (f"這只代表在目前樣本量下沒有偵測到優勢，並不證明沒有優勢：以約 {ev['n_eff']} 筆獨立觀察估計，"
            f"有八成把握能偵測到的最小排序相關(IC)約為 {x:.2f}，更小的效果無法與雜訊區分。")


# Reading aid for the signature "only the mean legs (IC, MSE) are significant, the probability legs (direction, Brier) are not".
# Both mean legs are functions of the same forecast mean.  On the real h = 1 data the three stocks with this signature (2412, 3045,
# 4904) have tick sizes of about half a daily sigma: IC(model mean) 0.09 - 0.12 against IC(-r_t) 0.13 - 0.16, i.e. the model learnt
# the one-day reversal of the bid / ask bounce.
BOUNCE_HINT = ("這種只有排序與誤差檢驗顯著的組合，常見於價格最小跳動單位相對於每日波動偏大的個股：收盤價在買價與賣價之間來回跳動，"
               "使報酬出現短期反轉，模型抓到的可能是這種價格跳動，而不是可交易的方向優勢。")


def mean_legs_only(ev: dict, cut: float) -> bool:
    """True when every test with a one-sided p-value below ``cut`` is a forecast-mean leg (``ic`` / ``mse``) and there is one."""
    hit = [k for k, v in ev["p"].items() if v < cut]
    return bool(hit) and all(k in ("ic", "mse") for k in hit)


def make_verdict(a: dict, ass: dict, *, h: int, exec_mode: str, n_trials: int = 1) -> dict:
    """Plain-Chinese verdict ``{"level", "title", "text", "bullets"}`` quoting the concrete numbers."""
    d, pr, rt = a["direction"], a["probabilistic"], a["returns"]
    per, ev = a["period"], a["evidence"]
    res = a["strategy_result"]
    level = ass["level"]
    hl = horizon_label(h)
    best = d["best_baseline"]
    best_txt = (f"最佳簡單基準「{BASELINE_LABELS[best]}」為 {pct(d['baselines'][best])}"
                if best and d["baselines"].get(best) is not None else "無可用基準")
    ci = d["ci95"]
    head = (f"在 {per['start']} 至 {per['end']} 的 {per['n_scored']:,} 次樣本外{hl}預測中，模型方向準確率為 "
            f"{pct(d['accuracy'])}（95%信賴區間 {pct(ci[0])}–{pct(ci[1])}），{best_txt}，"
            f"單尾二項檢定 {pval(d['p_value_vs_best_baseline'])}；Brier 技能分數 {pct(pr['brier_skill'], 2, True)}"
            f"（對照當時已知的上漲比例；對照含事後資訊的全期上漲比例為 {pct(pr.get('brier_skill_oracle'), 2, True)}）、"
            f"預測排序相關(IC) {num(rt['ic'], 3, True)}"
            f"（雙尾 {pval(rt['ic_pvalue'])}，p 值未校準僅供參考）。") if per["n_scored"] else \
        "回測期間沒有可評估的樣本外預測。"
    names = {k: TEST_LABELS[k] for k in TEST_LABELS}
    thr = THRESH_STOCK
    cl = f"{thr['confirm'] * 100:g}%"
    sig = [f"{names[k]} {pval(v)}" for k, v in sorted(ev["p"].items(), key=lambda kv: kv[1]) if v < thr["confirm"]]
    if level == "no_edge":
        if ass["reason"] == "small_sample":
            tail = (f"有效樣本只有約 {ev['n_eff']} 筆獨立觀察（共 {ev['n_scored']} 筆，預測期間互相重疊），"
                    f"樣本太少，無法得出可靠結論，因此不宣稱模型有預測優勢。")
        elif ass["reason"] == "no_tests":
            tail = "可用於檢驗的資料不足，因此不宣稱模型有預測優勢。"
        elif ass["reason"] == "not_consistent":
            lone = [f"{names[k]} {pval(v)}" for k, v in ev["p"].items() if v < thr["agree"]]
            tail = (f"只有 1 項檢驗（{'、'.join(lone)}）達到顯著，其餘檢驗沒有同步支持，較可能是偶然現象，"
                    f"因此不宣稱模型有預測優勢。")
        else:
            tail = ("方向、機率、排序與報酬誤差等檢驗經多重檢定校正後（最小調整後 "
                    f"{pval(ass['p_fw'])}），都未達到「可確認模型優於簡單基準」的標準，"
                    "因此不能說這個模型有預測優勢。")
        if ass["reason"] != "no_tests":
            tail += power_sentence(ev, h)
        text = head + tail + DISCLAIM
    elif level == "weak_edge" and ass.get("conflict"):
        adv = "、".join(f"{names[k]} {pval(ev['p'][k])}" for k in ass["conflict"])
        worse = any(ev["p"][k] > ADVERSE_P_HARD for k in ass["conflict"])
        text = (head + f"在 {ass['n_tests']} 項預測檢驗中，{len(sig)} 項達到 {cl} 顯著（{'、'.join(sig) or '—'}），"
                f"但{adv}（單尾 p 越接近 1，代表模型越不優於簡單基準），顯示模型的方向與機率預測"
                f"{'比簡單基準更差' if worse else '沒有比簡單基準更好'}，兩類檢驗互相矛盾，因此最多只列為微弱訊號，"
                f"不宣稱有可用的預測優勢。排序與誤差檢驗顯著、方向與機率檢驗卻不顯著，可能是價格最小跳動單位相對於"
                f"每日波動偏大的個股常見的現象：收盤價在買價與賣價之間來回跳動，使報酬出現短期反轉，模型抓到的也許是這種"
                f"價格跳動，而不是可交易的方向優勢，扣成本後通常無法獲利。" + DISCLAIM)
    elif level == "weak_edge":
        text = (head + f"在 {ass['n_tests']} 項預測檢驗中，{len(sig)} 項達到 {cl} 顯著（{'、'.join(sig) or '—'}），"
                f"經多重檢定校正後的最小調整後 {pval(ass['p_fw'])}。這代表模型可能帶有微弱訊號，但效果小、"
                f"證據有限，且可能隨市場環境改變而消失；實際能否扣除成本後獲利仍須看下方策略結果。"
                + (BOUNCE_HINT if mean_legs_only(ev, thr["agree"]) else "") + DISCLAIM)
    elif level == "moderate_edge":
        trade = ("且扣除成本後的策略夏普明顯高於買進持有" if ass["tradable"]
                 else "但扣除交易成本後，策略尚未能穩定勝過買進持有")
        # IC and MSE are both functions of the forecast mean: when only those two confirm, they are not independent evidence
        mean_only = mean_legs_only(ev, thr["confirm"])
        agree = ("；排序相關與報酬誤差兩項檢驗同為預測均值的函數，方向與機率檢驗並未同步支持，證據強度不如多項獨立檢驗"
                 if mean_only else "，統計證據相當一致")
        text = (head + f"{len(sig)} 項檢驗同時達到 {cl} 顯著（{'、'.join(sig)}）{agree}，{trade}。"
                f"仍需持續追蹤，市場結構改變時優勢可能消失。" + (BOUNCE_HINT if mean_only else "") + DISCLAIM)
    else:
        text = (head + f"{len(sig)} 項檢驗達到 {cl} 顯著（{'、'.join(sig)}），且扣除成本後策略的夏普比率顯著高於買進持有"
                f"（PSR {pct(ev.get('psr_vs_bh'), 1)}）。這是歷史樣本外的結果；市場結構改變、模型失效與無法成交等"
                f"實務限制，仍可能使實際結果與回測不同，並不保證獲利。" + DISCLAIM)
    b = []
    if per["n_scored"]:
        bl = d["baselines"]
        b.append(f"方向：準確率 {pct(d['accuracy'])}（n={d['n']:,}）；基準 — 一直猜漲 {pct(bl['always_up'])}、"
                 f"沿用前期漲跌方向 {pct(bl['prev_sign'])}、訓練期多數方向 {pct(bl['train_majority'])}；"
                 f"對最佳基準 {pval(d['p_value_vs_best_baseline'])}，對丟銅板(50%) {pval(d['p_value_vs_coin'])}。")
        hc = d["high_conf"]
        if hc["n"]:
            b.append(f"高信心預測（機率≥{hc['threshold']:.0%} 或 ≤{1 - hc['threshold']:.0%}）共 {hc['n']} 次，"
                     f"準確率 {pct(hc['accuracy'])}（{pval(hc['p_value'])}）。")
        floor = pr.get("ece_noise_floor")
        b.append(f"機率品質：Brier {num(pr['brier'], 4)}，因果氣候基準（每個預測日只用當時已知的上漲比例）"
                 f"{num(pr['brier_baseline'], 4)}，技能分數 {pct(pr['brier_skill'], 2, True)}（正值才代表優於只猜當時已知的整體上漲比例）；"
                 f"事後基準（驗證期間整體上漲比例，含事後資訊、最嚴格的比較，僅供參考）{num(pr.get('brier_baseline_oracle'), 4)}，"
                 f"技能分數 {pct(pr.get('brier_skill_oracle'), 2, True)}；Brier 檢定對因果基準單尾 "
                 f"{pval(ev.get('p_brier_causal'))}、對事後基準單尾 {pval(ev.get('p_brier_oracle'))}（需同時優於兩者）；"
                 f"對數損失 {num(pr['log_loss'], 4)} 對 {num(pr['log_loss_baseline'], 4)}；校準誤差 ECE {pct(pr['ece'])}"
                 + (f"（純雜訊水準約 {pct(floor)}）。" if floor is not None else "。"))
        nk = "零預測" if a.get("naive_kind") == "zero" else "歷史平均預測"
        dm_dir = "" if rt["dm_stat"] is None else ("，模型誤差較小" if rt["dm_stat"] > 0 else "，模型誤差較大")
        b.append(f"報酬預測：與實際報酬的 Spearman 相關 IC {num(rt['ic'], 3, True)}（雙尾 {pval(rt['ic_pvalue'])}，"
                 f"以歷史價格推得的預測其虛無分配略偏負，p 值未校準僅供參考）；"
                 f"相對「{nk}」基準的樣本外 R² {pct(rt['oos_r2'], 2, True)}；Diebold-Mariano 檢定 雙尾 "
                 f"{pval(rt['dm_pvalue'])}{dm_dir}。")
        cv = rt["coverage"]
        b.append(f"區間校準：50%/80%/90% 預測區間實際涵蓋 {pct(cv.get('0.50'), 0)}/{pct(cv.get('0.80'), 0)}/"
                 f"{pct(cv.get('0.90'), 0)}（理想為 50%/80%/90%），Kupiec 檢定 {pval(rt['kupiec_pvalue'])}；"
                 f"PIT 均勻性 KS 檢定 {pval(rt['pit_ks_pvalue'])}。")
    if res is not None:
        b.append(_strategy_bullet(res.section, exec_mode))
        if res.psr_vs_bh is not None:
            b.append(f"策略夏普高於買進持有夏普的機率(PSR) {pct(res.psr_vs_bh, 1)}，自助法單尾 {pval(res.diff_p)}"
                     + (f"；已用 {n_trials} 次試驗做選擇偏誤修正(DSR)。" if n_trials > 1 else "。"))
    if h > 1:
        b.append(f"{hl}預測的觀察值彼此重疊，p 值與信賴區間已依有效樣本數(約 n/{h})做保守調整。")
    floor = pr.get("ece_noise_floor")
    if pr["ece"] is not None and pr["ece"] > max(0.05, floor if floor is not None else 0.0):
        b.append(f"提醒：機率校準偏差偏大（ECE {pct(pr['ece'])}，已高於純雜訊水準約 {pct(floor)}），預測機率不宜照字面解讀。")
    c90 = rt["coverage"].get("0.90")
    if c90 is not None and abs(c90 - 0.90) > 0.05:
        b.append(f"提醒：90% 區間實際涵蓋 {pct(c90, 0)}，與理想值差距較大，風險區間可能{'偏窄' if c90 < 0.9 else '偏寬'}。")
    if ev["n_scored"] < MIN_N_VERDICT:
        b.append(f"提醒：僅 {ev['n_scored']} 筆樣本，結論可信度低。")
    if a.get("data_alerts"):
        b.insert(0, data_alert_text(a["data_alerts"]))
    return {"level": level, "title": LEVEL_TITLES[level], "text": text, "bullets": b}


def make_notes(a: dict, *, h: int, exec_mode: str, commission_discount: float, slippage_bps: float, kind: str,
               rt_cost: float) -> list:
    tax = config.TAX_ETF if kind == "etf" else config.TAX_STOCK
    notes = [
        f"已計入手續費（牌告 {config.COMMISSION_RATE * 100:.4f}% × {commission_discount * 10:g} 折，買賣各一次）、"
        f"證交稅 {tax * 100:.1f}%（賣出）與滑價（每邊 {slippage_bps:g} bp），來回交易成本約 {rt_cost * 100:.2f}%。",
        f"執行假設：{exec_label(exec_mode)}；訊號只使用當日收盤前已知的資料，最早於下一個交易日才產生報酬。",
        "未計入：漲跌停鎖死而無法成交、最低手續費、市場衝擊、融資融券與現金利息、期末持倉的賣出成本。",
        "買進持有基準不扣交易成本（較嚴格的比較標準）；報酬以還原權息價（含股利）計算。",
        "所有預測皆為前推式(walk-forward)樣本外預測：每一筆只使用當時已知的資料，包含參數估計與校準。",
        f"Brier 與對數損失的基準為「因果氣候基準」：每個預測日只用當時已知（已到期）的歷史上漲比例；"
        f"驗證期間整體上漲比例屬事後資訊（事後基準，是任何即時預測都做不到的最嚴格比較），另列於「事後基準」欄位且不作為主要數字；"
        f"判定是否具預測優勢的 Brier 檢定則同時對兩者比較，採對模型較不利者（需同時優於兩者）。"
        f"報酬預測的對照基準為{'零預測' if a.get('naive_kind') == 'zero' else '歷史平均預測'}（取誤差較小者）。",
        "此標的是您目前挑選的個股；若是因其近期表現亮眼才挑選，結果會有選擇偏誤，多檔回測請看整體驗證。",
        "滾動準確率圖中的「基準」為同一期間實際上漲的比例（也就是一直猜漲的準確率）。",
    ]
    if a.get("data_alerts"):
        notes.insert(0, data_alert_text(a["data_alerts"]))
    per = a["period"]
    if per["n_tie_excluded"]:
        rate = per["n_tie_excluded"] / per["n_forecasts"] if per["n_forecasts"] else float("nan")
        if a.get("p_up_event") == "up_given_move":
            notes.append(f"平盤（報酬恰為 0）共 {per['n_tie_excluded']} 筆（{pct(rate)}）：平盤既不是漲也不是跌，方向準確率、基準比較與機率評分"
                         f"（Brier、對數損失、校準）都不計入，因為模型預測的是「價格有變動時上漲的機率」P(報酬>0｜報酬≠0)。")
        else:
            notes.append(f"平盤（報酬恰為 0）共 {per['n_tie_excluded']} 筆（{pct(rate)}）：方向準確率與基準比較不計入；"
                         f"機率評分（Brier、對數損失、校準）把平盤視為「未上漲」，與模型預測的目標 P(報酬>0) 一致。")
    if a["n_pending"]:
        notes.append(f"最後 {a['n_pending']} 筆預測尚未到期，不列入績效統計。")
    if h > 1:
        notes.append(f"{h} 日預測彼此重疊：二項/Spearman/Kupiec/KS 檢定採有效樣本數 n/{h}，Diebold-Mariano 使用 "
                     f"Newey-West 異質變異數自相關穩健(HAC)變異數（落後 {h - 1} 期）。")
    ev = a["evidence"]
    x = M.min_detectable_ic(h, ev["n_scored"])
    if math.isfinite(x):
        notes.append(f"檢定力：以約 {ev['n_eff']} 筆獨立觀察估計，本驗證有八成把握能偵測到的最小排序相關(IC)約為 {x:.2f}；"
                     f"更小的效果無法與雜訊區分，因此「沒有偵測到優勢」不等於「沒有優勢」。")
    return notes


# ============================================================================= public API
def build_backtest_report(data: PriceData, wf: WalkForward, horizon: int, *, threshold: float = config.BUY_P_UP,
                          exec_mode: str = "next_open", commission_discount: float = config.COMMISSION_DISCOUNT,
                          slippage_bps: float = config.SLIPPAGE_BPS, sizing: str = "threshold", n_trials: int = 1,
                          n_boot: int = 1000) -> dict:
    """Build the DESIGN.md 6.4 ``report`` object (strictly JSON-serialisable: NaN/inf -> null).

    Raises ``ValueError`` for invalid parameters or when the walk-forward is not aligned with ``data``
    (realised returns recomputed from the adjusted closes must equal ``HorizonForecasts.realized``).
    """
    if exec_mode not in S.EXEC_MODES:
        raise ValueError(f"exec_mode must be one of {S.EXEC_MODES}, got {exec_mode!r}")
    if sizing not in S.SIZINGS:
        raise ValueError(f"sizing must be one of {S.SIZINGS}, got {sizing!r}")
    if not (0.0 < float(threshold) < 1.0):
        raise ValueError("threshold must lie in (0, 1)")
    if int(n_trials) < 1:
        raise ValueError("n_trials must be >= 1")
    a = analyse(data, wf, int(horizon), threshold=threshold, exec_mode=exec_mode,
                commission_discount=commission_discount, slippage_bps=slippage_bps, sizing=sizing,
                n_trials=int(n_trials), n_boot=n_boot)
    h = a["h"]
    kind = S.cost_kind(data.kind)
    rt_cost = config.roundtrip_cost(kind, commission_discount, slippage_bps)
    ass = assess_evidence(a["evidence"])
    verdict = make_verdict(a, ass, h=h, exec_mode=exec_mode, n_trials=int(n_trials))
    report = {
        "symbol": data.symbol, "name": data.name, "kind": data.kind, "horizon": int(h),
        "params": {"threshold": float(threshold), "exec": exec_mode, "commission_discount": float(commission_discount),
                   "slippage_bps": float(slippage_bps), "sizing": sizing, "n_trials": int(n_trials),
                   "roundtrip_cost": rt_cost, "min_train": int(wf.min_train), "refit_every": int(wf.refit_every)},
        "period": a["period"], "direction": a["direction"], "probabilistic": a["probabilistic"],
        "returns": a["returns"], "strategy": a["strategy_result"].section, "rolling": a["rolling"],
        "verdict": verdict,
        "notes": make_notes(a, h=h, exec_mode=exec_mode, commission_discount=commission_discount,
                            slippage_bps=slippage_bps, kind=kind, rt_cost=rt_cost),
    }
    return M.json_safe(report)


# ----------------------------------------------------------------------------- schema validation
_STATS_SPEC = {"total_return": "num?", "cagr": "num?", "vol": "num?", "sharpe": "num?", "sortino": "num?",
               "max_dd": "num?", "calmar": "num?", "exposure": "num?", "turnover_per_year": "num?", "n_trades": "int",
               "trade_hit_rate": "num?", "psr": "num?", "dsr": "num?", "sharpe_ci95": ("list", "num?", 2)}
_BASELINES = ("always_up", "prev_sign", "train_majority")
REPORT_SPEC = {
    "symbol": "str", "name": "str", "kind": "str", "horizon": "int",
    "params": {"threshold": "num", "exec": ("enum", S.EXEC_MODES), "commission_discount": "num",
               "slippage_bps": "num", "sizing": ("enum", S.SIZINGS), "n_trials": "int", "roundtrip_cost": "num",
               "min_train": "int", "refit_every": "int"},
    "period": {"start": "str?", "end": "str?", "n_forecasts": "int", "n_scored": "int", "n_tie_excluded": "int"},
    "direction": {
        "accuracy": "num?", "ci95": ("list", "num?", 2), "n": "int", "base_rate_up": "num?",
        "baselines": {k: "num?" for k in _BASELINES}, "best_baseline": ("enum?", _BASELINES),
        "p_value_vs_best_baseline": "num?", "p_value_vs_coin": "num?",
        "by_confidence": ("list", {"bucket": "str", "lo": "num", "hi": "num", "n": "int", "accuracy": "num?",
                                   "mean_p_up": "num?", "actual_up_rate": "num?"}),
        "high_conf": {"threshold": "num", "n": "int", "accuracy": "num?", "p_value": "num?"}},
    # brier_baseline / brier_skill / log_loss_baseline use the CAUSAL expanding base rate (DESIGN.md 9(a)); the optional keys
    # (absent in reports of older versions) add the log-loss skill, the ORACLE (test-sample base rate) references and the ECE
    # level an exactly calibrated forecaster reaches by binning noise
    "probabilistic": {"brier": "num?", "brier_baseline": "num?", "brier_skill": "num?", "log_loss": "num?",
                      "log_loss_baseline": "num?", "ece": "num?",
                      "calibration": ("list", {"p_mean": "num", "freq": "num", "n": "int"}),
                      "log_loss_skill?": "num?", "brier_baseline_oracle?": "num?", "brier_skill_oracle?": "num?",
                      "log_loss_baseline_oracle?": "num?", "log_loss_skill_oracle?": "num?", "ece_noise_floor?": "num?",
                      "p_up_event?": ("enum", P_UP_EVENTS), "n?": "int"},
    "returns": {"ic": "num?", "ic_pvalue": "num?", "oos_r2": "num?", "rmse_model": "num?", "rmse_naive": "num?",
                "dm_stat": "num?", "dm_pvalue": "num?",
                "coverage": {"0.50": "num?", "0.80": "num?", "0.90": "num?"}, "kupiec_pvalue": "num?",
                "pit_hist": ("list", "num?", 10), "pit_ks_pvalue": "num?", "pinball_loss": "num?"},
    "strategy": {"rule": "str", "stats": _STATS_SPEC, "benchmark": _STATS_SPEC,
                 "sharpe_diff_ci95": ("list", "num?", 2),
                 "curve": {"dates": ("list", "str"), "strategy": ("list", "num?"), "buyhold": ("list", "num?"),
                           "position": ("list", "num?"), "drawdown_strategy": ("list", "num?"),
                           "drawdown_buyhold": ("list", "num?")}},
    "rolling": {"window": "int", "dates": ("list", "str"), "accuracy": ("list", "num?"),
                "base_rate": ("list", "num?")},
    "verdict": {"level": ("enum", LEVELS), "title": "str", "text": "str", "bullets": ("list", "str")},
    "notes": ("list", "str"),
}


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_schema(spec, val, path: str, probs: list) -> None:
    """Recursive checker for the mini schema language used by :data:`REPORT_SPEC` (appends problems)."""
    if isinstance(spec, dict):
        if not isinstance(val, dict):
            probs.append(f"{path}: expected object, got {type(val).__name__}")
            return
        for k, sub in spec.items():
            optional_key = k.endswith("?")          # "name?": the key may be absent (additive contract extensions)
            name = k[:-1] if optional_key else k
            if name not in val:
                if not optional_key:
                    probs.append(f"{path}.{name}: missing key" if path else f"{name}: missing key")
            else:
                check_schema(sub, val[name], f"{path}.{name}" if path else name, probs)
        return
    if isinstance(spec, str):
        optional = spec.endswith("?")
        kind = spec.rstrip("?")
        if val is None:
            if not optional:
                probs.append(f"{path}: null not allowed")
            return
        if kind == "num" and not _is_num(val):
            probs.append(f"{path}: expected finite number, got {val!r}")
        elif kind == "int" and (not isinstance(val, int) or isinstance(val, bool)):
            probs.append(f"{path}: expected integer, got {val!r}")
        elif kind == "str" and not isinstance(val, str):
            probs.append(f"{path}: expected string, got {type(val).__name__}")
        elif kind == "bool" and not isinstance(val, bool):
            probs.append(f"{path}: expected bool, got {type(val).__name__}")
        return
    tag = spec[0]
    if tag == "opt":                       # null, or a value matching spec[1]
        if val is not None:
            check_schema(spec[1], val, path, probs)
        return
    if tag in ("enum", "enum?"):
        if val is None:
            if tag == "enum":
                probs.append(f"{path}: null not allowed")
        elif val not in spec[1]:
            probs.append(f"{path}: {val!r} not in {list(spec[1])}")
        return
    if tag == "list":
        if not isinstance(val, list):
            probs.append(f"{path}: expected list, got {type(val).__name__}")
            return
        if len(spec) > 2 and spec[2] is not None and len(val) != spec[2]:
            probs.append(f"{path}: expected {spec[2]} items, got {len(val)}")
        for i, item in enumerate(val):
            check_schema(spec[1], item, f"{path}[{i}]", probs)
        return
    probs.append(f"{path}: bad schema spec {spec!r}")


def validate_backtest_report(report: dict) -> list:
    """Schema problems of a backtest report (``[]`` when it satisfies DESIGN.md 6.4).

    Checks required keys, types (finite numbers or null), enums, fixed lengths (``pit_hist`` = 10, CI pairs),
    equal curve lengths, the 600-point cap, and strict JSON serialisability (no NaN/inf).
    """
    probs: list = []
    if not isinstance(report, dict):
        return [f"report: expected object, got {type(report).__name__}"]
    check_schema(REPORT_SPEC, report, "", probs)
    cur = (report.get("strategy") or {}).get("curve") if isinstance(report.get("strategy"), dict) else None
    if isinstance(cur, dict) and all(isinstance(cur.get(k), list) for k in REPORT_SPEC["strategy"]["curve"]):
        lens = {k: len(cur[k]) for k in REPORT_SPEC["strategy"]["curve"]}
        if len(set(lens.values())) != 1:
            probs.append(f"strategy.curve: arrays differ in length {lens}")
        elif lens["dates"] > M.MAX_CURVE_POINTS:
            probs.append(f"strategy.curve: {lens['dates']} points exceeds {M.MAX_CURVE_POINTS}")
    roll = report.get("rolling")
    if isinstance(roll, dict) and all(isinstance(roll.get(k), list) for k in ("dates", "accuracy", "base_rate")):
        lens = {k: len(roll[k]) for k in ("dates", "accuracy", "base_rate")}
        if len(set(lens.values())) != 1:
            probs.append(f"rolling: arrays differ in length {lens}")
        elif lens["dates"] > M.MAX_CURVE_POINTS:
            probs.append(f"rolling: {lens['dates']} points exceeds {M.MAX_CURVE_POINTS}")
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as exc:
        probs.append(f"report: not strictly JSON serialisable ({exc})")
    return probs
