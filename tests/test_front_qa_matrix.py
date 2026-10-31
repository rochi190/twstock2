"""Frontend QA matrix: EVERY view x {full, nulls, sparse, computing, error, abort-on-route-change}.

Each cell boots the real page (index.html + all scripts) in V8 with a fake DOM and a declarative fetch mock, then asserts the invariants that
matter for robustness: no exception / console.error, no "NaN" / "undefined" / "[object" / "Infinity" / "null" in the rendered text, valid ids and
aria references, labelled controls and canvases, exactly one <h1>.  The "nulls" scenario fuzzes REAL backend payloads (seeded: random leaves become
null, keys vanish, lists get truncated); "abort" leaves the page while requests are in flight and checks that nothing from the old view lands in the
new one and that no further requests / timers are started.
"""
from __future__ import annotations

import copy
import json
import random
import re
import unittest

from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp, audit, bad_text_hits, default_rules, fixture

OK = lambda body: {"status": 200, "body": body}
ERR = lambda status, code, msg="x": {"status": status, "body": {"status": "error", "error": {"code": code, "message": msg}}}
COMPUTING = lambda *a: OK({"status": "computing", "progress": {"done": a[0], "total": a[1], "message": a[2]}}) if a else OK({"status": "computing"})

VIEW_HASH = {"stock": "#/stock/2330", "picks": "#/picks", "backtest": "#/backtest/2330?h=1&thr=0.55", "method": "#/method"}
ENDPOINTS = {
    "stock": [r"^/api/forecast\?", r"^/api/history\?", r"^/api/backtest\?", r"^/api/status"],
    "picks": [r"^/api/recommendations\?", r"^/api/status"],
    "backtest": [r"^/api/backtest\?"],
    "method": [],
}
SETTLE = {"stock": 9000, "picks": 4000, "backtest": 4000, "method": 400}


# --------------------------------------------------------------------------------------------- payload helpers
def fuzz(x, rng, p=0.28, keep=("status",)):
    """Seeded fuzz of a JSON payload: leaves -> None, keys dropped, lists truncated.  Envelope `status` keys are kept."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in keep and isinstance(v, str):
                out[k] = v
                continue
            r = rng.random()
            if r < p / 4:
                continue                                  # key vanishes
            out[k] = fuzz(v, rng, p, keep)
        return out
    if isinstance(x, list):
        if x and rng.random() < p / 5:
            x = x[: rng.randrange(0, len(x))]
        return [fuzz(v, rng, p, keep) for v in x]
    if rng.random() < p:
        return None
    return x


def real(name):
    return fixture(f"real/{name}.json")


def payload_rules(view, bodies):
    """Rules answering every endpoint of `view` listed in `bodies` ({regex: body}) and the rest from the real fixtures."""
    return [{"re": r, "resp": OK(b) if not (isinstance(b, dict) and "status" in b and "body" in b) else b} for r, b in bodies.items()]


def boot(view, rules=(), settle=None, hash_=None, viewport=1280, delay=0, **kw):
    app = FrontApp(hash=hash_ or VIEW_HASH[view], viewport=viewport, boot=False, delay=delay, **kw)
    app.set_rules(list(rules) + default_rules(True))
    app.boot()
    app.settle(SETTLE[view] if settle is None else settle)
    return app


LABEL_FOR_JS = """(function () {
  var bad = [], ls = document.querySelectorAll('label');
  for (var i = 0; i < ls.length; i++) { var f = ls[i].getAttribute('for'); if (f && !document.getElementById(f)) bad.push(f); }
  return bad;
})()"""


class Base(unittest.TestCase):
    def invariants(self, app, what, expect_errors=False, h1=1):
        text = app.text("#view")
        hits = bad_text_hits(text)
        self.assertEqual(hits, [], f"{what}: bad token {hits} in rendered text")
        artefacts = re.findall(r"\d\.\d{7,}", text)
        self.assertEqual(artefacts, [], f"{what}: unformatted floating-point number in rendered text")
        if not expect_errors:
            self.assertEqual(app.errors(), [], f"{what}: console.error")
        a = audit(app)
        for key in ("dupIds", "badRefs", "unnamedButtons", "unlabelledCanvas", "unlabelledInputs", "emptyLinks"):
            self.assertEqual(a[key], [], f"{what}: {key}")
        self.assertEqual(a["h1"], h1, f"{what}: <h1> count")
        self.assertEqual(app.json(LABEL_FOR_JS), [], f"{what}: <label for> without target")
        self.assertGreater(len(text), 60, f"{what}: page is (almost) empty")
        return text


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class FullPayloads(Base):
    def test_every_view_renders_real_data_cleanly(self):
        for view in VIEW_HASH:
            with self.subTest(view=view):
                app = boot(view)
                self.invariants(app, f"{view}/full")
                self.assertEqual(app.count(f".view-{view}"), 1)

    def test_all_views_and_all_horizons_in_one_session(self):
        app = FrontApp(hash="#/stock/2330", boot=False)
        app.set_rules(default_rules(True))
        app.boot()
        app.settle(6000)
        for hash_, view in (("#/picks", "picks"), ("#/backtest/2330?h=5", "backtest"), ("#/method", "method"), ("#/stock/0050", "stock"), ("#/backtest/2330?h=1", "backtest")):
            app.goto(hash_, 3500)
            with self.subTest(hash=hash_):
                self.invariants(app, hash_)
                self.assertEqual(app.count(f".view-{view}"), 1)
                self.assertEqual(app.count(".view-host"), 1)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class NullsFuzzed(Base):
    BODIES = {
        "stock": lambda rng: {r"^/api/forecast\?": fuzz(real("forecast_2330"), rng), r"^/api/history\?": fuzz(real("history_2330"), rng),
                              r"^/api/backtest\?": fuzz(real("backtest_2330_h1"), rng), r"^/api/status": fuzz(real("status"), rng)},
        "picks": lambda rng: {r"^/api/recommendations\?": fuzz(real("recommendations_h5"), rng), r"^/api/status": fuzz(real("status"), rng)},
        "backtest": lambda rng: {r"^/api/backtest\?": fuzz(real("backtest_2330_h1"), rng)},
        "method": lambda rng: {},
    }

    def test_seeded_fuzz_of_real_payloads(self):
        for view in ("stock", "picks", "backtest"):
            for seed in range(1, 9):
                with self.subTest(view=view, seed=seed):
                    rng = random.Random(seed * 7919 + len(view))
                    bodies = self.BODIES[view](rng)
                    app = boot(view, payload_rules(view, bodies), settle=SETTLE[view] + 3000)
                    self.invariants(app, f"{view}/nulls/seed{seed}", expect_errors=False)

    def test_backtest_report_with_every_block_fuzzed_harder(self):
        for seed in range(100, 112):
            rng = random.Random(seed)
            body = fuzz(real("backtest_2330_h5"), rng, p=0.5)
            with self.subTest(seed=seed):
                app = boot("backtest", payload_rules("backtest", {r"^/api/backtest\?": body}))
                self.invariants(app, f"backtest/hard-nulls/{seed}")

    def test_picks_universe_panel_with_fuzzed_report(self):
        for seed in range(1, 6):
            rng = random.Random(seed)
            bodies = {r"^/api/backtest/universe": fuzz(real("backtest_universe_h5"), rng)}
            app = boot("picks", payload_rules("picks", bodies), settle=3000)
            if app.count(".uv-card .collapse-btn"):
                app.click(".uv-card .collapse-btn")
                if app.count(".uv-card .btn-primary"):
                    app.click(".uv-card .btn-primary")
                app.settle(3000)
            with self.subTest(seed=seed):
                self.invariants(app, f"picks/universe/nulls/{seed}")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class Sparse(Base):
    def test_minimal_envelopes(self):
        minimal = {
            "stock": [{r"^/api/forecast\?": {"status": "ready"}, r"^/api/history\?": {"status": "ready"}, r"^/api/backtest\?": {"status": "ready"}, r"^/api/status": {"status": "ready"}},
                      {r"^/api/forecast\?": fixture("forecast_sparse.json"), r"^/api/history\?": {"status": "ready", "bars": []}},
                      {r"^/api/forecast\?": {"status": "ready", "horizons": {}, "fan": {}}, r"^/api/history\?": {"status": "ready", "bars": [{}, None, {"d": "2026-01-01"}]}}],
            "picks": [{r"^/api/recommendations\?": {"status": "ready"}}, {r"^/api/recommendations\?": fixture("recommendations_empty.json")},
                      {r"^/api/recommendations\?": {"status": "ready", "buys": [None, {}, {"code": 5}], "all": "x", "avoids": {}, "portfolio": [], "market": None}}],
            "backtest": [{r"^/api/backtest\?": {"status": "ready", "report": {}}}, {r"^/api/backtest\?": {"status": "ready", "report": {"symbol": "2330", "verdict": {}}}},
                         {r"^/api/backtest\?": {"status": "ready", "report": {"direction": {}, "probabilistic": {}, "returns": {}, "strategy": {}, "rolling": {}, "params": {}, "period": {}}}},
                         {r"^/api/backtest\?": {"status": "ready", "report": {"strategy": {"curve": {"dates": ["2026-01-01"], "strategy": [1], "buyhold": [1]}, "stats": None, "benchmark": []}}}}],
            "method": [{}],
        }
        for view, variants in minimal.items():
            for i, bodies in enumerate(variants):
                with self.subTest(view=view, variant=i):
                    app = boot(view, payload_rules(view, bodies), settle=SETTLE[view] + 2000)
                    # a sparse payload may legitimately show an error card or an empty state, but never garbage text or exceptions
                    self.invariants(app, f"{view}/sparse/{i}")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class Computing(Base):
    def test_progress_is_shown_and_never_crashes(self):
        variants = {
            "with_total": COMPUTING(30, 1250, "前推式預測 30/1250"),
            "no_total": COMPUTING(0, 0, "排隊中…"),
            "no_progress": COMPUTING(),
            "garbage_progress": OK({"status": "computing", "progress": {"done": "x", "total": None, "message": 5}}),
            "overflow": COMPUTING(5000, 1250, "超出"),
            "negative": COMPUTING(-5, 10, "負數"),
        }
        for view in ("stock", "picks", "backtest"):
            for name, resp in variants.items():
                with self.subTest(view=view, progress=name):
                    rules = [{"re": r, "resp": resp} for r in ENDPOINTS[view] if "status" not in r]
                    app = boot(view, rules, settle=5000)
                    self.invariants(app, f"{view}/computing/{name}")
                    self.assertGreaterEqual(app.count(".progress, .skeleton"), 1, "a long computation must show progress or a skeleton")

    def test_computing_then_ready(self):
        ready = {"status": 200, "body": "real_recommendations_h5"}
        seq = [COMPUTING(1, 60, "a"), COMPUTING(30, 60, "b"), ready]
        app = boot("picks", [{"re": r"^/api/recommendations\?", "seq": seq, "resp": ready}], settle=6000)
        self.invariants(app, "picks/computing->ready")
        self.assertGreaterEqual(app.count(".card"), 4)


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class Errors(Base):
    KINDS = {
        "internal": ERR(500, "internal", "伺服器發生錯誤"),
        "unknown_symbol": ERR(404, "unknown_symbol", "找不到代號"),
        "insufficient": ERR(422, "insufficient_history", "歷史資料不足"),
        "bad_request": ERR(400, "bad_request", "參數錯誤"),
        "unavailable": ERR(503, "data_unavailable", "資料暫時無法取得"),
        "html_500": {"status": 500, "invalidJson": True, "body": None},
        "null_body": OK(None),
        "string_body": OK("oops"),
        "array_body": OK([1, 2]),
        "envelope_without_error": {"status": 200, "body": {"status": "error"}},
        "network": {"networkError": True},
    }

    def test_every_error_kind_in_every_data_view(self):
        for view in ("stock", "picks", "backtest"):
            for name, resp in self.KINDS.items():
                with self.subTest(view=view, error=name):
                    rules = [{"re": r, "resp": resp} for r in ENDPOINTS[view] if "status" not in r]
                    app = boot(view, rules, settle=22000 if name == "network" else 7000)
                    self.invariants(app, f"{view}/error/{name}", expect_errors=False)
                    self.assertGreaterEqual(app.count(".error-box, .banner-warn, .banner-error"), 1, "an error must be visible to the user")

    def test_error_then_retry_recovers_in_every_data_view(self):
        for view, ready_name in (("picks", "real_recommendations_h5"), ("backtest", "real_backtest_2330_h1")):
            with self.subTest(view=view):
                ready = {"status": 200, "body": ready_name}
                re_ = ENDPOINTS[view][0]
                app = boot(view, [{"re": re_, "seq": [ERR(500, "internal"), ready], "resp": ready}], settle=3000)
                self.assertGreaterEqual(app.count(".error-box"), 1)
                app.click(".error-box .btn")
                app.settle(4000)
                self.assertEqual(app.count(".error-box"), 0)
                self.invariants(app, f"{view}/retry")

    def test_unknown_stock_is_a_friendly_page(self):
        app = boot("stock", hash_="#/stock/9998", settle=3000)
        self.assertIn("找不到這檔股票", app.text("#view"))
        self.invariants(app, "stock/9998")


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class AbortOnRouteChange(Base):
    def test_leaving_while_requests_are_in_flight(self):
        for view in ("stock", "picks", "backtest"):
            for leave_after in (50, 700, 2500):
                with self.subTest(view=view, leave_after=leave_after):
                    app = boot(view, [], settle=leave_after, delay=900)
                    app.goto("#/method", 100)
                    n = len(app.fetched())
                    app.settle(12000)
                    self.assertEqual(len(app.fetched()), n, "the old view must not start requests after it was left")
                    self.assertEqual(app.count(".view-stock, .view-picks, .view-backtest"), 0, "nothing of the old view may reappear")
                    self.assertEqual(app.count(".view-method"), 1)
                    self.invariants(app, f"{view}/abort@{leave_after}")
                    self.assertLessEqual(app.pending_timers(), 4, "timers left behind by the old view")

    def test_route_hopping_does_not_leak_charts_or_timers(self):
        app = FrontApp(hash="#/stock/2330", boot=False, delay=200)
        app.set_rules(default_rules(True))
        app.boot()
        for hash_ in ("#/backtest/2330?h=1", "#/picks", "#/stock/0050", "#/backtest/2330?h=5", "#/method", "#/stock/2330", "#/backtest/2330?h=1"):
            app.goto(hash_, 700)
        app.goto("#/method", 100)
        app.settle(15000)
        self.assertEqual(app.errors(), [])
        self.assertEqual(app.json("TQ.charts.mountedCount()"), 0, "every chart of every left view must be destroyed")
        self.assertLessEqual(app.pending_timers(), 4)
        self.invariants(app, "hopping")

    def test_back_and_forward_between_views(self):
        app = FrontApp(hash="#/backtest/2330?h=1", boot=False)
        app.set_rules(default_rules(True))
        app.boot()
        app.settle(4000)
        app.goto("#/method", 300)
        app.goto("#/backtest/2330?h=1", 4000)          # coming back: re-mounted, re-run from the cache, no stale state
        self.invariants(app, "back")
        self.assertEqual(app.count(".bt-head-card"), 1)


if __name__ == "__main__":
    unittest.main()
