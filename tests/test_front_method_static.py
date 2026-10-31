"""The numbers the 方法說明 page states must be the numbers the backend uses.  The backend sources are parsed with ``ast`` (no numpy /
scipy import), the page's constants are read from V8.  Also: stated facts that can be checked arithmetically."""
from __future__ import annotations

import ast
import math
import re
import sys
import unittest
from pathlib import Path

from front_helpers import HAVE_RACER, SKIP_MSG, FrontApp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import config  # noqa: E402  (pure python, imports only pathlib)


def module_assign(path: Path, name: str):
    """Literal value of a top-level ``name = <literal>`` (or annotated) assignment, via ast."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name and node.value is not None:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def function_default(path: Path, func: str, arg: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            args = node.args.args
            defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
            for a, d in zip(args, defaults):
                if a.arg == arg and d is not None:
                    return ast.literal_eval(d)
    raise KeyError((func, arg))


QUANT = ROOT / "backend" / "quant"
EVAL = ROOT / "backend" / "evaluation"


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class ConstantsMatchTheBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FrontApp(scripts=["format.js", "charts.js", "core.js", "view_method.js"], fixtures=[])
        cls.C = app.json("TQ.views.method._internals.CONST")
        cls.features = app.json("TQ.views.method._internals.FEATURES")
        cls.cfg = module_assign(QUANT / "engine.py", "DEFAULT_CFG")

    def test_engine_default_cfg(self):
        mapping = {"garch_window": "garchWindow", "hmm_window": "hmmWindow", "hmm_states": "hmmStates", "dma_alpha": "dmaAlpha", "dma_window": "dmaWindow",
                   "dma_eta": "dmaEta", "cal_window": "calWindow", "cal_lambda": "calLambda", "cal_sigma_a": "calSigmaA", "cal_sigma_b": "calSigmaB",
                   "dlm_delta": "dlmDelta", "dlm_beta_v": "dlmBetaV", "dlm_tau": "dlmTau", "naive_nu": "naiveNu", "mc_paths": "mcPaths", "mc_pool": "mcPool",
                   "garch_skew_damping": "skewDamping"}
        for key, name in mapping.items():
            self.assertEqual(self.C[name], self.cfg[key], f"{name} on the page != DEFAULT_CFG[{key!r}]")

    def test_config_constants(self):
        C = self.C
        pairs = {"commission": config.COMMISSION_RATE, "taxStock": config.TAX_STOCK, "taxEtf": config.TAX_ETF, "slippageBps": config.SLIPPAGE_BPS,
                 "discount": config.COMMISSION_DISCOUNT, "dailyLimit": config.DAILY_LIMIT, "tradingDays": config.TRADING_DAYS_PER_YEAR,
                 "minTrain": config.MIN_TRAIN, "refitEvery": config.REFIT_EVERY, "testBars": config.TEST_BARS_DEFAULT, "buyP": config.BUY_P_UP,
                 "avoidP": config.AVOID_P_UP, "kelly": config.KELLY_FRACTION, "maxWeight": config.MAX_WEIGHT, "minAvgValue": config.MIN_AVG_VALUE_TWD,
                 "universe": config.UNIVERSE_SIZE}
        for name, value in pairs.items():
            self.assertEqual(C[name], value, name)

    def test_recommender_and_features(self):
        self.assertEqual(self.C["shrinkBars"], module_assign(EVAL / "recommend.py", "SHRINK_BARS"))
        keys = module_assign(QUANT / "features.py", "FEATURE_KEYS")
        self.assertEqual([f[0] for f in self.features], keys, "the feature table must list exactly the features of quant/features.py, in order")
        self.assertEqual(self.C["nFeatures"], len(keys))
        names = module_assign(QUANT / "features.py", "FEATURE_NAMES")
        self.assertEqual(set(names), set(keys))

    def test_stated_priors_of_the_naive_baseline(self):
        mu0 = function_default(QUANT / "features.py", "shrunk_drift", "mu0")
        tau0 = function_default(QUANT / "features.py", "shrunk_drift", "tau0")
        self.assertEqual((mu0, tau0), (0.0002, 0.0004))          # "每日 0.02%" / "每日 0.04%" on the page
        self.assertAlmostEqual(mu0 * config.TRADING_DAYS_PER_YEAR, 0.049, places=3)      # "約年化 5%"
        self.assertAlmostEqual(tau0 * config.TRADING_DAYS_PER_YEAR, 0.098, places=3)     # "約年化 10%"

    def test_verdict_thresholds_quoted_on_the_page(self):
        th = module_assign(EVAL / "report.py", "THRESH_STOCK")
        self.assertEqual(th, {"fw": 0.01, "confirm": 0.01, "strong": 0.001, "agree": 0.05})
        self.assertEqual(module_assign(EVAL / "report.py", "MIN_N_STRONG"), 500)
        self.assertEqual(module_assign(EVAL / "report.py", "ROLLING_WINDOW"), 250)         # "近 250 個預測" in the backtest view

    def test_arithmetic_claims(self):
        half_life_days = math.log(0.5) / math.log(self.C["dlmDelta"])
        self.assertAlmostEqual(half_life_days / config.TRADING_DAYS_PER_YEAR, 1.4, delta=0.05)             # "半衰期約 1.4 年"
        p = len(self.features)
        self.assertLess(p * self.C["dlmTau"] ** 2, 0.01)                                                   # "先驗的 R² 不到 1%"
        self.assertEqual(self.C["tradingDays"] * 5, 1225)
        self.assertEqual(config.roundtrip_cost("stock", 0.6, 5.0), 0.00571)
        self.assertAlmostEqual(config.roundtrip_cost("etf", 0.6, 5.0), 0.00371)

    def test_the_text_still_claims_what_the_code_does(self):
        eng = (QUANT / "engine.py").read_text(encoding="utf-8")
        self.assertIn('temper=float(cfg["dma_eta"]) / h', eng)           # "η / h" tempering of the model-averaging weights
        self.assertIn("Deterministic initialisations (no randomness)", (QUANT / "hmm.py").read_text(encoding="utf-8"))
        self.assertIn("only *filtered*", (QUANT / "hmm.py").read_text(encoding="utf-8").replace("Only *filtered*", "only *filtered*"))
        self.assertIn("2021-08", eng)                                    # the dev / evaluation split the page mentions
        cal = (QUANT / "calibration.py").read_text(encoding="utf-8")
        self.assertIn("regularised towards the identity", cal)
        self.assertIn("1/h", cal)
        self.assertIn("Fernandez-Steel", (QUANT / "ensemble.py").read_text(encoding="utf-8"))
        self.assertIn("Barone-Adesi", (QUANT / "mc.py").read_text(encoding="utf-8"))
        self.assertIn("James-Stein", (EVAL / "recommend.py").read_text(encoding="utf-8"))
        self.assertIn("Newey-West", (EVAL / "metrics.py").read_text(encoding="utf-8"))
        self.assertIn("Wilson", (EVAL / "metrics.py").read_text(encoding="utf-8"))


@unittest.skipUnless(HAVE_RACER, SKIP_MSG)
class ContentIsWellFormed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FrontApp(scripts=["format.js", "charts.js", "core.js", "view_method.js"], fixtures=[])
        cls.sections = cls.app.json("TQ.views.method._internals.SECTIONS")

    def walk(self, x, path="", out=None):
        out = [] if out is None else out
        if isinstance(x, str):
            out.append((path, x))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                self.walk(v, f"{path}[{i}]", out)
        elif isinstance(x, dict):
            for k, v in x.items():
                self.walk(v, f"{path}.{k}", out)
        return out

    def test_no_control_characters_from_bad_escapes(self):
        # "$q_\tau$" inside an ordinary JS string silently becomes "$q_<TAB>au$": every backslash must be doubled (or the text must be raw)
        for sec in self.sections:
            for path, text in self.walk(sec["blocks"], sec["id"]):
                self.assertIsNone(re.search(r"[\x00-\x1f]", text), f"control character in {path}: {text[:60]!r}")

    def test_every_inline_dollar_is_paired(self):
        # inline math is $...$: an odd number of dollar signs means a paragraph would show a raw "$" (or swallow text)
        for sec in self.sections:
            for path, text in self.walk(sec["blocks"], sec["id"]):
                self.assertEqual(text.count("$") % 2, 0, f"unpaired $ in {path}: {text[:80]!r}")

    def test_every_math_string_parses_without_unknown_commands(self):
        js = """(function () {
          var I = TQ.views.method._internals, bad = [];
          function check(src, where) {
            var d = document.createElement('div');
            I.mathNodes(src).forEach(function (n) { d.appendChild(typeof n === 'string' ? document.createTextNode(n) : n); });
            var s = JSON.parse(window.__serialize(d)).html;
            if (/class="bad"/.test(s)) bad.push(where + ' :: ' + src.slice(0, 70));
            if (/[\\\\$]/.test(d.textContent.replace(/\\\\\\\\/g, ''))) bad.push(where + ' (stray backslash/dollar) :: ' + src.slice(0, 70));
          }
          function inlineMath(text, where) { var re = /\\$([^$]+)\\$/g, m; while ((m = re.exec(text)) !== null) check(m[1], where); }
          function walk(x, where) {
            if (typeof x === 'string') { inlineMath(x, where); return; }
            if (Array.isArray(x)) { x.forEach(function (v, i) { walk(v, where); }); return; }
            if (x && typeof x === 'object') {
              Object.keys(x).forEach(function (k) {
                if (k === 'eq' && typeof x[k] === 'string') check(x[k], where);
                else if (k === 's' && x.t === 'eq') (Array.isArray(x.s) ? x.s : [x.s]).forEach(function (src) { check(src, where); });
                else walk(x[k], where);
              });
            }
          }
          I.SECTIONS.forEach(function (s) { walk(s.blocks, s.id); });
          return bad;
        })()"""
        self.assertEqual(self.app.json(js), [])

    def test_ids_are_unique_and_lower_case(self):
        ids = [s["id"] for s in self.sections]
        self.assertEqual(len(ids), len(set(ids)))
        for i in ids:
            self.assertRegex(i, r"^[a-z][a-z0-9-]*$")

    def test_every_equation_has_a_gloss_or_is_part_of_a_block_that_has_one(self):
        missing = []
        for sec in self.sections:
            for b in sec["blocks"]:
                if b.get("t") == "eq" and not b.get("note"):
                    missing.append((sec["id"], str(b.get("s"))[:50]))
        self.assertEqual(missing, [], "every displayed equation needs a plain-language explanation")


if __name__ == "__main__":
    unittest.main()
