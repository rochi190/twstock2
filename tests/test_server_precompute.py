"""``python -m backend.precompute`` (warm-up CLI) run in-process with the fake network and the fake engine."""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from server_fakes import FakeEngine, UNIVERSE, install_world  # noqa: E402
from data_fakes import DataCase  # noqa: E402

import schema_check as sc  # noqa: E402
from backend import config, data, jobs, precompute  # noqa: E402


class Precompute(DataCase):
    def setUp(self):
        super().setUp()
        self.engine = FakeEngine()
        for p in (mock.patch.object(jobs, "_engine", lambda: self.engine),
                  mock.patch.object(data, "get_universe",
                                    lambda n=config.UNIVERSE_SIZE, refresh=False: [
                                        dict(code=c, name=nm, market="TWSE", kind="stock") for c, nm in UNIVERSE][:n])):
            p.start()
            self.addCleanup(p.stop)
        install_world(self.net, 900)

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = precompute.main(["--n", "5", "--no-processes", "--workers", "2", *argv])
        return rc, out.getvalue(), err.getvalue()

    def test_default_downloads_prices_and_computes_forecasts(self):
        rc, out, err = self.run_cli()
        self.assertEqual(rc, 0, err)
        self.assertIn("行情：成功 5 檔，失敗 0 檔", out)
        self.assertIn("預測 h=5：評分 5 檔", out)
        self.assertIn("全部完成", out)
        self.assertEqual(self.engine.calls["forecast"], 6)  # five stocks + the TAIEX block
        self.assertEqual(self.engine.calls["wf"], 0)
        # the files the server will look for
        for sub in ("prices", "forecast"):
            self.assertTrue(os.listdir(os.path.join(self.tmp, sub)), sub)

    def test_server_finds_everything_precompute_cached(self):
        self.run_cli("--prices", "--forecasts", "--universe-bt", "--h", "5", "--top", "3")
        n_fc, n_wf, n_net = self.engine.calls["forecast"], self.engine.calls["wf"], len(self.net.calls)
        self.assertEqual(n_wf, 5)
        comp = jobs.Compute(workers=2, use_processes=False)
        comp.first_wait = 5.0
        comp.universe_size = 5
        try:
            rec = comp.recommendations(5, 10)
            self.assertEqual(rec["status"], "ready")
            self.assertEqual(sc.check_recommendations(rec, 5, 10), [])
            ub = comp.universe_backtest(5, 3, "next_open", config.COMMISSION_DISCOUNT, config.SLIPPAGE_BPS)
            self.assertEqual(ub["status"], "ready")
            self.assertEqual(sc.check_universe_report(ub, 5, 3), [])
        finally:
            comp.close()
        self.assertEqual((self.engine.calls["forecast"], self.engine.calls["wf"]), (n_fc, n_wf))  # all cache hits
        self.assertEqual(len(self.net.calls), n_net)  # no extra downloads either

    def test_universe_bt_flag_and_validate_alias(self):
        rc, out, err = self.run_cli("--universe-bt", "--h", "5")
        self.assertEqual(rc, 0, err)
        self.assertIn("整體驗證 h=5", out)
        self.assertEqual(self.engine.calls["wf"], 5)
        rc, out, err = self.run_cli("--validate", "--h", "1")  # old spelling still works; WFs are reused
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.engine.calls["wf"], 5)

    def test_extra_symbols_forecast_and_backtest(self):
        rc, out, err = self.run_cli("--symbols", "0050,6488", "--forecasts", "--backtest")
        self.assertEqual(rc, 0, err)
        self.assertIn("預測 0050 完成", out)
        self.assertIn("回測 6488 完成", out)
        self.assertEqual(self.engine.calls["forecast:0050"], 1)
        self.assertEqual(self.engine.calls["wf:6488"], 1)

    def test_refresh_flag_redownloads(self):
        self.run_cli("--prices")
        n = self.net.count("/chart/")
        self.run_cli("--prices")
        self.assertEqual(self.net.count("/chart/"), n)  # warm cache: nothing to do
        self.run_cli("--prices", "--refresh")
        self.assertGreater(self.net.count("/chart/"), n)

    def test_bad_arguments(self):
        rc, out, err = self.run_cli("--symbols", "../etc")
        self.assertEqual(rc, 2)
        self.assertIn("代號", err)
        rc, out, err = self.run_cli("--h", "7")
        self.assertEqual(rc, 2)

    def test_network_down_is_a_clean_failure(self):
        self.net.routes.clear()
        rc, out, err = self.run_cli("--forecasts")
        self.assertEqual(rc, 1)
        self.assertIn("失敗", err)


if __name__ == "__main__":
    unittest.main()
