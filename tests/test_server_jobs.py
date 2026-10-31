"""jobs.py building blocks: JSON sanitiser, errors, LRU/SingleFlight, JobManager (sharing, TTL, bounded memory),
ComputePool (real spawn processes, bounded window, worker death), parameter validation, housekeeping."""
import dataclasses
import json
import math
import os
import sys
import threading
import time
import unittest
from datetime import date, datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from data_fakes import TAIPEI, DataCase  # noqa: E402

from backend import config, data, jobs  # noqa: E402
from backend.jobs import ApiError, JobManager, to_api_error, to_jsonable  # noqa: E402


class ToJsonable(unittest.TestCase):
    def test_numpy_scalars_and_arrays(self):
        o = {"a": np.float64(1.5), "b": np.int64(3), "c": np.bool_(True), "d": np.arange(3), "e": np.array([[1.0, np.nan]]),
             "f": np.float32(0.25)}
        j = to_jsonable(o)
        self.assertEqual(j, {"a": 1.5, "b": 3, "c": True, "d": [0, 1, 2], "e": [[1.0, None]], "f": 0.25})
        self.assertIs(type(j["a"]), float)
        self.assertIs(type(j["b"]), int)
        json.dumps(j, allow_nan=False)

    def test_nan_and_infinities_become_null(self):
        j = to_jsonable({"x": float("nan"), "y": float("inf"), "z": -math.inf, "w": [np.nan, np.inf, 1.0], "v": np.float64("nan")})
        self.assertEqual(j, {"x": None, "y": None, "z": None, "w": [None, None, 1.0], "v": None})

    def test_dates(self):
        j = to_jsonable({"d": np.datetime64("2026-10-22"), "a": np.array(["2026-10-22", "NaT"], dtype="datetime64[D]"),
                         "p": date(2026, 1, 2), "t": datetime(2026, 1, 2, 3, 4, 5, tzinfo=TAIPEI),
                         "ns": np.datetime64("2026-10-22T10:00:00")})
        self.assertEqual(j["d"], "2026-10-22")
        self.assertEqual(j["a"], ["2026-10-22", None])
        self.assertEqual(j["p"], "2026-01-02")
        self.assertEqual(j["t"], "2026-01-02T03:04:05+08:00")
        self.assertEqual(j["ns"], "2026-10-22")

    def test_keys_containers_and_dataclasses(self):
        @dataclasses.dataclass
        class D:
            x: float
            y: tuple

        j = to_jsonable({1: "a", np.int64(5): "b", ("t",): "c", "s": {1, 2}, "dc": D(float("nan"), (1, 2))})
        self.assertEqual(j["1"], "a")
        self.assertEqual(j["5"], "b")
        self.assertEqual(sorted(j["s"]), [1, 2])
        self.assertEqual(j["dc"], {"x": None, "y": [1, 2]})

    def test_dumps_is_strict_utf8(self):
        raw = jobs.dumps_json({"name": "台積電", "v": float("nan")})
        self.assertEqual(raw.decode("utf-8"), '{"name":"台積電","v":null}')
        self.assertNotIn(b"NaN", raw)


class Errors(unittest.TestCase):
    def test_mapping(self):
        cases = [(data.BadSymbol("x"), "bad_request", 400), (data.UnknownSymbol("x"), "unknown_symbol", 404),
                 (data.DataUnavailable("x"), "data_unavailable", 503), (ValueError("insufficient_history"), "insufficient_history", 422),
                 (RuntimeError("x"), "internal", 500), (KeyError("bug"), "internal", 500), (ValueError("other"), "internal", 500)]
        for exc, code, status in cases:
            e = to_api_error(exc, "2330")
            self.assertEqual((e.code, e.status), (code, status), repr(exc))
            d = e.to_dict()
            self.assertEqual(d["status"], "error")
            self.assertTrue(d["error"]["message"])
            self.assertTrue(any("一" <= ch <= "鿿" for ch in d["error"]["message"]), "message must be Chinese")

    def test_internal_details_are_kept_out_of_the_message(self):
        e = to_api_error(RuntimeError("secret path /home/x"))
        self.assertNotIn("secret", e.message)
        self.assertIn("secret", e.detail)


class Containers(unittest.TestCase):
    def test_lru_is_bounded_and_recency_ordered(self):
        lru = jobs.LRU(3)
        for i in range(10):
            lru.put(i, i)
        self.assertEqual([lru.get(i) for i in range(10)], [None] * 7 + [7, 8, 9])
        lru.get(7)
        lru.put(10, 10)  # evicts 8, not the freshly read 7
        self.assertEqual((lru.get(7), lru.get(8)), (7, None))

    def test_single_flight_runs_once_and_shares_exceptions(self):
        sf = jobs.SingleFlight()
        n, gate, out = [0], threading.Event(), []

        def fn():
            n[0] += 1
            gate.wait(5)
            return "v"

        ts = [threading.Thread(target=lambda: out.append(sf.do("k", fn))) for _ in range(8)]
        [t.start() for t in ts]
        time.sleep(0.2)
        gate.set()
        [t.join(5) for t in ts]
        self.assertEqual((n[0], out), (1, ["v"] * 8))

        errs = []

        def bad():
            time.sleep(0.2)
            raise ValueError("nope")

        def go():
            try:
                sf.do("e", bad)
            except ValueError as exc:
                errs.append(str(exc))

        ts = [threading.Thread(target=go) for _ in range(4)]
        [t.start() for t in ts]
        [t.join(5) for t in ts]
        self.assertEqual(errs, ["nope"] * 4)
        self.assertEqual(sf.do("e", lambda: 5), 5)  # the failure is not remembered


class JobManagerTests(unittest.TestCase):
    def test_identical_keys_share_one_job_and_progress_flows(self):
        jm = JobManager(max_concurrent=2)
        gate, runs = threading.Event(), [0]

        def runner(progress):
            runs[0] += 1
            progress(1, 4, "一")
            gate.wait(5)
            progress(4, 4, "四")
            return {"x": 1}

        jobs_ = [jm.get_or_start("k", "t", runner) for _ in range(5)]
        self.assertTrue(all(j is jobs_[0] for j in jobs_))
        time.sleep(0.2)
        s = jobs_[0].snapshot()
        self.assertEqual(s["status"], "computing")
        self.assertEqual(s["progress"], {"done": 1, "total": 4, "message": "一"})
        gate.set()
        s = jobs_[0].snapshot(wait=5)
        self.assertEqual(s, {"status": "ready", "x": 1})
        self.assertEqual(runs[0], 1)
        self.assertIs(jm.get_or_start("k", "t", runner), jobs_[0])  # finished result is shared too
        self.assertEqual(runs[0], 1)

    def test_snapshot_wait_returns_early_when_finished(self):
        jm = JobManager()
        j = jm.get_or_start("a", "t", lambda p: {"v": 1})
        t0 = time.time()
        self.assertEqual(j.snapshot(wait=5)["status"], "ready")
        self.assertLess(time.time() - t0, 1.0)

    def test_errors_become_structured_and_expire(self):
        jm = JobManager(error_ttl=0.2)
        n = [0]

        def runner(progress):
            n[0] += 1
            raise data.UnknownSymbol("9999")

        j = jm.get_or_start("k", "t", runner)
        s = j.snapshot(wait=5)
        self.assertEqual((s["status"], s["error"]["code"]), ("error", "unknown_symbol"))
        self.assertIs(jm.get_or_start("k", "t", runner), j)  # remembered briefly (no hammering)
        time.sleep(0.3)
        j2 = jm.get_or_start("k", "t", runner)  # then retried
        self.assertIsNot(j2, j)
        j2.snapshot(wait=5)
        self.assertEqual(n[0], 2)

    def test_ready_results_expire(self):
        jm = JobManager(ready_ttl=0.2)
        n = [0]

        def runner(progress):
            n[0] += 1
            return {"n": n[0]}

        j = jm.get_or_start("k", "t", runner)
        self.assertEqual(j.snapshot(wait=5)["n"], 1)
        time.sleep(0.3)
        self.assertEqual(jm.get_or_start("k", "t", runner).snapshot(wait=5)["n"], 2)

    def test_registry_is_bounded(self):
        jm = JobManager(max_entries=10)
        for i in range(60):
            jm.get_or_start(f"k{i}", "t", lambda p: {"payload": "x" * 100}).snapshot(wait=5)
        self.assertLessEqual(sum(jm.stats().values()), 10)

    def test_running_jobs_are_never_evicted(self):
        jm = JobManager(max_entries=3, max_concurrent=10)
        gate = threading.Event()
        running = [jm.get_or_start(f"r{i}", "t", lambda p: gate.wait(5) and {"ok": 1}) for i in range(3)]
        for i in range(10):
            jm.get_or_start(f"x{i}", "t", lambda p: {"ok": 1})
        self.assertTrue(all(j.status == "computing" for j in running))
        self.assertGreaterEqual(jm.stats()["computing"], 3)
        gate.set()

    def test_concurrency_limit(self):
        jm = JobManager(max_concurrent=2)
        active, peak, lock, gate = [0], [0], threading.Lock(), threading.Event()

        def runner(progress):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.15)
            with lock:
                active[0] -= 1
            return {}

        js = [jm.get_or_start(f"k{i}", "t", runner) for i in range(6)]
        [j.snapshot(wait=10) for j in js]
        self.assertTrue(all(j.status == "ready" for j in js))
        self.assertEqual(peak[0], 2)

    def test_close_rejects_new_jobs(self):
        jm = JobManager()
        jm.close()
        with self.assertRaises(ApiError):
            jm.get_or_start("k", "t", lambda p: {})


class Params(unittest.TestCase):
    def test_horizon_validation(self):
        self.assertEqual(jobs.validate_horizon("5"), 5)
        for bad in ("2", "x", None, "0", "-1", "1.5"):
            with self.assertRaises(ApiError) as cm:
                jobs.validate_horizon(bad)
            self.assertEqual(cm.exception.code, "bad_request")

    def test_backtest_params(self):
        p = jobs.build_backtest_params("2330")
        self.assertEqual((p.symbol, p.h, p.thr, p.exec, p.disc, p.slip, p.years, p.sizing),
                         ("2330", 1, 0.55, "next_open", 0.6, 5.0, 5.0, "threshold"))
        self.assertEqual(jobs.build_backtest_params("2330.TW", "5", "0.6").key(), jobs.build_backtest_params("2330", 5, 0.6).key())
        self.assertNotEqual(p.key(), jobs.build_backtest_params("2330", thr=0.6).key())
        for kw in ({"thr": 0.49}, {"thr": 0.96}, {"thr": "nan"}, {"thr": "inf"}, {"disc": 1.1}, {"slip": -1}, {"years": 0.5},
                   {"years": 26}, {"exec_mode": "x"}, {"sizing": "x"}, {"h": 3}, {"thr": "abc"}):
            with self.assertRaises(ApiError) as cm:
                jobs.build_backtest_params("2330", **kw)
            self.assertEqual(cm.exception.code, "bad_request", kw)
        with self.assertRaises(ApiError):
            jobs.build_backtest_params("../etc/passwd")

    def test_test_bars(self):
        self.assertEqual(jobs.backtest_test_bars(6000, 5.0), config.TEST_BARS_DEFAULT)  # shared with the universe validation
        self.assertEqual(jobs.backtest_test_bars(6000, 2.0), 490)
        self.assertEqual(jobs.backtest_test_bars(900, 5.0), 400)  # capped at n - MIN_TRAIN

    def test_env_int(self):
        os.environ["TWQ_T"] = "7"
        try:
            self.assertEqual(jobs.env_int("TWQ_T", 60, 4, 500), 7)
            os.environ["TWQ_T"] = "2"
            self.assertEqual(jobs.env_int("TWQ_T", 60, 4, 500), 4)
            os.environ["TWQ_T"] = "abc"
            self.assertEqual(jobs.env_int("TWQ_T", 60, 4, 500), 60)
        finally:
            del os.environ["TWQ_T"]
        self.assertEqual(jobs.env_int("TWQ_NOT_SET_XYZ", 60, 4, 500), 60)


class EpochToken(DataCase):
    def test_changes_at_open_and_at_final(self):
        f = jobs.Compute.epoch_token
        a = f(datetime(2026, 10, 22, 8, 0, tzinfo=TAIPEI))    # before the open: yesterday's bar is the newest
        b = f(datetime(2026, 10, 22, 9, 30, tzinfo=TAIPEI))   # session running
        c = f(datetime(2026, 10, 22, 13, 59, tzinfo=TAIPEI))
        d = f(datetime(2026, 10, 22, 14, 0, tzinfo=TAIPEI))   # bar final
        e = f(datetime(2026, 10, 22, 23, 0, tzinfo=TAIPEI))
        self.assertEqual(b, c)
        self.assertNotEqual(a, b)
        self.assertNotEqual(c, d)
        self.assertEqual(d, e)
        self.assertEqual(f(datetime(2026, 10, 24, 12, 0, tzinfo=TAIPEI)), f(datetime(2026, 10, 25, 20, 0, tzinfo=TAIPEI)))  # weekend


class Housekeeping(unittest.TestCase):
    def test_prune_removes_old_cache_files_and_stray_temp_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for sub in ("forecast", "wf", "results", "prices"):
                os.makedirs(os.path.join(tmp, sub))
            old, new = time.time() - 40 * 86400, time.time()
            files = {"forecast/a.json": old, "wf/b.pkl.gz": old, "results/c.json": old, "forecast/fresh.json": new,
                     "prices/2330.json": old, "prices/.x.tmp": time.time() - 7200, "wf/.y.tmp": new}
            for rel, mt in files.items():
                p = os.path.join(tmp, rel)
                with open(p, "w") as fh:
                    fh.write("x")
                os.utime(p, (mt, mt))
            n = jobs.prune_cache(tmp)
            self.assertEqual(n, 3)
            left = sorted(os.path.join(sub, f) for sub in os.listdir(tmp) for f in os.listdir(os.path.join(tmp, sub)))
            self.assertEqual(sorted(left), sorted(["forecast/fresh.json", "prices/2330.json", "wf/.y.tmp"]))

    def test_disk_store_roundtrip_and_supersede(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ds = jobs.DiskStore(lambda: __import__("pathlib").Path(tmp))
            ds.save_json("forecast", "2330_2026-10-21", "k1", {"v": 1}, supersede="2330_*")
            ds.save_json("forecast", "2330_2026-10-22", "k2", {"v": float("nan")}, supersede="2330_*")
            self.assertEqual(os.listdir(os.path.join(tmp, "forecast")), ["2330_2026-10-22_k2.json"])
            self.assertEqual(ds.load_json("forecast", "2330_2026-10-22", "k2"), {"v": None})
            self.assertIsNone(ds.load_json("forecast", "2330_2026-10-21", "k1"))
            ds.save_pickle("wf", "x", "k", {"a": np.arange(5)})
            self.assertEqual(ds.load_pickle("wf", "x", "k")["a"].tolist(), [0, 1, 2, 3, 4])
            with open(ds.path("wf", "x", "bad", "pkl.gz"), "wb") as fh:
                fh.write(b"not a gzip")
            self.assertIsNone(ds.load_pickle("wf", "x", "bad"))  # corrupt -> recompute, never crash


class RealProcessPool(unittest.TestCase):
    """The pool really runs in separate spawned processes and survives a dying worker."""

    @classmethod
    def setUpClass(cls):
        cls.pool = jobs.ComputePool(workers=2, use_processes=True)

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def test_workers_are_other_processes_and_tasks_are_picklable(self):
        self.assertEqual(self.pool.run(jobs._task_selftest, 12), 144)
        pids = {self.pool.run(jobs._task_ping) for _ in range(6)}
        self.assertTrue(pids)
        self.assertNotIn(os.getpid(), pids)

    def test_default_worker_count_leaves_a_core_free(self):
        p = jobs.ComputePool()
        try:
            self.assertEqual(p.workers, max(1, (os.cpu_count() or 2) - 1))
        finally:
            p.close()

    def test_bounded_window_map_with_results_and_errors(self):
        results, errs = {}, {}

        def on_result(key, value, exc):
            if exc is not None:
                errs[key] = exc
            else:
                results[key] = value

        specs = [(i, jobs._task_selftest, (i,)) for i in range(7)] + [("bad", jobs._task_selftest, ("a", "b"))]
        self.pool.map_bounded(specs, on_result)
        self.assertEqual(results, {i: i * i for i in range(7)})
        self.assertEqual(list(errs), ["bad"])

    def test_a_dying_worker_is_survived(self):
        import os as _os
        self.assertEqual(self.pool.run(jobs._task_selftest, 3), 9)
        with self.assertRaises(Exception):  # the task kills its own process -> BrokenProcessPool after the retry
            self.pool.run(_os._exit, 1)
        self.assertEqual(self.pool.run(jobs._task_selftest, 4), 16)  # a fresh pool was created transparently

    def test_progress_slots_are_visible_across_processes(self):
        slot = self.pool.acquire_slot()
        try:
            self.assertGreaterEqual(slot, 0)
            self.assertEqual(self.pool.read_slot(slot), (0, 0))
            self.pool._arr[2 * slot] = 7
            self.assertEqual(self.pool.read_slot(slot)[0], 7)
        finally:
            self.pool.release_slot(slot)
        slots = [self.pool.acquire_slot() for _ in range(jobs._NSLOTS + 3)]
        self.assertEqual(slots[-3:], [-1, -1, -1])  # exhausted -> -1 (progress simply not reported)
        for s in slots:
            self.pool.release_slot(s)


class ThreadPoolWindow(unittest.TestCase):
    def test_window_bounds_in_flight_tasks_and_cancel_stops_submission(self):
        pool = jobs.ComputePool(workers=8, use_processes=False)
        try:
            active, peak, lock = [0], [0], threading.Lock()

            def task(i):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.05)
                with lock:
                    active[0] -= 1
                return i

            got = []
            pool.map_bounded([(i, task, (i,)) for i in range(20)], lambda k, v, e: got.append(k), window=3)
            self.assertEqual(sorted(got), list(range(20)))
            self.assertLessEqual(peak[0], 3)
            cancel = threading.Event()
            got.clear()

            def on_result(k, v, e):
                got.append(k)
                if len(got) == 2:
                    cancel.set()

            pool.map_bounded([(i, task, (i,)) for i in range(40)], on_result, window=2, cancel=cancel)
            self.assertLess(len(got), 40)
        finally:
            pool.close()


class MemoryIsBounded(DataCase):
    def test_price_record_memory_cache_is_an_lru(self):
        from data_fakes import random_walk
        from test_data_cache import history
        old = data.REC_MEM_MAX
        data.REC_MEM_MAX = 4
        try:
            ds = history(260)
            self.route_names(twse=[])
            for i in range(12):
                code = f"{1200 + i}"
                self.route_symbol(f"{code}.TW", ds, random_walk(260, i))
                data.get_price_data(code)
            self.assertLessEqual(len(data._REC_MEM), 4)
            self.assertEqual(list(data._REC_MEM)[-1], "1211")  # most recent kept
            n = len(self.net.calls)
            data.get_price_data("1200")  # evicted from memory, still on disk: no network
            self.assertEqual(len(self.net.calls), n)
        finally:
            data.REC_MEM_MAX = old

    def test_bookkeeping_dicts_are_capped(self):
        old = data.BOOKKEEPING_MAX
        d = {i: i for i in range(100)}
        data._cap(d, 10)
        self.assertEqual(list(d), list(range(90, 100)))
        for i in range(50):
            data._sym_lock(f"S{i}")
        self.assertLessEqual(len(data._SYM_LOCKS), old)

    def test_forecast_memory_cache_is_an_lru(self):
        c = jobs.Compute(workers=1, use_processes=False)
        try:
            for i in range(1000):
                c._forecast_mem.put(i, {"x": i})
            self.assertLessEqual(len(c._forecast_mem._d), 256)
            for i in range(50):
                c._wf_mem.put(i, object())
            self.assertLessEqual(len(c._wf_mem._d), 6)
        finally:
            c.close()


if __name__ == "__main__":
    unittest.main()
