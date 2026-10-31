"""``data._http_get`` against a real local HTTP server: retries, backoff, gzip, truncation, 404, concurrency cap."""
import gzip
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from backend import data  # noqa: E402


class Scripted(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    counts: dict = {}
    active = 0
    peak = 0
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _reply(self, status, body=b"", headers=None):
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path
        with Scripted.lock:
            n = Scripted.counts[path] = Scripted.counts.get(path, 0) + 1
        if path.startswith("/ok"):
            return self._reply(200, b'{"ok":1}')
        if path.startswith("/ua"):
            return self._reply(200, (self.headers.get("User-Agent", "") + "|" + self.headers.get("Accept-Encoding", "")).encode())
        if path.startswith("/gz"):
            return self._reply(200, gzip.compress(b'{"gz":true}'), {"Content-Encoding": "gzip"})
        if path.startswith("/flaky500"):
            return self._reply(500, b"boom") if n <= 2 else self._reply(200, b"fine")
        if path.startswith("/r429"):
            return self._reply(429, b"slow down", {"Retry-After": "3"}) if n == 1 else self._reply(200, b"fine")
        if path.startswith("/notfound"):
            return self._reply(404, b'{"chart":{"error":{"code":"Not Found"}}}')
        if path.startswith("/forbidden"):
            return self._reply(403, b"no")
        if path.startswith("/always500"):
            return self._reply(503, b"down")
        if path.startswith("/trunc"):
            if n == 1:  # promise 1000 bytes, deliver 10, hang up -> http.client.IncompleteRead
                self.send_response(200)
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b"0123456789")
                self.wfile.flush()
                self.close_connection = True
                return
            return self._reply(200, b"complete")
        if path.startswith("/badgz"):
            return self._reply(200, b"definitely not gzip", {"Content-Encoding": "gzip"})
        if path.startswith("/slow"):
            time.sleep(1.5)
            return self._reply(200, b"late")
        if path.startswith("/hold"):
            with Scripted.lock:
                Scripted.active += 1
                Scripted.peak = max(Scripted.peak, Scripted.active)
            time.sleep(0.12)
            with Scripted.lock:
                Scripted.active -= 1
            return self._reply(200, b"held")
        return self._reply(404, b"?")


class HttpGet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Scripted)
        cls.srv.daemon_threads = True
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.t = threading.Thread(target=cls.srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        cls.t.start()
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        Scripted.counts.clear()
        Scripted.peak = 0
        self.sleeps = []
        p = mock.patch.object(data, "_sleep", self.sleeps.append)
        p.start()
        self.addCleanup(p.stop)

    def test_ok_and_headers(self):
        self.assertEqual(data._http_get(self.base + "/ok"), b'{"ok":1}')
        body = data._http_get(self.base + "/ua").decode()
        self.assertEqual(body, "Mozilla/5.0|gzip")

    def test_gzip_is_decoded(self):
        self.assertEqual(data._http_get(self.base + "/gz"), b'{"gz":true}')
        self.assertEqual(data._get_json(self.base + "/gz"), {"gz": True})

    def test_retries_5xx_with_exponential_backoff(self):
        body = data._http_get(self.base + "/flaky500", retries=4)
        self.assertEqual(body, b"fine")
        self.assertEqual(Scripted.counts["/flaky500"], 3)
        self.assertEqual(len(self.sleeps), 2)
        self.assertLess(self.sleeps[0], self.sleeps[1])  # exponential
        self.assertGreaterEqual(self.sleeps[0], data.BACKOFF_BASE)

    def test_429_honours_retry_after(self):
        self.assertEqual(data._http_get(self.base + "/r429"), b"fine")
        self.assertEqual(Scripted.counts["/r429"], 2)
        self.assertGreaterEqual(max(self.sleeps), 3.0)

    def test_gives_up_after_retries(self):
        with self.assertRaises(data.DataUnavailable):
            data._http_get(self.base + "/always500", retries=3)
        self.assertEqual(Scripted.counts["/always500"], 3)
        self.assertEqual(len(self.sleeps), 2)

    def test_404_is_immediate_not_found(self):
        with self.assertRaises(data.NotFoundError):
            data._http_get(self.base + "/notfound")
        self.assertEqual(Scripted.counts["/notfound"], 1)
        self.assertEqual(self.sleeps, [])

    def test_other_4xx_not_retried(self):
        with self.assertRaises(data.DataUnavailable):
            data._http_get(self.base + "/forbidden")
        self.assertEqual(Scripted.counts["/forbidden"], 1)

    def test_truncated_body_is_retried(self):
        self.assertEqual(data._http_get(self.base + "/trunc"), b"complete")
        self.assertEqual(Scripted.counts["/trunc"], 2)

    def test_corrupt_gzip_is_retried_then_fails(self):
        with self.assertRaises(data.DataUnavailable):
            data._http_get(self.base + "/badgz", retries=2)
        self.assertEqual(Scripted.counts["/badgz"], 2)

    def test_timeout_is_retried_then_fails(self):
        t0 = time.time()
        with self.assertRaises(data.DataUnavailable):
            data._http_get(self.base + "/slow", timeout=0.25, retries=2)
        self.assertLess(time.time() - t0, 1.4)
        self.assertEqual(Scripted.counts["/slow"], 2)

    def test_connection_refused_is_unavailable(self):
        with self.assertRaises(data.DataUnavailable):
            data._http_get("http://127.0.0.1:9/never", timeout=1, retries=2)
        self.assertEqual(len(self.sleeps), 1)

    def test_semaphore_caps_concurrency(self):
        sem = threading.BoundedSemaphore(3)
        errs = []

        def worker(i):
            try:
                data._http_get(f"{self.base}/hold?{i}", semaphore=sem)
            except Exception as exc:  # noqa: BLE001
                errs.append(exc)

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errs, [])
        self.assertLessEqual(Scripted.peak, 3)
        self.assertGreaterEqual(Scripted.peak, 2)  # it really did run in parallel

    def test_yahoo_semaphore_is_six(self):
        self.assertEqual(data.YAHOO_CONCURRENCY, 6)
        # the module-level semaphore really is a bounded semaphore of that size
        sems = [data._YAHOO_SEM.acquire(blocking=False) for _ in range(6)]
        try:
            self.assertTrue(all(sems))
            self.assertFalse(data._YAHOO_SEM.acquire(blocking=False))
        finally:
            for ok in sems:
                if ok:
                    data._YAHOO_SEM.release()


class FakeYahoo(BaseHTTPRequestHandler):
    """Serves ``/chart/<sym>`` like Yahoo (gzip JSON) and records the concurrency and the request lines."""
    protocol_version = "HTTP/1.1"
    active = 0
    peak = 0
    seen: list = []
    lock = threading.Lock()
    hold = 0.08

    def log_message(self, *a):
        pass

    def do_GET(self):  # noqa: N802
        import json as _json
        from datetime import date as _date
        sys.path.insert(0, HERE)
        from data_fakes import bdays, random_walk, yahoo_payload
        with FakeYahoo.lock:
            FakeYahoo.active += 1
            FakeYahoo.peak = max(FakeYahoo.peak, FakeYahoo.active)
            FakeYahoo.seen.append((self.path, self.headers.get("User-Agent"), self.headers.get("Accept-Encoding")))
        try:
            time.sleep(FakeYahoo.hold)
            ds = [d for d in bdays(_date(2025, 1, 6), 400) if d <= _date(2026, 10, 22)][-300:]
            body = gzip.compress(yahoo_payload(ds, random_walk(len(ds), 5)))
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with FakeYahoo.lock:
                FakeYahoo.active -= 1


class YahooOverRealSockets(unittest.TestCase):
    """The production request path (urllib, gzip, bounded semaphore) against a local stand-in for Yahoo."""

    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeYahoo)
        cls.srv.daemon_threads = True
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.t = threading.Thread(target=cls.srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        cls.t.start()
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        import shutil
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="twq_sock_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        data.set_cache_dir(self.tmp)
        self.addCleanup(data.set_cache_dir, None)
        FakeYahoo.peak = 0
        FakeYahoo.seen = []
        for patcher in (mock.patch.object(data, "YAHOO_CHART", self.base + "/chart/{sym}"),
                        mock.patch.object(data, "_sleep", lambda s: None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        # a fresh, empty name table so no TWSE / TPEx request is attempted
        now = time.time()
        data._NAMES_MEM["table"] = {"items": {}, "fetched_ts": now, "expires_ts": now + 1e6, "sources": {}}
        data._HOL_MEM[2026] = frozenset()
        data._HOL_MEM[2027] = frozenset()

    def test_twenty_four_symbols_in_parallel_never_exceed_six_requests(self):
        codes = [f"{1500 + i}" for i in range(24)]
        errs = []

        def go(c):
            try:
                self.assertEqual(data.get_price_data(c).n, 300)
            except Exception as exc:  # noqa: BLE001
                errs.append(exc)

        ts = [threading.Thread(target=go, args=(c,)) for c in codes]
        [t.start() for t in ts]
        [t.join(60) for t in ts]
        self.assertEqual(errs, [])
        self.assertEqual(len(FakeYahoo.seen), 24)
        self.assertLessEqual(FakeYahoo.peak, data.YAHOO_CONCURRENCY)
        self.assertGreaterEqual(FakeYahoo.peak, 3)

    def test_request_shape_matches_the_design(self):
        data.get_price_data("2330")
        path, ua, enc = FakeYahoo.seen[0]
        self.assertTrue(path.startswith("/chart/2330.TW?period1=0&period2="), path)
        self.assertIn("&interval=1d&events=div%2Csplits", path)
        self.assertNotIn("range=max", path)
        self.assertEqual(ua, "Mozilla/5.0")
        self.assertEqual(enc, "gzip")
        data.get_market_data()
        self.assertTrue(FakeYahoo.seen[-1][0].startswith("/chart/%5ETWII?period1=0"), FakeYahoo.seen[-1][0])


if __name__ == "__main__":
    unittest.main()
