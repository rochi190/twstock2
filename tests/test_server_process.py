"""`python -m backend.server` as a real OS process: startup, env handling, clean shutdown on SIGINT / SIGTERM (also with
SIGINT inherited as "ignored"), port clash message, and a full end-to-end run with the REAL quant engine, the REAL
spawn process pool and the REAL evaluation modules on synthetic prices (no network: the launcher fakes Yahoo / TWSE)."""
import http.client
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import schema_check as sc  # noqa: E402

from backend import config, server  # noqa: E402

SLOW = os.environ.get("TWQ_SKIP_SLOW", "") not in ("", "0")

LAUNCHER = textwrap.dedent('''
    import sys
    sys.path[:0] = [{root!r}, {tests!r}]
    from datetime import datetime
    from unittest import mock
    from data_fakes import FakeNet, TAIPEI
    import server_fakes
    from backend import data, server


    def main():
        net = FakeNet()
        server_fakes.install_world(net, {n_bars})
        universe = [dict(code=c, name=nm, market="TWSE", kind="stock") for c, nm in server_fakes.UNIVERSE]
        fake_now = datetime(2026, 10, 22, 16, 0, tzinfo=TAIPEI)
        with mock.patch.object(data, "_http_get", net), mock.patch.object(data, "_now", lambda: fake_now), \\
                mock.patch.object(data, "_wall", lambda: fake_now.timestamp()), \\
                mock.patch.object(data, "get_universe", lambda n=60, refresh=False: universe[:n]):
            data._HOL_MEM[2026] = frozenset()   # no holidays: weekends only
            data._HOL_MEM[2027] = frozenset()
            return server.main(["--port", "0", "--no-warm", "--quiet"])


    if __name__ == "__main__":   # spawned pool workers import this file as __mp_main__: they must not start a server
        sys.exit(main())
''')


def start(env_extra=None, args=("--port", "0", "--quiet", "--no-warm"), sigint_ignored=False, launcher=None, tmpdir=None):
    env = dict(os.environ)
    env.pop("PORT", None)
    env.pop("HOST", None)
    env.update({"TWQ_OFFLINE": "1", "TWQ_CACHE_DIR": tmpdir or tempfile.mkdtemp(prefix="twq_proc_"), "PYTHONUNBUFFERED": "1"})
    env.update(env_extra or {})
    cmd = [sys.executable, launcher] if launcher else [sys.executable, "-m", "backend.server", *args]
    pre = (lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)) if sigint_ignored else None
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=pre)
    line = p.stdout.readline()
    m = re.search(r":(\d+)/", line)
    if not m:
        p.kill()
        raise AssertionError(f"server did not announce a port: {line!r} / {p.stderr.read()[:500]}")
    return p, int(m.group(1)), line


def fetch(port, path, timeout=60):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


def stop(p, sig=signal.SIGINT, timeout=15):
    t0 = time.time()
    p.send_signal(sig)
    try:
        rc = p.wait(timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        rc = "timeout"
    return rc, time.time() - t0


def close_pipes(p):
    for f in (p.stdout, p.stderr):
        if f is not None and not f.closed:
            f.close()


def child_pids(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


class Lifecycle(unittest.TestCase):
    def test_starts_serves_and_exits_cleanly_on_sigint(self):
        p, port, line = start()
        try:
            self.assertIn("已啟動", line)
            st, body = fetch(port, "/api/status")
            self.assertEqual(st, 200)
            self.assertEqual(sc.check_status(sc.strict_loads(body)), [])
            self.assertEqual(fetch(port, "/")[0] in (200, 404), True)
        finally:
            rc, dt = stop(p, signal.SIGINT)
        self.assertEqual(rc, 0)
        self.assertLess(dt, 5.0)
        self.assertNotIn("Traceback", p.stderr.read())
        close_pipes(p)

    def test_sigterm_also_shuts_down_cleanly(self):
        p, port, _ = start()
        self.assertEqual(fetch(port, "/api/status")[0], 200)
        rc, dt = stop(p, signal.SIGTERM)
        self.assertEqual(rc, 0)
        self.assertNotIn("Traceback", p.stderr.read())
        close_pipes(p)

    def test_sigint_works_even_when_started_in_the_background(self):
        # `./run.sh &` from a non-interactive shell starts the process with SIGINT ignored
        p, port, _ = start(sigint_ignored=True)
        rc, dt = stop(p, signal.SIGINT)
        self.assertEqual(rc, 0)
        close_pipes(p)

    def test_conda_style_HOST_variable_does_not_decide_the_bind_address(self):
        p, port, line = start({"HOST": "x86_64-conda-linux-gnu"}, args=("--port", "0", "--quiet", "--no-warm"))
        try:
            self.assertEqual(fetch(port, "/api/status")[0], 200)
            self.assertIn("localhost", line)
        finally:
            stop(p)
        self.assertIn("HOST", p.stderr.read())  # told the user it was ignored
        close_pipes(p)

    def test_port_in_use_is_a_clean_error(self):
        p, port, _ = start()
        try:
            q = subprocess.run([sys.executable, "-m", "backend.server", "--port", str(port), "--no-warm"], cwd=ROOT,
                               capture_output=True, text=True, timeout=30,
                               env=dict(os.environ, TWQ_OFFLINE="1", TWQ_CACHE_DIR=tempfile.mkdtemp(prefix="twq_proc_")))
            self.assertEqual(q.returncode, 2)
            self.assertIn("無法", q.stderr)
            self.assertNotIn("Traceback", q.stderr)
        finally:
            stop(p)
            close_pipes(p)


class EnvParsing(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in ("HOST", "PORT", "TWQ_HOST")}
        for k in self.saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_defaults(self):
        self.assertEqual(server.env_host(), config.DEFAULT_HOST)
        self.assertEqual(server.env_port(), config.DEFAULT_PORT)

    def test_host_accepts_ip_literals_and_localhost_only(self):
        for val, want in (("127.0.0.1", "127.0.0.1"), ("0.0.0.0", "0.0.0.0"), ("::1", "::1"), ("[::1]", "::1"),
                          ("localhost", "127.0.0.1"), ("192.168.1.5", "192.168.1.5")):
            os.environ["HOST"] = val
            self.assertEqual(server.env_host(), want, val)
        for val in ("x86_64-conda-linux-gnu", "myhost.example.com", "ubuntu", "  "):
            os.environ["HOST"] = val
            self.assertEqual(server.env_host(), config.DEFAULT_HOST, val)
        os.environ["HOST"] = "x86_64-conda-linux-gnu"
        os.environ["TWQ_HOST"] = "127.0.0.1"
        self.assertEqual(server.env_host(), "127.0.0.1")

    def test_port(self):
        for val, want in (("8123", 8123), ("0", 0), ("abc", config.DEFAULT_PORT), ("70000", config.DEFAULT_PORT), ("-5", config.DEFAULT_PORT)):
            os.environ["PORT"] = val
            self.assertEqual(server.env_port(), want, val)


@unittest.skipIf(SLOW, "TWQ_SKIP_SLOW is set")
class RealEngineEndToEnd(unittest.TestCase):
    """Real quant engine + real spawn pool + real evaluation, synthetic prices, faked network."""

    N_BARS = 800

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="twq_e2e_")
        cls.launcher = os.path.join(cls.tmp, "launcher.py")
        with open(cls.launcher, "w") as fh:
            fh.write(LAUNCHER.format(root=ROOT, tests=HERE, n_bars=cls.N_BARS))
        cls.proc, cls.port, _ = start({"TWQ_UNIVERSE_SIZE": "5"}, launcher=cls.launcher, tmpdir=os.path.join(cls.tmp, "cache"))

    proc_stopped = False

    @classmethod
    def tearDownClass(cls):
        if not cls.proc_stopped:
            stop(cls.proc)
        for f in (cls.proc.stdout, cls.proc.stderr):
            f.close()

    def poll(self, path, timeout=240):
        t0, last = time.time(), None
        seen = []
        while time.time() - t0 < timeout:
            st, body = fetch(self.port, path)
            d = sc.strict_loads(body)
            if d.get("status") != "computing":
                return st, d, seen
            self.assertEqual(sc.check_envelope(d), [])
            seen.append(d["progress"]["done"])
            time.sleep(0.5)
        self.fail(f"{path} did not finish in {timeout}s (last progress {seen[-3:]})")

    def test_01_real_forecast_matches_the_schema(self):
        t0 = time.time()
        st, body = fetch(self.port, "/api/forecast?symbol=2330")
        d = sc.strict_loads(body)
        self.assertEqual(st, 200, d)
        self.assertEqual(sc.check_forecast(d), [], sc.check_forecast(d))
        self.assertEqual(d["name"], "台積電")
        self.assertLess(time.time() - t0, 60)
        t1 = time.time()
        self.assertEqual(fetch(self.port, "/api/forecast?symbol=2330")[0], 200)
        self.assertLess(time.time() - t1, 1.0)  # cached

    def test_02_real_recommendations_in_the_pool(self):
        st, d, seen = self.poll("/api/recommendations?h=5")
        self.assertEqual(st, 200, d)
        self.assertEqual(sc.check_recommendations(d, 5, 10), [], sc.check_recommendations(d, 5, 10))
        self.assertEqual(d["n_scored"], 5)

    def test_03_real_backtest(self):
        st, d, seen = self.poll("/api/backtest?symbol=2330&h=5")
        self.assertEqual(st, 200, d)
        self.assertEqual(sc.check_backtest(d, 5), [], sc.check_backtest(d, 5))
        st, d1, _ = self.poll("/api/backtest?symbol=2330&h=1")  # reuses the walk-forward pickle
        self.assertEqual(sc.check_backtest(d1, 1), [])

    def test_04_real_universe_validation_then_sigint_leaves_no_orphans(self):
        st, d, seen = self.poll("/api/backtest/universe?h=5&top=2")
        self.assertEqual(st, 200, d)
        self.assertEqual(sc.check_universe_report(d, 5, 2), [], sc.check_universe_report(d, 5, 2))
        kids = child_pids(self.proc.pid)
        self.assertTrue(kids, "the process pool should have worker processes")
        self.assertLessEqual(len(kids) - 1, max(1, (os.cpu_count() or 2) - 1) + 1)  # <= CPU-1 workers (+ resource tracker)
        rc, dt = stop(self.proc)
        self.assertEqual(rc, 0)
        time.sleep(0.5)
        self.assertEqual([k for k in kids if alive(k)], [], "pool workers survived the server")
        type(self).proc_stopped = True


if __name__ == "__main__":
    unittest.main()
