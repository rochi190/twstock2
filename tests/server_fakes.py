"""Fixtures for the server tests: a fake quant engine, a fake Yahoo/TWSE network (via ``data_fakes``), and a base
class that starts a REAL ``ThreadingHTTPServer`` on an ephemeral port in-process.

* ``FakeEngine`` replaces ``backend.quant.engine`` (monkeypatched through ``jobs._engine``).  It returns payloads that
  honour DESIGN.md 6.1 / types.WalkForward, counts its calls and can be paused (``gate``) to test job sharing.
* The evaluation package (recommend / report / universe_bt) is the REAL one -- it is pure numpy and fast on small data.
* Every network call is faked, nothing is written outside a temp directory.
"""
from __future__ import annotations

import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from datetime import date, timedelta
from unittest import mock

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import schema_check as sc  # noqa: E402
from data_fakes import DataCase, bdays, yahoo_payload  # noqa: E402

from backend import config, data, jobs, server  # noqa: E402
from backend.types import HorizonForecasts, WalkForward  # noqa: E402

END = date(2026, 10, 22)  # Thursday; DataCase.START is that day 16:00 Taipei (after the close)
QKEYS = ("0.05", "0.25", "0.5", "0.75", "0.95")
LABELS = {1: "明日", 5: "未來5日", 20: "未來20日"}


def series_dates(n: int, end: date = END) -> list[date]:
    ds = bdays(end - timedelta(days=int(n * 1.6) + 20), 20_000)
    ds = [d for d in ds if d <= end][-n:]
    assert len(ds) == n and ds[-1] == end
    return ds


def price_series(n: int, seed: int, start: float = 100.0, drift: float = 0.0003, vol: float = 0.015) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.round(start * np.exp(np.cumsum(rng.normal(drift, vol, n))), 2)


# ============================================================================================ fake engine
class FakeEngine:
    """Stand-in for ``backend.quant.engine``: deterministic, instant, call-counting, pausable."""

    def __init__(self):
        self.calls: Counter = Counter()
        self.lock = threading.Lock()
        self.gate: threading.Event | None = None  # when set (cleared event) computations wait for it
        self.fail_for: set[str] = set()  # symbols whose computation raises
        self.nan_driver = False  # put a NaN into the payload to prove the JSON layer maps it to null
        self.zero_sigma = False  # every horizon reports sigma = 0 (what a frozen price series gives, audit P-14)
        self.delay = 0.0
        self.wf_symbols: list[str] = []

    def _pause(self):
        if self.delay:
            time.sleep(self.delay)
        if self.gate is not None:
            assert self.gate.wait(30), "FakeEngine gate never released"

    # ---- live forecast
    def forecast_latest(self, data_, market, horizons=config.HORIZONS, cfg=None, **kw) -> dict:
        with self.lock:
            self.calls["forecast"] += 1
            self.calls[f"forecast:{data_.symbol}"] += 1
        self._pause()
        if data_.symbol in self.fail_for:
            raise RuntimeError("boom (fake engine failure)")
        if data_.n < config.MIN_BARS_FORECAST:
            raise ValueError("insufficient_history")
        return self.core(data_, market)

    def core(self, d, market) -> dict:
        last = float(d.close[-1])
        r = np.diff(np.log(d.adjclose[-250:]))
        sig1 = float(np.std(r)) or 0.01
        hz = {}
        for h in config.HORIZONS:
            sg = sig1 * np.sqrt(h) * 1.05
            mu = 0.0004 * h
            p_raw = float(stats.norm.cdf(mu / sg))
            qs = {k: float(np.exp(mu + stats.norm.ppf(float(k)) * sg) - 1.0) for k in QKEYS}
            if h == 1:
                qs = {k: float(np.clip(v, -0.0999, 0.0999)) for k, v in qs.items()}
            hz[str(h)] = {
                "h": h, "days_label": LABELS[h], "p_up": float(np.clip(p_raw + 0.01, 0.01, 0.99)), "p_up_raw": p_raw,
                "exp_return": float(np.exp(mu + sg ** 2 / 2) - 1.0), "median_return": qs["0.5"], "sigma": float(sg),
                "quantiles": qs, "price_quantiles": {k: last * (1.0 + v) for k, v in qs.items()},
                "components": {c: {"p_up": p_raw, "mu": mu} for c in sc.COMPONENTS}}
        days = list(range(1, config.FAN_DAYS + 1))
        rets = [[float(np.exp(0.0004 * t + stats.norm.ppf(lv) * sig1 * np.sqrt(t)) - 1.0) for t in days]
                for lv in config.FAN_LEVELS]
        fc = {
            "symbol": d.symbol, "asof": str(d.dates[-1]), "last_close": last,
            "chg_1d": float(d.close[-1] / d.close[-2] - 1.0), "n_bars": int(d.n), "horizons": hz,
            "fan": {"days": days, "levels": list(config.FAN_LEVELS),
                    "prices": [[last * (1.0 + x) for x in row] for row in rets], "returns": rets},
            "regime": {"labels": ["低波動盤整", "多頭穩定", "高波動空頭"], "probs": [0.5, 0.3, 0.2], "current": "低波動盤整",
                       "transition": [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.1, 0.1, 0.8]]},
            "vol": {"sigma_daily": sig1, "sigma_annual": sig1 * np.sqrt(245.0), "percentile": 0.5, "regime_note": "測試用"},
            "risk": {"var95_1d": -1.645 * sig1, "cvar95_1d": -2.06 * sig1, "var95_20d": -0.1, "cvar95_20d": -0.14,
                     "median_max_drawdown_20d": -0.05, "prob_loss_gt_10pct_20d": 0.04},
            "models": {"weights": {str(h): {c: 0.25 for c in sc.COMPONENTS} for h in config.HORIZONS},
                       "names": {"garch": "GJR-GARCH-t", "hmm": "隱馬可夫regime", "dlm": "動態線性模型(貝氏)",
                                 "naive": "隨機漫步基準"}},
            "drivers": [{"name": "12-1個月動能", "key": "mom_12_1", "value": 0.3, "z": 1.2, "contribution": 0.0003}],
        }
        if self.nan_driver:
            fc["drivers"][0]["value"] = float("nan")
            fc["vol"]["percentile"] = float("inf")
        if self.zero_sigma:
            for hz_ in fc["horizons"].values():
                hz_["sigma"] = 0.0
        return fc

    # ---- walk-forward
    def walk_forward(self, d, market, horizons=config.HORIZONS, min_train=config.MIN_TRAIN,
                     refit_every=config.REFIT_EVERY, start_idx=None, progress=None, cfg=None) -> WalkForward:
        with self.lock:
            self.calls["wf"] += 1
            self.calls[f"wf:{d.symbol}"] += 1
            self.wf_symbols.append(d.symbol)
        if progress:
            for i in (0, 1, 2, 3, 4):
                progress(i * 25, 100)
        self._pause()
        if d.symbol in self.fail_for:
            raise RuntimeError("boom (fake engine failure)")
        n = d.n
        first = min_train if start_idx is None else int(start_idx)
        a = d.adjclose
        idx = np.arange(n)
        hfs = {}
        for h in horizons:
            realized = np.full(n, np.nan)
            realized[: n - h] = np.log(a[h:] / a[: n - h])
            sigma = np.full(n, float(np.std(np.diff(np.log(a)))) * np.sqrt(h) * 1.05)
            p_up = np.clip(0.5 + 0.09 * np.sin(idx / 9.0 + h), 0.05, 0.95)  # wanders across the 0.55 buy threshold
            mu = sigma * stats.norm.ppf(p_up)
            q = mu[:, None] + sigma[:, None] * stats.norm.ppf(np.asarray(config.QLEVELS))[None, :]
            with np.errstate(invalid="ignore"):
                pit = stats.norm.cdf((realized - mu) / sigma)
            for arr in (p_up, mu, sigma):
                arr[:first] = np.nan
            q[:first] = np.nan
            pit[np.isnan(realized)] = np.nan
            hfs[h] = HorizonForecasts(h=h, p_up=p_up, p_up_raw=p_up.copy(), mu=mu, sigma=sigma, q=q, pit=pit,
                                      realized=realized, comp_p_up={c: p_up.copy() for c in ("garch", "hmm", "dlm", "naive")},
                                      comp_mu={c: mu.copy() for c in ("garch", "hmm", "dlm", "naive")},
                                      weights={c: np.full(n, 0.25) for c in ("garch", "hmm", "dlm", "naive")})
        return WalkForward(symbol=d.symbol, dates=d.dates.copy(), horizons=hfs, first_idx=first, refit_every=refit_every,
                           min_train=min_train, model_names=["garch", "hmm", "dlm", "naive"],
                           regime_prob=np.full((n, 3), 1.0 / 3.0), regime_labels=["低波動盤整", "多頭穩定", "高波動空頭"],
                           vol1d=np.full(n, 0.015), config={}, qlevels=tuple(config.QLEVELS))


# ============================================================================================ fake world
UNIVERSE = [("1101", "台泥"), ("1102", "亞泥"), ("1216", "統一"), ("2002", "中鋼"), ("2330", "台積電")]


def install_world(net, n: int, extra_codes=()) -> list:
    """Route a small fake Taiwan market on ``net`` (a ``data_fakes.FakeNet``): name tables, five TWSE stocks, ETF 0050,
    OTC 6488 (``.TW`` -> 404 then ``.TWO``) and ^TWII, all with ``n`` bars ending 2026-10-22.  Returns the dates."""
    from data_fakes import twse_company, twse_day_all, tpex_day
    ds = series_dates(n)
    net.route("STOCK_DAY_ALL", twse_day_all([(c, nm, 10 ** 10 - i, 100.0) for i, (c, nm) in enumerate(UNIVERSE)]
                                            + [("0050", "元大台灣50", 10 ** 9, 100.0)]))
    net.route("t187ap03_L", twse_company([]))
    net.route("tpex_mainboard", tpex_day([("6488", "環球晶", 7 * 10 ** 9, 400.0)]))

    def chart(ysym, close, vol, **kw):
        net.route(f"/chart/{ysym}?", yahoo_payload(ds, close, volume=vol, **kw))

    for i, (c, _) in enumerate(UNIVERSE):
        chart(f"{c}.TW", price_series(n, seed=10 + i, start=50 + 40 * i), [2_000_000 + 1000 * j for j in range(n)])
    chart("0050.TW", price_series(n, seed=3, start=100), [3_000_000] * n)
    net.route("/chart/6488.TW?", data.NotFoundError("404"))
    chart("6488.TWO", price_series(n, seed=4, start=400), [1_000_000] * n)
    chart("%5ETWII", price_series(n, seed=99, start=15000), [0] * n, itype="INDEX")
    for code in extra_codes:
        net.route(f"/chart/{code}.TW?", data.NotFoundError("404"))
        net.route(f"/chart/{code}.TWO?", data.NotFoundError("404"))
    return ds


# ============================================================================================ server base class


class ServerCase(DataCase):
    """Real in-process HTTP server on 127.0.0.1:<ephemeral> with fake network + fake engine."""

    N_BARS = 900
    FIRST_WAIT = 0.05  # keep "computing" answers snappy in tests

    def setUp(self):
        super().setUp()
        self.engine = FakeEngine()
        p = mock.patch.object(jobs, "_engine", lambda: self.engine)
        p.start()
        self.addCleanup(p.stop)
        self.universe = [{"code": c, "name": nm, "market": "TWSE", "kind": "stock"} for c, nm in UNIVERSE]
        p = mock.patch.object(data, "get_universe", lambda n=config.UNIVERSE_SIZE, refresh=False: self.universe[:n])
        p.start()
        self.addCleanup(p.stop)
        self.route_world()
        # static files
        self.fe_root = tempfile.mkdtemp(prefix="twq_fe_")
        self.addCleanup(shutil.rmtree, self.fe_root, True)
        self.fe = os.path.join(self.fe_root, "frontend")
        os.makedirs(os.path.join(self.fe, "sub"))
        os.makedirs(os.path.join(self.fe, "emptydir"))
        for name, content in (("index.html", "<!doctype html><title>測試</title><h1>首頁</h1>"), ("app.js", "console.log('hi');"),
                              ("style.css", "body{color:red}"), ("data.json", '{"a":1}'), ("logo.svg", "<svg/>"),
                              (".secret", "top secret"), ("sub/page.html", "<p>sub</p>"), ("sub/.hidden", "hidden"),
                              ("notes.txt", "純文字"), ("font.woff2", "wOF2")):
            with open(os.path.join(self.fe, name), "w", encoding="utf-8") as fh:
                fh.write(content)
        with open(os.path.join(self.fe, "pic.png"), "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        with open(os.path.join(self.fe_root, "outside.txt"), "w") as fh:
            fh.write("outside the frontend directory")
        try:
            os.symlink(os.path.join(self.fe_root, "outside.txt"), os.path.join(self.fe, "link.txt"))
        except OSError:  # pragma: no cover - platforms without symlinks
            pass
        self.compute = jobs.Compute(workers=2, use_processes=False, max_jobs=2)
        self.compute.first_wait = self.FIRST_WAIT
        self.compute.universe_size = len(self.universe)
        self.srv = server.make_server("127.0.0.1", 0, compute=self.compute, frontend_dir=self.fe, quiet=True)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)

    def _stop_server(self):
        if self.engine.gate is not None:
            self.engine.gate.set()
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(5)
        self.compute.close()

    # ---- fake world ------------------------------------------------------------------------------------------
    def route_world(self, n=None, extra_codes=()):
        self.ds = install_world(self.net, n or self.N_BARS, extra_codes)

    def route_short(self, code, n):
        """A listed stock with only ``n`` bars."""
        ds = series_dates(n)
        self.route_symbol(f"{code}.TW", ds, price_series(n, seed=7, start=30), volume=[1_000_000] * n)

    # ---- HTTP client -----------------------------------------------------------------------------------------
    def request(self, path, method="GET", headers=None, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            raw = resp.read()
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, raw
        finally:
            conn.close()

    def get(self, path, headers=None):
        return self.request(path, "GET", headers)

    def json(self, path, expect=None):
        status, hdr, raw = self.get(path)
        d = sc.strict_loads(raw)
        if expect is not None:
            self.assertEqual(status, expect, f"{path} -> {status} {d}")
        return status, d, hdr

    def poll(self, path, timeout=60.0, every=0.02):
        """Poll a job endpoint until it leaves ``computing``; returns (http status, body, [progress snapshots])."""
        t0 = time.time()
        trace = []
        while True:
            status, d, _ = self.json(path)
            if d.get("status") != "computing":
                return status, d, trace
            trace.append(dict(d["progress"]))
            self.assertEqual(sc.check_envelope(d), [])
            if time.time() - t0 > timeout:
                self.fail(f"{path} still computing after {timeout}s: {d}")
            time.sleep(every)

    def raw_exchange(self, payload: bytes, read_timeout=5.0) -> bytes:
        """Send raw bytes on a fresh TCP connection and return the whole answer (headers + ``Content-Length`` bytes)."""
        import socket
        with socket.create_connection(("127.0.0.1", self.port), timeout=read_timeout) as s:
            s.sendall(payload)
            s.settimeout(read_timeout)
            buf = b""
            try:
                while True:
                    head, sep, rest = buf.partition(b"\r\n\r\n")
                    if sep:
                        length = 0
                        for line in head.split(b"\r\n")[1:]:
                            k, _, v = line.partition(b":")
                            if k.strip().lower() == b"content-length":
                                length = int(v.strip() or 0)
                        if len(rest) >= length:
                            break
                    c = s.recv(65536)
                    if not c:
                        break
                    buf += c
            except (socket.timeout, ConnectionError):
                pass
            return buf
