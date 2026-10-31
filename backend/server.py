"""HTTP server: JSON API under ``/api/*`` plus the static frontend (``python -m backend.server``).

Standard library only (``ThreadingHTTPServer``).  Environment: ``HOST`` (default 127.0.0.1), ``PORT`` (default 8000);
``--host`` / ``--port`` override them.  The server binds to localhost by default and refuses requests whose ``Host``
header is not a loopback name (DNS-rebinding guard) while bound to a loopback address.

Endpoints (exact shapes in DESIGN.md section 6)::

    GET /api/status
    GET /api/search?q=台積&limit=20                     (empty q: the popular list, always complete whatever ``limit`` is)
    GET /api/history?symbol=2330&limit=750
    GET /api/forecast?symbol=2330                       (synchronous, cached per symbol + last bar)
    GET /api/recommendations?h=5&top=10                 (job: computing -> ready)
    GET /api/backtest?symbol=2330&h=1&thr=0.55&exec=next_open&disc=0.6&slip=5&years=5&sizing=threshold   (job)
    GET /api/backtest/universe?h=5&top=5&exec=next_open&disc=0.6&slip=5                                  (job)

Every response is UTF-8 JSON with NaN/+-inf mapped to ``null``.  Errors use HTTP 4xx/5xx and
``{"status":"error","error":{"code":...,"message":"繁中訊息"}}`` with the codes ``bad_request``, ``unknown_symbol``,
``insufficient_history``, ``data_unavailable``, ``internal``.

Static files come from ``frontend/`` (``/`` -> ``index.html``): path traversal, dotfiles and symlinks leaving the
directory are refused, MIME types are explicit, HTML/JS/CSS carry ``Cache-Control: no-cache`` plus an ETag
(conditional requests answer 304).
"""
from __future__ import annotations

import argparse
import ipaddress
import logging
import mimetypes
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import config, data, jobs
from .jobs import ApiError, Compute, dumps_json, to_api_error

log = logging.getLogger("twquant.server")

VERSION = jobs.VERSION
MAX_HISTORY = 20000
CSP = ("default-src 'self' data: blob:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; "
       "base-uri 'self'; frame-ancestors 'none'")
SECURITY_HEADERS = (("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "no-referrer"),
                    ("X-Frame-Options", "DENY"), ("Content-Security-Policy", CSP))

MIME = {
    ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8", ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8", ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".ico": "image/x-icon",
    ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf", ".otf": "font/otf",
    ".webmanifest": "application/manifest+json", ".wasm": "application/wasm",
}
NO_CACHE_EXT = {".html", ".htm", ".js", ".mjs", ".css", ".json", ".map", ".webmanifest", ".svg"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


# ======================================================================================= query parsing
class Query:
    """Typed accessors over a parsed query string; violations raise ``bad_request`` (Traditional Chinese)."""

    def __init__(self, qs: dict[str, list[str]]):
        self._qs = qs

    def raw(self, name: str) -> str | None:
        v = self._qs.get(name)
        return v[0] if v else None

    def text(self, name: str, default: str | None = None, required: bool = False, maxlen: int = 80) -> str | None:
        v = self.raw(name)
        if v is None or v.strip() == "":
            if required:
                raise ApiError("bad_request", f"缺少必要參數 {name}。")
            return default
        v = v.strip()
        if len(v) > maxlen:
            raise ApiError("bad_request", f"參數 {name} 太長（上限 {maxlen} 字元）。")
        return v

    def integer(self, name: str, default: int, lo: int, hi: int) -> int:
        v = self.raw(name)
        if v is None or v.strip() == "":
            return default
        try:
            x = int(v)
        except ValueError:
            raise ApiError("bad_request", f"參數 {name} 必須是整數。") from None
        if x < lo or x > hi:
            raise ApiError("bad_request", f"參數 {name} 必須介於 {lo} 與 {hi} 之間。")
        return x

    def number(self, name: str, default: float, lo: float, hi: float) -> float:
        v = self.raw(name)
        if v is None or v.strip() == "":
            return default
        try:
            x = float(v)
        except ValueError:
            raise ApiError("bad_request", f"參數 {name} 必須是數字。") from None
        if not (lo <= x <= hi):  # also rejects NaN
            raise ApiError("bad_request", f"參數 {name} 必須介於 {lo:g} 與 {hi:g} 之間。")
        return x

    def choice(self, name: str, default: str, options: tuple) -> str:
        v = self.raw(name)
        if v is None or v.strip() == "":
            return default
        if v not in options:
            raise ApiError("bad_request", f"參數 {name} 必須是 {'、'.join(options)} 其中之一。")
        return v


# ======================================================================================= application state
class App:
    """Everything the request handlers share."""

    def __init__(self, compute: Compute | None = None, frontend_dir: Path | None = None,
                 allowed_hosts: set[str] | None = None, quiet: bool = False):
        self.compute = compute or Compute()
        self.frontend_dir = Path(frontend_dir or config.FRONTEND_DIR)
        self.allowed_hosts = allowed_hosts
        self.quiet = quiet
        self.started = time.time()

    # ------------------------------------------------------------------ API handlers (return dict payloads)
    def api_status(self, q: Query) -> dict:
        t = data.taipei_now()
        return {"status": "ready", "server_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "taipei_time": t.isoformat(timespec="seconds"), "market_open": data.market_is_open(t),
                "last_trading_date": data.last_trading_date_expected(t).isoformat(), "version": VERSION,
                "modules": {"engine": _module_available("backend.quant.engine"),
                            "evaluation": _module_available("backend.evaluation.report")},
                "jobs": self.compute.jobs.stats(), "uptime_s": round(time.time() - self.started, 1)}

    def api_search(self, q: Query) -> dict:
        text = q.text("q", "", maxlen=40) or ""
        limit = q.integer("limit", 20, 1, 50)
        if _fold_empty(text):
            limit = max(limit, data.POPULAR_COUNT)  # the documented popular list (12 universe names + 3 ETFs) is never cut
        try:
            results = data.search_instruments(text, limit=limit)
        except data.DataError as exc:
            raise to_api_error(exc) from None
        return {"status": "ready", "q": text, "results": results}

    def api_history(self, q: Query) -> dict:
        raw = q.text("symbol", required=True)
        limit = q.integer("limit", 750, 1, MAX_HISTORY)
        try:
            sym = data.resolve_symbol(raw)
        except data.BadSymbol as exc:
            raise to_api_error(exc, raw) from None
        try:
            pd = data.get_price_data(sym)
        except (data.DataError, KeyError) as exc:
            raise to_api_error(exc, raw) from None
        lo = max(0, pd.n - limit)
        d = [str(x) for x in pd.dates[lo:]]
        o, h, l, c = (a[lo:].round(4).tolist() for a in (pd.open, pd.high, pd.low, pd.close))
        v = [int(x) for x in pd.volume[lo:]]
        bars = [{"d": d[i], "o": o[i], "h": h[i], "l": l[i], "c": c[i], "v": v[i]} for i in range(len(d))]
        first = pd.dates[lo] if pd.n else None
        divs = [{"d": str(dd), "amt": round(float(a), 4)} for dd, a in pd.dividends if first is not None and dd >= first]
        return {"status": "ready", "symbol": pd.symbol, "name": pd.name, "market": pd.market, "kind": pd.kind,
                "asof": d[-1] if d else None, "last_bar_complete": bool(pd.last_bar_complete),
                "stale": bool(pd.stale), "n_total": int(pd.n), "bars": bars, "dividends": divs}

    def api_forecast(self, q: Query) -> dict:
        raw = q.text("symbol", required=True)
        payload = self.compute.forecast_payload(raw)
        return {"status": "ready", **payload}

    def api_recommendations(self, q: Query) -> dict:
        h = jobs.validate_horizon(q.raw("h") if q.raw("h") not in (None, "") else 5)
        top = q.integer("top", 10, 1, 50)
        return self.compute.recommendations(h, top)

    def api_backtest(self, q: Query) -> dict:
        p = jobs.build_backtest_params(
            q.text("symbol", required=True),
            h=q.raw("h") if q.raw("h") not in (None, "") else 1,
            thr=q.raw("thr") if q.raw("thr") not in (None, "") else config.BUY_P_UP,
            exec_mode=q.text("exec", "next_open") or "next_open",
            disc=q.raw("disc") if q.raw("disc") not in (None, "") else config.COMMISSION_DISCOUNT,
            slip=q.raw("slip") if q.raw("slip") not in (None, "") else config.SLIPPAGE_BPS,
            years=q.raw("years") if q.raw("years") not in (None, "") else jobs.DEFAULT_YEARS,
            sizing=q.text("sizing", "threshold") or "threshold")
        return self.compute.backtest(p)

    def api_universe_backtest(self, q: Query) -> dict:
        h = jobs.validate_horizon(q.raw("h") if q.raw("h") not in (None, "") else 5)
        top = q.integer("top", 5, 1, 20)
        exec_mode = q.choice("exec", "next_open", jobs.EXEC_MODES)
        disc = q.number("disc", config.COMMISSION_DISCOUNT, 0.0, 1.0)
        slip = q.number("slip", config.SLIPPAGE_BPS, 0.0, 100.0)
        return self.compute.universe_backtest(h, top, exec_mode, disc, slip)

    def routes(self) -> dict:
        return {"/api/status": self.api_status, "/api/search": self.api_search, "/api/history": self.api_history,
                "/api/forecast": self.api_forecast, "/api/recommendations": self.api_recommendations,
                "/api/backtest/universe": self.api_universe_backtest, "/api/backtest": self.api_backtest}

    def close(self) -> None:
        self.compute.close()


def _host_name(host: str) -> str:
    """Hostname part of a ``Host`` header value (port stripped, IPv6 literals keep their brackets)."""
    host = host.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    if host.count(":") == 1:
        return host.split(":", 1)[0]
    return host


def _fix_utf8(raw_path: str) -> str:
    """``http.server`` decodes the request line as latin-1; clients such as curl send raw UTF-8 (``?q=台積``).

    Re-decode such paths as UTF-8 (percent-encoded paths are pure ASCII and pass through untouched)."""
    if raw_path.isascii():
        return raw_path
    try:
        return raw_path.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw_path


def _fold_empty(text: str) -> bool:
    """Is the query empty after the folding ``data.search_instruments`` applies (so it returns the popular list)?"""
    return not data._fold(text or "")


def _module_available(name: str) -> bool:
    """Does the source file of ``backend.x.y`` exist?  (No import: ``backend.quant`` loads scipy eagerly.)"""
    rel = Path(*name.split(".")[1:]).with_suffix(".py")
    return (Path(config.ROOT) / "backend" / rel).is_file()


# ======================================================================================= request handler
class Handler(BaseHTTPRequestHandler):
    server_version = "TWQuantLab/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = 120  # idle keep-alive connections are closed after two minutes

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ logging
    def log_message(self, fmt, *args):  # noqa: D401 - silence the default stderr format
        pass

    def _log_request(self, status: int, started: float, quiet: bool = False) -> None:
        if self.app.quiet or quiet:
            return
        log.info("%s %s -> %d (%.0f ms)", self.command, self.path.split("?")[0][:80], status,
                 (time.time() - started) * 1000)

    # ------------------------------------------------------------------ low level responses
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in SECURITY_HEADERS:
            self.send_header(k, v)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head and status != HTTPStatus.NOT_MODIFIED:
            self.wfile.write(body)

    def _send_json(self, obj: dict, status: int = 200, head: bool = False) -> int:
        try:
            body = dumps_json(obj)
        except Exception as exc:  # noqa: BLE001 - a payload that cannot be serialised is a server bug
            log.error("cannot serialise response: %s", exc, exc_info=True)
            status = 500
            body = dumps_json(ApiError("internal", "伺服器內部發生錯誤，請稍後再試。").to_dict())
        self._send(status, body, "application/json; charset=utf-8", {"Cache-Control": "no-store"}, head)
        return status

    def _send_error_json(self, err: ApiError, head: bool = False) -> int:
        extra = {"Retry-After": "5"} if err.code == "data_unavailable" else None
        body = dumps_json(err.to_dict())
        self._send(err.status, body, "application/json; charset=utf-8",
                   dict({"Cache-Control": "no-store"}, **(extra or {})), head)
        return err.status

    # ------------------------------------------------------------------ dispatch
    def do_GET(self):
        self._handle(head=False)

    def do_HEAD(self):
        self._handle(head=True)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Allow", "GET, HEAD, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _method_not_allowed(self):
        self.close_connection = True  # the request body (if any) is not read: do not parse it as the next request
        body = dumps_json(ApiError("bad_request", "只支援 GET / HEAD 請求。", status=405).to_dict())
        self._send(405, body, "application/json; charset=utf-8",
                   {"Allow": "GET, HEAD, OPTIONS", "Cache-Control": "no-store", "Connection": "close"})

    do_POST = do_PUT = do_DELETE = do_PATCH = _method_not_allowed

    def _host_ok(self) -> bool:
        allowed = self.app.allowed_hosts
        if allowed is None:
            return True
        host = self.headers.get("Host")
        if not host:
            return True  # HTTP/1.0 clients without Host
        return _host_name(host) in allowed

    def _handle(self, head: bool) -> None:
        started = time.time()
        status = 500
        quiet = False
        try:
            if not self._host_ok():
                status = self._send_error_json(ApiError("bad_request", "不允許的 Host 標頭（僅限本機存取）。", status=403), head)
                return
            parts = urlsplit(_fix_utf8(self.path))
            path = "/" + parts.path.lstrip("/")
            if path.startswith("/api/") or path == "/api":
                status, quiet = self._handle_api(path.rstrip("/"), parts.query, head)
            else:
                status = self._serve_static(path, head)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return  # client went away
        except Exception as exc:  # noqa: BLE001 - last line of defence: structured 500
            log.error("unhandled error for %s: %s", self.path, exc, exc_info=True)
            try:
                status = self._send_error_json(to_api_error(exc), head)
            except Exception:  # noqa: BLE001
                return
        self._log_request(status, started, quiet)

    def _handle_api(self, path: str, query: str, head: bool) -> tuple[int, bool]:
        fn = self.app.routes().get(path)
        if fn is None:
            return self._send_error_json(ApiError("bad_request", f"找不到此 API 路徑：{path[:60]}", status=404), head), False
        try:
            qs = parse_qs(query, keep_blank_values=True, max_num_fields=50, encoding="utf-8", errors="replace")
        except ValueError:
            return self._send_error_json(ApiError("bad_request", "查詢字串格式錯誤。"), head), False
        try:
            payload = fn(Query(qs))
        except ApiError as err:
            return self._send_error_json(err, head), False
        except Exception as exc:  # noqa: BLE001
            err = to_api_error(exc, (qs.get("symbol") or [None])[0])
            if err.code == "internal":
                log.error("%s failed: %s", path, exc, exc_info=True)
            return self._send_error_json(err, head), False
        st = payload.get("status")
        if st == "error":
            err_info = payload.get("error") or {}
            err = ApiError(err_info.get("code", "internal"), err_info.get("message", "未知錯誤"))
            return self._send_error_json(err, head), False
        return self._send_json(payload, 200, head), st == "computing"

    # ------------------------------------------------------------------ static files
    def _resolve_static(self, path: str) -> Path | None:
        """Map a URL path to a file under ``frontend/`` or ``None``; never leaves the directory."""
        rel = unquote(path, errors="replace")
        if "\x00" in rel or "\\" in rel:
            return None
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if any(p == ".." or p.startswith(".") or len(p) > 255 for p in parts):
            return None
        root = self.app.frontend_dir.resolve()
        try:
            target = root.joinpath(*parts).resolve()
            if target != root and root not in target.parents:
                return None
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                return None
        except (OSError, RuntimeError, ValueError):  # ENAMETOOLONG, symlink loops, ...
            return None
        return target

    def _not_modified_since(self, mtime: float) -> bool:
        ims = self.headers.get("If-Modified-Since")
        if not ims:
            return False
        try:
            return int(mtime) <= int(parsedate_to_datetime(ims).timestamp())
        except (TypeError, ValueError, OverflowError):
            return False

    def _serve_static(self, path: str, head: bool) -> int:
        target = self._resolve_static(path)
        if target is None:
            msg = ("<!doctype html><meta charset=utf-8><title>找不到頁面</title><h1>404 找不到頁面</h1>"
                   "<p>前端檔案不存在或路徑不正確。</p>").encode("utf-8")
            self._send(404, msg, "text/html; charset=utf-8", {"Cache-Control": "no-cache"}, head)
            return 404
        ext = target.suffix.lower()
        ctype = MIME.get(ext) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        st = target.stat()
        etag = f'W/"{st.st_size:x}-{st.st_mtime_ns:x}"'
        cache = "no-cache" if ext in NO_CACHE_EXT else "public, max-age=3600"
        extra = {"Cache-Control": cache, "ETag": etag,
                 "Last-Modified": self.date_time_string(int(st.st_mtime))}
        inm = self.headers.get("If-None-Match")
        if inm is not None:
            if inm.strip() == "*" or any(tag.strip().removeprefix("W/") == etag.removeprefix("W/")
                                          for tag in inm.split(",")):
                self._send(304, b"", ctype, extra, True)
                return 304
        elif self._not_modified_since(st.st_mtime):  # only consulted when there is no If-None-Match (RFC 9110)
            self._send(304, b"", ctype, extra, True)
            return 304
        with open(target, "rb") as fh:
            body = fh.read()
        self._send(200, body, ctype, extra, head)
        return 200


# ======================================================================================= server factory / main
class QuantHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64
    app: App

    def handle_error(self, request, client_address):
        """A browser closing a tab mid-response is not an error worth a traceback on stderr."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError, socket.timeout, BrokenPipeError)):
            return
        log.error("error while handling a request from %s", client_address, exc_info=True)


def make_server(host: str = config.DEFAULT_HOST, port: int = config.DEFAULT_PORT, *, compute: Compute | None = None,
                frontend_dir: Path | None = None, quiet: bool = False) -> QuantHTTPServer:
    """Create (but do not start) the server; ``port=0`` picks a free port (``server.server_address[1]``)."""
    allowed = set(LOOPBACK_HOSTS) if host in LOOPBACK_HOSTS else None  # DNS-rebinding guard only for loopback binds
    app = App(compute=compute, frontend_dir=frontend_dir, allowed_hosts=allowed, quiet=quiet)
    cls = QuantHTTPServer
    if ":" in host:
        cls = type("QuantHTTPServer6", (QuantHTTPServer,), {"address_family": socket.AF_INET6})
    srv = cls((host, port), Handler)
    srv.app = app
    return srv


def _warm(app: App) -> None:
    try:
        data.warm_up()
        app.compute.prune()
    except Exception as exc:  # noqa: BLE001
        log.warning("warm-up failed: %s", exc)


def env_host() -> str:
    """Bind address from ``HOST`` (DESIGN.md).  Only IP literals and ``localhost`` are honoured: conda and some shells
    export ``HOST`` as a machine / build-triplet *name* (``x86_64-conda-linux-gnu``) which must never decide where a
    server listens.  ``TWQ_HOST`` is an unambiguous override."""
    for var in ("TWQ_HOST", "HOST"):
        val = (os.environ.get(var) or "").strip()
        if not val:
            continue
        try:
            ipaddress.ip_address(val.strip("[]"))
            return val.strip("[]")
        except ValueError:
            if val.lower() == "localhost":
                return "127.0.0.1"
            print(f"注意：忽略環境變數 {var}={val!r}（不是 IP 位址），改用 {config.DEFAULT_HOST}。", file=sys.stderr)
    return config.DEFAULT_HOST


def env_port() -> int:
    try:
        port = int(os.environ.get("PORT", config.DEFAULT_PORT))
    except ValueError:
        print(f"注意：環境變數 PORT={os.environ.get('PORT')!r} 不是整數，改用 {config.DEFAULT_PORT}。", file=sys.stderr)
        return config.DEFAULT_PORT
    return port if 0 <= port <= 65535 else config.DEFAULT_PORT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.server", description="台股量化實驗室 — 本機網頁伺服器")
    ap.add_argument("--host", default=env_host(), help="綁定位址（預設 127.0.0.1，僅限本機；環境變數 HOST）")
    ap.add_argument("--port", type=int, default=env_port(), help="連接埠（預設 8000；環境變數 PORT；0 = 自動挑選）")
    ap.add_argument("--quiet", action="store_true", help="不印出每個請求")
    ap.add_argument("--no-warm", action="store_true", help="啟動時不預先載入名稱/行事曆")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        srv = make_server(args.host, args.port, quiet=args.quiet)
    except OSError as exc:
        print(f"無法在 {args.host}:{args.port} 啟動伺服器：{exc}\n請改用其他連接埠，例如：PORT=8001 ./run.sh", file=sys.stderr)
        return 2
    host, port = srv.server_address[:2]
    shown = "localhost" if host in ("127.0.0.1", "::1", "") else host
    print(f"台股量化實驗室已啟動：http://{shown}:{port}/  （Ctrl-C 結束）", flush=True)
    if host not in LOOPBACK_HOSTS:
        print("警告：伺服器綁定在非本機位址，區域網路中的其他裝置也能存取。", flush=True)
    if not args.no_warm:
        threading.Thread(target=_warm, args=(srv.app,), name="warmup", daemon=True).start()

    def _term(signum, frame):  # SIGTERM -> same clean path as Ctrl-C
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _term)
        if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:  # started as `cmd &` from a non-interactive shell
            signal.signal(signal.SIGINT, signal.default_int_handler)
    except ValueError:
        pass
    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n收到中斷訊號，正在關閉…", flush=True)
    finally:
        try:
            srv.server_close()
        finally:
            srv.app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
