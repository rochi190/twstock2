"""Compute orchestration: forecast cache, walk-forward cache, process pool, background jobs with progress.

Layers
------
* :func:`to_jsonable` -- the JSON sanitiser used for every payload (numpy -> python, NaN/+-inf -> null,
  dates -> ``YYYY-MM-DD``).
* :class:`ApiError` / :func:`to_api_error` -- structured errors (``bad_request``, ``unknown_symbol``,
  ``insufficient_history``, ``data_unavailable``, ``internal``) with Traditional-Chinese messages.
* :class:`ComputePool` -- a lazily created ``ProcessPoolExecutor`` (spawn context, <= CPU-1 workers, BLAS pinned to
  one thread, SIGINT ignored in children) for the CPU heavy walk-forwards / forecasts.  Worker entry points are the
  top level ``_task_*`` functions below (picklable).  Walk-forward progress is reported from the child through a
  shared integer array (one slot per running task) that the parent polls.
* :class:`JobManager` -- background jobs keyed by a string: identical concurrent requests share one job, running
  jobs expose ``done/total/message``, finished results stay in memory (TTL) and errors are kept for a few seconds.
* :class:`Compute` -- the facade used by the HTTP layer and the ``precompute`` CLI.  It owns the caches
  (forecast cores: memory + ``forecast/*.json``; walk-forwards: memory + ``wf/*.pkl.gz``; finished job payloads:
  ``results/*.json``) and implements, strictly through the DESIGN.md signatures of ``quant.engine`` and
  ``evaluation.*``, the pipelines data -> engine -> evaluation for forecast / recommendations / backtest /
  universe validation.

The quant and evaluation packages are imported lazily (``_engine()`` etc.) so the server starts without them.

Cache keys never mention wall-clock "today": they combine the *actual* last completed bar date, a hash of ALL price
columns (open / high / low / close / adjusted close / volume), the request parameters and a fingerprint of the source files
of ``backend/quant`` (and ``evaluation`` for reports) together with the ``backend.config`` constants and ``types.py``
(audit C-05), so an updated model, a changed constant or a revised price history can never serve stale numbers.
"""
from __future__ import annotations

import concurrent.futures as cf
from concurrent.futures.process import BrokenProcessPool
import copy
import ctypes
import dataclasses
import gc
import gzip
import hashlib
import importlib
import inspect
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import signal
import threading
import time
import traceback
from collections import Counter, OrderedDict
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import numpy as np

from . import config, data
from .types import PriceData, WalkForward

log = logging.getLogger("twquant.jobs")

VERSION = "1.0"
DISCLAIMER = ("本工具為統計模型之研究用途，非投資建議。預測為機率分布而非保證，過去績效不代表未來，"
              "投資一定有風險，請自行判斷並承擔盈虧。")
DEFAULT_YEARS = 5.0
ASOF_SUPPORT = 0.3  # a last-bar date must be carried by this share of the scanned names (and >= 2) to be the reference
EVIDENCE_MAX_AGE_DAYS = 14.0  # a cached universe validation older than this is not shown as evidence (= prune_cache's limit)
BIG_MOVE = 0.11  # |1-day change| above this exceeds the +-10 % daily limit: flagged as a data-quality note
MIN_BACKTEST_TEST_BARS = 120  # fewer out-of-sample bars than this -> insufficient_history
MIN_UNIVERSE_BARS = config.MIN_TRAIN + 250
SHORT_HISTORY_BARS = config.MIN_TRAIN + 250


# ======================================================================================= JSON sanitiser
def to_jsonable(o):
    """Recursively convert ``o`` into JSON-safe python objects.

    numpy scalars/arrays -> python, NaN / +-inf -> ``None``, ``datetime64`` / ``date`` -> ``YYYY-MM-DD``,
    ``datetime`` -> ISO-8601, dict keys -> ``str``, dataclasses -> dicts, sets/tuples -> lists, ``Path`` -> str.
    """
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, int):
        return o
    if isinstance(o, float):  # also np.float64, which subclasses float: hand out a plain python float
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.generic):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            f = float(o)
            return f if math.isfinite(f) else None
        if isinstance(o, np.datetime64):
            return None if np.isnat(o) else str(o.astype("datetime64[D]"))
        return to_jsonable(o.item())
    if isinstance(o, np.ndarray):
        if o.dtype.kind == "M":
            return [None if np.isnat(x) else str(x.astype("datetime64[D]")) for x in o.ravel()] \
                if o.ndim == 1 else to_jsonable(o.tolist())
        return to_jsonable(o.tolist())
    if isinstance(o, dict):
        return {_json_key(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, datetime):
        return o.isoformat(timespec="seconds")
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return to_jsonable(dataclasses.asdict(o))
    return str(o)


def _json_key(k) -> str:
    if isinstance(k, str):
        return k
    if isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)):
        return str(int(k))
    return str(k)


def dumps_json(o) -> bytes:
    """UTF-8 JSON bytes of ``to_jsonable(o)`` (``allow_nan=False`` guarantees valid JSON)."""
    return json.dumps(to_jsonable(o), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


# ======================================================================================= errors
HTTP_STATUS = {"bad_request": 400, "unknown_symbol": 404, "insufficient_history": 422, "data_unavailable": 503,
               "internal": 500}


class ApiError(Exception):
    """Structured API error (code, Traditional-Chinese message, HTTP status)."""

    def __init__(self, code: str, message: str, status: int | None = None, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status or HTTP_STATUS.get(code, 500)
        self.detail = detail

    def to_dict(self) -> dict:
        """The wire format.  ``detail`` (exception text, URLs, paths) stays server side: it is logged, never sent."""
        return {"status": "error", "error": {"code": self.code, "message": self.message}}


def to_api_error(exc: BaseException, symbol: str | None = None) -> ApiError:
    """Map any exception to an :class:`ApiError` (unknown ones become ``internal``)."""
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, data.BadSymbol):
        return ApiError("bad_request", f"股票代號格式不正確：{symbol or exc}（請輸入 4～6 碼的上市櫃代號，例如 2330、0050）。")
    if isinstance(exc, data.UnknownSymbol):  # a bare KeyError from a bug elsewhere must stay an `internal` error
        shown = symbol or (str(exc.args[0]) if exc.args else "")
        return ApiError("unknown_symbol", f"找不到股票代號「{shown}」，請確認是否為上市(櫃)股票或 ETF 代號。")
    if isinstance(exc, data.DataUnavailable):
        return ApiError("data_unavailable", "無法取得行情資料（網路或資料來源暫時無法使用），請稍後再試。", detail=str(exc))
    if isinstance(exc, ValueError) and exc.args and exc.args[0] == "insufficient_history":
        return ApiError("insufficient_history", "歷史資料不足，無法產生可靠的預測。")
    if isinstance(exc, ModuleNotFoundError) and str(getattr(exc, "name", "")).startswith("backend."):
        return ApiError("internal", "運算模組尚未安裝或載入失敗，請聯絡維護者。", detail=repr(exc))
    return ApiError("internal", "伺服器內部發生錯誤，請稍後再試。", detail=f"{type(exc).__name__}: {exc}")


def insufficient(name: str, n: int, need: int, what: str = "預測") -> ApiError:
    return ApiError("insufficient_history",
                    f"「{name}」只有 {n} 個交易日的歷史資料，少於{what}所需的 {need} 個，無法產生可靠的結果。")


# ======================================================================================= lazy module seams
def _engine():
    return importlib.import_module("backend.quant.engine")


def _report():
    return importlib.import_module("backend.evaluation.report")


def _recommend():
    return importlib.import_module("backend.evaluation.recommend")


def _universe_bt():
    return importlib.import_module("backend.evaluation.universe_bt")


_FP: dict[tuple, str] = {}
_FP_LOCK = threading.Lock()


_PATH_CONSTANTS = ("ROOT", "CACHE_DIR", "FRONTEND_DIR")  # machine specific, not model inputs
CACHE_SCHEMA = 2  # bumped whenever the layout / meaning of a cached payload changes (v2: audit fixes, placeholder-free prices)


def config_fingerprint() -> str:
    """Hash of the numbers the models and the evaluation take from ``backend.config`` (horizons, ``MIN_TRAIN``,
    ``REFIT_EVERY``, quantile grid, daily limit, tax / commission / slippage ...) and of the source of ``roundtrip_cost`` and
    ``backend/types.py`` (audit C-05): editing one of them changes every forecast / walk-forward / report key, while a
    comment-only edit or moving the project folder does not."""
    h = hashlib.sha1()
    for name in sorted(n for n in dir(config) if n.isupper() and n not in _PATH_CONSTANTS):
        h.update(f"{name}={getattr(config, name)!r};".encode())
    try:
        h.update(inspect.getsource(config.roundtrip_cost).encode())
        h.update((Path(__file__).resolve().parent / "types.py").read_bytes())
    except (OSError, TypeError):
        h.update(b"no-source")
    return h.hexdigest()[:10]


def code_fingerprint(*packages: str) -> str:
    """Hash of (name, size, mtime) of every ``*.py`` in the given ``backend`` sub-packages plus :func:`config_fingerprint`,
    computed once per process.

    Keys of cached forecasts / walk-forwards / reports contain it, so editing the model or a configuration constant
    invalidates old caches.
    """
    key = tuple(packages)
    with _FP_LOCK:
        if key not in _FP:
            h = hashlib.sha1()
            h.update(config_fingerprint().encode())
            root = Path(__file__).resolve().parent
            for pkg in packages:
                for f in sorted((root / pkg).glob("*.py")):
                    try:
                        st = f.stat()
                    except OSError:
                        continue
                    h.update(f"{pkg}/{f.name}:{st.st_size}:{st.st_mtime_ns};".encode())
            _FP[key] = h.hexdigest()[:10]
        return _FP[key]


def _fp_quant() -> str:
    return code_fingerprint("quant")


def _fp_all() -> str:
    return code_fingerprint("quant", "evaluation")


# ======================================================================================= small utilities
def release_memory() -> None:
    """Hand freed heap back to the OS after a big job (glibc keeps per-thread arenas; a 60-stock validation would
    otherwise leave the server at ~400 MB RSS for good).  No-op where ``malloc_trim`` does not exist."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class LRU:
    """Tiny thread-safe LRU mapping."""

    def __init__(self, cap: int):
        self.cap = cap
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.cap:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


class SingleFlight:
    """Run ``fn`` once per key even when many threads ask simultaneously; the others get the same result/exception."""

    def __init__(self):
        self._lock = threading.Lock()
        self._calls: dict = {}

    def do(self, key, fn):
        with self._lock:
            fut = self._calls.get(key)
            owner = fut is None
            if owner:
                fut = self._calls[key] = cf.Future()
        if not owner:
            return fut.result()
        try:
            res = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised for every waiter
            fut.set_exception(exc)
            raise
        else:
            fut.set_result(res)
            return res
        finally:
            with self._lock:
                self._calls.pop(key, None)


def _hash(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


class DiskStore:
    """Atomic JSON / gzip-pickle files under ``root`` (``kind/<label>_<hash>.<ext>``)."""

    def __init__(self, root_fn: Callable[[], Path]):
        self._root_fn = root_fn

    @staticmethod
    def safe(label: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)

    def path(self, kind: str, label: str, key: str, ext: str) -> Path:
        return self._root_fn() / kind / f"{self.safe(label)}_{key}.{ext}"

    def load_json(self, kind: str, label: str, key: str):
        return data.read_json_file(self.path(kind, label, key, "json"))

    def save_json(self, kind: str, label: str, key: str, obj, supersede: str | None = None) -> None:
        """Write atomically; ``supersede`` is a glob (inside the same directory, already ``safe``) of older
        generations to delete."""
        p = self.path(kind, label, key, "json")
        data.atomic_write_json(p, to_jsonable(obj))
        if supersede:
            self._prune_siblings(p, supersede)

    def load_pickle(self, kind: str, label: str, key: str):
        p = self.path(kind, label, key, "pkl.gz")
        try:
            with gzip.open(p, "rb") as fh:
                return pickle.load(fh)
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 - corrupt / incompatible pickle -> recompute
            log.warning("unreadable pickle cache %s (%s)", p, exc)
            return None

    def save_pickle(self, kind: str, label: str, key: str, obj, supersede: str | None = None) -> None:
        p = self.path(kind, label, key, "pkl.gz")
        buf = gzip.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=1)
        data.atomic_write_bytes(p, buf)
        if supersede:
            self._prune_siblings(p, supersede)

    @staticmethod
    def _prune_siblings(path: Path, pattern: str) -> None:
        """Delete the other files of the same directory matching ``pattern`` (superseded cache generations)."""
        try:
            for f in path.parent.glob(pattern):
                if f != path and not f.name.startswith("."):
                    f.unlink(missing_ok=True)
        except OSError:
            pass


def prune_cache(root: Path, max_age_days: float = 14.0) -> int:
    """Remove job-cache files (forecast/, wf/, results/) older than ``max_age_days`` and stray temp files."""
    n = 0
    cutoff = time.time() - max_age_days * 86400
    for sub in ("forecast", "wf", "results"):
        d = Path(root) / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    n += 1
            except OSError:
                pass
    for d in (Path(root), Path(root) / "prices", Path(root) / "forecast", Path(root) / "wf", Path(root) / "results"):
        if not d.is_dir():
            continue
        for f in d.glob(".*.tmp"):
            try:
                if f.stat().st_mtime < time.time() - 3600:
                    f.unlink()
            except OSError:
                pass
    return n


# ======================================================================================= process pool
_NSLOTS = 64
_P_ARR = None  # shared progress array (set in children by the initializer; in thread mode by ComputePool)


def _pool_init(arr) -> None:
    """Child initializer: ignore SIGINT (the parent terminates us), lower priority, remember the progress array."""
    global _P_ARR
    _P_ARR = arr
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    try:
        os.nice(5)
    except (OSError, AttributeError):
        pass


def _slot_progress(slot: int):
    arr = _P_ARR
    if arr is None or slot < 0:
        return None

    def cb(done, total):
        try:
            arr[2 * slot] = int(done)
            arr[2 * slot + 1] = int(total)
        except Exception:  # noqa: BLE001 - progress must never break the computation
            pass

    return cb


def _task_ping() -> int:
    return os.getpid()


def _task_selftest(x):  # used by the tests to prove that the spawn pool works
    return x * x


def _task_forecast(dc: PriceData, mc: PriceData | None) -> dict:
    """Worker: ``engine.forecast_latest`` -> JSON-safe dict."""
    return to_jsonable(_engine().forecast_latest(dc, mc))


def _task_walk_forward(dc: PriceData, mc: PriceData | None, start_idx: int, slot: int) -> WalkForward:
    """Worker: ``engine.walk_forward`` over all default horizons with progress reported through ``slot``."""
    return _engine().walk_forward(dc, mc, start_idx=start_idx, progress=_slot_progress(slot))


class ComputePool:
    """Process pool (spawn) for CPU-heavy tasks, or a thread pool when ``use_processes=False`` (tests)."""

    def __init__(self, workers: int | None = None, use_processes: bool = True):
        cpu = os.cpu_count() or 2
        self.workers = max(1, int(workers) if workers else max(1, cpu - 1))
        self.use_processes = use_processes
        self._lock = threading.Lock()
        self._pool = None
        self._closed = False
        self._ctx = mp.get_context("spawn")
        self._arr = self._ctx.Array("i", 2 * _NSLOTS, lock=False) if use_processes else [0] * (2 * _NSLOTS)
        self._free = list(range(_NSLOTS))
        self._slot_lock = threading.Lock()

    # ---- pool lifecycle
    def _ensure(self):
        global _P_ARR
        with self._lock:
            if self._closed:
                raise RuntimeError("compute pool is closed")
            if self._pool is None:
                if self.use_processes:
                    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                        os.environ.setdefault(var, "1")  # inherited by the children; the parent's BLAS is already up
                    self._pool = cf.ProcessPoolExecutor(max_workers=self.workers, mp_context=self._ctx,
                                                        initializer=_pool_init, initargs=(self._arr,))
                else:
                    _P_ARR = self._arr
                    self._pool = cf.ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="compute")
            return self._pool

    def _reset(self, broken) -> None:
        with self._lock:
            if self._pool is broken:
                self._pool = None
        try:
            broken.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass

    def submit(self, fn, *args) -> cf.Future:
        for attempt in (0, 1):
            pool = self._ensure()
            try:
                return pool.submit(fn, *args)
            except (BrokenProcessPool, RuntimeError) as exc:
                if self._closed:
                    raise
                log.warning("compute pool unusable (%s) -- recreating", exc)
                self._reset(pool)
        raise RuntimeError("compute pool unavailable")

    def warm(self) -> None:
        """Start every worker (imports numpy/scipy/engine in the children) without waiting."""
        for _ in range(self.workers):
            self.submit(_task_ping)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            pool, self._pool = self._pool, None
        if pool is None:
            return
        procs = list(getattr(pool, "_processes", {}).values()) if self.use_processes else []
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass

    # ---- progress slots
    def acquire_slot(self) -> int:
        with self._slot_lock:
            if not self._free:
                return -1
            s = self._free.pop()
        self._arr[2 * s] = 0
        self._arr[2 * s + 1] = 0
        return s

    def release_slot(self, slot: int) -> None:
        if slot >= 0:
            with self._slot_lock:
                self._free.append(slot)

    def read_slot(self, slot: int) -> tuple[int, int]:
        return int(self._arr[2 * slot]), int(self._arr[2 * slot + 1])

    # ---- helpers
    def run(self, fn, *args, retries: int = 1):
        """Submit and wait; re-submit once if the pool broke (a worker was killed)."""
        for attempt in range(retries + 1):
            fut = self.submit(fn, *args)
            try:
                return fut.result()
            except BrokenProcessPool:
                log.warning("worker died while running %s -- retrying", getattr(fn, "__name__", fn))
                with self._lock:
                    broken = self._pool
                if broken is not None:
                    self._reset(broken)
                if attempt >= retries:
                    raise
        raise RuntimeError("unreachable")

    def run_walk_forward(self, dc: PriceData, mc, start_idx: int, on_progress: Callable | None = None):
        """Walk-forward in the pool, polling the child's progress slot every 0.4 s."""
        slot = self.acquire_slot()
        try:
            for attempt in (0, 1):
                fut = self.submit(_task_walk_forward, dc, mc, start_idx, slot)
                try:
                    while True:
                        try:
                            return fut.result(timeout=0.4)
                        except cf.TimeoutError:
                            if on_progress is not None and slot >= 0:
                                on_progress(*self.read_slot(slot))
                except BrokenProcessPool:
                    log.warning("worker died during walk-forward -- retrying once")
                    with self._lock:
                        broken = self._pool
                    if broken is not None:
                        self._reset(broken)
                    if attempt:
                        raise
        finally:
            self.release_slot(slot)

    def map_bounded(self, specs: list[tuple], on_result: Callable, window: int | None = None,
                    cancel: threading.Event | None = None) -> None:
        """Run ``specs = [(key, fn, args)]`` keeping at most ``window`` tasks in flight (default: workers).

        ``on_result(key, value, exception)`` is called as each finishes.  A bounded window keeps interactive
        requests (which share this pool) from queueing behind a whole universe's worth of work.  A task whose
        worker process died is re-submitted once.
        """
        window = max(1, window or self.workers)
        pending: dict = {}
        todo = list(reversed(specs))  # pop() from the end keeps the original order
        retried: set = set()
        while todo or pending:
            while todo and len(pending) < window and not (cancel and cancel.is_set()):
                spec = todo.pop()
                try:
                    pending[self.submit(spec[1], *spec[2])] = spec
                except Exception as exc:  # noqa: BLE001
                    on_result(spec[0], None, exc)
            if not pending:
                return  # cancelled (or nothing left)
            done, _ = cf.wait(list(pending), timeout=0.5, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                spec = pending.pop(fut)
                key = spec[0]
                try:
                    value = fut.result()
                except BrokenProcessPool as exc:
                    if key in retried:
                        on_result(key, None, exc)
                        continue
                    retried.add(key)
                    log.warning("worker died while computing %s -- retrying once", key)
                    with self._lock:
                        broken = self._pool
                    if broken is not None:
                        self._reset(broken)
                    todo.append(spec)
                except Exception as exc:  # noqa: BLE001
                    on_result(key, None, exc)
                else:
                    on_result(key, value, None)


# ======================================================================================= job manager
class Job:
    """One background computation (state is guarded by the owning :class:`JobManager`)."""

    def __init__(self, key: str, kind: str):
        self.key = key
        self.kind = kind
        self.status = "computing"  # computing | ready | error
        self.done = 0
        self.total = 0
        self.message = "排隊中…"
        self.result: dict | None = None
        self.error: ApiError | None = None
        self.created = time.time()
        self.finished: float | None = None
        self.cancel = threading.Event()
        self.finished_event = threading.Event()  # set when the job leaves the "computing" state
        self._lock: threading.Lock | None = None  # the manager's lock, set on registration

    def snapshot(self, wait: float = 0.0) -> dict:
        """Current envelope.  With ``wait`` > 0 a still-running job is given that many seconds to finish first, so
        cache hits and quick failures are answered by the very first request instead of after a polling round trip."""
        if wait > 0 and not self.finished_event.is_set():
            self.finished_event.wait(wait)
        if self._lock is None:
            return self._snapshot()
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> dict:
        if self.status == "ready":
            return {"status": "ready", **(self.result or {})}
        if self.status == "error":
            return self.error.to_dict() if self.error else {"status": "error", "error": {
                "code": "internal", "message": "未知錯誤"}}
        return {"status": "computing", "progress": {"done": int(self.done), "total": int(self.total),
                                                    "message": self.message},
                "elapsed": round(time.time() - self.created, 1)}


class JobManager:
    """Registry of background jobs; identical keys share one job (running, freshly finished or freshly failed)."""

    def __init__(self, max_concurrent: int = 2, ready_ttl: float = 1800.0, error_ttl: float = 15.0,
                 max_entries: int = 128, max_queue: int = 64):
        self._lock = threading.Lock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self.ready_ttl = ready_ttl
        self.error_ttl = error_ttl
        self.max_entries = max_entries  # finished jobs kept (oldest evicted first); running jobs are never evicted
        self.max_queue = max_queue  # running + queued jobs: beyond this new keys are refused instead of piling up threads
        self._closed = False

    def get_or_start(self, key: str, kind: str, runner: Callable[[Callable], dict]) -> Job:
        """Return the live job for ``key`` or start ``runner(progress)`` in a daemon thread.

        ``progress(done, total, message=None)`` updates the job's progress.
        """
        with self._lock:
            if self._closed:
                raise ApiError("internal", "伺服器正在關閉。")
            job = self._jobs.get(key)
            if job is not None:
                age = time.time() - (job.finished or job.created)
                if job.status == "computing" or (job.status == "ready" and age < self.ready_ttl) or \
                        (job.status == "error" and age < self.error_ttl):
                    return job
                self._jobs.pop(key, None)
            if sum(1 for j in self._jobs.values() if j.status == "computing") >= self.max_queue:
                raise ApiError("internal", "目前計算中的工作過多，請稍後再試。", status=503)
            job = Job(key, kind)
            job._lock = self._lock
            self._jobs[key] = job
            self._evict_locked()
        t = threading.Thread(target=self._run, args=(job, runner), name=f"job-{kind}", daemon=True)
        t.start()
        return job

    def _evict_locked(self) -> None:
        while len(self._jobs) > self.max_entries:
            for k, j in self._jobs.items():
                if j.status != "computing":
                    self._jobs.pop(k)
                    break
            else:
                break

    def _run(self, job: Job, runner: Callable) -> None:
        def progress(done, total, message=None):
            with self._lock:
                job.done, job.total = int(done), int(total)
                if message is not None:
                    job.message = str(message)

        try:
            with self._sem:
                if job.cancel.is_set():
                    raise ApiError("internal", "工作已取消。")
                progress(0, 0, "開始計算…")
                result = runner(progress)
            with self._lock:
                job.result = result
                job.done = job.total = max(job.total, job.done)
                job.status = "ready"
                job.finished = time.time()
            job.finished_event.set()
            release_memory()  # ~50 ms; keeps the long-running server's RSS near its working set
        except BaseException as exc:  # noqa: BLE001 - every failure becomes a structured error
            err = to_api_error(exc)
            if err.code == "internal":
                log.error("job %s failed:\n%s", job.key, traceback.format_exc())
            else:
                log.warning("job %s failed: %s%s", job.key, err.message, f" ({err.detail})" if err.detail else "")
            with self._lock:
                job.error = err
                job.status = "error"
                job.finished = time.time()
            job.finished_event.set()

    def stats(self) -> dict:
        with self._lock:
            c = Counter(j.status for j in self._jobs.values())
        return {"computing": c.get("computing", 0), "ready": c.get("ready", 0), "error": c.get("error", 0)}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for j in self._jobs.values():
                j.cancel.set()


# ======================================================================================= parameters
def _num_param(v, name, lo, hi):
    f = float(v)
    if not math.isfinite(f) or f < lo or f > hi:
        raise ApiError("bad_request", f"參數 {name} 必須介於 {lo:g} 與 {hi:g} 之間。")
    return f


@dataclasses.dataclass(frozen=True)
class BacktestParams:
    """Validated parameters of ``/api/backtest`` (``years`` = length of the out-of-sample window)."""

    symbol: str
    h: int = 1
    thr: float = config.BUY_P_UP
    exec: str = "next_open"
    disc: float = config.COMMISSION_DISCOUNT
    slip: float = config.SLIPPAGE_BPS
    years: float = DEFAULT_YEARS
    sizing: str = "threshold"

    def key(self) -> str:
        return _hash(self.symbol, self.h, round(self.thr, 6), self.exec, round(self.disc, 6), round(self.slip, 6),
                     round(self.years, 6), self.sizing)


EXEC_MODES = ("next_open", "close")
SIZINGS = ("threshold", "kelly")


def validate_horizon(h) -> int:
    try:
        hi = int(h)
    except (TypeError, ValueError):
        raise ApiError("bad_request", f"參數 h（預測天數）必須是 {'、'.join(map(str, config.HORIZONS))} 其中之一。") from None
    if hi not in config.HORIZONS:
        raise ApiError("bad_request", f"參數 h（預測天數）必須是 {'、'.join(map(str, config.HORIZONS))} 其中之一。")
    return hi


def build_backtest_params(symbol: str, h=1, thr=config.BUY_P_UP, exec_mode="next_open", disc=config.COMMISSION_DISCOUNT,
                          slip=config.SLIPPAGE_BPS, years=DEFAULT_YEARS, sizing="threshold") -> BacktestParams:
    try:
        sym = data.resolve_symbol(symbol)
    except data.BadSymbol as exc:
        raise to_api_error(exc, str(symbol)) from None
    try:
        thr_f = _num_param(thr, "thr", 0.5, 0.95)
        disc_f = _num_param(disc, "disc", 0.0, 1.0)
        slip_f = _num_param(slip, "slip", 0.0, 100.0)
        years_f = _num_param(years, "years", 1.0, 25.0)
    except (TypeError, ValueError):
        raise ApiError("bad_request", "數值參數格式錯誤（thr、disc、slip、years 必須是數字）。") from None
    if exec_mode not in EXEC_MODES:
        raise ApiError("bad_request", f"參數 exec 必須是 {' 或 '.join(EXEC_MODES)}。")
    if sizing not in SIZINGS:
        raise ApiError("bad_request", f"參數 sizing 必須是 {' 或 '.join(SIZINGS)}。")
    return BacktestParams(sym, validate_horizon(h), thr_f, exec_mode, disc_f, slip_f, years_f, sizing)


def backtest_test_bars(n: int, years: float) -> int:
    """Out-of-sample bars for a backtest: ``years * 245`` (5 years -> ``config.TEST_BARS_DEFAULT`` so that the
    single-stock walk-forward is the same object the universe validation uses), capped at ``n - MIN_TRAIN``."""
    want = config.TEST_BARS_DEFAULT if abs(years - DEFAULT_YEARS) < 1e-9 else int(round(years * config.TRADING_DAYS_PER_YEAR))
    return min(want, n - config.MIN_TRAIN)


# ======================================================================================= the facade
def env_int(name: str, default: int, lo: int, hi: int) -> int:
    """Integer environment variable clamped to ``[lo, hi]`` (``default`` when unset or malformed)."""
    try:
        v = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return max(lo, min(hi, v))


def _call_with_supported_kwargs(fn, *args, **kwargs):
    """Call ``fn`` passing only the keyword arguments its signature accepts (optional engine/eval extensions)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def pick_asof_ref(dates: list):
    """Reference as-of date of a universe scan: the NEWEST last-bar date carried by at least ``ASOF_SUPPORT`` of the names
    (and by at least two of them), the most common date when none qualifies (audit C-04).

    The most common date alone is wrong when a refresh problem makes half of the names one bar old: the correct, newer
    half would be the one that gets skipped.  Names behind the reference are skipped as "not synchronised"."""
    counts = Counter(dates)
    need = max(2, math.ceil(ASOF_SUPPORT * len(dates)))
    supported = [d for d, c in counts.items() if c >= need]
    return max(supported) if supported else max(counts, key=lambda d: (counts[d], d))


class Compute:
    """Caches + pipelines used by the HTTP server and the precompute CLI."""

    def __init__(self, workers: int | None = None, use_processes: bool = True, max_jobs: int = 2,
                 forecast_threads: int = 2):
        self.pool = ComputePool(workers, use_processes)
        self.jobs = JobManager(max_concurrent=max_jobs)
        self.disk = DiskStore(data.cache_dir)
        self._forecast_mem = LRU(256)
        self._wf_mem = LRU(6)
        self._flight = SingleFlight()
        self._evidence_memo: dict = {}
        self._forecast_sem = threading.BoundedSemaphore(forecast_threads)
        self.universe_size = env_int("TWQ_UNIVERSE_SIZE", config.UNIVERSE_SIZE, 4, 500)
        self.first_wait = 0.4  # seconds a job endpoint waits before answering "computing" (see Job.snapshot)
        self._closed = False
        self._cancel = threading.Event()  # set by close(): running jobs stop submitting work

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel.set()
        self.jobs.close()
        self.pool.close()

    def _check_open(self) -> None:
        if self._closed:
            raise ApiError("internal", "伺服器正在關閉。")

    # ------------------------------------------------------------------ data preparation
    @staticmethod
    def market_series() -> PriceData | None:
        """Completed TAIEX bars or None when unavailable (market features are then disabled)."""
        try:
            return data.completed(data.get_market_data())
        except (data.DataError, KeyError) as exc:
            log.warning("market data unavailable: %s", exc)
            return None

    @staticmethod
    def stock_series(symbol: str) -> tuple[PriceData, PriceData]:
        """(raw fetched data, completed data) of ``symbol``; raises DataError subclasses."""
        full = data.get_price_data(symbol)
        return full, data.completed(full)

    # ------------------------------------------------------------------ forecast cores
    def _forecast_key(self, dc: PriceData, mc: PriceData | None) -> tuple[str, str]:
        msig = data.price_signature(mc) if mc is not None else "nomarket"
        key = _hash("fc", dc.symbol, dc.dates[-1], data.price_signature(dc), msig, _fp_quant(), VERSION, CACHE_SCHEMA)
        return f"{dc.symbol}_{dc.dates[-1]}", key

    def cached_forecast_core(self, dc: PriceData, mc: PriceData | None) -> dict | None:
        label, key = self._forecast_key(dc, mc)
        hit = self._forecast_mem.get(key)
        if hit is not None:
            return hit
        hit = self.disk.load_json("forecast", label, key)
        if isinstance(hit, dict) and hit.get("horizons"):
            self._forecast_mem.put(key, hit)
            return hit
        return None

    def store_forecast_core(self, dc: PriceData, mc: PriceData | None, fc: dict) -> dict:
        label, key = self._forecast_key(dc, mc)
        fc = to_jsonable(fc)
        self._forecast_mem.put(key, fc)
        try:
            self.disk.save_json("forecast", label, key, fc, supersede=f"{DiskStore.safe(dc.symbol)}_*")
        except OSError as exc:
            log.warning("cannot persist forecast: %s", exc)
        return fc

    def forecast_core(self, dc: PriceData, mc: PriceData | None, use_pool: bool = False) -> dict:
        """The engine's forecast core for the last bar of ``dc`` (cached per symbol / last bar / data / code)."""
        if dc.n < config.MIN_BARS_FORECAST:
            raise insufficient(dc.name, dc.n, config.MIN_BARS_FORECAST)
        hit = self.cached_forecast_core(dc, mc)
        if hit is not None:
            return hit
        _, key = self._forecast_key(dc, mc)

        def compute():
            again = self.cached_forecast_core(dc, mc)
            if again is not None:
                return again
            if use_pool:
                fc = self.pool.run(_task_forecast, dc, mc)
            else:
                with self._forecast_sem:
                    fc = to_jsonable(_engine().forecast_latest(dc, mc))
            return self.store_forecast_core(dc, mc, fc)

        return self._flight.do(("fc", key), compute)

    # ------------------------------------------------------------------ /api/forecast
    def forecast_payload(self, symbol: str) -> dict:
        """Full 6.1 payload (without ``status``) for one symbol; synchronous."""
        try:
            sym = data.resolve_symbol(symbol)
        except data.BadSymbol as exc:
            raise to_api_error(exc, str(symbol)) from None
        try:
            full, dc = self.stock_series(sym)
        except (data.DataError, KeyError) as exc:
            raise to_api_error(exc, sym) from None
        if dc.n < config.MIN_BARS_FORECAST:
            raise insufficient(dc.name, dc.n, config.MIN_BARS_FORECAST)
        mc = None if sym == data.INDEX_SYMBOL else self.market_series()
        try:
            core = self.forecast_core(dc, mc)
        except ValueError as exc:
            raise to_api_error(exc, sym) from None
        self.check_degenerate(core, dc)
        return self.assemble_forecast(core, full, dc, mc)

    @staticmethod
    def check_degenerate(core: dict, dc: PriceData) -> None:
        """A forecast with a zero / non-finite spread (a series that never moves: suspended stock, frozen feed) is not a
        prediction: ``p_up`` of such a model is an artefact of the fit (audit P-14: 7.6 % with sigma 0).  422 rather than a
        200 with a confident-looking probability."""
        for hz in (core.get("horizons") or {}).values():
            sg = hz.get("sigma") if isinstance(hz, dict) else None
            if not (isinstance(sg, (int, float)) and not isinstance(sg, bool) and math.isfinite(sg) and sg > 0):
                raise ApiError("insufficient_history",
                               f"「{dc.name}」的價格序列近期完全沒有變動（可能停牌，或資料來源異常），模型無法算出有意義的預測。")

    def assemble_forecast(self, core: dict, full: PriceData, dc: PriceData, mc: PriceData | None) -> dict:
        fc = copy.deepcopy(core)
        asof = dc.dates[-1].astype(object)
        last_close = float(dc.close[-1])
        chg = float(dc.close[-1] / dc.close[-2] - 1.0) if dc.n >= 2 else None
        fc.setdefault("symbol", dc.symbol)
        fc.setdefault("asof", asof.isoformat())
        fc.setdefault("last_close", last_close)
        fc.setdefault("chg_1d", chg)
        fc.setdefault("n_bars", dc.n)
        fc.update(name=dc.name, market=dc.market, kind=dc.kind, asof_complete=True,
                  next_trading_day=data.next_trading_day(asof).isoformat(), short_history=bool(dc.n < SHORT_HISTORY_BARS),
                  stale=bool(full.stale), intraday_bar_dropped=bool(full.n != dc.n), data_through=full.dates[-1].astype(object).isoformat(),
                  disclaimer=DISCLAIMER)
        try:
            fc = _recommend().annotate_forecast(fc, dc.kind)
        except ImportError as exc:
            log.error("recommend module unavailable: %s", exc)
            for hz in (fc.get("horizons") or {}).values():
                hz.update(signal="hold", score=None, net_exp_return=None, confidence="低")
        fc["notes"] = self.forecast_notes(fc, full, dc, mc)
        return fc

    @staticmethod
    def forecast_notes(fc: dict, full: PriceData, dc: PriceData, mc: PriceData | None) -> list[str]:
        asof = dc.dates[-1].astype(object)
        notes = [f"資料截至 {asof.isoformat()} 收盤；預測起點為該日收盤價，對象是下一個交易日（{fc['next_trading_day']}）起算的 1、5、20 個交易日。"]
        if full.n != dc.n:
            notes.append("目前為盤中（尚未收盤）：未完成的當日K棒不納入模型，預測以前一個交易日收盤為基準。")
        notes.append("預測以還原權息價（含股利再投資）計算報酬率，圖表顯示的是未還原的實際價格；機率與區間為統計估計，並非保證。")
        notes.append("台股單日漲跌幅限制為 ±10%，1 日預測分布已截斷在漲跌停範圍內。")
        if full.stale:
            notes.append(f"網路暫時無法更新，目前使用本機快取（最後成功更新：{full.fetched_at or '未知'}），資料可能落後。")
        else:
            now = data.taipei_now()
            exp = data.last_trading_date_expected(now)
            exp_final = exp if now >= data.final_time(exp) else data.prev_trading_day(exp)
            if asof < exp_final:
                notes.append(f"此標的最新一根完整日K為 {asof.isoformat()}，比目前應有的最新交易日 {exp_final.isoformat()} 更舊"
                             f"（可能停牌、下市或資料來源延遲）：預測的起點是舊資料，「明日」的目標日 {fc['next_trading_day']} 已過去或就是今天，請當成歷史資料看待。")
        chg = fc.get("chg_1d")
        if isinstance(chg, (int, float)) and math.isfinite(chg) and abs(chg) > BIG_MOVE:
            notes.append(f"最新一日的漲跌幅 {chg * 100:+.1f}% 超過台股單日 ±10% 的漲跌幅限制（新上市初期等特殊情形除外），"
                         f"資料可能有誤，請先確認價格再參考預測。")
        if fc.get("short_history"):
            notes.append(f"此標的僅有 {dc.n} 個交易日的歷史資料，模型參數的不確定性較高，請謹慎看待。")
        if mc is None and dc.symbol != data.INDEX_SYMBOL:
            notes.append("大盤(加權指數)資料暫時無法取得，本次預測未使用大盤因子。")
        notes.append("多數個股的預測並沒有統計上顯著的優勢，請搭配「歷史驗證」頁面檢視同一模型在過去的真實表現。")
        return notes

    # ------------------------------------------------------------------ walk-forward cache
    def _wf_key(self, dc: PriceData, mc: PriceData | None, test_bars: int) -> tuple[str, str]:
        msig = data.price_signature(mc) if mc is not None else "nomarket"
        key = _hash("wf", dc.symbol, dc.dates[-1], dc.n, data.price_signature(dc), msig, test_bars, _fp_quant(),
                    config.HORIZONS, config.MIN_TRAIN, config.REFIT_EVERY, CACHE_SCHEMA)
        return f"{dc.symbol}_{dc.dates[-1]}_{test_bars}", key

    def cached_walk_forward(self, dc: PriceData, mc: PriceData | None, test_bars: int):
        label, key = self._wf_key(dc, mc, test_bars)
        wf = self._wf_mem.get(key)
        if wf is None:
            wf = self.disk.load_pickle("wf", label, key)
            if wf is not None:
                ok = (isinstance(wf, WalkForward) and len(wf.dates) == dc.n and wf.symbol == dc.symbol
                      and np.array_equal(wf.dates, dc.dates))
                if not ok:
                    return None
                self._wf_mem.put(key, wf)
        return wf

    def walk_forward(self, dc: PriceData, mc: PriceData | None, test_bars: int,
                     on_progress: Callable | None = None) -> WalkForward:
        """Cached walk-forward (all horizons) with ``test_bars`` out-of-sample rows, computed in the process pool."""
        wf = self.cached_walk_forward(dc, mc, test_bars)
        if wf is not None:
            return wf
        label, key = self._wf_key(dc, mc, test_bars)

        def compute():
            again = self.cached_walk_forward(dc, mc, test_bars)
            if again is not None:
                return again
            res = self.pool.run_walk_forward(dc, mc, dc.n - test_bars, on_progress)
            return self.store_walk_forward(dc, mc, test_bars, res)

        return self._flight.do(("wf", key), compute)

    def store_walk_forward(self, dc: PriceData, mc: PriceData | None, test_bars: int, wf: WalkForward) -> WalkForward:
        label, key = self._wf_key(dc, mc, test_bars)
        self._wf_mem.put(key, wf)
        try:
            self.disk.save_pickle("wf", label, key, wf,
                                  supersede=f"{DiskStore.safe(dc.symbol)}_*_{test_bars}_*")
        except OSError as exc:
            log.warning("cannot persist walk-forward: %s", exc)
        return wf

    # ------------------------------------------------------------------ result cache (finished job payloads)
    def _result_key(self, kind: str, **parts) -> tuple[str, str]:
        return kind, _hash(kind, json.dumps(parts, sort_keys=True, default=str), _fp_all(), VERSION, CACHE_SCHEMA)

    def load_result(self, kind: str, **parts):
        label, key = self._result_key(kind, **parts)
        hit = self.disk.load_json("results", label, key)
        if not isinstance(hit, dict):
            return None
        hit.pop("_meta", None)  # bookkeeping of the file (code fingerprint ...), not part of the API payload
        return hit

    def save_result(self, kind: str, payload: dict, meta: dict | None = None, **parts) -> None:
        label, key = self._result_key(kind, **parts)
        try:
            self.disk.save_json("results", label, key, dict(payload, _meta=meta) if meta else payload)
        except OSError as exc:
            log.warning("cannot persist result: %s", exc)

    # ------------------------------------------------------------------ epoch (job key ingredient)
    @staticmethod
    def epoch_token(now=None) -> str:
        """Changes when a new trading day starts (09:00) and when that day's bars turn final (14:00)."""
        t = data.taipei_now(now)
        exp = data.last_trading_date_expected(t)
        return f"{exp.isoformat()}{'F' if t >= data.final_time(exp) else 'P'}"

    # ------------------------------------------------------------------ bulk loading
    @staticmethod
    def load_universe_series(universe: list[dict], progress: Callable | None = None, base: int = 0,
                             total: int = 0, refresh: bool = False) -> tuple[dict, dict]:
        """Fetch (cached) price data for the universe with a small thread pool (``refresh`` forces a re-download).

        Returns ``({code: (full, completed)}, {code: exception})``.
        """
        ok: dict = {}
        bad: dict = {}
        done = 0
        lock = threading.Lock()

        def one(item):
            return item["code"], data.get_price_data(item["code"], refresh=refresh)

        with cf.ThreadPoolExecutor(max_workers=6, thread_name_prefix="fetch") as ex:
            futs = {ex.submit(one, it): it for it in universe}
            for fut in cf.as_completed(futs):
                it = futs[fut]
                try:
                    code, full = fut.result()
                    # keep the curated Chinese name even if the live table briefly lacks it
                    if it.get("name") and full.name != it["name"]:
                        full.name = it["name"]
                    ok[code] = (full, data.completed(full))
                except Exception as exc:  # noqa: BLE001 - one bad symbol must not sink the universe
                    bad[it["code"]] = exc
                with lock:
                    done += 1
                    if progress is not None:
                        progress(base + done, total, f"下載行情 {done}/{len(universe)}")
        return ok, bad

    # ------------------------------------------------------------------ recommendations
    def recommendations(self, h: int, top: int) -> dict:
        """Job envelope for ``/api/recommendations``.

        The cached universe validation of the same horizon (never started from here) is part of the job key: a validation
        finished after the list was built gives a new job (cheap: every forecast core is cached), so the evidence banner
        and the notes never lag behind the validation result."""
        ev = self.universe_evidence(h)
        key = f"recs|{h}|{top}|{self.epoch_token()}|{ev.get('computed_at') or 'none'}"
        job = self.jobs.get_or_start(key, "recommendations",
                                     lambda progress: self._run_recommendations(h, top, progress, evidence=ev))
        return job.snapshot(self.first_wait)

    # ------------------------------------------------------------------ evidence (DESIGN.md 9.4)
    def universe_evidence(self, h: int) -> dict:
        """The ``evidence`` block of the picks page for horizon ``h``: ``{"horizon", "source": "none"}`` or the block of
        ``universe_bt.evidence_from_report`` of the newest cached universe validation that (a) is for horizon ``h``, (b) was
        computed with the default execution / cost parameters, (c) was computed by the same model and configuration
        (``code_fp``), (d) is at most ``EVIDENCE_MAX_AGE_DAYS`` old.  Only the cache is read: the heavy job is never started
        and no price data is loaded.  Additive keys: ``data_asof`` (last bar date of the validation panel), ``top_k``."""
        h = int(h)
        none = {"horizon": h, "source": "none", "verdict_level": None, "ic": None, "ic_pvalue": None, "hit_rate": None,
                "base_rate": None, "computed_at": None}
        root = data.cache_dir() / "results"
        try:
            files = sorted(root.glob("universe_backtest_*.json"))
        except OSError:
            return none
        fp, now, best = _fp_all(), time.time(), None
        for f in files:
            try:
                st = f.stat()
            except OSError:
                continue
            if now - st.st_mtime > EVIDENCE_MAX_AGE_DAYS * 86400:
                continue
            meta = self._evidence_meta(f, st)
            if meta is None or meta["h"] != h or meta["code_fp"] != fp:
                continue
            rank = (meta["evidence"].get("computed_at") or "", st.st_mtime_ns)
            if best is None or rank > best[0]:
                best = (rank, meta["evidence"])
        return copy.deepcopy(best[1]) if best else none

    def _evidence_meta(self, path: Path, st) -> dict | None:
        """(horizon, code fingerprint, evidence block) of one cached universe result file (memoised per file version)."""
        sig = (str(path), st.st_mtime_ns, st.st_size)
        memo = self._evidence_memo
        if sig in memo:
            return memo[sig]
        meta = None
        try:
            payload = data.read_json_file(path)
            rep = payload.get("report") if isinstance(payload, dict) else None
            m = payload.get("_meta") if isinstance(payload, dict) else None
            prm = payload.get("params") if isinstance(payload, dict) else None
            if (isinstance(rep, dict) and isinstance(m, dict) and isinstance(prm, dict)
                    and m.get("cache_schema") == CACHE_SCHEMA and prm.get("exec") == "next_open"
                    and abs(float(prm.get("disc", -1)) - config.COMMISSION_DISCOUNT) < 1e-9
                    and abs(float(prm.get("slip", -1)) - config.SLIPPAGE_BPS) < 1e-9):
                ev = _universe_bt().evidence_from_report(payload, int(rep.get("horizon")), payload.get("computed_at"))
                if ev.get("source") == "universe_validation":
                    ev["data_asof"] = (rep.get("period") or {}).get("end")
                    ev["top_k"] = rep.get("top_k")
                    meta = {"h": int(rep["horizon"]), "code_fp": m.get("code_fp"), "evidence": ev}
        except Exception as exc:  # noqa: BLE001 - a malformed cache file is simply not evidence
            log.warning("unusable universe result %s: %s", path.name, exc)
        if len(memo) > 64:
            memo.clear()
        memo[sig] = meta
        return meta

    def _run_recommendations(self, h: int, top: int, progress: Callable, evidence: dict | None = None) -> dict:
        evidence = copy.deepcopy(evidence) if evidence is not None else self.universe_evidence(h)
        universe = data.get_universe(self.universe_size)
        total = len(universe) * 2 + 2
        progress(0, total, "載入大盤與股票行情…")
        mc = self.market_series()
        ok, bad = self.load_universe_series(universe, progress, 0, total)
        if not ok:
            raise ApiError("data_unavailable", "無法取得任何股票的行情資料（網路或資料來源暫時無法使用），請稍後再試。")
        # all names are scored at the reference last completed bar; laggards (suspended / stale feed / a refresh that lost
        # its last bar) are skipped -- see pick_asof_ref: the newest date many names carry, not the most common one
        dates = [c.dates[-1] for _, c in ok.values() if c.n]
        if not dates:
            raise ApiError("data_unavailable", "取得的行情都沒有任何資料列，無法評分，請稍後再試。")
        asof_ref = pick_asof_ref(dates)
        skipped: list[dict] = []
        work: dict[str, tuple[PriceData, PriceData]] = {}
        stale = False
        for it in universe:
            code = it["code"]
            if code in bad:
                skipped.append({"code": code, "name": it["name"], "reason": "行情下載失敗"})
                continue
            full, dc = ok[code]
            if dc.n < config.MIN_BARS_FORECAST:
                skipped.append({"code": code, "name": it["name"], "reason": f"歷史資料僅 {dc.n} 日，不足 {config.MIN_BARS_FORECAST} 日"})
            elif dc.dates[-1] != asof_ref:
                skipped.append({"code": code, "name": it["name"], "reason": f"資料僅到 {dc.dates[-1]}，未與其他標的同步"})
            else:
                work[code] = (full, dc)
                stale = stale or full.stale
        # ---- forecast cores: cache hits are instant, misses go to the process pool (bounded window)
        cores: dict[str, dict] = {}
        base = len(universe)
        step = [0]
        misses = []
        for it in universe:
            code = it["code"]
            if code not in work:
                step[0] += 1
                continue
            dc = work[code][1]
            hit = self.cached_forecast_core(dc, mc)
            if hit is not None:
                cores[code] = hit
                step[0] += 1
            else:
                misses.append((code, _task_forecast, (dc, mc)))
        progress(base + step[0], total, f"預測 {len(cores)}/{len(work)}（快取）")
        errors: dict[str, Exception] = {}

        def on_result(code, value, exc):
            if self._closed:
                return
            step[0] += 1
            if exc is not None:
                errors[code] = exc
            else:
                cores[code] = self.store_forecast_core(work[code][1], mc, value)
            progress(base + step[0], total, f"預測 {len(cores)}/{len(work)}：{code} {work[code][0].name}")

        if misses:
            self.pool.map_bounded(misses, on_result, cancel=self._cancel)
            self._check_open()
        for code, exc in errors.items():
            log.warning("forecast failed for %s: %r", code, exc)
            skipped.append({"code": code, "name": work[code][0].name, "reason": "模型運算失敗"})
        if not cores:
            raise ApiError("internal", "所有股票的預測運算都失敗，請查看伺服器日誌。")
        # ---- build
        progress(total - 1, total, "整理推薦名單…")
        fcs = []
        returns = {}
        for it in universe:
            code = it["code"]
            if code not in cores:
                continue
            full, dc = work[code]
            fc = copy.deepcopy(cores[code])
            fc.update(symbol=code, name=it["name"], kind=dc.kind, market=dc.market, asof_complete=True)
            fc.setdefault("asof", dc.dates[-1].astype(object).isoformat())
            av = data.avg_traded_value(dc)
            if av is not None:
                fc["avg_value_20d"] = av
            fcs.append(fc)
            r = dc.logret()[1:][-250:]
            if len(r) >= 30 and np.all(np.isfinite(r)):
                returns[code] = r
        market_fc = None
        if mc is not None and mc.n >= config.MIN_BARS_FORECAST:
            try:
                market_fc = self.forecast_core(mc, None)
                market_fc = dict(market_fc, symbol=data.INDEX_SYMBOL, name=data.INDEX_NAME, kind="index")
            except Exception as exc:  # noqa: BLE001 - the market block is optional
                log.warning("market forecast failed: %s", exc)
        if evidence.get("data_asof"):
            evidence["current"] = bool(str(evidence["data_asof"]) >= str(asof_ref))
        rec = _recommend().build_recommendations(fcs, h, top_k=top, market_forecast=market_fc, returns=returns,
                                                 evidence=evidence)
        rec = to_jsonable(rec)
        rec["universe_size"] = len(universe)
        rec["skipped"] = skipped
        rec["stale"] = bool(stale)
        if skipped:
            rec.setdefault("notes", []).append(
                f"另有 {len(skipped)} 檔因資料缺漏、資料未同步或運算失敗而略過：" +
                "、".join(f"{s['code']}{s['name']}" for s in skipped[:8]) + ("…" if len(skipped) > 8 else "") + "。")
        if stale:
            rec.setdefault("notes", []).append("部分行情因網路問題使用本機快取，可能不是最新資料。")
        rec["notes"] = rec.get("notes", []) + [
            "本工具為統計模型之研究用途，非投資建議；樣本內看似漂亮的排名未必能重現，請參考「推薦整體歷史驗證」。"]
        progress(total, total, "完成")
        return rec

    # ------------------------------------------------------------------ single-stock backtest
    def backtest(self, params: BacktestParams) -> dict:
        key = f"bt|{params.key()}|{self.epoch_token()}"
        job = self.jobs.get_or_start(key, "backtest", lambda progress: self._run_backtest(params, progress))
        return job.snapshot(self.first_wait)

    def _run_backtest(self, p: BacktestParams, progress: Callable) -> dict:
        progress(0, 4, f"載入 {p.symbol} 行情…")
        try:
            full, dc = self.stock_series(p.symbol)
        except (data.DataError, KeyError) as exc:
            raise to_api_error(exc, p.symbol) from None
        need = config.MIN_TRAIN + MIN_BACKTEST_TEST_BARS
        if dc.n < need:
            raise insufficient(dc.name, dc.n, need, "回測")
        mc = None if p.symbol == data.INDEX_SYMBOL else self.market_series()
        test_bars = backtest_test_bars(dc.n, p.years)
        cached = self.load_result("backtest", params=p.key(), asof=str(dc.dates[-1]), sig=data.price_signature(dc),
                                  msig=data.price_signature(mc) if mc is not None else "-", test_bars=test_bars)
        if cached is not None:
            progress(4, 4, "已有快取結果")
            return cached
        progress(1, 4, f"前推式預測（{test_bars} 個樣本外交易日）…")
        wf = self.walk_forward(dc, mc, test_bars,
                               lambda d, t: progress(1, 4, f"前推式預測 {d}/{t if t else '?'}"))
        progress(3, 4, "計算驗證指標與策略模擬…")
        report = _report().build_backtest_report(
            dc, wf, p.h, threshold=p.thr, exec_mode=p.exec, commission_discount=p.disc, slippage_bps=p.slip,
            sizing=p.sizing, n_trials=1)
        report = to_jsonable(report)
        payload = {"report": report, "asof": str(dc.dates[-1]), "computed_at": data.taipei_now().isoformat(timespec="seconds"),
                   "stale": bool(full.stale)}
        try:
            problems = _report().validate_backtest_report(report)
        except Exception as exc:  # noqa: BLE001
            problems = [f"validator failed: {exc!r}"]
        if problems:
            log.warning("backtest report schema problems for %s: %s", p.symbol, problems[:5])
            payload["schema_problems"] = problems[:20]
        self.save_result("backtest", payload, params=p.key(), asof=str(dc.dates[-1]), sig=data.price_signature(dc),
                         msig=data.price_signature(mc) if mc is not None else "-", test_bars=test_bars)
        progress(4, 4, "完成")
        return payload

    # ------------------------------------------------------------------ universe validation
    def universe_backtest(self, h: int, top: int, exec_mode: str, disc: float, slip: float) -> dict:
        key = f"ubt|{h}|{top}|{exec_mode}|{round(disc, 6)}|{round(slip, 6)}|{self.epoch_token()}"
        job = self.jobs.get_or_start(key, "universe_backtest", lambda progress: self._run_universe_backtest(
            h, top, exec_mode, disc, slip, progress))
        return job.snapshot(self.first_wait)

    def _run_universe_backtest(self, h: int, top: int, exec_mode: str, disc: float, slip: float,
                               progress: Callable) -> dict:
        universe = data.get_universe(self.universe_size)
        total = len(universe) * 2 + 2
        progress(0, total, "載入大盤與股票行情…")
        mc = self.market_series()
        ok, bad = self.load_universe_series(universe, progress, 0, total)
        elig = {}
        skipped = [{"code": c, "name": next((u["name"] for u in universe if u["code"] == c), c), "reason": "行情下載失敗"}
                   for c in bad]
        for it in universe:
            code = it["code"]
            if code not in ok:
                continue
            full, dc = ok[code]
            if dc.n < MIN_UNIVERSE_BARS:
                skipped.append({"code": code, "name": it["name"], "reason": f"歷史資料僅 {dc.n} 日"})
            else:
                elig[code] = (full, dc, backtest_test_bars(dc.n, DEFAULT_YEARS))
        if len(elig) < 4:
            raise ApiError("data_unavailable", "可用於驗證的股票少於 4 檔（資料下載失敗或歷史不足），無法進行整體驗證。")
        asof_key = _hash(*sorted(f"{c}:{dc.dates[-1]}:{data.price_signature(dc)}:{tb}" for c, (_, dc, tb) in elig.items()))
        parts = dict(h=h, top=top, exec=exec_mode, disc=round(disc, 6), slip=round(slip, 6), data=asof_key,
                     msig=data.price_signature(mc) if mc is not None else "-")
        cached = self.load_result("universe_backtest", **parts)
        if cached is not None:
            progress(total, total, "已有快取結果")
            return cached
        # ---- walk-forwards: cache hits are instant, misses are computed in the pool with a bounded window
        panel: dict[str, tuple[PriceData, WalkForward]] = {}
        base = len(universe)
        step = [0]
        misses = []
        for code, (full, dc, tb) in elig.items():
            wf = self.cached_walk_forward(dc, mc, tb)
            if wf is not None:
                panel[code] = (dc, wf)
                step[0] += 1
            else:
                misses.append((code, _task_walk_forward, (dc, mc, dc.n - tb, -1)))
        progress(base + step[0], total, f"前推式預測 {len(panel)}/{len(elig)}（快取）")
        errors: dict[str, Exception] = {}

        def on_result(code, value, exc):
            if self._closed:
                return
            step[0] += 1
            if exc is not None:
                errors[code] = exc
                log.warning("walk-forward failed for %s: %r", code, exc)
            else:
                _, dc, tb = elig[code]
                panel[code] = (dc, self.store_walk_forward(dc, mc, tb, value))
            progress(base + step[0], total, f"前推式預測 {len(panel)}/{len(elig)}：{code} {elig[code][0].name}")

        if misses:
            self.pool.map_bounded(misses, on_result, cancel=self._cancel)
            self._check_open()
        for code in errors:
            skipped.append({"code": code, "name": elig[code][0].name, "reason": "模型運算失敗"})
        if len(panel) < 4:
            raise ApiError("internal", "前推式預測成功的股票少於 4 檔，無法進行整體驗證，請查看伺服器日誌。")
        progress(total - 1, total, "評估推薦名單的歷史績效…")
        ub = _universe_bt()
        report = _call_with_supported_kwargs(
            ub.evaluate_universe, panel, h, top_k=top, commission_discount=disc, slippage_bps=slip,
            exec_mode=exec_mode, taiex=mc)
        report = to_jsonable(report)
        if skipped and isinstance(report.get("notes"), list):
            report["notes"].append(f"有 {len(skipped)} 檔因資料不足或運算失敗未納入驗證：" +
                                   "、".join(f"{s['code']}{s['name']}" for s in skipped[:8]) + ("…" if len(skipped) > 8 else "") + "。")
        payload = {"report": report, "computed_at": data.taipei_now().isoformat(timespec="seconds"),
                   "skipped": skipped,
                   "params": {"h": h, "top": top, "exec": exec_mode, "disc": round(disc, 6), "slip": round(slip, 6)}}
        try:
            problems = ub.validate_universe_report(report) if hasattr(ub, "validate_universe_report") else []
        except Exception as exc:  # noqa: BLE001
            problems = [f"validator failed: {exc!r}"]
        if problems:
            log.warning("universe report schema problems: %s", problems[:5])
            payload["schema_problems"] = problems[:20]
        self.save_result("universe_backtest", payload, meta={"code_fp": _fp_all(), "cache_schema": CACHE_SCHEMA}, **parts)
        progress(total, total, "完成")
        return payload

    # ------------------------------------------------------------------ housekeeping
    def prune(self) -> int:
        return prune_cache(data.cache_dir())
