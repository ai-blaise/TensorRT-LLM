import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _rt_signal(offset: int) -> Optional[int]:
    sig_rtmin = getattr(signal, "SIGRTMIN", None)
    if sig_rtmin is None:
        return None
    base = sig_rtmin() if callable(sig_rtmin) else int(sig_rtmin)
    return base + offset


class SnapshotHookController:
    """Opt-in TensorRT-LLM CRIU/CUDA-checkpoint hook controller.

    The controller is deliberately conservative: it only installs when
    OPTRT_SNAPSHOT_HOOKS=1 and records proof files for canary validation. It
    does not make production snapshots safe by itself; restore-specific NIXL,
    LayerSplit, and KVarN proofs are separate gates.
    """

    def __init__(
        self,
        executor: Any,
        *,
        component: Optional[str] = None,
        status_dir: Optional[str] = None,
        timeout_s: Optional[float] = None,
        cuda_synchronize: Optional[Callable[[], None]] = None,
    ) -> None:
        self.executor = executor
        self.component = component or os.environ.get("DYN_COMPONENT",
                                                     "unknown")
        self.status_dir = Path(
            status_dir or os.environ.get("OPTRT_SNAPSHOT_HOOK_DIR",
                                         "/tmp/optrt-snapshot-hooks"))
        self.timeout_s = float(
            timeout_s if timeout_s is not None else os.environ.get(
                "OPTRT_SNAPSHOT_HOOK_TIMEOUT_S", "30"))
        self.cuda_synchronize = cuda_synchronize
        self.rank = int(getattr(executor, "global_rank", -1))
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._phase_threads: dict[str, threading.Thread] = {}
        self._installed = False
        self._previous_handlers: dict[int, Any] = {}

    @classmethod
    def maybe_install(cls, executor: Any) -> Optional["SnapshotHookController"]:
        if not _env_enabled("OPTRT_SNAPSHOT_HOOKS"):
            return None
        controller = cls(executor)
        controller.install()
        return controller

    @staticmethod
    def pre_snapshot_signal() -> Optional[int]:
        return _rt_signal(5)

    @staticmethod
    def post_restore_signal() -> Optional[int]:
        return _rt_signal(6)

    def install(self) -> None:
        if self._installed:
            return
        signals = {
            "pre_snapshot": self.pre_snapshot_signal(),
            "post_restore": self.post_restore_signal(),
        }
        missing = [phase for phase, sig in signals.items() if sig is None]
        if missing:
            raise RuntimeError(
                "OPTRT snapshot hooks require Linux real-time signals; "
                f"missing: {','.join(missing)}")

        for phase, sig in signals.items():
            assert sig is not None
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._make_signal_handler(phase))
        self._installed = True
        self._write_phase("installed", "ready", extra={
            "pre_snapshot_signal": signals["pre_snapshot"],
            "post_restore_signal": signals["post_restore"],
        })
        logger.info(
            "OPTRT snapshot hooks installed "
            f"(component={self.component}, rank={self.rank}, pid={self.pid})")

    def _make_signal_handler(self, phase: str) -> Callable[[int, FrameType],
                                                           None]:

        def _handler(signum: int, frame: FrameType) -> None:
            del frame
            logger.info(
                f"OPTRT snapshot hook signal received: phase={phase} "
                f"signal={signum} rank={self.rank} pid={self.pid}")
            self.start_phase(phase, signum=signum)

        return _handler

    def start_phase(self, phase: str, *, signum: Optional[int] = None) -> None:
        with self._lock:
            prior = self._phase_threads.get(phase)
            if prior is not None and prior.is_alive():
                self._write_phase(phase, "already_running",
                                  extra={"signal": signum})
                return
            thread = threading.Thread(target=self._run_phase,
                                      args=(phase, signum),
                                      name=f"optrt-snapshot-{phase}",
                                      daemon=True)
            self._phase_threads[phase] = thread
            thread.start()

    def wait_phase(self, phase: str, timeout: Optional[float] = None) -> bool:
        thread = self._phase_threads.get(phase)
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def run_phase_sync(self, phase: str) -> None:
        self._run_phase(phase, None)

    def _run_phase(self, phase: str, signum: Optional[int]) -> None:
        started = time.monotonic()
        self._write_phase(phase, "start", extra={"signal": signum})
        try:
            if phase == "pre_snapshot":
                self._pre_snapshot()
            elif phase == "post_restore":
                self._post_restore()
            else:
                raise ValueError(f"unknown snapshot hook phase: {phase}")
            self._write_phase(
                phase,
                "ready",
                extra={"elapsed_s": round(time.monotonic() - started, 6)})
        except BaseException as exc:
            self._write_phase(
                phase,
                "error",
                extra={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_s": round(time.monotonic() - started, 6),
                })
            logger.exception(f"OPTRT snapshot hook failed: phase={phase}")

    def _pre_snapshot(self) -> None:
        queue = getattr(self.executor, "executor_request_queue", None)
        if queue is not None:
            with queue.enqueue_lock:
                queue.active = False
            queue.enqueue_control_request()

        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self._is_quiesced():
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(
                "pre-snapshot hook timed out waiting for executor quiescence")

        self._cuda_sync()

    def _post_restore(self) -> None:
        self._cuda_sync()
        queue = getattr(self.executor, "executor_request_queue", None)
        if queue is not None:
            with queue.enqueue_lock:
                queue.active = True

    def _is_quiesced(self) -> bool:
        active_requests = getattr(self.executor, "active_requests", [])
        if len(active_requests) > 0:
            return False

        async_transfer_manager = getattr(self.executor,
                                         "async_transfer_manager", None)
        if (async_transfer_manager is not None
                and async_transfer_manager.has_any_inflight_requests()):
            return False

        transceiver = getattr(self.executor, "kv_cache_transceiver", None)
        if transceiver is not None:
            check_gen = getattr(transceiver, "check_gen_transfer_complete",
                                None)
            if callable(check_gen) and not check_gen():
                return False
            check_context = getattr(transceiver,
                                    "check_context_transfer_status", None)
            if callable(check_context):
                # check_context_transfer_status() has side effects in the real
                # transceiver, so do not call it from a signal path.
                pass
        return True

    def _cuda_sync(self) -> None:
        if self.cuda_synchronize is not None:
            self.cuda_synchronize()
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(
                f"OPTRT snapshot hook CUDA synchronize skipped: {exc}")

    def _write_phase(self,
                     phase: str,
                     state: str,
                     *,
                     extra: Optional[dict[str, Any]] = None) -> None:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "component": self.component,
            "rank": self.rank,
            "pid": self.pid,
            "phase": phase,
            "state": state,
            "time_unix_s": time.time(),
        }
        if extra:
            payload.update(extra)
        name = (
            f"optrt_snapshot_{self.component}_rank{self.rank}_pid{self.pid}_"
            f"{phase}.{state}.json")
        (self.status_dir / name).write_text(json.dumps(payload, sort_keys=True)
                                            + "\n")
