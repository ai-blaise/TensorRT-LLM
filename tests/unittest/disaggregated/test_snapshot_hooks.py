import json
import importlib.util
import threading
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_HOOKS_PATH = (
    REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" /
    "snapshot_hooks.py")
spec = importlib.util.spec_from_file_location("snapshot_hooks",
                                              SNAPSHOT_HOOKS_PATH)
snapshot_hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot_hooks)
SnapshotHookController = snapshot_hooks.SnapshotHookController


class _FakeQueue:

    def __init__(self):
        self.enqueue_lock = threading.Lock()
        self.active = True
        self.control_count = 0

    def enqueue_control_request(self):
        with self.enqueue_lock:
            self.control_count += 1


class _FakeAsyncTransferManager:

    def __init__(self, inflight=False):
        self.inflight = inflight

    def has_any_inflight_requests(self):
        return self.inflight


class _FakeTransceiver:

    def __init__(self, gen_complete=True):
        self.gen_complete = gen_complete

    def check_gen_transfer_complete(self):
        return self.gen_complete


class _FakeExecutor:

    def __init__(self):
        self.global_rank = 3
        self.executor_request_queue = _FakeQueue()
        self.active_requests = []
        self.async_transfer_manager = _FakeAsyncTransferManager()
        self.kv_cache_transceiver = _FakeTransceiver()


def _read_json(path):
    return json.loads(path.read_text())


def test_snapshot_hook_pre_and_post_restore_proof_files(tmp_path):
    executor = _FakeExecutor()
    sync_calls = []
    controller = SnapshotHookController(
        executor,
        component="decode",
        status_dir=str(tmp_path),
        timeout_s=0.2,
        cuda_synchronize=lambda: sync_calls.append("sync"),
    )

    controller.run_phase_sync("pre_snapshot")
    controller.run_phase_sync("post_restore")

    pre_ready = next(tmp_path.glob("*pre_snapshot.ready.json"))
    post_ready = next(tmp_path.glob("*post_restore.ready.json"))
    pre_payload = _read_json(pre_ready)
    post_payload = _read_json(post_ready)

    assert pre_payload["component"] == "decode"
    assert pre_payload["rank"] == 3
    assert pre_payload["phase"] == "pre_snapshot"
    assert pre_payload["state"] == "ready"
    assert post_payload["phase"] == "post_restore"
    assert post_payload["state"] == "ready"
    assert executor.executor_request_queue.active is True
    assert executor.executor_request_queue.control_count == 1
    assert sync_calls == ["sync", "sync"]


def test_snapshot_hook_times_out_when_executor_is_not_quiesced(tmp_path):
    executor = _FakeExecutor()
    executor.active_requests.append(object())
    controller = SnapshotHookController(
        executor,
        component="prefill",
        status_dir=str(tmp_path),
        timeout_s=0.01,
        cuda_synchronize=lambda: None,
    )

    controller.run_phase_sync("pre_snapshot")

    error_payload = _read_json(next(tmp_path.glob("*pre_snapshot.error.json")))
    assert error_payload["state"] == "error"
    assert error_payload["error_type"] == "TimeoutError"
    assert "quiescence" in error_payload["error"]
    assert executor.executor_request_queue.active is False


def test_snapshot_hook_install_is_opt_in(monkeypatch, tmp_path):
    executor = _FakeExecutor()
    monkeypatch.delenv("OPTRT_SNAPSHOT_HOOKS", raising=False)
    assert SnapshotHookController.maybe_install(executor) is None

    installed = {}
    monkeypatch.setenv("OPTRT_SNAPSHOT_HOOKS", "1")
    monkeypatch.setenv("OPTRT_SNAPSHOT_HOOK_DIR", str(tmp_path))
    monkeypatch.setattr(snapshot_hooks, "_rt_signal",
                        lambda offset: 1000 + offset)
    monkeypatch.setattr(snapshot_hooks.signal, "getsignal",
                        lambda signum: None)
    monkeypatch.setattr(snapshot_hooks.signal, "signal",
                        lambda signum, handler: installed.setdefault(
                            signum, handler))

    controller = SnapshotHookController.maybe_install(executor)

    assert controller is not None
    assert set(installed) == {1005, 1006}
    assert next(tmp_path.glob("*installed.ready.json")).exists()
