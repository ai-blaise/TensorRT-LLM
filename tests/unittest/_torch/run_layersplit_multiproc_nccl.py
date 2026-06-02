"""Driver to run the LayerSplit multi-process NCCL integration test.

Invocation::

    NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=<gpu_a>,<gpu_b> \
        python3.11 tests/unittest/_torch/run_layersplit_multiproc_nccl.py

Designed to coexist with another NCCL tenant on the same B200 / GB200
node (e.g. a serving deployment that already owns NVSwitch resources):
``NCCL_NVLS_ENABLE=0`` is forwarded to the child processes to bypass
the NVLink SHARP Multicast collision.

The test itself lives in ``test_layersplit_multiproc_nccl.py``; this
runner exists because the project's pytest conftest pulls in the heavy
``tensorrt_llm`` package (mpi4py, tensorrt, etc.) at collection time,
so it cannot be invoked through plain ``pytest tests/...``. The runner
side-steps that by exec'ing the test module via ``importlib`` and
driving a subprocess fan-out for each policy.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
WORKER_FILE = "test_layersplit_multiproc_nccl.py"


def run_one(policy: str,
            port: int,
            num_layers: int = 8,
            worker_fn: str = "_worker",
            world_size: int = 2) -> bool:
    """Spawn an ``N``-rank torch.distributed.run subprocess and verify
    every rank reported PASS for the given owner-assignment policy.

    ``worker_fn`` selects which entry point to invoke inside the worker
    module: ``_worker`` for the M5 sync-broadcast test, ``_worker_m6``
    for the M6 prefetch-overlap test, ``_worker_m8b`` for the 2-channel
    M8b test.

    ``world_size`` defaults to 2 for backwards compatibility with the
    CP=2 invocations; pass ``world_size=4`` to drive the M10 CP=4
    validation. The caller is responsible for ensuring
    ``CUDA_VISIBLE_DEVICES`` exposes at least ``world_size`` GPUs.
    """
    with tempfile.TemporaryDirectory() as td:
        # Copy the worker module into /tmp so the spawned children can
        # import it as a top-level module ("_ls_mp_worker"), avoiding the
        # nested-package importlib dance that mp.spawn cannot pickle.
        shutil.copy(
            ROOT / "tests/unittest/_torch" / WORKER_FILE,
            "/tmp/_ls_mp_worker.py",
        )

        driver = f"""
import os, sys
os.environ['NCCL_NVLS_ENABLE'] = '0'
sys.path.insert(0, '{ROOT}')
sys.path.insert(0, '/tmp')

# Pre-load LayerSplit module without triggering the heavy tensorrt_llm
# package __init__ chain (which needs mpi4py / tensorrt at import time).
import importlib.util
spec = importlib.util.spec_from_file_location(
    'tensorrt_llm._torch.attention_backend.sparse.layersplit',
    '{ROOT}/tensorrt_llm/_torch/attention_backend/sparse/layersplit.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

import _ls_mp_worker as w
rank = int(os.environ['RANK'])
getattr(w, '{worker_fn}')(rank, int(os.environ['WORLD_SIZE']),
          int(os.environ['MASTER_PORT']),
          '{policy}', {num_layers}, '{td}')
"""
        driver_path = (f"/tmp/_ls_driver_{policy}_{worker_fn}_"
                       f"cp{world_size}.py")
        with open(driver_path, "w") as f:
            f.write(driver)

        result = subprocess.run(
            [
                "python3.11", "-m", "torch.distributed.run",
                f"--nproc_per_node={world_size}",
                f"--master_port={port}", driver_path,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            print(f"FAIL fn={worker_fn} policy={policy} cp={world_size} "
                  f"returncode={result.returncode}")
            print("STDOUT (tail):", result.stdout[-1500:])
            print("STDERR (tail):", result.stderr[-1500:])
            return False

        ok = True
        for rank in range(world_size):
            p = pathlib.Path(td) / f"rank{rank}.result"
            if not p.exists():
                ok = False
                print(f"  rank{rank}: MISSING result file")
                continue
            content = p.read_text()
            if not content.startswith("PASS"):
                ok = False
                print(f"  rank{rank}: {content}")
        if ok:
            print(f"PASS fn={worker_fn} policy={policy} cp={world_size}")
        return ok


def main() -> int:
    overall = True
    # Resolve the world_size we can drive from the visible-devices set.
    # Default CP=2 (the existing baseline); detect CP=4 when 4+ GPUs are
    # visible so the M10 CP=4 validation runs automatically when the
    # caller widens the pin (e.g. CUDA_VISIBLE_DEVICES=3,4,5,6).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    n_visible = len([v for v in visible if v.strip()])
    cp_sizes = [2]
    if n_visible >= 4:
        cp_sizes.append(4)
    for cp_size in cp_sizes:
        port_base = 29540 if cp_size == 2 else 29640
        # M5 sync-broadcast test, both policies
        for i, policy in enumerate(["round_robin", "contiguous"]):
            if not run_one(policy,
                           port_base + i,
                           worker_fn="_worker",
                           world_size=cp_size):
                overall = False
        # M6 prefetch-overlap test, both policies
        for i, policy in enumerate(["round_robin", "contiguous"]):
            if not run_one(policy,
                           port_base + 10 + i,
                           worker_fn="_worker_m6",
                           world_size=cp_size):
                overall = False
        # M8b 2-channel test, both policies
        for i, policy in enumerate(["round_robin", "contiguous"]):
            if not run_one(policy,
                           port_base + 20 + i,
                           worker_fn="_worker_m8b",
                           world_size=cp_size):
                overall = False
        # M5e active-block broadcast test, both policies
        for i, policy in enumerate(["round_robin", "contiguous"]):
            if not run_one(policy,
                           port_base + 30 + i,
                           worker_fn="_worker_m5e",
                           world_size=cp_size):
                overall = False
        # M5f dual-cache (indexer + dense KV) broadcast test, both policies
        for i, policy in enumerate(["round_robin", "contiguous"]):
            if not run_one(policy,
                           port_base + 40 + i,
                           worker_fn="_worker_m5f",
                           world_size=cp_size):
                overall = False
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
