# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Driver for the LayerSplit overlap benchmark (M9 measurement scaffold).

Spawns two ranks via ``torch.distributed.run`` on the GPUs visible to the
caller (set ``CUDA_VISIBLE_DEVICES=<gpu_a>,<gpu_b>``), drives the worker
in ``bench_layersplit_overlap.py`` across a small matrix of
(payload_bytes, compute_us) shapes, and prints a consolidated report
showing the wall-clock overlap savings of M6 / M8b vs M5.

Invocation (mirrors the integration-test runner pattern)::

    NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=<gpu_a>,<gpu_b> \\
        python3.11 tests/unittest/_torch/run_bench_layersplit_overlap.py

Goes through the same NCCL-NVLS-disabled rendezvous as the M5/M6/M8b
correctness tests so it coexists with a sibling NCCL tenant on the same
node (e.g. a serving deployment that already holds NVSwitch resources).
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
WORKER_FILE = "bench_layersplit_overlap.py"


def run_one(num_layers: int,
            payload_bytes: int,
            compute_us: int,
            port: int,
            warmup: int = 3,
            measure: int = 10,
            world_size: int = 2) -> None:
    """Spawn the N-rank bench for one (num_layers, payload_bytes,
    compute_us) shape and print rank 0's report.

    ``world_size`` defaults to 2 for backwards compatibility; pass 4 to
    drive the bench under CP=4 (the caller must have at least
    ``world_size`` GPUs visible via ``CUDA_VISIBLE_DEVICES``).
    """
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(
            ROOT / "tests/unittest/_torch" / WORKER_FILE,
            "/tmp/_ls_bench_worker.py",
        )
        result_path = pathlib.Path(td) / "report.txt"
        driver = f"""
import os, sys
os.environ['NCCL_NVLS_ENABLE'] = '0'
sys.path.insert(0, '{ROOT}')
sys.path.insert(0, '/tmp')

import importlib.util
spec = importlib.util.spec_from_file_location(
    'tensorrt_llm._torch.attention_backend.sparse.layersplit',
    '{ROOT}/tensorrt_llm/_torch/attention_backend/sparse/layersplit.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

import _ls_bench_worker as w
rank = int(os.environ['RANK'])
w._bench_worker(rank, int(os.environ['WORLD_SIZE']),
                int(os.environ['MASTER_PORT']),
                {num_layers}, {payload_bytes}, {compute_us},
                {warmup}, {measure}, '{result_path}')
"""
        driver_path = (f"/tmp/_ls_bench_driver_{payload_bytes}_"
                       f"{compute_us}.py")
        with open(driver_path, "w") as f:
            f.write(driver)

        result = subprocess.run(
            [
                "python3.11", "-m", "torch.distributed.run",
                f"--nproc_per_node={world_size}", f"--master_port={port}",
                driver_path,
            ],
            capture_output=True,
            text=True,
            timeout=360,
        )
        if result.returncode != 0:
            print(f"FAIL cp={world_size} payload_bytes={payload_bytes} "
                  f"compute_us={compute_us} returncode={result.returncode}")
            print("STDOUT (tail):", result.stdout[-1500:])
            print("STDERR (tail):", result.stderr[-1500:])
            return
        if result_path.exists():
            print(f"\n=== cp={world_size} payload_bytes={payload_bytes} "
                  f"compute_us={compute_us} ===")
            print(result_path.read_text())


def main() -> int:
    # Matrix sized for DeepSeek-V3.2-REAP-345B production shapes. The
    # heartbeat-class rows (16B - 1MB) cover the wiring + overhead
    # regime; the realistic rows (4MB - 64MB) cover the per-layer
    # active-KV sizes expected at long context under CP=2/CP=4 deployments
    # (rough estimate: 8K active tokens / CP rank × 656 B / token ≈ 5 MB
    # for V3.2, growing to ~50 MB at 128K context with cp_size=2). This
    # is the regime where M6 / M8b overlap is supposed to actually win
    # over M5 sync, so the bench must cover it explicitly.
    num_layers = 61
    matrix = [
        (16, 0),
        (16, 500),
        (1024, 0),
        (1024, 500),
        (64 * 1024, 0),
        (64 * 1024, 500),
        (1024 * 1024, 0),
        (1024 * 1024, 500),
        # Realistic per-layer active-KV sizes (M5d-full payload regime)
        (4 * 1024 * 1024, 500),
        (16 * 1024 * 1024, 500),
        (64 * 1024 * 1024, 500),
    ]

    # Auto-detect CP size from visible devices. Run CP=2 by default; add
    # CP=4 when 4+ GPUs are visible (matches the user's planned 2- or
    # 4-GPU deployment shape).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    n_visible = len([v for v in visible if v.strip()])
    cp_sizes = [2]
    if n_visible >= 4:
        cp_sizes.append(4)

    for cp_size in cp_sizes:
        print(f"\n{'='*70}\n=== CP={cp_size} sweep\n{'='*70}")
        port_base = 29600 if cp_size == 2 else 29700
        for i, (payload_bytes, compute_us) in enumerate(matrix):
            run_one(num_layers=num_layers,
                    payload_bytes=payload_bytes,
                    compute_us=compute_us,
                    port=port_base + i,
                    world_size=cp_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
