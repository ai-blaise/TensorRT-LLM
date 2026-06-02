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
"""Driver for the LayerSplit M5e-vs-M5d wire-byte + wall-time bench."""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
WORKER_FILE = "bench_layersplit_m5e_vs_m5d.py"


def run_one(num_layers: int,
            num_blocks_per_layer: int,
            block_bytes: int,
            active_per_step: int,
            port: int) -> None:
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(
            ROOT / "tests/unittest/_torch" / WORKER_FILE,
            "/tmp/_ls_m5e_bench_worker.py",
        )
        result_path = pathlib.Path(td) / "report.txt"
        driver = f"""
import os, sys
os.environ['NCCL_NVLS_ENABLE'] = '0'
sys.path.insert(0, '{ROOT}')
sys.path.insert(0, '/tmp')

import _ls_m5e_bench_worker as w
rank = int(os.environ['RANK'])
w._worker(rank, int(os.environ['WORLD_SIZE']),
          int(os.environ['MASTER_PORT']),
          {num_layers}, {num_blocks_per_layer}, {block_bytes},
          {active_per_step}, '{result_path}')
"""
        driver_path = (f"/tmp/_ls_m5e_bench_driver_"
                       f"nb{num_blocks_per_layer}_act{active_per_step}.py")
        with open(driver_path, "w") as f:
            f.write(driver)

        result = subprocess.run(
            [
                "python3.11", "-m", "torch.distributed.run",
                "--nproc_per_node=2", f"--master_port={port}", driver_path,
            ],
            capture_output=True,
            text=True,
            timeout=360,
        )
        if result.returncode != 0:
            print(f"FAIL num_blocks={num_blocks_per_layer} "
                  f"active={active_per_step} returncode={result.returncode}")
            print("STDOUT (tail):", result.stdout[-1500:])
            print("STDERR (tail):", result.stderr[-1500:])
            return
        if result_path.exists():
            print(f"\n=== num_blocks_per_layer={num_blocks_per_layer} "
                  f"block_bytes={block_bytes} active_per_step={active_per_step} ===")
            print(result_path.read_text())


def main() -> int:
    # Matrix sweeps the active/total ratio. Block sizes mimic V3.2 indexer K
    # (~8 KB per block: tokens_per_block=128, head_dim=64 bytes NVFP4 + 4
    # scale = 68 bytes/token, 128*68=8.7KB rounded to 8192 for power-of-2).
    num_layers = 61  # DeepSeek-V3.2
    block_bytes = 8192
    matrix = [
        # (num_blocks_per_layer, active_per_step)
        (1024, 1),     # tiny decode: 1 active out of 1k blocks
        (1024, 256),   # batch=256, decode: 1 block per req
        (4096, 256),   # batch=256, longer context: blocks grow
        (16384, 256),  # batch=256, much longer context
        (65536, 1024),  # very large pool, modest active
        (262144, 4096),  # V3.2 production-scale (~2GB pool / layer)
    ]
    for i, (nblocks, active) in enumerate(matrix):
        run_one(num_layers=num_layers,
                num_blocks_per_layer=nblocks,
                block_bytes=block_bytes,
                active_per_step=active,
                port=29800 + i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
