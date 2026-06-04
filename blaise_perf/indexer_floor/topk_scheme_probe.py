"""r12: WHICH topk scheme runs at prod, and is the win from scheme or scan?

At fixed seq_lens=4608, width 8192 -> insertion (7.5us), width 132096 -> radix
(14us). Confirm via TRTLLM_SCHEMEX_DEBUG which path each takes, and time the
exact kernel under nsys-free CUDA events with a per-launch breakdown.
Also test: does forcing the SAME scheme (radix) at BOTH widths still show the
gap? If radix@8192 ~= radix@132096 (both ~14us) the win is purely the scheme
switch (host dispatch by numColumns); if radix@8192 ~= 7.5us the win is the
shorter scan and numColumns only selects scheme.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from logits_launch_probe import build_inputs, call  # noqa: E402

TOPK = torch.ops.trtllm.indexer_topk_decode


def time_topk(logits, seq_lens, next_n, index_topk, iters=300):
    num_rows = logits.shape[0]
    indices = torch.empty((num_rows, index_topk), dtype=torch.int32,
                          device="cuda")
    for _ in range(30):
        TOPK(logits, seq_lens, indices, next_n, index_topk, None, None)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        TOPK(logits, seq_lens, indices, next_n, index_topk, None, None)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def main():
    B, next_n, index_topk = 32, 1, 2048
    print(f"# B={B} index_topk={index_topk}; "
          f"TRTLLM_SCHEMEX_DEBUG={os.environ.get('TRTLLM_SCHEMEX_DEBUG')}",
          flush=True)
    # The dispatch keys on numColumns = logits.size(1) = the alloc WIDTH.
    # Insertion threshold = 12288. So width 8192 -> insertion, 16384+ -> radix.
    for (width, seq) in [(8192, 4608), (16384, 4608), (132096, 4608),
                         (8192, 8192), (16384, 8192)]:
        logits = torch.randn((B, width), dtype=torch.float32, device="cuda")
        sl = torch.full((B,), seq, dtype=torch.int32, device="cuda")
        # one debug call to print the scheme line
        idx = torch.empty((B, index_topk), dtype=torch.int32, device="cuda")
        TOPK(logits, sl, idx, next_n, index_topk, None, None)
        torch.cuda.synchronize()
        t = time_topk(logits, sl, next_n, index_topk)
        scheme = "insertion" if width < 12288 else "radix"
        print(f"  width={width:>7} seq={seq:>6} -> {scheme:>9} : {t:6.2f}us",
              flush=True)


if __name__ == "__main__":
    main()
