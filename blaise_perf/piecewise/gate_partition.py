"""P3 gate (partition-logic level): does a DSA-shaped prefill graph keep
captured spans, or does stop_partition collapse it to all-eager?

This replicates the EXACT node-classification loop from
piecewise_optimizer.piecewise_optimizer (the part that decides which submodules
are captured vs excluded) and runs it against an FX graph whose op skeleton
mirrors a DSA decoder-layer prefill forward:

  linear(in) -> mla_dsa_proj -> mla_dsa_attn_inplace -> linear(moe) -> linear(out)

It then reports captured-span count vs excluded(eager) count. This isolates the
load-bearing op-trt-specific risk the plan flagged (stop_partition triggered by
an early aten.index/aten.cumsum) WITHOUT needing model weights or executing the
custom-op bodies.

Two graphs are tested:
  A) clean DSA prefill skeleton (no raw aten index/cumsum on the path)
  B) same but with an early aten.cumsum injected BEFORE attention (the
     pathological case) -- to prove the loop's stop_partition behavior and
     bound the coverage risk.
"""
import torch
import torch.fx as fx
from tensorrt_llm._torch.compilation.utils import is_call_function


def classify(graph):
    """Verbatim copy of the node-classification loop in piecewise_optimizer."""
    stop_partition = False
    node_to_graph_id = {}
    idx = 0
    exclude_modules_id = []
    SPLIT_CONTINUE = [
        torch.ops.trtllm.attn_custom_op_inplace.default,
        torch.ops.trtllm.mla_custom_op_inplace.default,
        torch.ops.trtllm.mla_dsa_attn_inplace.default,
    ]
    SPLIT_STOP_TRIGGERS = SPLIT_CONTINUE + [
        torch.ops.aten.index.Tensor,
        torch.ops.aten.cumsum.default,
    ]
    for node in graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if (not stop_partition and is_call_function(node, SPLIT_STOP_TRIGGERS)):
            idx += 1
            node_to_graph_id[node] = idx
            exclude_modules_id.append(idx)
            if node.target not in SPLIT_CONTINUE:
                stop_partition = True
            else:
                idx += 1
        else:
            node_to_graph_id[node] = idx
    total_submods = len(set(node_to_graph_id.values()))
    captured = total_submods - len(exclude_modules_id)
    return captured, len(exclude_modules_id), total_submods, stop_partition


def build_dsa_prefill_graph(inject_early_cumsum=False):
    g = fx.Graph()
    hs = g.placeholder("hidden_states")
    pos = g.placeholder("position_ids")
    out = g.placeholder("output")
    # pre-attn projections / norms (captured span 1): emulate a few pointwise+mm
    x = g.call_function(torch.ops.aten.mul.Tensor, (hs, 2.0))
    x = g.call_function(torch.ops.aten.add.Tensor, (x, hs))
    if inject_early_cumsum:
        # Pathological: a raw aten.cumsum on the captured path BEFORE attention.
        x = g.call_function(torch.ops.aten.cumsum.default, (x, 0))
    # Op 1 (capturable custom op)
    proj = g.call_function(torch.ops.trtllm.mla_dsa_proj.default, (x, pos, "0"))
    q = g.call_function(torch.ops.aten.add.Tensor, (proj, 0.0))
    # Op 2 (eager split point; capture CONTINUES after it)
    attn = g.call_function(
        torch.ops.trtllm.mla_dsa_attn_inplace.default,
        (q, q, q, q, [q], pos, "0", out))
    # post-attn MoE-ish (captured span 2): pointwise + mm emulation
    y = g.call_function(torch.ops.aten.add.Tensor, (out, hs))
    y = g.call_function(torch.ops.aten.mul.Tensor, (y, 0.5))
    y = g.call_function(torch.ops.aten.add.Tensor, (y, hs))
    g.output(y)
    return g


for label, inj in [("A clean DSA prefill", False),
                   ("B early-cumsum pathological", True)]:
    g = build_dsa_prefill_graph(inject_early_cumsum=inj)
    captured, excluded, total, stopped = classify(g)
    print(f"[{label}] captured_spans={captured} excluded_eager={excluded} "
          f"total_submods={total} stop_partition_tripped={stopped}")
    if not inj:
        assert captured >= 2, ("clean DSA prefill must keep >=2 captured spans "
                               "(pre-attn proj + post-attn MoE)")
        print("  => PASS: DSA prefill keeps captured spans around the eager "
              "mla_dsa_attn_inplace op (no collapse).")
    else:
        assert captured <= 1, "early cumsum should collapse downstream coverage"
        print("  => CONFIRMED: an early raw aten.cumsum/index on the captured "
              "path collapses downstream coverage (the op-trt-specific risk).")
print("GATE_PARTITION_DONE")
