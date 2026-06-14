"""P3-DEFINITIVE gate for DSA-prefill piecewise CUDA graph (op-trt-hisparse).

Drives a faithful DSA decoder-layer-shaped module through op-trt's REAL piecewise
machinery (compilation/backend.py Backend + piecewise_optimizer) with RANDOM
weights at real prefill shapes, and measures the three weight-independent things
that gate the benefit:

  (a) the REAL captured-span count produced by the production partitioner
      (piecewise_optimizer), for a graph built from the REAL custom ops
      trtllm::mla_dsa_proj / trtllm::mla_dsa_attn_inplace in the real decoder
      forward order -- plus the exact eager-vs-piecewise kernel-LAUNCH count
      derived from the partitioned submodules (each captured span replays as one
      cudaGraphLaunch instead of its N constituent kernel launches);
  (b) BIT-EXACT eager-vs-piecewise output (capture must not perturb numerics;
      TRUE reference = the eager run of the SAME module, same weights);
  (c) per-step LATENCY eager-vs-piecewise (the realized benefit), CUDA-event
      timed over many iters, extrapolated to ~61 layers.

Why random weights gate (a)-(c): captured-span count is graph-structural; bit-
exactness and the launch/latency delta are about whether capture changes
execution, not about weight VALUES. Only end-to-end MODEL ACCURACY needs the
real 345B weights, and piecewise changes no numerics, so that is separable and
deferred to where the weights are resident.

The op BODIES are real GPU kernels at the production tensor shapes (proj returns
the 9-tensor straight-line bundle; attn_inplace writes the attn output in
place). What differs from the 345B model is only the indexer/attention numerics
inside the EAGER (uncaptured) attn op -- which by construction cannot affect the
captured-span count, capture bit-exactness, or the launch/latency delta.
"""
import argparse

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention_backend.trtllm import \
    TrtllmAttentionMetadata
from tensorrt_llm._torch.modules.attention import MLA
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.compilation.utils import (
    is_call_function, set_capture_piecewise_cuda_graph_flag)
from tensorrt_llm._torch.utils import (model_extra_attrs,
                                       set_per_request_piecewise_cuda_graph_flag,
                                       set_piecewise_cuda_graph_flag)

DEV = "cuda"
DT = torch.bfloat16

# representative DeepSeek-V3.2 DSA dims (config of the 345B graft, tp1)
HIDDEN = 7168
N_HEADS = 128
QK_NOPE = 128
QK_ROPE = 64
QK_HEAD = QK_NOPE + QK_ROPE  # 192
KV_LORA = 512
V_HEAD = 128
IDX_HEADS = 64
IDX_HEAD_DIM = 128
MOE_INTER = 2048


class _Indexer:
    use_fp4 = False
    n_heads = IDX_HEADS
    head_dim = IDX_HEAD_DIM


class FaithfulMLA(MLA):
    """Real shapes + real GEMMs, random weights; implements the two op bodies.

    Subclasses the real MLA so extract_extra_attrs' isinstance(layer, MLA)
    check passes, but skips MLA.__init__ (which needs a full ModelConfig +
    quantized weight pipeline) -- we set only the fields the two op bodies and
    the proj fake-impl read. The op bodies are overridden below, so none of
    MLA's real forward methods run on the gated path."""

    def __init__(self, layer_idx: int):
        nn.Module.__init__(self)  # skip MLA.__init__ (heavy: needs ModelConfig)
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.num_heads_tp = N_HEADS
        self.qk_head_dim = QK_HEAD
        self.kv_lora_rank = KV_LORA
        self.qk_rope_head_dim = QK_ROPE
        self.v_head_dim = V_HEAD

        class _MQA:
            indexer = _Indexer()

        self.mqa = _MQA()
        self.w_q = (torch.randn(HIDDEN, N_HEADS * QK_HEAD, device=DEV, dtype=DT)
                    * 0.02)
        self.w_kv = torch.randn(HIDDEN, KV_LORA, device=DEV, dtype=DT) * 0.02
        self.w_kpe = torch.randn(HIDDEN, QK_ROPE, device=DEV, dtype=DT) * 0.02
        self.w_idxq = (torch.randn(HIDDEN, IDX_HEADS * IDX_HEAD_DIM, device=DEV,
                                   dtype=DT) * 0.02)
        self.w_idxk = torch.randn(HIDDEN, IDX_HEAD_DIM, device=DEV, dtype=DT) * 0.02
        self.w_o = torch.randn(N_HEADS * V_HEAD, HIDDEN, device=DEV, dtype=DT) * 0.02

    def forward_dsa_proj(self, position_ids, hidden_states, metadata):
        nt = hidden_states.shape[0]
        q = hidden_states @ self.w_q
        compressed_kv = hidden_states @ self.w_kv
        k_pe = hidden_states @ self.w_kpe
        latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
        idx_q = (hidden_states @ self.w_idxq).view(nt, IDX_HEADS, IDX_HEAD_DIM)
        idx_k = hidden_states @ self.w_idxk
        q_fp8 = idx_q.to(torch.float8_e4m3fn)
        k_fp8 = idx_k.to(torch.float8_e4m3fn)
        k_scale = torch.ones(nt, 1, device=DEV, dtype=torch.float32)
        weights = torch.ones(nt, IDX_HEADS, device=DEV, dtype=torch.float32)
        q_scale = torch.ones(nt, IDX_HEADS, 1, device=DEV, dtype=torch.float32)
        return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale,
                weights, q_scale]

    def forward_dsa_attn(self, q, compressed_kv, k_pe, latent_cache,
                         indexer_intermediates, position_ids, metadata, output):
        nt = q.shape[0]
        qh = q.view(nt, N_HEADS, QK_HEAD)[:, :, :V_HEAD]
        ctx = torch.tanh(qh) + compressed_kv[:, :V_HEAD].unsqueeze(1)
        attn = ctx.reshape(nt, N_HEADS * V_HEAD) @ self.w_o
        output.copy_(attn)


class _Meta(TrtllmAttentionMetadata):
    """Subclass so extract_extra_attrs' isinstance check passes. We bypass the
    heavy __init__ (constructed via object.__new__ in build_meta) because the
    DSA op bodies + the fake-impl read only the MLA layer, never metadata
    fields, on the partition/capture path being gated."""
    num_contexts = 0
    num_ctx_tokens = 0


def build_meta():
    m = object.__new__(_Meta)  # skip TrtllmAttentionMetadata.__init__
    return m


class DSADecoderLayerShaped(nn.Module):
    """Mirrors DeepseekV3DecoderLayer.forward op order; attention via real ops."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.in_norm_w = [torch.ones(HIDDEN, device=DEV, dtype=DT)
                          for _ in range(n_layers)]
        self.post_norm_w = [torch.ones(HIDDEN, device=DEV, dtype=DT)
                            for _ in range(n_layers)]
        self.w_up = [torch.randn(HIDDEN, MOE_INTER, device=DEV, dtype=DT) * 0.02
                     for _ in range(n_layers)]
        self.w_down = [torch.randn(MOE_INTER, HIDDEN, device=DEV, dtype=DT) * 0.02
                       for _ in range(n_layers)]

    @staticmethod
    def _rmsnorm(x, w):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (v.to(x.dtype)) * w

    def forward(self, hidden_states, position_ids):
        h = hidden_states
        for i in range(self.n_layers):
            li = str(i)
            residual = h
            x = self._rmsnorm(h, self.in_norm_w[i])
            proj = torch.ops.trtllm.mla_dsa_proj(x, position_ids, li)
            q, compressed_kv, k_pe, latent_cache = proj[:4]
            indexer_intermediates = proj[4:]
            # Allocate the attn output buffer INSIDE the forward, per layer --
            # mirrors MLA.forward (attention.py:3454 attn_output =
            # self.create_output(...)): the buffer is internal to the captured
            # region, never crosses the eager/captured boundary as an external
            # arg (which would alias a stale captured address under replay).
            attn_out = x.new_empty(x.shape)
            torch.ops.trtllm.mla_dsa_attn_inplace(q, compressed_kv, k_pe,
                                                  latent_cache,
                                                  indexer_intermediates,
                                                  position_ids, li, attn_out)
            h = residual + attn_out
            residual = h
            x = self._rmsnorm(h, self.post_norm_w[i])
            x = torch.relu(x @ self.w_up[i]) @ self.w_down[i]
            h = residual + x
        return h


def build_extra_attrs(n_layers, meta, mlas):
    import weakref
    attrs = {}
    attrs["attention_metadata"] = weakref.ref(meta)
    attrs["mla_layers"] = {str(i): weakref.ref(mlas[i]) for i in range(n_layers)}
    return attrs


# ============================ TIER 1 ====================================== #
def tier1_real_partitioner(n_layers, num_tokens):
    print(f"\n===== TIER 1: REAL piecewise_optimizer partition + launch count "
          f"(n_layers={n_layers}, num_tokens={num_tokens}) =====")
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.fx.passes.split_module import split_module

    meta = build_meta()
    mlas = [FaithfulMLA(i) for i in range(n_layers)]
    attrs = build_extra_attrs(n_layers, meta, mlas)

    mod = DSADecoderLayerShaped(n_layers)
    hs = torch.randn(num_tokens, HIDDEN, device=DEV, dtype=DT)
    pos = torch.arange(num_tokens, device=DEV, dtype=torch.int32)

    with model_extra_attrs(attrs):
        gm = make_fx(mod, tracing_mode="fake", _allow_non_fake_inputs=True)(
            hs, pos)

    n_proj = sum(1 for n in gm.graph.nodes
                 if is_call_function(n, torch.ops.trtllm.mla_dsa_proj.default))
    n_attn = sum(
        1 for n in gm.graph.nodes
        if is_call_function(n, torch.ops.trtllm.mla_dsa_attn_inplace.default))
    n_index = sum(1 for n in gm.graph.nodes
                  if is_call_function(n, torch.ops.aten.index.Tensor))
    n_cumsum = sum(1 for n in gm.graph.nodes
                   if is_call_function(n, torch.ops.aten.cumsum.default))
    total_nodes = len(list(gm.graph.nodes))
    print(f"  graph nodes total: {total_nodes}")
    print(f"  mla_dsa_proj nodes:        {n_proj}")
    print(f"  mla_dsa_attn_inplace nodes:{n_attn} (eager split points)")
    print(f"  aten.index.Tensor nodes:   {n_index} (stop_partition trigger)")
    print(f"  aten.cumsum nodes:         {n_cumsum} (stop_partition trigger)")

    # --- Production classifier (verbatim from piecewise_optimizer.py:251-281) -- #
    stop_partition = False
    node_to_graph_id = {}
    idx = 0
    exclude_modules_id = []
    SPLIT_CONTINUE = [
        torch.ops.trtllm.attn_custom_op_inplace.default,
        torch.ops.trtllm.mla_custom_op_inplace.default,
        torch.ops.trtllm.mla_dsa_attn_inplace.default,
    ]
    TRIGGERS = SPLIT_CONTINUE + [
        torch.ops.aten.index.Tensor,
        torch.ops.aten.cumsum.default,
    ]
    for node in gm.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if (not stop_partition and is_call_function(node, TRIGGERS)):
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

    # --- ACTUALLY split with the production split_module to get per-span ----- #
    # --- kernel-launch counts (call_function nodes per captured submodule). -- #
    split = split_module(gm, None, lambda node: node_to_graph_id[node],
                         keep_original_order=True)
    eager_names = {f"submod_{i}" for i in exclude_modules_id}
    captured_launches = 0  # one cudaGraphLaunch per captured span
    eager_launches = 0     # raw per-op launches in eager spans + eager ops
    eager_total_kernels = 0  # what the fully-eager forward launches
    per_span = []
    for name, sub in split.named_children():
        if not name.startswith("submod_"):
            continue
        k = sum(1 for n in sub.graph.nodes if n.op == "call_function")
        eager_total_kernels += k
        if name in eager_names:
            eager_launches += k
        else:
            captured_launches += 1  # collapses to a single graph replay
            per_span.append(k)
    piecewise_launches = captured_launches + eager_launches

    print(f"  >> captured_spans={captured}  eager_spans={len(exclude_modules_id)}"
          f"  total_submods={total_submods}  stop_partition={stop_partition}")
    print(f"  >> per-captured-span call_function kernel counts: {per_span}")
    print(f"  >> EAGER total kernel launches (fully eager forward): "
          f"{eager_total_kernels}")
    print(f"  >> PIECEWISE launches = {captured_launches} graph-replays "
          f"+ {eager_launches} eager-op launches = {piecewise_launches}")
    if piecewise_launches > 0:
        reduction = 1.0 - piecewise_launches / eager_total_kernels
        print(f"  >> LAUNCH-COUNT reduction eager->piecewise: "
              f"{reduction*100:.1f}%  ({eager_total_kernels} -> "
              f"{piecewise_launches})")
    # Expected structure: the post-attn MoE of layer N and the pre-attn proj of
    # layer N+1 are contiguous capturable nodes with NO split point between
    # them, so they MERGE into one captured span. For N layers the partitioner
    # therefore yields exactly N+1 captured spans (boundary span0=pre-attn-L0,
    # spanN=post-attn-L{N-1}, and N-1 merged interior spans) + N eager attn
    # spans. This is strictly better than 2 spans/layer (fewer, larger captured
    # graphs => fewer cudaGraphLaunch calls). It is NOT a collapse.
    ok = (captured == n_layers + 1 and not stop_partition
          and len(exclude_modules_id) == n_attn)
    print(f"  TIER1 {'PASS' if ok else 'CHECK'}: real-op DSA prefill keeps "
          f"{captured} captured spans (== n_layers+1; interior MoE+proj spans "
          f"merge across the layer boundary) around {len(exclude_modules_id)} "
          f"eager attn ops, no stop_partition collapse.")
    return dict(captured=captured, eager_spans=len(exclude_modules_id),
                total_submods=total_submods, stop_partition=stop_partition,
                n_attn=n_attn, eager_total_kernels=eager_total_kernels,
                piecewise_launches=piecewise_launches,
                launch_reduction=(reduction if piecewise_launches else None),
                per_span=per_span, ok=ok)


# ============================ TIER 2 ====================================== #
def _time_ms(fn, args, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn(*args)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def tier2_bitexact_and_latency(n_layers, num_tokens):
    print(f"\n===== TIER 2: REAL Backend capture -> bit-exact + latency "
          f"(n_layers={n_layers}, num_tokens={num_tokens}) =====")

    meta = build_meta()
    mlas = [FaithfulMLA(i) for i in range(n_layers)]
    attrs = build_extra_attrs(n_layers, meta, mlas)

    mod = DSADecoderLayerShaped(n_layers).eval()
    # STABLE input buffers (fixed addresses) -- piecewise capture binds the
    # captured graph to the input tensor addresses; to feed new data you copy
    # into the SAME buffer and replay. We fix hs_data once and always pass the
    # same `hs`/`pos` objects, so the eager ref and the piecewise replay read
    # byte-identical inputs.
    hs_data = torch.randn(num_tokens, HIDDEN, device=DEV, dtype=DT)
    hs = hs_data.clone()
    pos = torch.arange(num_tokens, device=DEV, dtype=torch.int32)

    with model_extra_attrs(attrs), torch.no_grad():
        ref = mod(hs, pos).clone()  # eager ref on the stable input
    torch.cuda.synchronize()
    print(f"  eager ref: shape={tuple(ref.shape)} finite="
          f"{torch.isfinite(ref).all().item()} mean={ref.float().mean():.6f}")

    # eager latency (true reference path)
    def _eager_call(a, b):
        with model_extra_attrs(attrs), torch.no_grad():
            return mod(a, b)

    eager_ms = _time_ms(_eager_call, (hs, pos))
    print(f"  EAGER per-step latency: {eager_ms:.3f} ms "
          f"({eager_ms/n_layers:.4f} ms/layer)")

    # ---- piecewise via the REAL Backend ----
    set_piecewise_cuda_graph_flag(True)
    set_per_request_piecewise_cuda_graph_flag(True)
    set_capture_piecewise_cuda_graph_flag(True)

    backend = Backend(enable_inductor=False, enable_userbuffers=False,
                      enable_piecewise_cuda_graph=True,
                      capture_num_tokens=[num_tokens], max_num_streams=1,
                      mapping=None)

    class _Wrapped(nn.Module):
        # Backend reads input_num_tokens from a placeholder named
        # l_input_ids_ / l_kwargs_input_ids_; expose first arg as input_ids.
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, position_ids):
            return self.inner(input_ids, position_ids)

    wrapped = _Wrapped(mod).eval()
    compiled = torch.compile(wrapped, backend=backend, fullgraph=False)

    out_pw, err = None, None
    with model_extra_attrs(attrs), torch.no_grad():
        try:
            # First few runs: warmup_count<3 in PiecewiseRunner returns eager;
            # the graph captures on the 4th. We always pass the SAME hs/pos
            # objects (stable addresses) and re-fill hs with hs_data each time
            # so the captured graph and the eager ref read identical inputs.
            for _ in range(6):
                hs.copy_(hs_data)
                out_pw = compiled(hs, pos)
            torch.cuda.synchronize()
            hs.copy_(hs_data)
            out_pw = compiled(hs, pos).clone()
            torch.cuda.synchronize()
        except Exception as e:
            import traceback
            traceback.print_exc()
            err = f"{type(e).__name__}: {e}"

    if err is not None:
        print(f"  PIECEWISE compile/capture FAILED: {err}")
        return dict(bitexact=None, err=err, eager_ms=eager_ms)

    print(f"  piecewise out: shape={tuple(out_pw.shape)} finite="
          f"{torch.isfinite(out_pw).all().item()} mean={out_pw.float().mean():.6f}")
    exact = torch.equal(ref, out_pw)
    maxdiff = (ref.float() - out_pw.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), out_pw.float().flatten(), dim=0).item()
    print(f"  BIT-EXACT eager-vs-piecewise: {exact}  max_abs_diff={maxdiff:.3e}"
          f"  cos={cos:.8f}")

    def _pw_call(a, b):
        with model_extra_attrs(attrs), torch.no_grad():
            return compiled(a, b)

    pw_ms = _time_ms(_pw_call, (hs, pos))
    print(f"  PIECEWISE per-step latency: {pw_ms:.3f} ms "
          f"({pw_ms/n_layers:.4f} ms/layer)")
    speedup = eager_ms / pw_ms if pw_ms > 0 else float("nan")
    saved_per_layer = (eager_ms - pw_ms) / n_layers
    print(f"  >> latency speedup eager->piecewise: {speedup:.3f}x "
          f"(saved {eager_ms-pw_ms:.3f} ms/step, {saved_per_layer:.4f} ms/layer)")
    print(f"  >> extrapolated to 61 layers: eager~{eager_ms/n_layers*61:.2f} ms "
          f"-> piecewise~{pw_ms/n_layers*61:.2f} ms "
          f"(saved ~{saved_per_layer*61:.2f} ms/prefill-step)")
    return dict(bitexact=exact, maxdiff=maxdiff, cos=cos, eager_ms=eager_ms,
                pw_ms=pw_ms, speedup=speedup, saved_per_layer=saved_per_layer,
                err=None)


# ============================ TIER 4 (P4 / BCG) =========================== #
def tier4_dynamic_conditions(n_layers, captured_tokens, dynamic_tokens):
    """op-trt's answer to BCG (#25195) for the DSA-prefill scope: it does NOT
    have a within-graph break-and-resume (no torch.cond / cudaGraphConditional /
    breakable capture). It handles dynamic prefill conditions by (1) capturing a
    separate piecewise graph per num_tokens bucket in capture_num_tokens, and
    (2) GRACEFULLY FALLING BACK TO EAGER for any runtime token count not in the
    buckets (PiecewiseRunner.__call__: `runtime_num_of_token not in self.entries
    -> return self.default_callable(*args)`).

    This test proves both: a bucketed shape captures + replays bit-exact, and an
    UN-bucketed shape (the dynamic condition) runs correctly via eager fallback
    WITHOUT a recapture or a crash. That is op-trt's dynamic-condition coverage
    for prefill; a true BCG would instead keep ONE graph and break/resume inside
    it -- valuable for decode-side variable-draft-length (MTP), which is out of
    scope here (decode = full-graph). See the doc for the scope-gap statement.
    """
    print(f"\n===== TIER 4 (P4/BCG): dynamic-condition handling "
          f"(n_layers={n_layers}, bucket={captured_tokens}, "
          f"dynamic={dynamic_tokens}) =====")
    meta = build_meta()
    mlas = [FaithfulMLA(i) for i in range(n_layers)]
    attrs = build_extra_attrs(n_layers, meta, mlas)
    mod = DSADecoderLayerShaped(n_layers).eval()

    set_piecewise_cuda_graph_flag(True)
    set_per_request_piecewise_cuda_graph_flag(True)
    set_capture_piecewise_cuda_graph_flag(True)
    backend = Backend(enable_inductor=False, enable_userbuffers=False,
                      enable_piecewise_cuda_graph=True,
                      capture_num_tokens=[captured_tokens], max_num_streams=1,
                      mapping=None)

    class _Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, position_ids):
            return self.inner(input_ids, position_ids)

    compiled = torch.compile(_Wrapped(mod).eval(), backend=backend,
                             fullgraph=False)

    results = {}
    for tag, nt in [("bucketed", captured_tokens), ("dynamic", dynamic_tokens)]:
        hs_data = torch.randn(nt, HIDDEN, device=DEV, dtype=DT)
        hs = hs_data.clone()
        pos = torch.arange(nt, device=DEV, dtype=torch.int32)
        with model_extra_attrs(attrs), torch.no_grad():
            ref = mod(hs, pos).clone()
            ok = True
            err = None
            try:
                out = None
                for _ in range(6):
                    hs.copy_(hs_data)
                    out = compiled(hs, pos)
                torch.cuda.synchronize()
                hs.copy_(hs_data)
                out = compiled(hs, pos).clone()
                torch.cuda.synchronize()
            except Exception as e:
                ok = False
                err = f"{type(e).__name__}: {e}"
        if not ok:
            print(f"  [{tag} nt={nt}] FAILED: {err}")
            results[tag] = dict(ok=False, err=err)
            continue
        exact = torch.equal(ref, out)
        maxdiff = (ref.float() - out.float()).abs().max().item()
        print(f"  [{tag} nt={nt}] bit-exact={exact} max_abs_diff={maxdiff:.3e} "
              f"({'captured graph replay' if tag=='bucketed' else 'EAGER fallback (unbucketed)'})")
        results[tag] = dict(ok=True, bitexact=exact, maxdiff=maxdiff)
    bucketed_exact = bool(results.get("bucketed", {}).get("bitexact"))
    dynamic_exact = bool(results.get("dynamic", {}).get("bitexact"))
    verdict = bucketed_exact and dynamic_exact
    print(f"  TIER4 {'PASS' if verdict else 'CHECK'}: bucketed shape captures "
          f"bit-exact AND unbucketed dynamic shape falls back to eager bit-exact"
          f" (no recapture/crash). op-trt has NO within-graph break/resume; "
          f"dynamic prefill = bucket + eager-fallback.")
    return dict(results=results, verdict=verdict)


# ============================ TIER 5 (P5) ================================= #
def tier5_coverage_and_buckets(n_layers, buckets):
    """P5: maximize coverage + tune buckets.

    (a) Coverage fraction: of all kernel launches in the eager forward, what
        fraction lands in CAPTURED spans vs the irreducible eager attn region.
        Confirms the eager region is EXACTLY the N mla_dsa_attn_inplace ops and
        nothing leaks (no proj/MoE/norm forced eager).
    (b) Why the eager region is already minimal: forward_dsa_attn
        (attention.py:2014) begins with q = q[:num_tokens] -- an aten.slice
        whose length is a runtime int from batch metadata -- and every
        downstream op depends on it, plus the num_contexts>0 / num_generations>0
        data-dependent branches and the variable-seqlen sparse kernel. None can
        be hoisted into a fixed-shape captured graph; the proj (Op 1) is
        capturable precisely because it runs on the FULL PADDED tensor. So the
        split point sits at the padded->actual boundary, which is the
        theoretical coverage floor: only attention stays eager.
    (c) Multi-bucket capture: the prefill bucket-tuning lever -- capture several
        num_tokens buckets; each bucketed shape replays from its own graph,
        unbucketed shapes fall back to eager (validated in tier4).
    """
    print(f"\n===== TIER 5 (P5): coverage + bucket tuning "
          f"(n_layers={n_layers}, buckets={buckets}) =====")
    # (a) coverage fraction via the real partition (reuse tier1's classifier).
    # Captured-span kernels = sum of call_function nodes in the captured
    # submodules; eager-region kernels = the remainder (the attn ops).
    t1 = tier1_real_partitioner(n_layers, buckets[0])
    captured_span_kernels = sum(t1["per_span"])
    eager_region_kernels = t1["eager_total_kernels"] - captured_span_kernels
    coverage = captured_span_kernels / t1["eager_total_kernels"]
    print(f"  >> COVERAGE: {captured_span_kernels}/{t1['eager_total_kernels']} "
          f"kernels captured = {coverage*100:.1f}%  | eager region = "
          f"{eager_region_kernels} kernels across {t1['eager_spans']} attn ops")
    eager_is_only_attn = (t1["eager_spans"] == t1["n_attn"])
    print(f"  >> eager region is EXACTLY the {t1['n_attn']} attn ops "
          f"(no proj/MoE/norm leakage): {eager_is_only_attn}")

    # (c) multi-bucket capture through the real Backend
    print(f"  -- multi-bucket capture test (buckets={buckets}) --")
    meta = build_meta()
    mlas = [FaithfulMLA(i) for i in range(n_layers)]
    attrs = build_extra_attrs(n_layers, meta, mlas)
    mod = DSADecoderLayerShaped(n_layers).eval()
    set_piecewise_cuda_graph_flag(True)
    set_per_request_piecewise_cuda_graph_flag(True)
    set_capture_piecewise_cuda_graph_flag(True)
    backend = Backend(enable_inductor=False, enable_userbuffers=False,
                      enable_piecewise_cuda_graph=True,
                      capture_num_tokens=list(buckets), max_num_streams=1,
                      mapping=None)

    class _Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, position_ids):
            return self.inner(input_ids, position_ids)

    compiled = torch.compile(_Wrapped(mod).eval(), backend=backend,
                             fullgraph=False)
    bucket_ok = {}
    for nt in buckets:
        hs_data = torch.randn(nt, HIDDEN, device=DEV, dtype=DT)
        hs = hs_data.clone()
        pos = torch.arange(nt, device=DEV, dtype=torch.int32)
        with model_extra_attrs(attrs), torch.no_grad():
            ref = mod(hs, pos).clone()
            out = None
            for _ in range(6):
                hs.copy_(hs_data)
                out = compiled(hs, pos)
            torch.cuda.synchronize()
            hs.copy_(hs_data)
            out = compiled(hs, pos).clone()
            torch.cuda.synchronize()
        exact = torch.equal(ref, out)
        bucket_ok[nt] = exact
        print(f"     bucket nt={nt}: bit-exact={exact}")
    all_ok = all(bucket_ok.values()) and eager_is_only_attn
    print(f"  TIER5 {'PASS' if all_ok else 'CHECK'}: coverage maximal "
          f"({coverage*100:.1f}%, eager==attn-only), all {len(buckets)} buckets "
          f"capture bit-exact.")
    return dict(coverage=coverage, captured_span_kernels=captured_span_kernels,
                eager_region_kernels=eager_region_kernels,
                eager_is_only_attn=eager_is_only_attn, buckets=bucket_ok,
                ok=all_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--dynamic-tokens", type=int, default=1536)
    ap.add_argument("--buckets", type=int, nargs="+",
                    default=[1024, 2048, 4096])
    ap.add_argument("--tier", choices=["1", "2", "4", "5", "both", "all"],
                    default="both")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
          f"layers={args.layers} tokens={args.tokens}")
    res = {}
    if args.tier in ("1", "both", "all"):
        res["tier1"] = tier1_real_partitioner(args.layers, args.tokens)
    if args.tier in ("2", "both", "all"):
        res["tier2"] = tier2_bitexact_and_latency(args.layers, args.tokens)
    if args.tier in ("4", "all"):
        res["tier4"] = tier4_dynamic_conditions(args.layers, args.tokens,
                                                args.dynamic_tokens)
    if args.tier in ("5", "all"):
        res["tier5"] = tier5_coverage_and_buckets(args.layers, args.buckets)
    print("\n===== SUMMARY =====")
    import pprint
    pprint.pprint(res)
    print("GATE_DEFINITIVE_DONE")


if __name__ == "__main__":
    main()
