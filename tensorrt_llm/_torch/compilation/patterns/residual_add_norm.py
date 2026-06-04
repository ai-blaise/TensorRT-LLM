from operator import getitem

import torch
from torch._inductor.pattern_matcher import (MULTIPLE, CallFunction, KeywordArg,
                                             Match, MultiOutputPattern,
                                             PatternMatcherPass, fwd_only,
                                             register_replacement)

aten = torch.ops.aten
from torch._higher_order_ops.auto_functionalize import auto_functionalized


def register_add_norm(custom_pass: PatternMatcherPass):
    residual = KeywordArg("residual")
    add_Tensor = CallFunction(aten.add.Tensor,
                              KeywordArg("input"),
                              residual,
                              _users=MULTIPLE)
    flashinfer_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        add_Tensor,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE)
    add_norm_pattern = MultiOutputPattern([flashinfer_norm_default, add_Tensor])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
    ):
        at = auto_functionalized(
            torch.ops.trtllm.flashinfer_fused_add_rmsnorm.default,
            input=input,
            residual=residual,
            weight=norm_weight,
            eps=eps)
        return at[1], at[2]

    def extra_check(match: Match):
        # Check the original residual and hidden has no other users since we will inplace update them
        residual_node = match.ctx.pattern_to_node[add_Tensor]
        if not isinstance(residual_node, torch.fx.graph.Node):
            return False

        # torch uses dict here to guarantee the order of the uses
        if list(residual_node.args[0].users.keys()
                )[-1] != residual_node or list(
                    residual_node.args[1].users.keys())[-1] != residual_node:
            return False

        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=add_norm_pattern,
        extra_check=extra_check,
    )


def register_add_norm_quant(custom_pass: PatternMatcherPass):
    residual_out = CallFunction(aten.add.Tensor,
                                KeywordArg("input"),
                                KeywordArg("residual"),
                                _users=MULTIPLE)

    flashinfer_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        residual_out,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=1)

    static_quantize = CallFunction(
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        flashinfer_norm_default,
        KeywordArg("scale"),
        _users=1)

    quant_out = CallFunction(getitem, static_quantize, 0, _users=1)
    add_norm_quant_pattern = MultiOutputPattern([quant_out, residual_out])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        scale: torch.Tensor,
        eps: float,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        scale: torch.Tensor,
        eps: float,
    ):
        out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        at = auto_functionalized(
            torch.ops.trtllm.flashinfer_fused_add_rmsnorm_quant.default,
            out=out,
            input=input,
            residual=residual,
            weight=norm_weight,
            scale=scale,
            eps=eps,
        )
        # at[1]=out (fp8 quant), at[2]=residual (updated)
        return at[1], at[2]

    def extra_check(match: Match) -> bool:
        # flashinfer_fused_add_rmsnorm_quant mutates residual in-place. Check that the original
        # residual tensor has the add node as its last user so no downstream node sees a stale pre-mutation value.
        add_node = match.ctx.pattern_to_node[residual_out]
        if not isinstance(add_node, torch.fx.graph.Node):
            return False

        if list(add_node.args[1].users.keys())[-1] != add_node:
            return False

        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=add_norm_quant_pattern,
        extra_check=extra_check,
    )


def register_add_norm_fp4_quant(custom_pass: PatternMatcherPass):
    """Fuse residual-add + RMSNorm + NVFP4 quantize into one kernel.

    The NVFP4 MoE/attention input is produced by three separate launches:

        add(input, residual)            -> aten.add
        flashinfer_rmsnorm(.)           -> 1 launch
        fp4_quantize(., scale)          -> 1 launch  (standalone quant)

    On Blackwell this collapses to a single ``trtllm.fused_add_rms_norm_quant``
    kernel (residual-add + RMSNorm + NVFP4 quant + block-scale-factor output),
    removing the standalone quant launch and the intermediate hp norm
    round-trip. At decode batch sizes this is ~3x faster for the sub-block
    (measured -64..-72%, ~20-23us saved per occurrence on B200).

    Unlike the FP8 sibling above, ``fused_add_rms_norm_quant`` is functional
    (no in-place mutation), so the residual is returned as a fresh tensor.
    """
    residual_out = CallFunction(aten.add.Tensor,
                                KeywordArg("input"),
                                KeywordArg("residual"),
                                _users=MULTIPLE)

    flashinfer_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        residual_out,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=1)

    # NVFP4 group size 16 is the only layout the trtllm MoE/dense kernels read,
    # so match it as a literal. The trailing fp4_quantize defaults
    # (sfUseUE8M0=False, isSfSwizzledLayout=True) are not materialized as graph
    # args, so they are validated in extra_check against the matched node.
    fp4_quantize = CallFunction(
        torch.ops.trtllm.fp4_quantize.default,
        flashinfer_norm_default,
        KeywordArg("scale"),
        16,
        _users=MULTIPLE)

    fp4_out = CallFunction(getitem, fp4_quantize, 0, _users=MULTIPLE)
    fp4_sf = CallFunction(getitem, fp4_quantize, 1, _users=MULTIPLE)
    add_norm_fp4_pattern = MultiOutputPattern([fp4_out, fp4_sf, residual_out])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        scale: torch.Tensor,
        eps: float,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        scale: torch.Tensor,
        eps: float,
    ):
        # fused_add_rms_norm_quant returns (fp4_i32, residual_out, sf, norm_out?)
        # with NVFP4 16-wide swizzled scale factors. fp4_i32 is int32-packed
        # (8 fp4 per int32); reinterpret to the uint8 packing fp4_quantize
        # produces (2 fp4 per byte) so downstream consumers are unchanged.
        fp4_i32, residual_o, sf, _ = torch.ops.trtllm.fused_add_rms_norm_quant(
            input,
            residual,
            norm_weight,
            scale,
            True,
            eps,
            False,
        )
        hidden_size = input.shape[-1]
        fp4_u8 = fp4_i32.view(torch.uint8).reshape(*input.shape[:-1],
                                                   hidden_size // 2)
        return fp4_u8, sf, residual_o

    def extra_check(match: Match) -> bool:
        # Only fuse the canonical NVFP4 quant: 16-wide (matched as a literal),
        # no UE8M0, swizzled SF (the layout the trtllm MoE/dense kernels read).
        # The last two fp4_quantize args are defaulted and absent from the graph
        # node, so validate them positionally against the matched node, falling
        # back to the unfused three-op path on any mismatch.
        quant_node = match.ctx.pattern_to_node.get(fp4_quantize)
        if not isinstance(quant_node, torch.fx.graph.Node):
            return False
        args = quant_node.args
        # args = (input, scale, sf_vec_size[, sf_use_ue8m0[, is_sf_swizzled]])
        sf_use_ue8m0 = args[3] if len(args) > 3 else quant_node.kwargs.get(
            "sfUseUE8M0", False)
        is_sf_swizzled = args[4] if len(args) > 4 else quant_node.kwargs.get(
            "isSfSwizzledLayout", True)
        if sf_use_ue8m0 not in (False, 0):
            return False
        if is_sf_swizzled not in (True, 1):
            return False
        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=add_norm_fp4_pattern,
        extra_check=extra_check,
    )
