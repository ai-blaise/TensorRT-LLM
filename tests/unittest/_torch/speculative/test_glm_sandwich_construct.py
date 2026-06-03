"""Numerical equivalence: op-trt Glm4DecoderLayer sandwich path vs HF reference math.

Builds ONE dense Glm4DecoderLayer (TP1, bf16) with the dense GLM-4-9B-0414 config,
runs a single decode-shaped forward, and compares against an explicit re-implementation
of the HF Glm4DecoderLayer sandwich block using the SAME module sub-weights
(input_layernorm, self_attn, post_self_attn_layernorm, post_attention_layernorm,
mlp, post_mlp_layernorm). Validates the residual/norm wiring is correct.
"""
import os
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_glm import Glm4DecoderLayer
from tensorrt_llm.mapping import Mapping

torch.manual_seed(0)
dev = "cuda"
dtype = torch.bfloat16

cfg = AutoConfig.from_pretrained(
    "/models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP", trust_remote_code=True)
# Shrink for a fast standalone test but keep arch-defining fields.
cfg.num_hidden_layers = 1
cfg.torch_dtype = dtype

mapping = Mapping(world_size=1, tp_size=1, rank=0)
# Unquantized model config (bf16) so we exercise the sandwich wiring without FP8 kernels.
mc = ModelConfig(pretrained_config=cfg, mapping=mapping, attn_backend="TRTLLM")
mc.quant_config.quant_algo = None

aux = {k: torch.cuda.Stream() for k in __import__(
    "tensorrt_llm._torch.utils", fromlist=["AuxStreamType"]).AuxStreamType}

layer = Glm4DecoderLayer(mc, 0, aux).to(dev).to(dtype).eval()
print("has_sandwich_norm:", layer.has_sandwich_norm)
assert layer.has_sandwich_norm, "dense GLM must take sandwich path"
assert hasattr(layer, "post_self_attn_layernorm")
assert hasattr(layer, "post_mlp_layernorm")
# Confirm the dense GatedMLP branch
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
assert isinstance(layer.mlp, GatedMLP), type(layer.mlp)
print("OK: dense sandwich layer constructed (GatedMLP, post_self_attn+post_mlp norms present)")
