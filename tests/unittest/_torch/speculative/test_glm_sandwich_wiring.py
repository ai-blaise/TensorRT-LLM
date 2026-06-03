"""Verify Glm4DecoderLayer.forward_sandwich residual/norm WIRING matches HF."""
import os, torch
from torch import nn
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_glm import Glm4DecoderLayer
from tensorrt_llm.mapping import Mapping
from tensorrt_llm._torch.utils import AuxStreamType
import tensorrt_llm._torch.models.modeling_glm as gm

torch.manual_seed(1)
dev, dtype = "cuda", torch.float32

cfg = AutoConfig.from_pretrained("/models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP", trust_remote_code=True)
cfg.num_hidden_layers = 1; cfg.hidden_size = 64; cfg.intermediate_size = 128
cfg.num_attention_heads = 4; cfg.num_key_value_heads = 2; cfg.head_dim = 16
cfg.torch_dtype = dtype

mapping = Mapping(world_size=1, tp_size=1, rank=0)
mc = ModelConfig(pretrained_config=cfg, mapping=mapping, attn_backend="TRTLLM")
mc.quant_config.quant_algo = None
aux = {k: torch.cuda.Stream() for k in AuxStreamType}
layer = Glm4DecoderLayer(mc, 0, aux).to(dev).to(dtype).eval()
H = cfg.hidden_size
for nm in ["input_layernorm","post_self_attn_layernorm","post_attention_layernorm","post_mlp_layernorm"]:
    getattr(layer, nm).weight.data.copy_(torch.randn(H, device=dev, dtype=dtype)*0.1 + 1.0)

Wa = torch.randn(H, H, device=dev, dtype=dtype) * 0.05
Wm = torch.randn(H, H, device=dev, dtype=dtype) * 0.05
class AttnStub(nn.Module):
    def forward(self, position_ids, hidden_states, attn_metadata, all_reduce_params=None, **kw):
        return hidden_states @ Wa.T
    def __call__(self, position_ids, hidden_states, attn_metadata, all_reduce_params=None, **kw):
        return hidden_states @ Wa.T
class MlpStub(nn.Module):
    def forward(self, hidden_states, final_all_reduce_params=None, **kw):
        return hidden_states @ Wm.T
    def __call__(self, hidden_states, final_all_reduce_params=None, **kw):
        return hidden_states @ Wm.T
layer.self_attn = AttnStub().to(dev); layer.mlp = MlpStub().to(dev)

def rmsnorm_ref(x, w, eps):
    v = x.float(); v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * w.float()).to(x.dtype)

x = torch.randn(3, H, device=dev, dtype=dtype); eps = cfg.rms_norm_eps
r = x
h = rmsnorm_ref(x, layer.input_layernorm.weight, eps); h = h @ Wa.T
h = rmsnorm_ref(h, layer.post_self_attn_layernorm.weight, eps); h = r + h
r = h
h2 = rmsnorm_ref(h, layer.post_attention_layernorm.weight, eps); h2 = h2 @ Wm.T
h2 = rmsnorm_ref(h2, layer.post_mlp_layernorm.weight, eps); ref = r + h2

orig = gm.GatedMLP
gm.GatedMLP = (MlpStub,)  # make the isinstance assert pass
try:
    out, resid = layer.forward_sandwich(position_ids=None, hidden_states=x, attn_metadata=None, spec_metadata=None)
finally:
    gm.GatedMLP = orig

err = (out - ref).abs().max().item(); rel = err / (ref.abs().max().item() + 1e-9)
print(f"residual returned: {resid}")
print(f"max abs err vs HF reference: {err:.3e}  rel: {rel:.3e}")
assert resid is None, "sandwich must return residual=None"
assert rel < 1e-5, f"WIRING MISMATCH rel={rel}"
print("PASS: forward_sandwich wiring matches HF Glm4DecoderLayer exactly")
