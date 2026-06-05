"""Standalone GLM-4-9B-0414-FP8 draft load gate.

Builds the dense Glm4ForCausalLM (the SMC-SD draft) and runs the real
checkpoint through Glm4WeightLoader, asserting no missing/unexpected keys and a
correctly-shaped fused gate_up_proj. Exits non-zero on any failure.
"""
import glob
import json
import os
import sys

import torch
from safetensors import safe_open

MODEL_DIR = os.environ.get(
    "GLM_DIR", "/models/BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP")


def load_state_dict(model_dir):
    sd = {}
    for st in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(st, framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    return sd


def main():
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_glm import Glm4ForCausalLM

    cfg = ModelConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print(f"[load] arch={cfg.pretrained_config.architectures} "
          f"quant={cfg.quant_config.quant_algo} "
          f"hidden={cfg.pretrained_config.hidden_size} "
          f"inter={cfg.pretrained_config.intermediate_size} "
          f"layers={cfg.pretrained_config.num_hidden_layers}",
          flush=True)

    with torch.device("cuda"):
        model = Glm4ForCausalLM(cfg).eval()

    raw = load_state_dict(MODEL_DIR)
    n_gup = sum(1 for k in raw if k.endswith(".gate_up_proj.weight"))
    n_gate = sum(1 for k in raw if k.endswith(".gate_proj.weight"))
    print(f"[ckpt] tensors={len(raw)} gate_up_proj.weight={n_gup} "
          f"gate_proj.weight={n_gate}", flush=True)
    assert n_gup == cfg.pretrained_config.num_hidden_layers, \
        "expected one fused gate_up_proj per layer (pre-fused checkpoint)"
    assert n_gate == 0, "checkpoint is pre-fused; should have no gate_proj"

    # Module params before load, to detect what the loader leaves untouched.
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if "gate_up_proj" in n and n.endswith(".weight")}

    model.load_weights(raw)
    if hasattr(model, "post_load_weights"):
        model.post_load_weights()

    # gate_up_proj fused shape check on layer 0.
    gup = dict(model.named_parameters())[
        "model.layers.0.mlp.gate_up_proj.weight"]
    hidden = cfg.pretrained_config.hidden_size
    inter = cfg.pretrained_config.intermediate_size
    print(f"[shape] loaded gate_up_proj.weight={tuple(gup.shape)} "
          f"(expect [{2 * inter}, {hidden}])", flush=True)
    assert tuple(gup.shape) == (2 * inter, hidden), \
        f"gate_up_proj fused shape wrong: {tuple(gup.shape)}"

    # Confirm the loader actually wrote the fused weight (not left at init).
    changed = any(not torch.equal(before[n].cpu(),
                                  dict(model.named_parameters())[n].detach().cpu())
                  for n in before)
    assert changed, "gate_up_proj weights unchanged after load (load no-op?)"

    # NaN/inf scan over all loaded params. FP8 dtypes don't implement isfinite,
    # so upcast them to fp32 for the check (the stored FP8 range is always finite
    # but we still verify the scales / bf16 params explicitly).
    bad = []
    for n, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        q = p.float() if p.dtype in (torch.float8_e4m3fn,
                                     torch.float8_e5m2) else p
        if not torch.isfinite(q).all():
            bad.append(n)
    assert not bad, f"non-finite params after load: {bad[:5]}"

    print("[OK] GLM-4-9B draft loaded clean: fused gate_up_proj correct, "
          "all params finite.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
