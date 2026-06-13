import pytest

from tensorrt_llm._torch.attention_backend.sparse.hisparse import (
    OPTRTHiSparseCoordinator)
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig


def test_hisparse_coordinator_disabled_path_is_noop():
    cfg = DeepSeekSparseAttentionConfig(index_head_dim=128, index_topk=1024)
    coordinator = OPTRTHiSparseCoordinator(cfg)

    assert coordinator.enabled is False
    assert coordinator.step_id == 0
    coordinator.assert_startup_ready()
    coordinator.reset_step()
    assert coordinator.step_id == 1
    assert coordinator.map_topk_to_hot_pool(topk_indices=None,
                                            metadata=None,
                                            layer_idx=0,
                                            skip_topk=False,
                                            is_generation=True) is None


def test_hisparse_coordinator_enabled_path_fails_closed_until_kernel_ready():
    cfg = DeepSeekSparseAttentionConfig(
        index_head_dim=128,
        index_topk=1024,
        indexer_k_dtype="fp4",
        indexer_mode="indexcache-hisa",
        enable_nvfp4_hisa=True,
        mla_latent_kv_dtype="kvarn_k2v2",
        mla_latent_kv_amortize=True,
        hisparse_enabled=True,
    )
    coordinator = OPTRTHiSparseCoordinator(cfg)

    assert coordinator.enabled is True
    with pytest.raises(NotImplementedError, match="fail-closed"):
        coordinator.assert_startup_ready()
    with pytest.raises(NotImplementedError, match="hot-pool TopK mapping"):
        coordinator.map_topk_to_hot_pool(topk_indices=None,
                                         metadata=None,
                                         layer_idx=0,
                                         skip_topk=False,
                                         is_generation=True)
