# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import tensorrt_llm  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_resident_handle_refreshes_attention_metadata():
    handle = torch.classes.trtllm.DeepseekResidentDecodeHandle(
        [0],
        [],
        [0],
        [],
        [],
        [torch.empty((1, ), device="cuda")],
    )
    seq_lens_cuda = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    kv_lens_cuda = torch.tensor([5, 7], dtype=torch.int32, device="cuda")
    req_idx_per_token = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    indexer_k_cache_block_offsets = torch.tensor(
        [[10, 11], [20, 21]],
        dtype=torch.int32,
        device="cuda",
    )
    slot_mapping_fp8 = torch.empty((2, ), dtype=torch.int64, device="cuda")
    slot_mapping_scale = torch.empty((2, ), dtype=torch.int64, device="cuda")
    gen_kv_indptr = torch.empty((3, ), dtype=torch.int64, device="cuda")
    gen_cached_token_indptr = torch.empty(
        (3, ),
        dtype=torch.int64,
        device="cuda",
    )
    kv_lens_cuda_2d = torch.empty((2, 2), dtype=torch.int32, device="cuda")

    handle.run_decode_window_attention_metadata_device_refresh(
        seq_lens_cuda,
        kv_lens_cuda,
        req_idx_per_token,
        indexer_k_cache_block_offsets,
        slot_mapping_fp8,
        slot_mapping_scale,
        gen_kv_indptr,
        gen_cached_token_indptr,
        kv_lens_cuda_2d,
        2,
        2,
        0,
        2,
        16,
        4,
        4,
        8,
    )

    torch.testing.assert_close(
        slot_mapping_fp8.cpu(),
        torch.tensor([1056, 2032], dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        slot_mapping_scale.cpu(),
        torch.tensor([1088, 2080], dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        gen_kv_indptr.cpu(),
        torch.tensor([0, 5, 12], dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        gen_cached_token_indptr.cpu(),
        torch.tensor([0, 4, 10], dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        kv_lens_cuda_2d.cpu(),
        torch.tensor([[5, 5], [7, 7]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_resident_handle_runs_indexer_topk_decode():
    handle = torch.classes.trtllm.DeepseekResidentDecodeHandle(
        [0],
        [],
        [0],
        [],
        [],
        [torch.empty((1, ), device="cuda")],
    )
    logits = torch.tensor(
        [
            [0.0, 4.0, 3.0, 2.0, -10.0, -11.0],
            [1.0, 9.0, 2.0, 8.0, 7.0, 6.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    kv_lens_cuda = torch.tensor([4, 6], dtype=torch.int32, device="cuda")
    topk_indices = torch.empty((2, 3), dtype=torch.int32, device="cuda")

    result = handle.run_indexer_topk_decode(
        logits,
        kv_lens_cuda,
        topk_indices,
        1,
        3,
    )

    assert result is topk_indices
    torch.testing.assert_close(
        torch.sort(topk_indices, dim=1).values.cpu(),
        torch.tensor([[1, 2, 3], [1, 3, 4]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_resident_handle_runs_indexer_xstep_recency_patch():
    handle = torch.classes.trtllm.DeepseekResidentDecodeHandle(
        [0],
        [],
        [0],
        [],
        [],
        [torch.empty((1, ), device="cuda")],
    )
    cached_topk = torch.tensor(
        [[10, 9, 8, 7], [20, 19, 18, 17]],
        dtype=torch.int32,
        device="cuda",
    )
    refresh_end = torch.tensor([11, 21], dtype=torch.int32, device="cuda")
    cur_kv_lens = torch.tensor([13, 22], dtype=torch.int32, device="cuda")

    result = handle.run_indexer_xstep_recency_patch(
        cached_topk,
        refresh_end,
        cur_kv_lens,
        1,
        2,
    )

    assert result is cached_topk
    torch.testing.assert_close(
        cached_topk.cpu(),
        torch.tensor([[10, 9, 12, 11], [20, 19, 21, 17]],
                     dtype=torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_resident_handle_runs_indexer_fp4_projection():
    device = torch.device("cuda")
    wq_b_weight = torch.arange(
        128 * 4,
        dtype=torch.float32,
        device=device,
    ).reshape(128, 4).to(torch.bfloat16) / 512
    wk_weight = torch.eye(128, 4, dtype=torch.float32, device=device)
    weights_proj_weight = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0]],
        dtype=torch.float32,
        device=device,
    )
    k_norm_weight = torch.ones((128, ), dtype=torch.float32, device=device)
    k_norm_bias = torch.zeros((128, ), dtype=torch.float32, device=device)
    rotary_cos_sin = torch.cat(
        (
            torch.ones((8, 32), dtype=torch.float32, device=device),
            torch.zeros((8, 32), dtype=torch.float32, device=device),
        ),
        dim=1,
    )
    handle = torch.classes.trtllm.DeepseekResidentDecodeHandle(
        [0, 6],
        [0],
        [0, 6],
        [83, 88, 93, 98, 99, 100],
        [0, 1, 2, 3, 4, 5],
        [
            wq_b_weight,
            wk_weight,
            weights_proj_weight,
            k_norm_weight,
            k_norm_bias,
            rotary_cos_sin,
        ],
    )
    q_lora = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 1.5, -2.0]],
        dtype=torch.bfloat16,
        device=device,
    )
    hidden_states = torch.tensor(
        [[2.0, 1.0, 0.0, -1.0], [1.0, -3.0, 2.0, 0.5]],
        dtype=torch.bfloat16,
        device=device,
    )
    position_ids = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
    q_fp4 = torch.empty((2, 1, 64), dtype=torch.int8, device=device)
    k_fp4 = torch.empty((2, 64), dtype=torch.int8, device=device)
    k_scale = torch.empty((2, 1), dtype=torch.int32, device=device)
    weights = torch.empty((2, 1), dtype=torch.float32, device=device)
    q_scale = torch.empty((2, 1, 1), dtype=torch.int32, device=device)

    result = handle.run_layer_dsa_indexer_fp4_projection(
        0,
        q_lora,
        hidden_states,
        position_ids,
        q_fp4,
        k_fp4,
        k_scale,
        weights,
        q_scale,
        2,
        1,
        128,
        64,
        1e-6,
        0.5,
        16,
        "cutlass,cublaslt,cuda_core",
    )

    assert result[0].data_ptr() == q_fp4.data_ptr()
    assert result[1].data_ptr() == k_fp4.data_ptr()
    assert result[2].data_ptr() == k_scale.data_ptr()
    assert result[3].data_ptr() == weights.data_ptr()
    assert result[4].data_ptr() == q_scale.data_ptr()
    assert q_fp4.shape == (2, 1, 64)
    assert k_fp4.shape == (2, 64)
    assert q_scale.shape == (2, 1, 1)
    torch.testing.assert_close(
        weights.cpu(),
        (hidden_states.float() @ weights_proj_weight.t() * 0.5).cpu(),
        rtol=1e-4,
        atol=1e-4,
    )
