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

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.modules.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.modules.fused_moe import moe_scheduler
from tensorrt_llm._torch.modules.fused_moe.moe_scheduler import (
    ExternalCommMoEScheduler,
)


class _FakeRoutingMethod:

    experts_per_token = 2

    def apply(self, router_logits):
        raise AssertionError("precomputed route path must not recompute routing")


class _FakeBackend:

    def __init__(self):
        self.seen_selected_experts = None
        self.seen_final_scales = None

    def _supports_load_balancer(self):
        return True

    def quantize_input(self, x, post_quant_comm=False):
        del post_quant_comm
        return x, torch.ones_like(x)

    def run_moe(self, *, x, token_selected_experts, token_final_scales, x_sf):
        del x_sf
        self.seen_selected_experts = token_selected_experts
        self.seen_final_scales = token_final_scales
        return x + token_final_scales.sum(dim=1, keepdim=True)


class _FakeMoe:

    def __init__(self):
        self.backend = _FakeBackend()
        self.routing_method = _FakeRoutingMethod()
        self.comm = None
        self.layer_load_balancer = None
        self.layer_idx = 0
        self.num_slots = 4
        self.use_dp = False
        self.parallel_size = 1
        self.reduce_results = False
        self.enable_dummy_allreduce = False
        self.enable_alltoall = False
        self.apply_router_weight_on_input = False
        self.aux_stream = None
        self.repeat_idx = 0
        self.repeat_count = 1
        self.rank = 0
        self.mapping = SimpleNamespace(tp_rank=0)

    def calculate_num_chunks(self, all_rank_num_tokens):
        del all_rank_num_tokens
        return 1

    def determine_communication_method(self, all_rank_num_tokens, num_chunks):
        del all_rank_num_tokens, num_chunks

    def _load_balancer_start_wait_gpu_stage(self, is_first_call):
        del is_first_call

    def _load_balancer_start_set_cpu_stage(self, is_last_call):
        del is_last_call

    def _load_balancer_done_set_cpu_stage(self, is_last_call):
        del is_last_call

    def _using_load_balancer(self):
        return False


def test_external_comm_scheduler_uses_precomputed_route(monkeypatch):
    monkeypatch.setattr(moe_scheduler, "try_run_warp_decode",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        moe_scheduler,
        "get_calibrator",
        lambda: SimpleNamespace(
            maybe_collect_or_replay_slots=lambda num_slots, slots: slots),
    )
    moe = _FakeMoe()
    scheduler = ExternalCommMoEScheduler(moe)
    x = torch.ones((3, 4), dtype=torch.float32)
    token_selected_experts = torch.tensor([[0, 1], [1, 2], [2, 3]],
                                          dtype=torch.int64)
    token_final_scales = torch.tensor([[0.25, 0.75], [0.5, 0.5],
                                       [0.125, 0.875]],
                                      dtype=torch.float32)

    output = scheduler.forward_precomputed_route(
        x,
        token_selected_experts,
        token_final_scales,
        do_finalize=True,
        output_dtype=torch.float32,
        all_rank_num_tokens=None,
        use_dp_padding=False,
    )

    assert torch.equal(moe.backend.seen_selected_experts,
                       token_selected_experts.to(torch.int32))
    assert torch.equal(moe.backend.seen_final_scales, token_final_scales)
    assert torch.equal(output, x + torch.ones((3, 1), dtype=torch.float32))


def test_configurable_moe_precomputed_route_advances_repeat_idx():
    def _forward_precomputed_route(x, *args, **kwargs):
        del args, kwargs
        return x + 1

    moe = object.__new__(ConfigurableMoE)
    moe.scheduler = SimpleNamespace(
        forward_precomputed_route=_forward_precomputed_route)
    moe.enable_dwdp = False
    moe.repeat_idx = 1
    moe.repeat_count = 3
    x = torch.zeros((1, 2), dtype=torch.float32)
    token_selected_experts = torch.zeros((1, 2), dtype=torch.int32)
    token_final_scales = torch.ones((1, 2), dtype=torch.float32)

    output = ConfigurableMoE.forward_precomputed_route(
        moe,
        x,
        token_selected_experts,
        token_final_scales,
    )

    assert torch.equal(output, x + 1)
    assert moe.repeat_idx == 2
