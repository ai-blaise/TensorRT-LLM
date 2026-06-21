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

import tensorrt_llm
import tensorrt_llm._torch.compilation.backend as backend_module
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm.mapping import Mapping


def test_distributed_backend_registers_fp4_add_norm_quant_before_plain_add_norm(
        monkeypatch):
    calls = []

    monkeypatch.setattr(tensorrt_llm, "mpi_world_size", lambda: 4)
    monkeypatch.setattr(backend_module, "register_ar_fusions",
                        lambda *_args, **_kwargs: calls.append("ar"))
    monkeypatch.setattr(
        backend_module,
        "register_add_norm_fp4_quant",
        lambda *_args, **_kwargs: calls.append("fp4_add_norm_quant"),
    )
    monkeypatch.setattr(backend_module, "register_add_norm",
                        lambda *_args, **_kwargs: calls.append("add_norm"))

    Backend._custom_pass_instances = None
    try:
        Backend.get_custom_pass(
            enable_userbuffers=False,
            mapping=Mapping(world_size=4, tp_size=4, rank=0),
        )
    finally:
        Backend._custom_pass_instances = None

    assert calls == ["ar", "fp4_add_norm_quant", "add_norm"]
