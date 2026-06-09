# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager


class _Connector:

    def __init__(self, should_add: bool):
        self._should_add = should_add

    def should_add_sequence(self, request):
        return self._should_add


def _manager(connector=None):
    manager = object.__new__(KVCacheManager)
    manager.kv_connector_manager = connector
    return manager


def _request(*, first_context: bool, disagg_gen_init: bool):
    return SimpleNamespace(is_first_context_chunk=first_context,
                           is_disagg_generation_init_state=disagg_gen_init)


def test_disagg_generation_init_adds_sequence_without_first_context_chunk():
    request = _request(first_context=False, disagg_gen_init=True)

    assert _manager()._should_add_sequence_for_context_prepare(request)


def test_connector_skip_prevents_disagg_generation_init_add_sequence():
    request = _request(first_context=False, disagg_gen_init=True)

    assert not _manager(_Connector(False))._should_add_sequence_for_context_prepare(
        request)


def test_non_first_non_disagg_request_does_not_add_sequence():
    request = _request(first_context=False, disagg_gen_init=False)

    assert not _manager()._should_add_sequence_for_context_prepare(request)
