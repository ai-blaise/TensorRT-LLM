"""
This module contains capturable drafting loops for speculative decoding.

These are torch modules wrap another draft model. The wrapped module
is supposed to invoke the draft model autoregressively and invoke
a sampling algorithm to obtain draft tokens. By structuring the code
like this, we are able to avoid host overhead: the entire drafting process
for speculation can be launched as a single CUDA graph.
"""

from abc import ABC, abstractmethod
import os
import math
from contextlib import contextmanager
from typing import Optional, final

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.speculative.eagle3 import Eagle3SpecMetadata
from tensorrt_llm._torch.speculative.interface import SpecMetadata
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager

# Enable capture_scalar_outputs to avoid graph breaks from Tensor.item() calls
torch._dynamo.config.capture_scalar_outputs = True


def _smc_cuda_sync_probe(label: str) -> None:
    if os.environ.get("SMC_CUDA_SYNC_PROBE", "0") != "1":
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"SMC CUDA sync probe failed after {label}") from exc


def _smc_probe_check_cuda_range(name: str, tensor: torch.Tensor, low: int,
                                high: int) -> None:
    if os.environ.get("SMC_CUDA_SYNC_PROBE", "0") != "1":
        return
    if tensor.numel() == 0:
        raise RuntimeError(f"SMC probe {name}: empty tensor")
    min_value = int(tensor.min().item())
    max_value = int(tensor.max().item())
    if min_value < low or max_value >= high:
        raise RuntimeError(
            f"SMC probe {name}: values out of range [{low}, {high}); "
            f"min={min_value}, max={max_value}, shape={tuple(tensor.shape)}")


def _mark_attn_metadata_generation_only(
        attn_metadata: AttentionMetadata) -> None:
    old_num_contexts = attn_metadata.num_contexts
    attn_metadata.host_request_types[:old_num_contexts].fill_(1)
    attn_metadata.num_contexts = 0

    if (hasattr(attn_metadata, "num_context_blocks")
            and hasattr(attn_metadata, "num_generation_blocks")):
        total_blocks = (attn_metadata.num_context_blocks +
                        attn_metadata.num_generation_blocks)
        attn_metadata.num_context_blocks = 0
        attn_metadata.num_generation_blocks = total_blocks


class BaseDraftingLoopWrapper(ABC, torch.nn.Module):

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata,
                **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def sample(self,
               logits: torch.Tensor,
               max_top_k: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def prepare_for_generation(
        self,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata,
        position_ids: torch.Tensor,
        spec_tree_manager: Optional[SpecTreeManager] = None
    ) -> torch.Tensor | None:
        raise NotImplementedError

    @final
    def load_weights_from_target_model(self, target_model) -> None:
        loader = getattr(self.draft_model, "load_weights_from_target_model",
                         None)
        if callable(loader):
            self.draft_model.load_weights_from_target_model(target_model)


@contextmanager
def save_metadata_state(attn_metadata: AttentionMetadata,
                        spec_metadata: SpecMetadata) -> None:
    attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
    batch_size = attn_metadata.num_seqs
    # Do not use prepare_for_spec_dec for this special field.
    # TRTLLM attention uses views of this tensor internally and prepare_for_spec_dec
    # creates a copy. If you write to the copy, TRTLLM attention won't see the updates.
    kv_lens = attn_metadata.kv_lens_cuda[:batch_size].clone()

    if attn_metadata.is_cuda_graph:
        assert spec_metadata.is_cuda_graph
        num_tokens = spec_metadata.num_tokens
        if isinstance(spec_metadata, Eagle3SpecMetadata):
            read_indices = spec_metadata.hidden_states_read_indices[:
                                                                    batch_size].clone(
                                                                    )
            write_indices = spec_metadata.hidden_states_write_indices[:
                                                                      batch_size].clone(
                                                                      )

    try:
        yield
    finally:
        attn_metadata.restore_from_spec_dec()
        attn_metadata.kv_lens_cuda[:batch_size].copy_(kv_lens)
        attn_metadata.on_update()
        if attn_metadata.is_cuda_graph:
            spec_metadata.num_tokens = num_tokens
            if isinstance(spec_metadata, Eagle3SpecMetadata):
                spec_metadata.hidden_states_read_indices[:batch_size].copy_(
                    read_indices)
                spec_metadata.hidden_states_write_indices[:batch_size].copy_(
                    write_indices)

        # This restore has to happen even if the spec_metadata is not being used
        # for CUDA graphs. It won't be reset by spec_metadata.prepare().
        if isinstance(spec_metadata, Eagle3SpecMetadata):
            spec_metadata.is_first_draft = True
            spec_metadata.eagle3_resource_manager.is_first_draft = True


class LinearDraftingLoopWrapper(BaseDraftingLoopWrapper):

    def __init__(self, max_draft_len: int, max_total_draft_tokens: int,
                 draft_model: torch.nn.Module):
        super().__init__()
        self.draft_model = draft_model
        self.config = self.draft_model.config
        self.model_config = self.draft_model.model_config
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata,
                **kwargs) -> dict[str, torch.Tensor]:
        logits = self.draft_model.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata,
                                          spec_metadata=spec_metadata,
                                          return_context_logits=True)
        logits = logits[spec_metadata.gather_ids]

        new_draft_tokens = [self.sample(logits)]
        draft_logits = [logits]
        if self.max_draft_len > 1:
            is_eagle3 = isinstance(spec_metadata, Eagle3SpecMetadata)
            with save_metadata_state(attn_metadata, spec_metadata):
                batch_size = attn_metadata.num_seqs

                new_position_ids = self.prepare_for_generation(
                    attn_metadata, spec_metadata, position_ids)
                for i in range(self.max_draft_len - 1):
                    logits = self.draft_model.forward(
                        input_ids=new_draft_tokens[-1],
                        position_ids=new_position_ids,
                        attn_metadata=attn_metadata,
                        spec_metadata=spec_metadata)
                    new_draft_tokens.append(self.sample(logits))
                    draft_logits.append(logits)
                    new_position_ids += 1
                    attn_metadata.kv_lens_cuda[:batch_size] += 1
                    if i == 0 and is_eagle3:
                        spec_metadata.hidden_states_read_indices[:batch_size].copy_(
                            spec_metadata.
                            hidden_states_write_indices[:batch_size])

        return {
            "new_draft_tokens": torch.stack(new_draft_tokens),
            "draft_logits": torch.stack(draft_logits)
        }

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        # TODO: inject the sampler here so we can support non-greedy
        tokens = torch.argmax(logits, dim=-1)
        if hasattr(self.draft_model.model, "d2t"):
            d2t = self.draft_model.model.d2t.data
            return tokens + d2t[tokens]

        return tokens

    def prepare_for_generation(self, attn_metadata: AttentionMetadata,
                               spec_metadata: SpecMetadata,
                               position_ids: torch.Tensor) -> torch.Tensor:
        batch_size = attn_metadata.num_seqs
        num_accepted_draft_tokens = spec_metadata.num_accepted_draft_tokens[:
                                                                            batch_size]
        seq_lens = attn_metadata.seq_lens_cuda[:batch_size]
        attn_metadata.kv_lens_cuda[:
                                   batch_size] -= seq_lens - num_accepted_draft_tokens - 1
        # Calculate last accepted token indices
        last_tokens_idx = torch.cumsum(
            seq_lens, dim=0,
            dtype=torch.long) - seq_lens + num_accepted_draft_tokens
        new_position_ids = position_ids[0, last_tokens_idx] + 1

        attn_metadata._seq_lens[:batch_size].fill_(1)
        attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
        attn_metadata.on_update()
        attn_metadata.kv_lens_cuda[:batch_size] += 1

        attn_metadata.host_request_types[:attn_metadata.num_contexts].fill_(1)
        attn_metadata.num_contexts = 0
        # The next inference of draft model will not use spec decoding and the number of input tokens is 1
        attn_metadata.use_spec_decoding = False

        spec_metadata.num_tokens = batch_size

        if isinstance(spec_metadata, Eagle3SpecMetadata):
            spec_metadata.eagle3_resource_manager.is_first_draft = False
            spec_metadata.is_first_draft = False

            old_write_indices = spec_metadata.hidden_states_write_indices

            spec_metadata.hidden_states_read_indices[:batch_size].copy_(
                old_write_indices[last_tokens_idx])
            spec_metadata.hidden_states_write_indices[:batch_size].copy_(
                torch.arange(
                    batch_size,
                    dtype=spec_metadata.hidden_states_write_indices.dtype,
                    device=spec_metadata.hidden_states_write_indices.device))

        return new_position_ids


class StaticTreeDraftingLoopWrapper(BaseDraftingLoopWrapper):

    def __init__(self, max_draft_len: int, max_total_draft_tokens: int,
                 max_batch_size: int, draft_model: torch.nn.Module):
        super().__init__()
        self.draft_model = draft_model
        self.config = self.draft_model.config
        self.model_config = self.draft_model.model_config
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_batch_size = max_batch_size

        self.draft_tokens_buffer = torch.zeros(
            (max_batch_size, max_total_draft_tokens + 1),
            dtype=torch.int64,
            device='cuda')
        self.position_ids_buffer = torch.zeros(
            (max_batch_size, max_total_draft_tokens + 1),
            dtype=torch.int64,
            device='cuda')

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata,
                **kwargs) -> dict[str, torch.Tensor]:
        assert isinstance(spec_metadata, Eagle3SpecMetadata)
        spec_tree_manager = spec_metadata.eagle3_resource_manager.spec_tree_manager

        logits = self.draft_model.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata,
                                          spec_metadata=spec_metadata,
                                          return_context_logits=True)
        batch_size = attn_metadata.num_seqs
        vocab_size = logits.shape[-1]
        logits = logits[spec_metadata.gather_ids]  # [batch_size, vocab_size]

        # new_draft_tokens: [batch_size * max_top_k]
        new_draft_tokens = self.sample(logits=logits,
                                       max_top_k=spec_tree_manager.max_top_k)

        self.extract_real_draft_tokens(
            cur_draft_idx=0,
            batch_size=batch_size,
            new_draft_tokens=new_draft_tokens,
            use_cuda_graph=attn_metadata.is_cuda_graph,
            spec_tree_manager=spec_tree_manager)
        return_draft_logits = None
        with save_metadata_state(attn_metadata, spec_metadata):
            batch_size = attn_metadata.num_seqs

            self.prepare_for_generation(attn_metadata=attn_metadata,
                                        spec_metadata=spec_metadata,
                                        spec_tree_manager=spec_tree_manager,
                                        position_ids=position_ids)

            for layer_idx in range(1, self.max_draft_len):
                # input_ids: [batch_size * (max_total_draft_tokens + 1)]
                # position_ids: [batch_size * (max_total_draft_tokens + 1)]
                # logits: [batch_size * (max_total_draft_tokens + 1), vocab_size]
                logits = self.draft_model.forward(
                    input_ids=self.draft_tokens_buffer[:batch_size, :self.
                                                       max_total_draft_tokens +
                                                       1].reshape(-1),
                    position_ids=self.
                    position_ids_buffer[:batch_size, :self.
                                        max_total_draft_tokens + 1].reshape(-1),
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    return_context_logits=True)

                # new_draft_tokens: [batch_size * (max_total_draft_tokens + 1) * max_top_k]
                new_draft_tokens = self.sample(
                    logits=logits, max_top_k=spec_tree_manager.max_top_k)
                # Keep updating
                self.extract_real_draft_tokens(
                    cur_draft_idx=layer_idx,
                    batch_size=batch_size,
                    new_draft_tokens=new_draft_tokens,
                    use_cuda_graph=attn_metadata.is_cuda_graph,
                    spec_tree_manager=spec_tree_manager)

                if layer_idx == self.max_draft_len - 1:
                    return_draft_logits = logits

        # self.draft_tokens_buffer[:batch_size, :]: [batch_size, max_total_draft_tokens + 1]
        # return_new_draft_tokens: [max_total_draft_tokens, batch_size]
        return_new_draft_tokens = torch.transpose(
            self.draft_tokens_buffer[:batch_size, :-1], 0, 1)

        # return_draft_logits: [batch_size, max_total_draft_tokens + 1, vocab_size] -> [max_total_draft_tokens, batch_size, vocab_size]
        if return_draft_logits is None:
            # When max_draft_len == 1, the loop doesn't execute.
            # Expand the initial logits to match the expected shape.
            return_draft_logits = logits.unsqueeze(1).expand(
                batch_size, self.max_total_draft_tokens + 1,
                vocab_size).reshape(-1, vocab_size)

        return_draft_logits = return_draft_logits.reshape(
            batch_size, self.max_total_draft_tokens + 1, vocab_size)
        return_draft_logits = torch.transpose(return_draft_logits[:, :-1, :], 0,
                                              1)

        assert return_new_draft_tokens.shape == (self.max_total_draft_tokens,
                                                 batch_size)
        assert return_draft_logits.shape == (self.max_total_draft_tokens,
                                             batch_size, vocab_size)

        return {
            "new_draft_tokens": return_new_draft_tokens,
            "draft_logits": return_draft_logits
        }

    def sample(self, logits: torch.Tensor, max_top_k: int) -> torch.Tensor:
        # TODO: inject the sampler here so we can support non-greedy

        # for draft_layer_idx == 0, logits is of shape [batch_size, vocab_size]
        # for draft_layer_idx > 0, logits is of shape [batch_size * (max_total_draft_tokens + 1), vocab_size]
        indices = torch.topk(
            logits, k=max_top_k, dim=-1
        ).indices  # [batch_size, max_top_k] or [batch_size * max_total_draft_tokens, max_top_k]
        tokens = indices.reshape(-1)

        if hasattr(self.draft_model.model, "d2t"):
            d2t = self.draft_model.model.d2t.data
            tokens = tokens + d2t[tokens]

        return tokens

    def extract_real_draft_tokens(self, cur_draft_idx: int, batch_size: int,
                                  new_draft_tokens: torch.Tensor,
                                  use_cuda_graph: bool,
                                  spec_tree_manager: SpecTreeManager):
        '''
        Extract the real draft tokens from the new draft tokens to self.draft_tokens_buffer.
        '''
        # After the first drafter layer, new_draft_tokens: [batch_size * max_top_k]
        # For other drafter layers, new_draft_tokens: [batch_size * (max_total_draft_tokens + 1) * max_top_k]
        if cur_draft_idx == 0:
            assert new_draft_tokens.shape[0] == (batch_size *
                                                 spec_tree_manager.max_top_k)
        else:
            assert new_draft_tokens.shape[0] == (
                batch_size * (self.max_total_draft_tokens + 1) *
                spec_tree_manager.max_top_k)

        # reshape the new_draft_tokens to [batch_size, -1, spec_tree_manager.max_top_k]
        new_draft_tokens = new_draft_tokens.reshape(batch_size, -1,
                                                    spec_tree_manager.max_top_k)

        # If using cuda graph, we need to use a torch op to implement this logic
        if use_cuda_graph:
            if os.environ.get("SMC_CUDA_SYNC_PROBE", "0") == "1":
                gather_idx = spec_tree_manager.tokens_gather_idx_for_drafter_model[
                    cur_draft_idx]
                top_k = spec_tree_manager.top_k_list_cuda[cur_draft_idx]
                cumsum = spec_tree_manager.draft_tokens_indices_cumsum
                if self.draft_tokens_buffer.size(0) < batch_size:
                    raise RuntimeError(
                        "SMC probe extract_real_draft_tokens: "
                        f"draft_tokens_buffer batch dim {self.draft_tokens_buffer.size(0)} "
                        f"is smaller than batch_size={batch_size}")
                _smc_probe_check_cuda_range("tokens_gather_idx", gather_idx, 0,
                                            new_draft_tokens.size(1))
                _smc_probe_check_cuda_range("top_k_list", top_k, 1,
                                            spec_tree_manager.max_top_k + 1)
                _smc_probe_check_cuda_range(
                    "draft_tokens_indices_cumsum", cumsum, 0,
                    self.max_total_draft_tokens + 1)
                if int(cumsum[-1].item()) != self.max_total_draft_tokens:
                    raise RuntimeError(
                        "SMC probe draft_tokens_indices_cumsum: final value "
                        f"{int(cumsum[-1].item())} != max_total_draft_tokens "
                        f"{self.max_total_draft_tokens}")
            torch.ops.trtllm.extract_real_draft_tokens_op(
                new_draft_tokens, self.draft_tokens_buffer, spec_tree_manager.
                tokens_gather_idx_for_drafter_model[cur_draft_idx],
                spec_tree_manager.top_k_list_cuda[cur_draft_idx],
                spec_tree_manager.draft_tokens_indices_cumsum, cur_draft_idx,
                batch_size, self.max_draft_len, self.max_total_draft_tokens,
                spec_tree_manager.max_top_k)
        else:
            # 1) Gather the real tokens processed by this layer
            if os.environ.get("SMC_CUDA_SYNC_PROBE", "0") == "1":
                gather_idx = spec_tree_manager.tokens_gather_idx_for_drafter_model[
                    cur_draft_idx]
                _smc_probe_check_cuda_range("tokens_gather_idx", gather_idx, 0,
                                            new_draft_tokens.size(1))
            process_tokens = new_draft_tokens[:, spec_tree_manager.
                                              tokens_gather_idx_for_drafter_model[
                                                  cur_draft_idx], :]  # [batch_size, num_tokens_process_this_layer, max_top_k]
            process_tokens = process_tokens.reshape(
                -1, spec_tree_manager.max_top_k
            )  # [batch_size * num_tokens_process_this_layer, max_top_k]

            # 2) Gather the real draft tokens samples by these processed tokens' logits
            top_k_list = spec_tree_manager.top_k_list_cuda[
                cur_draft_idx].repeat(
                    batch_size)  # [batch_size * num_tokens_process_this_layer]
            assert top_k_list.shape[0] == process_tokens.shape[0]
            _smc_probe_check_cuda_range("top_k_list", top_k_list, 1,
                                        spec_tree_manager.max_top_k + 1)

            # [batch_size * num_tokens_process_this_layer, spec_tree_manager.max_top_k]
            col_indices = torch.arange(
                spec_tree_manager.max_top_k,
                device=new_draft_tokens.device).unsqueeze(0).repeat(
                    top_k_list.shape[0], 1)

            mask = col_indices < top_k_list.unsqueeze(
                1
            )  # [batch_size * num_tokens_process_this_layer, spec_tree_manager.max_top_k]

            real_new_draft_tokens = process_tokens[
                mask]  # [batch_size * sum(spec_tree_manager.top_k_list_cuda[cur_draft_idx])]
            real_new_draft_tokens = real_new_draft_tokens.reshape(
                batch_size, -1
            )  # [batch_size, sum(spec_tree_manager.top_k_list_cuda[cur_draft_idx])]

            self.draft_tokens_buffer[:batch_size, spec_tree_manager.
                                     draft_tokens_indices_cumsum[cur_draft_idx]:
                                     spec_tree_manager.
                                     draft_tokens_indices_cumsum[
                                         cur_draft_idx +
                                         1]] = real_new_draft_tokens[:, :]

    def prepare_for_generation(self, attn_metadata: AttentionMetadata,
                               spec_metadata: SpecMetadata,
                               spec_tree_manager: SpecTreeManager,
                               position_ids: torch.Tensor):
        '''
        Prepare the inputs for the subsequent draft layers.
        Note: Except for the 0th drafter layer, in each subsequent drafter layer,
        we take 'max_total_drafter_tokens + 1' draft tokens as input.
        Only the first part of the draft tokens is meaningful, and the later tokens can be regarded as padding
        until we continuously write the correct value.

        This introduces additional redundant computation, but it makes it compatible with cuda graphs.

        What we need to prepare are:
            1) position_ids
            2) attn_metadata
                2.1) kv_lens_cuda
                2.2) _seq_lens, _seq_lens_cuda
                2.3) host_request_types
                2.4) num_contexts
                2.5) use_spec_decoding
                2.6) spec_decoding_position_offsets
                2.7) spec_decoding_packed_mask
                2.8) spec_decoding_generation_lengths
            3) spec_metadata
                3.1) num_tokens
                3.2) hidden_states_read_indices, hidden_states_write_indices
                3.3) is_first_draft
        '''
        batch_size = attn_metadata.num_seqs

        # 1) Prepare the position_ids
        num_accepted_draft_tokens = spec_metadata.num_accepted_draft_tokens[:
                                                                            batch_size]
        seq_lens = attn_metadata.seq_lens_cuda[:batch_size]
        # Calculate last accepted token indices
        last_tokens_idx = torch.cumsum(
            seq_lens, dim=0,
            dtype=torch.long) - seq_lens + num_accepted_draft_tokens
        position_start_idx = position_ids[0,
                                          last_tokens_idx] + 1  # [batch_size]
        self.position_ids_buffer[:batch_size, :-1] = position_start_idx.unsqueeze(
            1) + spec_tree_manager.spec_dec_position_offsets[0, 1:].unsqueeze(
                0) - 1  # exclude the root node
        # 2) Prepare the attn_metadata
        ## 2.1) kv_lens_cuda
        attn_metadata.kv_lens_cuda[:
                                   batch_size] -= seq_lens - num_accepted_draft_tokens - 1
        attn_metadata.kv_lens_cuda[:batch_size] += (
            self.max_total_draft_tokens + 1)

        ## 2.2) _seq_lens, _seq_lens_cuda
        attn_metadata._seq_lens[:batch_size].fill_(self.max_total_draft_tokens +
                                                   1)
        attn_metadata._seq_lens_cuda[:batch_size].fill_(
            self.max_total_draft_tokens + 1)
        attn_metadata.on_update()

        ## 2.3) host_request_types
        attn_metadata.host_request_types[:attn_metadata.num_contexts].fill_(1)

        ## 2.4) num_contexts
        attn_metadata.num_contexts = 0

        ## 2.5) use_spec_decoding
        attn_metadata.use_spec_decoding = True

        ## 2.6) spec_decoding_position_offsets
        ### attn_metadata.spec_decoding_position_offsets: [max_num_requests, max_total_draft_tokens + 1]
        attn_metadata.spec_decoding_position_offsets[:batch_size, :self.
                                                     max_total_draft_tokens] = spec_tree_manager.spec_dec_position_offsets[
                                                         0, 1:self.
                                                         max_total_draft_tokens +
                                                         1].unsqueeze(
                                                             0
                                                         ) - 1  # exclude the root node
        attn_metadata.spec_decoding_position_offsets[:batch_size, self.
                                                     max_total_draft_tokens] = 0  # padding

        ## 2.7) spec_decoding_packed_mask
        ### attn_metadata.spec_decoding_packed_mask: [max_num_requests, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32)]
        attn_metadata.spec_decoding_packed_mask[:
                                                batch_size, :, :] = spec_tree_manager.spec_dec_packed_mask_for_drafter_model

        ## 2.8) spec_decoding_generation_lengths
        ### attn_metadata.spec_decoding_generation_lengths: [max_num_requests]
        attn_metadata.spec_decoding_generation_lengths[:
                                                       batch_size] = self.max_total_draft_tokens + 1

        # 3) Update spec_metadata
        ## 3.1) num_tokens
        spec_metadata.num_tokens = batch_size * (self.max_total_draft_tokens +
                                                 1)
        ## 3.2) hidden_states_read_indices, hidden_states_write_indices
        ### spec_metadata.hidden_states_read_indices: [self.max_num_tokens]
        ### spec_metadata.hidden_states_write_indices: [self.max_num_tokens]
        old_write_indices = spec_metadata.hidden_states_write_indices
        start_idx = old_write_indices[
            last_tokens_idx]  # [batch_size], already take the accepted tokens into account.

        ### shape: [batch_size, self.max_total_draft_tokens + 1]
        hidden_states_read_indices_offset = spec_tree_manager.hidden_states_read_indices_offset_for_drafter_model[:self.max_total_draft_tokens + 1].repeat(
            batch_size).reshape(batch_size, self.max_total_draft_tokens + 1)
        hidden_states_read_indices_offset = hidden_states_read_indices_offset + start_idx.unsqueeze(
            1)
        spec_metadata.hidden_states_read_indices[:batch_size * (
            self.max_total_draft_tokens +
            1)] = hidden_states_read_indices_offset.reshape(-1)

        hidden_states_write_offset = torch.arange(
            1, self.max_total_draft_tokens + 1 + 1,
            device=position_ids.device).unsqueeze(0).repeat(
                batch_size, 1) + start_idx.unsqueeze(1)
        spec_metadata.hidden_states_write_indices[:batch_size * (
            self.max_total_draft_tokens +
            1)] = hidden_states_write_offset.reshape(-1)

        ## 3.3) is_first_draft
        spec_metadata.eagle3_resource_manager.is_first_draft = False
        spec_metadata.is_first_draft = False

        return


class SMCStaticParticleDraftingLoopWrapper(StaticTreeDraftingLoopWrapper):
    """Static-tree drafter for two-model SMC-SD particle proposals."""

    @staticmethod
    def _gather_logits_for_batch(logits: torch.Tensor,
                                 gather_ids: torch.Tensor,
                                 batch_size: int,
                                 tokens_per_request: int) -> torch.Tensor:
        if gather_ids.shape[0] > batch_size:
            if gather_ids.shape[0] % tokens_per_request == 0:
                gather_ids = gather_ids[::tokens_per_request]
            elif gather_ids.shape[0] % batch_size == 0:
                gather_ids = gather_ids[::gather_ids.shape[0] // batch_size]
            else:
                gather_ids = gather_ids[:batch_size]

        if gather_ids.shape[0] == batch_size:
            return logits[gather_ids]

        assert gather_ids.shape[0] < batch_size
        pad_ids = gather_ids.new_full((batch_size - gather_ids.shape[0], ),
                                      gather_ids[0])
        padded_gather_ids = torch.cat((gather_ids, pad_ids), dim=0)
        return logits[padded_gather_ids]

    @staticmethod
    def _host_draft_layout(spec_tree_manager: SpecTreeManager):
        """Host-side per-layer draft-token layout, memoized on the manager.

        Returns ``(offsets, repeats_per_layer)`` where ``offsets`` is the
        host-side prefix sum of per-layer draft-token counts (identical to the
        device tensor ``draft_tokens_indices_cumsum``) and
        ``repeats_per_layer[layer]`` is the per-parent child-count list for that
        layer. Both are plain Python ints, so reading them never triggers a
        device->host sync. The values are static for the lifetime of the static
        spec tree, hence computed once and cached.
        """
        cache = getattr(spec_tree_manager, "_record_logprob_host_layout", None)
        if cache is not None:
            return cache
        repeats_per_layer = [[int(r) for r in tk.tolist()]
                             for tk in spec_tree_manager.top_k_list]
        offsets = spec_tree_manager.draft_tokens_indices_cumsum.tolist()
        offsets = [int(o) for o in offsets]
        # The device prefix sum and the host re-sum of top_k_list must agree;
        # they index the same draft_tokens_buffer columns elsewhere.
        assert offsets[0] == 0
        for layer_idx, repeats in enumerate(repeats_per_layer):
            assert offsets[layer_idx + 1] - offsets[layer_idx] == sum(repeats)
        cache = (offsets, repeats_per_layer)
        spec_tree_manager._record_logprob_host_layout = cache
        return cache

    # Opt-in: collapse the per-parent logsumexp+gather into a single batched op
    # when the layer is uniform (every parent has exactly one child). This saves
    # n_particles-1 kernel launches per draft layer but the batched logsumexp
    # reduces a *contiguous* tensor whereas the per-parent path reduces
    # non-contiguous slices, so PyTorch may pick a different reduction tiling and
    # the normalizer can differ by up to 1 ULP (~4.77e-7) from the scalar path.
    # Off by default so the recorded draft log-probs stay bit-for-bit identical
    # to the reference; flip on the spec_tree_manager when that ULP is
    # acceptable. See test_smc_record_logprobs_equiv.py.
    smc_vectorize_logprob_record: bool = False

    def _record_layer_draft_log_probs(self, draft_log_prob_buffer: torch.Tensor,
                                      logits: torch.Tensor, batch_size: int,
                                      cur_draft_idx: int,
                                      spec_tree_manager: SpecTreeManager) -> None:
        # Host-side, sync-free offsets and per-parent child counts. Replaces the
        # former O(cur_draft_idx) loop that re-summed top_k_list[*].tolist()
        # (one device->host sync per layer) on every call: per decode step the
        # offset bookkeeping drops from O(gamma^2) device->host syncs to O(1)
        # (a one-time cache build), with bit-for-bit identical results.
        offsets, repeats_per_layer = self._host_draft_layout(spec_tree_manager)
        start = offsets[cur_draft_idx]
        end = offsets[cur_draft_idx + 1]
        if start == end:
            return
        repeats = repeats_per_layer[cur_draft_idx]

        if cur_draft_idx == 0:
            parent_logits = logits.reshape(batch_size,
                                           logits.shape[-1]).unsqueeze(1)
        else:
            parent_indices = spec_tree_manager.tokens_gather_idx_for_drafter_model[
                cur_draft_idx].long()
            parent_logits = logits.reshape(
                batch_size, self.max_total_draft_tokens + 1,
                logits.shape[-1]).index_select(1, parent_indices)

        child_tokens = self.draft_tokens_buffer[:batch_size, start:end].long()

        num_parents = parent_logits.shape[1]
        vectorize = getattr(spec_tree_manager,
                            "smc_vectorize_logprob_record",
                            self.smc_vectorize_logprob_record)
        # Uniform fast path (opt-in): every parent has exactly one child, so the
        # layer collapses to one batched logsumexp + one batched gather. The
        # gather is bit-exact; the batched logsumexp is within 1 ULP of the
        # per-parent reduction (see the note on smc_vectorize_logprob_record).
        if vectorize and num_parents == (end - start):
            parents = parent_logits.float()
            normalizer = torch.logsumexp(parents, dim=-1)
            selected = parents.gather(
                -1, child_tokens.unsqueeze(-1)).squeeze(-1)
            draft_log_prob_buffer[:, start:end] = selected - normalizer
            return

        # Default path: per-parent logsumexp+gather, bit-for-bit identical to
        # the original implementation. Only the sync-free offsets above changed.
        cursor = 0
        for parent_idx, repeat in enumerate(repeats):
            parent = parent_logits[:, parent_idx, :].float()
            normalizer = torch.logsumexp(parent, dim=-1, keepdim=True)
            next_cursor = cursor + int(repeat)
            selected = parent.gather(1, child_tokens[:, cursor:next_cursor])
            draft_log_prob_buffer[:, start + cursor:start + next_cursor] = (
                selected - normalizer)
            cursor = next_cursor

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata,
                **kwargs) -> dict[str, torch.Tensor]:
        spec_resource_manager = getattr(spec_metadata, "smc_resource_manager",
                                        None)
        assert spec_resource_manager is not None
        spec_tree_manager = spec_resource_manager.spec_tree_manager

        logits = self.draft_model.forward(input_ids=input_ids,
                                          position_ids=position_ids,
                                          attn_metadata=attn_metadata,
                                          spec_metadata=spec_metadata,
                                          return_context_logits=True)
        _smc_cuda_sync_probe("smc initial draft_model.forward")
        batch_size = attn_metadata.num_seqs
        logits = self._gather_logits_for_batch(
            logits, spec_metadata.gather_ids, batch_size,
            self.max_total_draft_tokens + 1)
        _smc_cuda_sync_probe("smc initial gather_logits")

        new_draft_tokens = self.sample(logits=logits,
                                       max_top_k=spec_tree_manager.max_top_k)
        _smc_cuda_sync_probe("smc initial sample")
        draft_log_prob_buffer = torch.empty(
            (batch_size, self.max_total_draft_tokens),
            dtype=torch.float32,
            device=logits.device)
        self.extract_real_draft_tokens(
            cur_draft_idx=0,
            batch_size=batch_size,
            new_draft_tokens=new_draft_tokens,
            use_cuda_graph=attn_metadata.is_cuda_graph,
            spec_tree_manager=spec_tree_manager)
        _smc_cuda_sync_probe("smc initial extract_real_draft_tokens")
        self._record_layer_draft_log_probs(draft_log_prob_buffer, logits,
                                           batch_size, 0, spec_tree_manager)
        _smc_cuda_sync_probe("smc initial record_log_probs")

        with save_metadata_state(attn_metadata, spec_metadata):
            batch_size = attn_metadata.num_seqs
            self.prepare_for_generation(attn_metadata=attn_metadata,
                                        spec_metadata=spec_metadata,
                                        spec_tree_manager=spec_tree_manager,
                                        position_ids=position_ids)

            for layer_idx in range(1, self.max_draft_len):
                logits = self.draft_model.forward(
                    input_ids=self.draft_tokens_buffer[:batch_size, :self.
                                                       max_total_draft_tokens +
                                                       1].reshape(-1),
                    position_ids=self.
                    position_ids_buffer[:batch_size, :self.
                                        max_total_draft_tokens + 1].reshape(-1),
                                          attn_metadata=attn_metadata,
                                          spec_metadata=spec_metadata,
                                          return_context_logits=True)
                _smc_cuda_sync_probe(
                    f"smc layer {layer_idx} draft_model.forward")
                new_draft_tokens = self.sample(
                    logits=logits, max_top_k=spec_tree_manager.max_top_k)
                _smc_cuda_sync_probe(f"smc layer {layer_idx} sample")
                self.extract_real_draft_tokens(
                    cur_draft_idx=layer_idx,
                    batch_size=batch_size,
                    new_draft_tokens=new_draft_tokens,
                    use_cuda_graph=attn_metadata.is_cuda_graph,
                    spec_tree_manager=spec_tree_manager)
                _smc_cuda_sync_probe(
                    f"smc layer {layer_idx} extract_real_draft_tokens")
                self._record_layer_draft_log_probs(draft_log_prob_buffer, logits,
                                                   batch_size, layer_idx,
                                                   spec_tree_manager)
                _smc_cuda_sync_probe(
                    f"smc layer {layer_idx} record_log_probs")

        return_new_draft_tokens = torch.transpose(
            self.draft_tokens_buffer[:batch_size, :-1], 0, 1)
        return_draft_log_probs = torch.transpose(draft_log_prob_buffer, 0, 1)

        assert return_new_draft_tokens.shape == (self.max_total_draft_tokens,
                                                 batch_size)
        assert return_draft_log_probs.shape == (self.max_total_draft_tokens,
                                                batch_size)

        return {
            "new_draft_tokens": return_new_draft_tokens,
            "draft_token_log_probs": return_draft_log_probs,
        }

    def prepare_for_generation(self, attn_metadata: AttentionMetadata,
                               spec_metadata: SpecMetadata,
                               spec_tree_manager: SpecTreeManager,
                               position_ids: torch.Tensor):
        batch_size = attn_metadata.num_seqs
        num_accepted_draft_tokens = spec_metadata.num_accepted_draft_tokens[:
                                                                            batch_size]
        seq_lens = attn_metadata.seq_lens_cuda[:batch_size]
        last_tokens_idx = torch.cumsum(
            seq_lens, dim=0,
            dtype=torch.long) - seq_lens + num_accepted_draft_tokens
        position_start_idx = position_ids[0, last_tokens_idx] + 1
        self.position_ids_buffer[:batch_size, :-1] = position_start_idx.unsqueeze(
            1) + spec_tree_manager.spec_dec_position_offsets[0, 1:].unsqueeze(
                0) - 1

        attn_metadata.kv_lens_cuda[:
                                   batch_size] -= seq_lens - num_accepted_draft_tokens - 1
        attn_metadata.kv_lens_cuda[:batch_size] += (
            self.max_total_draft_tokens + 1)
        attn_metadata._seq_lens[:batch_size].fill_(self.max_total_draft_tokens +
                                                   1)
        attn_metadata._seq_lens_cuda[:batch_size].fill_(
            self.max_total_draft_tokens + 1)
        # Recompute backend-specific decode page-table state with all active
        # SMC replay requests classified as generation.
        _mark_attn_metadata_generation_only(attn_metadata)
        attn_metadata.use_spec_decoding = True
        if attn_metadata.spec_decoding_position_offsets is None:
            attn_metadata.spec_decoding_position_offsets = torch.empty(
                [attn_metadata.max_num_requests, self.max_total_draft_tokens + 1],
                dtype=torch.int,
                device=position_ids.device)
        if attn_metadata.spec_decoding_packed_mask is None:
            attn_metadata.spec_decoding_packed_mask = torch.zeros(
                [attn_metadata.max_num_requests, self.max_total_draft_tokens + 1,
                 math.ceil((self.max_total_draft_tokens + 1) / 32)],
                dtype=torch.int,
                device=position_ids.device)
        if attn_metadata.spec_decoding_generation_lengths is None:
            attn_metadata.spec_decoding_generation_lengths = torch.empty(
                [attn_metadata.max_num_requests],
                dtype=torch.int,
                device=position_ids.device)
        attn_metadata.spec_decoding_position_offsets[:batch_size, :self.
                                                     max_total_draft_tokens] = spec_tree_manager.spec_dec_position_offsets[
                                                         0, 1:self.
                                                         max_total_draft_tokens +
                                                         1].unsqueeze(0) - 1
        attn_metadata.spec_decoding_position_offsets[:batch_size, self.
                                                     max_total_draft_tokens] = 0
        attn_metadata.spec_decoding_packed_mask[:
                                                batch_size, :, :] = spec_tree_manager.spec_dec_packed_mask_for_drafter_model
        attn_metadata.spec_decoding_generation_lengths[:
                                                       batch_size] = self.max_total_draft_tokens + 1
        spec_metadata.num_tokens = batch_size * (self.max_total_draft_tokens +
                                                 1)
