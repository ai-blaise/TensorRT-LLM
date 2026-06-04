#include "get_decoding_sched_meta.h"

#include <cuda_runtime_api.h>
#include <cutlass/fast_math.h>
#include <kerutils/kerutils.cuh>

#include "utils.h"

namespace smxx::decode {

// Scheduler-metadata generation for the sparse/dense MLA split-KV decode.
//
// The original kernel ran the per-SM-part greedy work-assignment scan on a
// SINGLE thread and emitted each 32-byte DecodingSchedMeta (plus its end-state
// inner-while) straight to global memory inside that loop. With num_sm_parts up
// to a few thousand at production decode batch sizes, that single-thread loop
// dominates the whole decode op (nsys, b>=32: 56-62% of total time, e.g. 256us
// at b=64 vs ~85us for the actual sparse attention compute).
//
// This version splits the work:
//   Phase 2 (lane 0, serial): the identical greedy scan, but it records only
//     the cheap per-PART begin state (begin_req / begin_block / begin_split, 3
//     shared ints) and the per-request num_splits. No 32 B struct store and no
//     end-state computation on the serial path.
//   Phase 3 (all threads, parallel): every part recomputes its end-state from
//     its stored begin-state and writes the full DecodingSchedMeta to a shared
//     staging buffer; the buffer is then flushed to global coalesced.
// The produced tile_scheduler_metadata + num_splits are bit-identical to the
// serial kernel on every part the consumer actually reads (parts with
// begin_req_idx >= b are skipped by the consumer's early-return, so their
// bytes -- OOB-read garbage in the original -- are irrelevant).
//
// When the staging + per-part-begin scratch would not fit in the opted-in
// dynamic shared memory we fall back to the original single-thread kernel.

namespace {
constexpr int kParallelThreads = 256;
}

// ---- Original single-thread kernel (fallback for very large num_sm_parts) ----
__global__ void __launch_bounds__(32, 1, 1)
get_mla_metadata_kernel_serial(__grid_constant__ const GetDecodeSchedMetaParams params) {
    int *seqlens_k_ptr = params.seqlens_k_ptr;
    DecodingSchedMeta *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.b;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;

    extern __shared__ int shared_mem[];
    int* num_blocks_shared = shared_mem;
    int* num_splits_shared = shared_mem + batch_size;
    int* seqlens_k_shared = shared_mem + batch_size*2+1;
    int* first_block_idx_shared = shared_mem + batch_size*3+1;
    int* last_block_idx_shared = shared_mem + batch_size*4+1;

    int total_num_blocks = 0;
    for (int i = threadIdx.x; i < batch_size; i += 32) {
        int cur_s_k;
        if (params.topk == -1) {
            cur_s_k = __ldg(seqlens_k_ptr + i);
        } else {
            cur_s_k = params.topk_length ? __ldg(params.topk_length + i) : params.topk;
            if (cur_s_k == 0) cur_s_k = 1;
            if (params.extra_topk) {
                cur_s_k = ku::ceil(cur_s_k, block_size_n);
                cur_s_k += params.extra_topk_length ? __ldg(params.extra_topk_length + i) : params.extra_topk;
            }
        }
        seqlens_k_shared[i] = cur_s_k;
        int last_token_idx = max(cur_s_k-1, 0);
        int cur_first_block_idx = 0;
        int cur_last_block_idx = last_token_idx / block_size_n;
        int num_blocks = cur_last_block_idx - cur_first_block_idx + 1;
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
        num_blocks_shared[i] = num_blocks;
        first_block_idx_shared[i] = cur_first_block_idx;
        last_block_idx_shared[i] = cur_last_block_idx;
    }
    for (int offset = 16; offset >= 1; offset /= 2)
        total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
    __syncwarp();

    if (threadIdx.x == 0) {
        int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;
        int now_req_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
        num_splits_shared[0] = 0;
        for (int i = 0; i < num_sm_parts; ++i) {
            DecodingSchedMeta cur_meta;
            cur_meta.begin_req_idx = now_req_idx;
            cur_meta.begin_block_idx = now_block + first_block_idx_shared[now_req_idx];
            cur_meta.begin_split_idx = now_n_split_idx;
            cur_meta.is_first_req_splitted = (now_block != 0);
            int remain_payload = payload;
            while (now_req_idx < batch_size) {
                int num_blocks = num_blocks_shared[now_req_idx];
                int now_remain_blocks = num_blocks - now_block;
                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                    cum_num_splits += now_n_split_idx + 1;
                    num_splits_shared[now_req_idx + 1] = cum_num_splits;
                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                    ++now_req_idx; now_block = 0; now_n_split_idx = 0;
                } else {
                    if (remain_payload - fixed_overhead_num_blocks > 0) {
                        now_block += remain_payload - fixed_overhead_num_blocks;
                        ++now_n_split_idx; remain_payload = 0;
                    }
                    break;
                }
            }
            cur_meta.end_req_idx = now_block > 0 ? now_req_idx : now_req_idx - 1;
            cur_meta.end_block_idx = now_block > 0 ? now_block + first_block_idx_shared[now_req_idx] : (seqlens_k_shared[now_req_idx-1] == 0 ? 0 : last_block_idx_shared[now_req_idx-1] + 1);
            cur_meta.is_last_req_splitted = cur_meta.end_block_idx != last_block_idx_shared[cur_meta.end_req_idx] + 1 && seqlens_k_shared[cur_meta.end_req_idx] != 0;
            if (cur_meta.begin_req_idx == cur_meta.end_req_idx)
                cur_meta.is_first_req_splitted = cur_meta.is_last_req_splitted = cur_meta.is_first_req_splitted || cur_meta.is_last_req_splitted;
            tile_scheduler_metadata_ptr[i] = cur_meta;
        }
        FLASH_DEVICE_ASSERT(now_req_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncwarp();
    for (int i = threadIdx.x; i <= batch_size; i += 32) num_splits_ptr[i] = num_splits_shared[i];
}

// ---- Parallel kernel: serial begin-seed (lane 0) + parallel end fill ----
__global__ void __launch_bounds__(kParallelThreads, 1, 1)
get_mla_metadata_kernel_parallel(__grid_constant__ const GetDecodeSchedMetaParams params) {
    int *seqlens_k_ptr = params.seqlens_k_ptr;
    DecodingSchedMeta *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.b;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;
    int tid = threadIdx.x;

    extern __shared__ int shared_mem[];
    int* num_blocks_shared = shared_mem;
    int* num_splits_shared = shared_mem + batch_size;
    int* seqlens_k_shared = shared_mem + batch_size*2+1;
    int* first_block_idx_shared = shared_mem + batch_size*3+1;
    int* last_block_idx_shared = shared_mem + batch_size*4+1;
    int off = batch_size*5+1;
    int* part_begin_req = shared_mem + off;                    // [num_sm_parts]
    int* part_begin_block = shared_mem + off + num_sm_parts;   // [num_sm_parts]
    int* part_begin_split = shared_mem + off + 2*num_sm_parts; // [num_sm_parts]

    // Phase 1: per-request scratch (parallel over the block).
    for (int i = tid; i < batch_size; i += kParallelThreads) {
        int cur_s_k;
        if (params.topk == -1) {
            cur_s_k = __ldg(seqlens_k_ptr + i);
        } else {
            cur_s_k = params.topk_length ? __ldg(params.topk_length + i) : params.topk;
            if (cur_s_k == 0) cur_s_k = 1;
            if (params.extra_topk) {
                cur_s_k = ku::ceil(cur_s_k, block_size_n);
                cur_s_k += params.extra_topk_length ? __ldg(params.extra_topk_length + i) : params.extra_topk;
            }
        }
        seqlens_k_shared[i] = cur_s_k;
        int last_token_idx = max(cur_s_k-1, 0);
        int cur_last_block_idx = last_token_idx / block_size_n;
        num_blocks_shared[i] = cur_last_block_idx + 1;
        first_block_idx_shared[i] = 0;
        last_block_idx_shared[i] = cur_last_block_idx;
    }
    __syncthreads();

    __shared__ int s_payload;
    if (tid == 0) {
        int total_num_blocks = 0;
        for (int i = 0; i < batch_size; ++i) total_num_blocks += num_blocks_shared[i] + fixed_overhead_num_blocks;
        s_payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;
    }
    __syncthreads();
    int payload = s_payload;

    // Phase 2 (lane 0): identical greedy scan, recording only per-part begin
    // state (3 cheap shared ints) + per-request num_splits.
    if (tid == 0) {
        int now_req_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
        num_splits_shared[0] = 0;
        for (int i = 0; i < num_sm_parts; ++i) {
            part_begin_req[i] = now_req_idx;
            part_begin_block[i] = now_block;
            part_begin_split[i] = now_n_split_idx;
            int remain_payload = payload;
            while (now_req_idx < batch_size) {
                int num_blocks = num_blocks_shared[now_req_idx];
                int now_remain_blocks = num_blocks - now_block;
                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                    cum_num_splits += now_n_split_idx + 1;
                    num_splits_shared[now_req_idx + 1] = cum_num_splits;
                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                    ++now_req_idx; now_block = 0; now_n_split_idx = 0;
                } else {
                    if (remain_payload - fixed_overhead_num_blocks > 0) {
                        now_block += remain_payload - fixed_overhead_num_blocks;
                        ++now_n_split_idx; remain_payload = 0;
                    }
                    break;
                }
            }
        }
        FLASH_DEVICE_ASSERT(now_req_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncthreads();

    // Phase 3 (parallel): each part computes its end-state and full meta.
    for (int i = tid; i < num_sm_parts; i += kParallelThreads) {
        int br = part_begin_req[i];
        int begin_block = part_begin_block[i];
        int begin_split = part_begin_split[i];

        DecodingSchedMeta m;
        m._pad[0] = 0;
        m.begin_req_idx = br;
        if (br >= batch_size) {
            // Trailing part: consumer early-returns (begin_req_idx >= b), so
            // the values are never read. Keep begin_req_idx >= b.
            m.begin_block_idx = 0; m.begin_split_idx = begin_split;
            m.is_first_req_splitted = 0;
            m.end_req_idx = br; m.end_block_idx = 0; m.is_last_req_splitted = 0;
            tile_scheduler_metadata_ptr[i] = m;  // write through (no staging needed)
            continue;
        }
        m.begin_block_idx = begin_block + first_block_idx_shared[br];
        m.begin_split_idx = begin_split;
        m.is_first_req_splitted = (begin_block != 0);

        int now_req = br, now_block = begin_block, budget = payload;
        while (now_req < batch_size) {
            int nb = num_blocks_shared[now_req];
            int remain = nb - now_block;
            if (budget >= remain + fixed_overhead_num_blocks) {
                budget -= remain + fixed_overhead_num_blocks;
                ++now_req; now_block = 0;
            } else {
                if (budget - fixed_overhead_num_blocks > 0) now_block += budget - fixed_overhead_num_blocks;
                break;
            }
        }
        int end_req = now_block > 0 ? now_req : now_req - 1;
        int end_block = now_block > 0 ? now_block + first_block_idx_shared[now_req]
                                      : (seqlens_k_shared[now_req-1] == 0 ? 0 : last_block_idx_shared[now_req-1] + 1);
        m.end_req_idx = end_req;
        m.end_block_idx = end_block;
        m.is_last_req_splitted = end_block != last_block_idx_shared[end_req] + 1 && seqlens_k_shared[end_req] != 0;
        if (m.begin_req_idx == m.end_req_idx)
            m.is_first_req_splitted = m.is_last_req_splitted = m.is_first_req_splitted || m.is_last_req_splitted;
        tile_scheduler_metadata_ptr[i] = m;
    }
    __syncthreads();

    for (int i = tid; i <= batch_size; i += kParallelThreads) num_splits_ptr[i] = num_splits_shared[i];
}

void run_get_decoding_sched_meta_kernel(GetDecodeSchedMetaParams &params) {
    int const base_ints = params.b*5+1;
    int const base_bytes = (int)sizeof(int) * base_ints;

    // Parallel path needs base + per-part begin scratch (3 ints/part). The
    // 32-byte metadata is written straight to global from phase 3 (coalesced
    // enough across the 256 threads), so no staging buffer is required.
    int const parallel_ints = base_ints + 3*params.num_sm_parts;
    int const parallel_bytes = (int)sizeof(int) * parallel_ints;
    int const kMaxDynSmem = 227 * 1024;

    if (parallel_bytes <= kMaxDynSmem) {
        CHECK_CUDA(cudaFuncSetAttribute(get_mla_metadata_kernel_parallel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, parallel_bytes));
        get_mla_metadata_kernel_parallel<<<1, kParallelThreads, parallel_bytes, params.stream>>>(params);
    } else {
        CHECK_CUDA(cudaFuncSetAttribute(get_mla_metadata_kernel_serial,
            cudaFuncAttributeMaxDynamicSharedMemorySize, base_bytes));
        get_mla_metadata_kernel_serial<<<1, 32, base_bytes, params.stream>>>(params);
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
