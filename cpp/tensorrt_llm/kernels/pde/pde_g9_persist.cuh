// PDE G9 — CROSS-STEP PERSISTENCE (standalone, sm_100).
//
// The capstone PDE capability. Every prior gate (G0..G6) still relaunched the
// forward once PER decode step: the host loop does, every step, a kernel launch
// + a decode-metadata rebuild (active-request list, positions, seq-lens) + a
// host<->device sync to read/advance state. That per-step host round-trip is the
// bulk of the decode "execution gap" (decode ~20.8ms/tok = 20-40x the HBM-BW
// floor; almost all of it is per-step overhead, not the compute).
//
// G9 proves the engine can run the decode loop DEVICE-SIDE ACROSS STEPS: ONE
// persistent cooperative launch that loops over steps internally, each step
//   (1) polls a device-visible REQUEST RING the host writes new requests into
//       (host-mapped pinned memory; system-scope atomic head/tail + acquire
//        fence on the device side),
//   (2) runs a synthetic-but-representative per-step "decode" compute over the
//       currently-active slots (a small GEMV stand-in + a deterministic integer
//       token recurrence),
//   (3) advances a DEVICE-RESIDENT decode state (per-slot position, remaining
//       length, hash accumulator) — NO host metadata rebuild,
//   (4) emits the step's output token(s) into a device->host COMPLETION RING the
//       host drains (system-scope atomic + release fence on the device side).
// The kernel returns only after all admitted requests finish AND the host has
// signalled no-more-requests (a device-visible stop flag). The host side only
// ENQUEUES requests over time + DRAINS completions (continuous-batching
// admission) — it NEVER relaunches per step and NEVER rebuilds metadata.
//
// (B) the reference is the SAME synthetic compute but host-driven: relaunch the
// step kernel S times; the host rebuilds the active list + reads/writes per-slot
// state across the launch boundary every step (today's pattern).
//
// Correctness is bit-exact and NEVER a self-compare: the emitted token stream is
// a deterministic integer hash-chain recurrence keyed on (request_id, token_pos)
// that an independent CPU reference reproduces EXACTLY; A's stream == B's stream
// == CPU's stream, and the request ring + completion ring must deliver every
// request and every completion exactly once (no drops, no dupes) under a
// time-staggered admission pattern with a variable active count.
//
// Standalone nvcc -arch=sm_100. Does NOT touch the decode/model path; the
// ABI-frozen files (SparseMlaDecodeKvarnHotOp.cpp / hisparseKvarnBdrRead.cuh)
// are untouched.
#pragma once

#include "pde_substrate.cuh"

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cstdint>

namespace pde {
namespace g9 {

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Shapes. A "slot" is a continuous-batching slot (one in-flight request). The
// per-step compute is a GEMV of a per-slot state vector (dim D) against a shared
// weight matrix (H x D) -> a length-H activation, reduced to a scalar that
// perturbs the next state. Decode-representative: tiny per-step math, lots of
// per-step *control* (poll queue, pick active slots, advance state, emit).
// ---------------------------------------------------------------------------
constexpr int kD = 128;            // per-slot state dim
constexpr int kH = 128;            // GEMV output rows (representative width)
constexpr int kMaxSlots = 256;     // concurrent in-flight slots (batch ceiling)
constexpr int kBlockThreads = 256;
constexpr int kWarps = kBlockThreads / 32;

// ---------------------------------------------------------------------------
// Deterministic integer recurrence. The emitted *token* is a pure integer
// hash-chain keyed on (request_id, token_pos) so device A, device B and the CPU
// reference produce BIT-IDENTICAL token streams regardless of float order /
// parallel reduction. splitmix64 finalizer (associative-free, exactly
// reproducible) seeded from the request id; the chain advances one step per
// emitted token. This is what the HARD gate checks: it stresses the queue/ring
// + the device-resident state machine, which is the actual G9 capability — not
// floating-point GEMM throughput.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ unsigned long long splitmix64(
    unsigned long long z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Token at (request_id, pos): chain so token(pos) depends on token(pos-1),
// forcing the device to actually carry per-slot state across steps (a stateless
// closed form would not prove cross-step state). state0 = mix(request_id*K+1).
__host__ __device__ __forceinline__ unsigned long long seed_for_request(
    unsigned long long request_id) {
  return splitmix64(request_id * 0x100000001b3ull + 0xD1B54A32D192ED03ull);
}
__host__ __device__ __forceinline__ unsigned long long advance_token(
    unsigned long long prev_state, int pos) {
  // fold the position in so the chain is position-sensitive (and so a dropped or
  // duplicated step would diverge immediately).
  return splitmix64(prev_state ^ (0x2545F4914F6CDD1Dull * (unsigned long long)(pos + 1)));
}

// ---------------------------------------------------------------------------
// Host-mapped REQUEST RING (host producer -> device consumer).
//
// SPSC ring in pinned/host-mapped memory. The host writes a RequestDesc into
// slot (tail % cap) then publishes by bumping `tail` with a RELEASE store
// (system scope); the device reads `tail` with an ACQUIRE load (system scope),
// consumes [head, tail), then bumps `head` with a release store so the host can
// reclaim space. `done` is a host->device stop flag: set once the host has
// enqueued its last request. All counters are cuda::atomic<…, system> living in
// host-mapped memory so the store/load cross the PCIe coherence boundary with
// the right fences.
// ---------------------------------------------------------------------------
struct RequestDesc {
  unsigned long long request_id;  // logical id (for the hash chain + dedupe)
  int slot;                       // continuous-batching slot to occupy
  int gen_len;                    // number of tokens to decode for this request
  int pad;
};

// We keep the ring control ints as plain device-visible ints and drive them with
// cuda::atomic_ref<…, system> at the use site (works on host-mapped memory and
// avoids needing the struct itself to be an atomic type).
struct RequestRing {
  RequestDesc* buf;          // [cap]  host-mapped
  unsigned int* head;        // [1]    device advances (consumed)
  unsigned int* tail;        // [1]    host advances   (produced)
  int* done_flag;            // [1]    host sets 1 = no more requests
  unsigned int cap;
};

// ---------------------------------------------------------------------------
// Device->host COMPLETION RING (device producer -> host consumer).
//
// The device emits one Completion per generated token. SPSC: device reserves a
// slot with a RELAXED atomicAdd on the shared `prod` counter (MPSC: every active
// compute CTA is a producer; the atomic add hands each a UNIQUE, monotone slot =>
// exactly-once, no inter-producer coordination), writes the payload, then stamps
// a per-slot `ready` flag with a RELEASE store (system scope). The host consumer
// holds a read cursor and drains buf[cursor] while ready[cursor] is set (ACQUIRE),
// i.e. it consumes the dense ready-prefix WITHOUT the producers serializing on a
// shared publish counter. (An earlier design advanced a single contiguous `pub`
// via a producer CAS-chain — that forced all B per-step publishes into a strict
// serial order, O(B) latency/step; the per-slot `ready` stamp removes it.)
// Capacity is sized to total tokens so it never wraps in the bench (wrap handling
// is the same MPSC, omitted to keep the exactly-once proof auditable).
// ---------------------------------------------------------------------------
struct Completion {
  unsigned long long request_id;
  unsigned long long token;     // the emitted token (hash-chain value)
  int slot;
  int pos;                      // token position within the request
};

struct CompletionRing {
  Completion* buf;           // [cap]  host-mapped
  unsigned int* prod;        // [1]    shared MPSC reservation counter (atomicAdd)
  unsigned int* ready;       // [cap]  per-slot ready stamp (device release / host acquire)
  unsigned int cap;
};

// ---------------------------------------------------------------------------
// DEVICE-RESIDENT decode state. Per slot: occupied?, request_id, current pos,
// gen_len, and the running hash-chain state. Advanced ON-DEVICE every step; the
// host NEVER rebuilds it. Also a float state vector per slot for the GEMV
// stand-in (kept device-side; its evolution is deterministic but only checked by
// cosine, not the bit-exact token gate).
// ---------------------------------------------------------------------------
struct DecodeState {
  int* occupied;                 // [kMaxSlots]  1 if a live request sits here
  unsigned long long* req_id;    // [kMaxSlots]
  int* pos;                      // [kMaxSlots]  next token position
  int* gen_len;                  // [kMaxSlots]
  unsigned long long* hstate;    // [kMaxSlots]  running token hash-chain state
  float* svec;                   // [kMaxSlots * kD]  GEMV state vector
};

struct EngineParams {
  RequestRing rq;
  CompletionRing cq;
  DecodeState st;
  const float* W;                // [kH * kD]  shared GEMV weight (device const-ish)
  unsigned int total_requests;   // for the engine's "all finished" termination
  unsigned int total_tokens;     // == sum of gen_len (completion-ring capacity check)
  int* engine_stop;              // [1] device scratch the engine sets when fully drained
};

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

// Representative per-step GEMV: act[h] = sum_d W[h,d] * svec[slot,d]; then the
// state vector is nudged by a deterministic function of the new token so it
// keeps evolving (proves the float state is also carried across steps). One CTA
// handles one slot's GEMV (warp per row-group). Returns nothing; writes svec.
// The reduction order is fixed (warp-shuffle tree) so A and B match in float
// too — but only the integer token is the HARD gate.
__device__ __forceinline__ void gemv_advance_svec(const EngineParams& p, int slot,
                                                   unsigned long long token,
                                                   float* s_act /*[kH]*/) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  float* sv = p.st.svec + (size_t)slot * kD;
  // act[h] = dot(W[h,:], sv)
  for (int h = warp; h < kH; h += kWarps) {
    const float* wrow = p.W + (size_t)h * kD;
    float acc = 0.0f;
#pragma unroll
    for (int d = lane; d < kD; d += 32) acc += wrow[d] * sv[d];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s_act[h] = acc;
  }
  __syncthreads();
  // nudge svec by a bounded, deterministic function of the token + activation so
  // the vector evolves without blowing up: sv[d] = tanh(sv[d]*0.5 + g(token,d) +
  // act[d % kH]*1e-3). g is a small bounded float derived from the token bits.
  for (int d = threadIdx.x; d < kD; d += blockDim.x) {
    unsigned long long m = splitmix64(token ^ (0x9E3779B1u * (unsigned long long)(d + 1)));
    float g = ((float)(m & 0xFFFF) / 32768.0f) - 1.0f;   // in [-1,1)
    float nv = sv[d] * 0.5f + g + s_act[d % kH] * 1.0e-3f;
    // bounded map (no libm dependency mismatch risk: use a rational squash)
    sv[d] = nv / (1.0f + fabsf(nv));
  }
  __syncthreads();
}

// ===========================================================================
// (A) PERSISTENT CROSS-STEP ENGINE — ONE cooperative launch, loops over steps.
//
// Grid roles (all CTAs cooperative-launched, all hit the SAME barrier count):
//   - CTA 0 is the ADMISSION + EMIT controller: each step it drains the request
//     ring into free decode-state slots (device-side admission), and after the
//     compute it publishes one completion per active slot into the completion
//     ring (single producer => exactly-once monotone seq). It also detects
//     global termination and sets engine_stop.
//   - ALL CTAs (incl. CTA 0) cooperate on the per-step compute: a grid-stride
//     over active slots; each active slot's GEMV+state-advance is done by one
//     CTA. Slots are advanced ON-DEVICE; the host is never consulted.
//
// Barrier discipline (cooperative grid => identical sync sequence on every CTA):
//   per step: [sync after admission] -> [compute] -> [sync after compute] ->
//             [sync after emit/termination]. CTA 0 does the admission/emit work;
//             the others no-op those regions but STILL execute every grid.sync.
//
// Termination: the engine keeps looping steps until (a) the host has set the
// request-ring done flag AND (b) every admitted request has emitted all its
// tokens AND the ring head == tail (all requests consumed). A device-resident
// counter `tokens_emitted` (in engine_stop[1..]) tracks progress; when it
// reaches total_tokens and the done flag is set, CTA 0 sets engine_stop[0]=1 and
// every CTA breaks after the next barrier. A safety cap on step count guards
// against a never-satisfied condition (returns; host detects via stop flag).
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads)
    kPersistentEngine(EngineParams p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  float* s_act = reinterpret_cast<float*>(smem);   // [kH]

  const bool ctrl = (blockIdx.x == 0);

  // device-resident progress counter lives in engine_stop[1]; stop flag in [0].
  // (host zeroed both before launch.)
  // hard safety cap: total_tokens + cap idle spins; each step emits >=0 tokens.
  const unsigned int kMaxSteps = p.total_tokens + 4u * p.total_requests + 1024u;

  for (unsigned int step = 0; step < kMaxSteps; ++step) {
    // ---------- (1) ADMISSION: CTA 0 drains the request ring into free slots ----------
    if (ctrl && threadIdx.x == 0) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_system> a_tail(*p.rq.tail);
      cuda::atomic_ref<unsigned int, cuda::thread_scope_system> a_head(*p.rq.head);
      unsigned int tail = a_tail.load(cuda::memory_order_acquire);
      unsigned int head = a_head.load(cuda::memory_order_relaxed);
      while (head != tail) {
        const RequestDesc rd = p.rq.buf[head % p.rq.cap];   // payload published before tail bump
        int slot = rd.slot;
        // occupy the slot + initialize its device-resident decode state.
        p.st.occupied[slot] = 1;
        p.st.req_id[slot] = rd.request_id;
        p.st.pos[slot] = 0;
        p.st.gen_len[slot] = rd.gen_len;
        p.st.hstate[slot] = seed_for_request(rd.request_id);
        head += 1u;
        a_head.store(head, cuda::memory_order_release);  // reclaim ring space
      }
    }
    grid.sync();  // (S0) admission visible to all compute CTAs

    // ---------- (2)+(3) COMPUTE + ON-DEVICE STATE ADVANCE ----------
    // grid-stride over slots; one CTA per active slot. Compute the next token
    // (integer hash-chain) and advance svec (float GEMV stand-in) ON-DEVICE.
    __shared__ int s_token_ready;       // did this CTA produce a token this step?
    __shared__ unsigned long long s_token;
    __shared__ int s_emit_slot;
    __shared__ int s_emit_pos;
    if (threadIdx.x == 0) { s_token_ready = 0; }
    __syncthreads();

    for (int slot = blockIdx.x; slot < kMaxSlots; slot += gridDim.x) {
      bool live;
      int pos, gen_len;
      unsigned long long hs;
      if (threadIdx.x == 0) {
        live = (p.st.occupied[slot] != 0) && (p.st.pos[slot] < p.st.gen_len[slot]);
      }
      // broadcast liveness decision (only thread 0 read it)
      live = __syncthreads_or(threadIdx.x == 0 ? (live ? 1 : 0) : 0) != 0;
      if (!live) continue;
      if (threadIdx.x == 0) {
        pos = p.st.pos[slot];
        gen_len = p.st.gen_len[slot];
        hs = p.st.hstate[slot];
        unsigned long long tok = advance_token(hs, pos);
        s_token = tok;
        s_emit_slot = slot;
        s_emit_pos = pos;
        // advance device-resident state ON-DEVICE (no host rebuild)
        p.st.hstate[slot] = tok;
        p.st.pos[slot] = pos + 1;
        s_token_ready = 1;
        if (pos + 1 >= gen_len) { p.st.occupied[slot] = 0; }
      }
      __syncthreads();
      // float GEMV stand-in (carries svec across steps); uses the new token.
      gemv_advance_svec(p, s_emit_slot, s_token, s_act);
      // EMIT (MPSC, exactly-once, no inter-producer serialization): every active
      // compute CTA publishes its own token. Reserve a UNIQUE slot with an atomic
      // add on the shared prod counter, write the payload, then stamp ready[slot]
      // with a release store so the host consumer can drain the dense ready-prefix
      // in cursor order. No CAS-chain: producers never wait on each other.
      if (threadIdx.x == 0 && s_token_ready) {
        // prod is DEVICE memory (host drains via the per-slot ready stamp, never
        // reads prod), so a device-scope atomic suffices — far cheaper than a
        // system-scope RMW on host-mapped memory under B-way contention/step.
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> a_prod(*p.cq.prod);
        unsigned int idx = a_prod.fetch_add(1u, cuda::memory_order_relaxed);
        Completion c;
        c.request_id = p.st.req_id[s_emit_slot];
        c.token = s_token;
        c.slot = s_emit_slot;
        c.pos = s_emit_pos;
        p.cq.buf[idx % p.cq.cap] = c;     // payload first
        // publish this slot: stamp ready = idx+1 (nonzero) with system-scope
        // release so the payload write is visible before the host sees ready.
        cuda::atomic_ref<unsigned int, cuda::thread_scope_system> a_ready(
            p.cq.ready[idx % p.cq.cap]);
        a_ready.store(idx + 1u, cuda::memory_order_release);
        // count global progress
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> a_prog(
            *reinterpret_cast<unsigned int*>(&p.engine_stop[1]));
        a_prog.fetch_add(1u, cuda::memory_order_relaxed);
      }
      if (threadIdx.x == 0) { s_token_ready = 0; }
      __syncthreads();
    }
    grid.sync();  // (S1) all compute + emits for this step done & visible

    // ---------- termination check (CTA0) ----------
    if (ctrl && threadIdx.x == 0) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> a_prog(
          *reinterpret_cast<unsigned int*>(&p.engine_stop[1]));
      unsigned int emitted = a_prog.load(cuda::memory_order_relaxed);
      cuda::atomic_ref<int, cuda::thread_scope_system> a_done(*p.rq.done_flag);
      cuda::atomic_ref<unsigned int, cuda::thread_scope_system> a_tail(*p.rq.tail);
      cuda::atomic_ref<unsigned int, cuda::thread_scope_system> a_head(*p.rq.head);
      int done = a_done.load(cuda::memory_order_acquire);
      unsigned int head = a_head.load(cuda::memory_order_relaxed);
      unsigned int tail = a_tail.load(cuda::memory_order_relaxed);
      bool ring_drained = (head == tail);
      if (done && ring_drained && emitted >= p.total_tokens) {
        p.engine_stop[0] = 1;     // plain store; broadcast via grid.sync below
      }
    }
    grid.sync();  // (S2) stop decision visible to all CTAs
    if (p.engine_stop[0] != 0) break;
  }
}

inline size_t engine_smem_bytes() { return (size_t)kH * sizeof(float); }

// ===========================================================================
// (B) PER-STEP-RELAUNCH BASELINE step kernel (today's pattern).
//
// ONE kernel launch == ONE decode step over a host-supplied active-slot list.
// The HOST rebuilds the active list every step (continuous-batching metadata
// rebuild), passes it in, and reads/writes the per-slot state across the launch
// boundary. The kernel: for each active slot in the list, compute the next token
// (same hash-chain), advance svec, and write the token + advanced state to global
// for the host to read back. No grid barriers (single step, plain grid).
//
// `active_slots[active_count]` : slots the host decided are live this step.
// `out_token[active_count]`    : this step's token per active slot (host drains).
// state arrays are the same DecodeState; the host reads pos/hstate back each step
// to rebuild the next active list (the metadata round-trip being measured).
// ===========================================================================
struct StepParams {
  DecodeState st;
  const float* W;
  const int* active_slots;   // [active_count]  host-built each step
  Completion* out_step;      // [active_count]  this step's emissions (d->h)
  int active_count;
};

// gemv stand-in for the relaunch step kernel (identical math to the engine's;
// duplicated only because it takes StepParams). Kept byte-identical so A and B
// produce the same float svec evolution too.
__device__ __forceinline__ void gemv_advance_svec_step(const StepParams& p,
                                                       int slot,
                                                       unsigned long long token,
                                                       float* s_act) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  float* sv = p.st.svec + (size_t)slot * kD;
  for (int h = warp; h < kH; h += kWarps) {
    const float* wrow = p.W + (size_t)h * kD;
    float acc = 0.0f;
#pragma unroll
    for (int d = lane; d < kD; d += 32) acc += wrow[d] * sv[d];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s_act[h] = acc;
  }
  __syncthreads();
  for (int d = threadIdx.x; d < kD; d += blockDim.x) {
    unsigned long long m = splitmix64(token ^ (0x9E3779B1u * (unsigned long long)(d + 1)));
    float g = ((float)(m & 0xFFFF) / 32768.0f) - 1.0f;
    float nv = sv[d] * 0.5f + g + s_act[d % kH] * 1.0e-3f;
    sv[d] = nv / (1.0f + fabsf(nv));
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kBlockThreads) kRelaunchStep(StepParams p) {
  extern __shared__ unsigned char smem[];
  float* s_act = reinterpret_cast<float*>(smem);
  __shared__ unsigned long long s_token;
  for (int ai = blockIdx.x; ai < p.active_count; ai += gridDim.x) {
    const int slot = p.active_slots[ai];
    int pos, gen_len;
    unsigned long long hs;
    if (threadIdx.x == 0) {
      pos = p.st.pos[slot];
      gen_len = p.st.gen_len[slot];
      hs = p.st.hstate[slot];
      unsigned long long tok = advance_token(hs, pos);
      s_token = tok;
      p.st.hstate[slot] = tok;
      p.st.pos[slot] = pos + 1;
      if (pos + 1 >= gen_len) p.st.occupied[slot] = 0;
      Completion c;
      c.request_id = p.st.req_id[slot];
      c.token = tok;
      c.slot = slot;
      c.pos = pos;
      p.out_step[ai] = c;
    }
    __syncthreads();
    gemv_advance_svec_step(p, slot, s_token, s_act);
  }
}

}  // namespace g9
}  // namespace pde
