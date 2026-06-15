// PDE G8 — MTP / SPECULATIVE-DECODE device loop (standalone, sm_100).
//
// Spec-decode / Multi-Token-Prediction (MTP) is the single biggest validated
// tok/s/user lever for this decode stack (EAGLE-3.1-class ~2x). Its control flow
// is the hard part for a persistent decode engine: per step it
//   (1) DRAFTS K candidate tokens for positions pos..pos+K-1 (a cheap draft
//       model / MTP head),
//   (2) VERIFIES them against the target model in ONE batched forward, and
//   (3) ACCEPTS a VARIABLE-LENGTH prefix — the number n of drafts that pass
//       verification is DATA-DEPENDENT and differs every step — then advances
//       the decode position by the accepted count and emits exactly the accepted
//       tokens (+ one corrected/bonus target token).
//
// The naive implementation does step (3) with a HOST ROUND-TRIP: the verify
// kernel writes an accept-length, the host d2h's it, BRANCHES on it (advance the
// position by n, decide whether the request continues, rebuild the next draft
// window), then h2d's the new state. That per-step d2h-of-a-control-value +
// host branch + h2d is exactly what is CUDA-graph-capture-ILLEGAL (a readback
// inside capture is illegal) and is a large part of the decode execution gap.
//
// G8 keeps the WHOLE draft -> verify -> variable-length-accept -> advance loop
// DEVICE-RESIDENT inside one persistent cooperative launch: the accept-length is
// computed on the device, consumed on the device as a control decision (the loop
// advances the per-slot position by n and decides continuation device-side), and
// only the ACCEPTED TOKENS are emitted to a device->host completion ring. The
// accept-length NEVER crosses to the host as a control decision. This builds on
// the G9 cross-step device loop (request/completion rings + device-resident
// state) and the G3 device-resident data-dependent control flow (the accept
// count is the same kind of device-side branch as G3's top-k selection).
//
//   (A) DEVICE MTP ENGINE  : one cudaLaunchCooperativeKernel; per step per slot
//        draft K -> verify -> device-side variable accept n -> advance pos by n
//        -> emit n tokens. No per-step host contact, no d2h of the accept-length.
//   (B) HOST-ORCHESTRATED  : the SAME synthetic draft/verify, but each step the
//        verify kernel writes the accept-length, the host d2h's it, BRANCHES
//        (advance pos by n, rebuild the active list + next draft window), h2d's
//        the advanced state, and relaunches. Today's spec-decode control pattern.
//
// Correctness is bit-exact and NEVER a self-compare. The draft + verify are
// DETERMINISTIC integer hash-chains so an independent CPU reference reproduces,
// per step, BOTH the accepted-token stream AND the per-step accept-length n
// EXACTLY; A's stream + per-step n == B's == CPU's, with a VARIABLE n actually
// exercised (we report the accept-length histogram). The target token stream is
// the same splitmix64 hash-chain as G9 so a request's final emitted sequence is
// also identical to a non-speculative decode of the same request (spec-decode is
// output-equivalent to greedy decode — the gate checks that invariant too).
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
namespace g8 {

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Shapes. A "slot" is a continuous-batching slot (one in-flight request). Per
// step a slot drafts K candidates, verifies, accepts a variable prefix. The
// verify "compute" is a representative GEMV stand-in (target-model forward over
// the K+1 candidate positions) reduced to a scalar — the value of G8 is the
// VARIABLE-LENGTH device-side accept control, not GEMM throughput.
// ---------------------------------------------------------------------------
constexpr int kD = 128;            // per-slot state dim (verify GEMV width)
constexpr int kH = 128;            // verify GEMV output rows
constexpr int kMaxSlots = 256;     // concurrent in-flight slots (batch ceiling)
constexpr int kBlockThreads = 256;
constexpr int kWarps = kBlockThreads / 32;
constexpr int kMaxK = 8;           // max draft length per step (MTP depth ceiling)

// ---------------------------------------------------------------------------
// Deterministic integer hash-chains (splitmix64 finalizer: exactly reproducible
// regardless of float order / parallelism). Two distinct chains:
//   * TARGET chain  : the ground-truth token a non-speculative greedy decode of
//                     this request would emit at each position. token(pos)
//                     depends on token(pos-1) so the device must carry per-slot
//                     state across steps (identical to G9).
//   * DRAFT proposal: the draft model's candidate for a position, derived from
//                     the last ACCEPTED target token + the draft offset. A draft
//                     candidate is ACCEPTED iff it equals the target token at
//                     that position — the standard greedy spec-decode acceptance
//                     rule. To get a realistic, controllable accept-length
//                     DISTRIBUTION (not always 0 or K), a deterministic
//                     per-(request,pos) "agreement" predicate decides whether the
//                     draft matches the target at each candidate offset; the
//                     accept-length is the count of LEADING matches, capped at K,
//                     and the step ALWAYS emits one extra corrected/bonus target
//                     token (so n_emit in [1, K+1]). All of this is a pure
//                     function of (request_id, pos, K) => CPU == A == B.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ unsigned long long splitmix64(
    unsigned long long z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__host__ __device__ __forceinline__ unsigned long long seed_for_request(
    unsigned long long request_id) {
  return splitmix64(request_id * 0x100000001b3ull + 0xD1B54A32D192ED03ull);
}

// Target token at position `pos`, given the running target chain state
// (state == the token emitted at pos-1, or the seed for pos==0). Position-folded
// so a dropped/duplicated token diverges immediately. Identical recurrence shape
// to G9's advance_token (kept separate namespace so G8 is self-contained).
__host__ __device__ __forceinline__ unsigned long long target_token(
    unsigned long long prev_state, int pos) {
  return splitmix64(prev_state ^ (0x2545F4914F6CDD1Dull * (unsigned long long)(pos + 1)));
}

// Deterministic AGREEMENT predicate: does the draft's candidate for position
// `pos` (request_id) match the target token there? A pure hash of (request_id,
// pos) compared to a threshold gives a per-position accept probability `p_acc`
// (tunable via accept_num/accept_den) WITHOUT any floats — fully reproducible.
// This models a draft model with acceptance rate ~p_acc. (We separately expose
// the actual emitted draft token below so the emitted stream is well-defined.)
__host__ __device__ __forceinline__ bool draft_agrees(
    unsigned long long request_id, int pos, unsigned int accept_num,
    unsigned int accept_den) {
  unsigned long long h = splitmix64(request_id * 0xBF58476D1CE4E5B9ull ^
                                    (0x9E3779B97F4A7C15ull * (unsigned long long)(pos + 1)));
  // uniform in [0,den): accept iff < num  => P(accept) = num/den.
  return (unsigned int)(h % (unsigned long long)accept_den) < accept_num;
}

// The token the draft model PROPOSES for `pos`. When draft_agrees() is true this
// is defined to equal the target token (accepted); when false it is a distinct
// "wrong" token (rejected). Making the proposed token explicit lets a reference
// reconstruct exactly what the draft emitted, but only the ACCEPTED tokens (==
// target tokens) are ever emitted to the completion ring, so the emitted stream
// equals the greedy target stream regardless. Returned for completeness / debug.
__host__ __device__ __forceinline__ unsigned long long draft_token(
    unsigned long long request_id, unsigned long long target_tok_at_pos, int pos,
    unsigned int accept_num, unsigned int accept_den) {
  if (draft_agrees(request_id, pos, accept_num, accept_den)) return target_tok_at_pos;
  // a wrong proposal: perturb the target token deterministically.
  return splitmix64(target_tok_at_pos ^ 0xD1B54A32D192ED03ull ^
                    (0x100000001b3ull * (unsigned long long)(pos + 1)));
}

// ---------------------------------------------------------------------------
// THE CORE DEVICE-SIDE CONTROL DECISION: given a slot's current position `pos`,
// its remaining budget, and the draft length K, compute the data-dependent
// accept-length n (# of leading draft candidates that match the target) and the
// number of tokens to EMIT this step (n accepted + 1 bonus target token, clamped
// to the request's remaining length). This is what the host would otherwise d2h
// + branch on; in G8 it is computed and consumed entirely on the device. Pure
// function of (request_id, pos, K, remaining) so CPU/A/B agree bit-for-bit.
//
//   n_accept  = max j in [0,K] s.t. drafts[0..j-1] all agree (leading run)
//   n_emit    = min(n_accept + 1, remaining)   // +1 = corrected/bonus token
// Always >=1 (the bonus token) so forward progress is guaranteed every step.
// ---------------------------------------------------------------------------
struct StepDecision {
  int n_accept;   // # of draft candidates accepted (the variable-length prefix)
  int n_emit;     // # of tokens emitted this step (n_accept + 1 bonus, clamped)
};

__host__ __device__ __forceinline__ StepDecision decide_step(
    unsigned long long request_id, int pos, int K, int remaining,
    unsigned int accept_num, unsigned int accept_den) {
  int n_accept = 0;
  // count the leading run of agreements over the K draft offsets, BUT never
  // propose past the request's remaining-1 budget (the bonus token consumes the
  // last slot). The draft for emit-offset j targets position pos+j.
  int max_draft = remaining - 1;          // leave room for the bonus token
  if (max_draft > K) max_draft = K;
  for (int j = 0; j < max_draft; ++j) {
    if (draft_agrees(request_id, pos + j, accept_num, accept_den)) n_accept++;
    else break;
  }
  StepDecision d;
  d.n_accept = n_accept;
  d.n_emit = n_accept + 1;                // +1 corrected/bonus target token
  if (d.n_emit > remaining) d.n_emit = remaining;
  return d;
}

// ---------------------------------------------------------------------------
// Host-mapped REQUEST RING (host producer -> device consumer). Same SPSC design
// as G9: write descriptor, publish with a system-scope release on `tail`; device
// acquires `tail`, consumes [head,tail), releases `head`. `done` = host stop flag.
// ---------------------------------------------------------------------------
struct RequestDesc {
  unsigned long long request_id;
  int slot;
  int gen_len;
  int pad;
};
struct RequestRing {
  RequestDesc* buf;          // [cap]  host-mapped
  unsigned int* head;        // [1]    device advances (consumed)
  unsigned int* tail;        // [1]    host advances   (produced)
  int* done_flag;            // [1]    host sets 1 = no more requests
  unsigned int cap;
};

// ---------------------------------------------------------------------------
// Device->host COMPLETION RING. Same MPSC exactly-once design as G9: each emitted
// token reserves a unique slot with a device-scope atomicAdd on `prod`, writes
// the payload, then stamps a per-slot `ready` with a system-scope release; the
// host drains the dense ready-prefix in cursor order (no shared publish-counter
// serialization). A step emits n_emit tokens => n_emit reservations.
//
// G8 adds a per-completion `accept_len` field: the accept-length of the STEP that
// produced this token, stamped on EACH token of that step. The host reconstructs
// the per-step accept-length sequence from the completions WITHOUT ever using it
// as a control decision (it's pure telemetry on the device->host side); the gate
// compares this reconstructed sequence against the CPU reference's per-step n.
// ---------------------------------------------------------------------------
struct Completion {
  unsigned long long request_id;
  unsigned long long token;     // emitted (accepted/bonus) target token
  int slot;
  int pos;                      // token position within the request
  int accept_len;               // accept-length n of the producing step (telemetry)
  int step_idx;                 // which step of this request produced it (telemetry)
};
struct CompletionRing {
  Completion* buf;           // [cap]  host-mapped
  unsigned int* prod;        // [1]    device-scope MPSC reservation counter
  unsigned int* ready;       // [cap]  per-slot ready stamp (device release / host acquire)
  unsigned int cap;
};

// ---------------------------------------------------------------------------
// DEVICE-RESIDENT decode state. Per slot, advanced ON-DEVICE every step; the host
// NEVER rebuilds it in path (A). `pos` jumps by the VARIABLE n_emit each step
// (the data-dependent advance). `step_ctr` counts MTP steps for telemetry.
// ---------------------------------------------------------------------------
struct DecodeState {
  int* occupied;                 // [kMaxSlots]
  unsigned long long* req_id;    // [kMaxSlots]
  int* pos;                      // [kMaxSlots]  next token position (jumps by n_emit)
  int* gen_len;                  // [kMaxSlots]
  unsigned long long* hstate;    // [kMaxSlots]  running TARGET hash-chain state
  int* step_ctr;                 // [kMaxSlots]  # MTP steps taken (telemetry)
  float* svec;                   // [kMaxSlots * kD]  verify-GEMV state vector
};

struct EngineParams {
  RequestRing rq;
  CompletionRing cq;
  DecodeState st;
  const float* W;                // [kH * kD]  shared verify-GEMV weight
  unsigned int total_requests;
  unsigned int total_tokens;     // == sum of gen_len (completion-ring sizing)
  int K;                         // draft length per step
  unsigned int accept_num;       // acceptance-rate numerator   (p_acc = num/den)
  unsigned int accept_den;       // acceptance-rate denominator
  int* engine_stop;              // [2]: [0]=stop flag, [1]=tokens_emitted progress
};

// ---------------------------------------------------------------------------
// Representative per-step VERIFY GEMV stand-in: one CTA verifies one slot's draft
// window. act[h] = sum_d W[h,d]*svec[slot,d]; then svec is nudged by the last
// emitted token (bounded rational squash, no libm mismatch). Fixed reduction
// order so A and B match in float too — but the integer token + accept-length are
// the HARD gate. We scale the work by n_emit (a longer accept = a wider verify),
// to keep the compute representative of "verify K+1 positions".
// ---------------------------------------------------------------------------
__device__ __forceinline__ void verify_gemv_advance(const float* W, float* svec_base,
                                                    int slot, unsigned long long token,
                                                    int n_emit, float* s_act /*[kH]*/) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  float* sv = svec_base + (size_t)slot * kD;
  for (int h = warp; h < kH; h += kWarps) {
    const float* wrow = W + (size_t)h * kD;
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
    float nv = sv[d] * 0.5f + g + s_act[d % kH] * 1.0e-3f + (float)n_emit * 1.0e-4f;
    sv[d] = nv / (1.0f + fabsf(nv));
  }
  __syncthreads();
}

// ===========================================================================
// (A) DEVICE MTP ENGINE — ONE cooperative launch, loops steps device-side.
//
// Grid roles (cooperative => identical grid.sync() sequence on every CTA):
//   - CTA 0 ADMITS new requests from the request ring into free slots, and
//     detects global termination + sets engine_stop. (No host metadata rebuild.)
//   - ALL CTAs (incl. 0) cooperate on the per-step MTP work: grid-stride over
//     active slots; each active slot is handled by one CTA, which:
//        (i)   reads pos/hstate, advances the TARGET chain to materialize the
//              K candidate target tokens for pos..pos+K-1 (device-side),
//        (ii)  computes the DATA-DEPENDENT accept-length n via decide_step()
//              (the device-side variable-length control decision),
//        (iii) ADVANCES pos by n_emit and the target chain state by n_emit
//              tokens, ON-DEVICE (the variable jump),
//        (iv)  emits the n_emit accepted/bonus tokens to the completion ring
//              (each tagged with this step's accept_len), and
//        (v)   retires the slot device-side when pos reaches gen_len.
//     The accept-length n is COMPUTED and CONSUMED on the device; it is never
//     d2h'd as a control value. Only the accepted tokens leave (to the host).
//
// Barrier discipline: per step [sync after admission] -> [MTP compute+emit] ->
//   [sync after compute] -> [sync after termination]. Inactive regions still hit
//   every grid.sync() to keep the cooperative grid in lockstep.
//
// Termination: loops until the host set done AND every admitted request emitted
// all its tokens (progress counter reaches total_tokens) AND the request ring is
// drained. Safety step cap guards a never-satisfied condition.
// ===========================================================================
__global__ void __launch_bounds__(kBlockThreads)
    kDeviceMtpEngine(EngineParams p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem[];
  float* s_act = reinterpret_cast<float*>(smem);   // [kH]

  const bool ctrl = (blockIdx.x == 0);
  // each MTP step emits up to K+1 tokens, so #steps <= total_tokens + slack.
  const unsigned int kMaxSteps = p.total_tokens + 4u * p.total_requests + 1024u;

  for (unsigned int step = 0; step < kMaxSteps; ++step) {
    // ---------- (1) ADMISSION (CTA 0) ----------
    if (ctrl && threadIdx.x == 0) {
      cuda::atomic_ref<unsigned int, cuda::thread_scope_system> a_tail(*p.rq.tail);
      cuda::atomic_ref<unsigned int, cuda::thread_scope_system> a_head(*p.rq.head);
      unsigned int tail = a_tail.load(cuda::memory_order_acquire);
      unsigned int head = a_head.load(cuda::memory_order_relaxed);
      while (head != tail) {
        const RequestDesc rd = p.rq.buf[head % p.rq.cap];
        int slot = rd.slot;
        p.st.occupied[slot] = 1;
        p.st.req_id[slot] = rd.request_id;
        p.st.pos[slot] = 0;
        p.st.gen_len[slot] = rd.gen_len;
        p.st.hstate[slot] = seed_for_request(rd.request_id);
        p.st.step_ctr[slot] = 0;
        head += 1u;
        a_head.store(head, cuda::memory_order_release);
      }
    }
    grid.sync();  // (S0) admission visible

    // ---------- (2) MTP draft -> verify -> variable accept -> advance -> emit ----------
    for (int slot = blockIdx.x; slot < kMaxSlots; slot += gridDim.x) {
      // liveness decided by thread 0, broadcast via __syncthreads_or.
      int live_flag = 0;
      if (threadIdx.x == 0)
        live_flag = (p.st.occupied[slot] != 0) && (p.st.pos[slot] < p.st.gen_len[slot]) ? 1 : 0;
      bool live = __syncthreads_or(live_flag) != 0;
      if (!live) continue;

      __shared__ int s_n_emit;
      __shared__ int s_n_accept;
      __shared__ int s_pos0;
      __shared__ int s_step_idx;
      __shared__ unsigned long long s_req;
      __shared__ unsigned long long s_last_tok;
      __shared__ unsigned long long s_toks[kMaxK + 1];  // the n_emit emitted tokens

      if (threadIdx.x == 0) {
        int pos = p.st.pos[slot];
        int gen_len = p.st.gen_len[slot];
        int remaining = gen_len - pos;
        unsigned long long hs = p.st.hstate[slot];
        unsigned long long req = p.st.req_id[slot];

        // (i)+(ii) device-side data-dependent accept decision.
        StepDecision dec = decide_step(req, pos, p.K, remaining, p.accept_num, p.accept_den);

        // (iii) materialize the n_emit TARGET tokens (the accepted prefix + bonus)
        // by advancing the target chain n_emit times, ON-DEVICE. These are the
        // tokens a greedy decode would emit at pos..pos+n_emit-1 — accepted drafts
        // equal them by construction, and the bonus is the target's own token.
        unsigned long long state = hs;
        for (int j = 0; j < dec.n_emit; ++j) {
          unsigned long long tok = target_token(state, pos + j);
          s_toks[j] = tok;
          state = tok;
        }
        // advance device-resident state by the VARIABLE n_emit (the data-dependent
        // jump): pos += n_emit, hstate = last emitted token.
        s_n_emit = dec.n_emit;
        s_n_accept = dec.n_accept;
        s_pos0 = pos;
        s_step_idx = p.st.step_ctr[slot];
        s_req = req;
        s_last_tok = (dec.n_emit > 0) ? s_toks[dec.n_emit - 1] : hs;
        p.st.pos[slot] = pos + dec.n_emit;
        p.st.hstate[slot] = s_last_tok;
        p.st.step_ctr[slot] = s_step_idx + 1;
        if (pos + dec.n_emit >= gen_len) p.st.occupied[slot] = 0;
      }
      __syncthreads();

      // verify-GEMV stand-in (carries svec across steps); scaled by n_emit.
      verify_gemv_advance(p.W, p.st.svec, slot, s_last_tok, s_n_emit, s_act);

      // (iv) EMIT the n_emit tokens (MPSC exactly-once). Thread 0 publishes the
      // step's tokens; each tagged with this step's accept_len. The accept-length
      // is NOT sent as a control value — the device already consumed it above to
      // advance pos; here it rides along as device->host telemetry only.
      if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> a_prod(*p.cq.prod);
        for (int j = 0; j < s_n_emit; ++j) {
          unsigned int idx = a_prod.fetch_add(1u, cuda::memory_order_relaxed);
          Completion c;
          c.request_id = s_req;
          c.token = s_toks[j];
          c.slot = slot;
          c.pos = s_pos0 + j;
          c.accept_len = s_n_accept;
          c.step_idx = s_step_idx;
          p.cq.buf[idx % p.cq.cap] = c;     // payload first
          cuda::atomic_ref<unsigned int, cuda::thread_scope_system> a_ready(
              p.cq.ready[idx % p.cq.cap]);
          a_ready.store(idx + 1u, cuda::memory_order_release);
        }
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> a_prog(
            *reinterpret_cast<unsigned int*>(&p.engine_stop[1]));
        a_prog.fetch_add((unsigned int)s_n_emit, cuda::memory_order_relaxed);
      }
      __syncthreads();
    }
    grid.sync();  // (S1) all MTP compute + emits for this step done & visible

    // ---------- termination (CTA 0) ----------
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
      if (done && ring_drained && emitted >= p.total_tokens) p.engine_stop[0] = 1;
    }
    grid.sync();  // (S2) stop decision visible
    if (p.engine_stop[0] != 0) break;
  }
}

inline size_t engine_smem_bytes() { return (size_t)kH * sizeof(float); }

// ===========================================================================
// (B) HOST-ORCHESTRATED MTP step kernel — today's spec-decode control pattern.
//
// ONE launch == ONE MTP step over a host-supplied active-slot list. The kernel
// drafts/verifies and computes the accept-length per active slot, writes the
// emitted tokens AND the per-slot accept-length / n_emit to global, and DOES NOT
// advance pos itself — the HOST reads the accept-length back (d2h), BRANCHES on
// it (advance pos by n_emit, retire finished slots, rebuild the active list +
// next draft window), and h2d's the advanced state for the next step. The d2h of
// the accept-length + the host branch + h2d is the round-trip G8 eliminates.
//
//   active_slots[active_count] : host-built each step.
//   out_tok[active_count*(kMaxK+1)] : emitted tokens per active slot (d->h).
//   out_nemit[active_count]    : n_emit per active slot (d->h CONTROL value).
//   out_naccept[active_count]  : n_accept per active slot (d->h, telemetry).
// state arrays = DecodeState; host reads pos/hstate to advance (the round-trip).
// ===========================================================================
struct StepParams {
  DecodeState st;
  const float* W;
  const int* active_slots;   // [active_count]
  unsigned long long* out_tok;   // [active_count * (kMaxK+1)]
  int* out_nemit;            // [active_count]  <-- the accept-driven advance the host d2h's
  int* out_naccept;          // [active_count]
  int active_count;
  int K;
  unsigned int accept_num;
  unsigned int accept_den;
};

__global__ void __launch_bounds__(kBlockThreads) kHostMtpStep(StepParams p) {
  extern __shared__ unsigned char smem[];
  float* s_act = reinterpret_cast<float*>(smem);
  __shared__ int s_n_emit;
  __shared__ unsigned long long s_last_tok;
  __shared__ unsigned long long s_toks[kMaxK + 1];

  for (int ai = blockIdx.x; ai < p.active_count; ai += gridDim.x) {
    const int slot = p.active_slots[ai];
    if (threadIdx.x == 0) {
      int pos = p.st.pos[slot];
      int gen_len = p.st.gen_len[slot];
      int remaining = gen_len - pos;
      unsigned long long hs = p.st.hstate[slot];
      unsigned long long req = p.st.req_id[slot];
      StepDecision dec = decide_step(req, pos, p.K, remaining, p.accept_num, p.accept_den);
      unsigned long long state = hs;
      for (int j = 0; j < dec.n_emit; ++j) {
        unsigned long long tok = target_token(state, pos + j);
        s_toks[j] = tok;
        p.out_tok[(size_t)ai * (kMaxK + 1) + j] = tok;
        state = tok;
      }
      s_n_emit = dec.n_emit;
      s_last_tok = (dec.n_emit > 0) ? s_toks[dec.n_emit - 1] : hs;
      // write the accept-driven control value for the HOST to read back + branch.
      p.out_nemit[ai] = dec.n_emit;
      p.out_naccept[ai] = dec.n_accept;
      // NOTE: deliberately do NOT advance p.st.pos/hstate here — the HOST does it
      // after d2h'ing out_nemit (today's host-orchestrated control pattern).
    }
    __syncthreads();
    verify_gemv_advance(p.W, p.st.svec, slot, s_last_tok, s_n_emit, s_act);
  }
}

}  // namespace g8
}  // namespace pde
