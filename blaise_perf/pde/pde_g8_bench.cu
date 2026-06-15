// PDE G8 gate microbench (standalone, sm_100). REAL numbers on GPU0.
//
// G8 = MTP / SPECULATIVE-DECODE DEVICE LOOP: ONE persistent cooperative launch
// runs the draft -> verify -> VARIABLE-LENGTH-accept -> advance loop device-side
// across steps. The accept-length n (how many of the K drafts pass verification)
// is DATA-DEPENDENT and differs per step; G8 computes AND consumes it on the
// device (advances the per-slot position by the variable n device-side) and emits
// only the accepted tokens. Compared to:
//
//   (A) DEVICE MTP ENGINE  : one cudaLaunchCooperativeKernel; a host producer
//        enqueues requests, a host consumer drains completions. The accept-length
//        NEVER goes to host as a control decision; no per-step relaunch.
//   (B) HOST-ORCHESTRATED  : the SAME synthetic draft/verify, but each step the
//        kernel writes the accept-length, the host d2h's it, BRANCHES (advance
//        pos by n, rebuild active list + next draft window), h2d's, relaunches.
//        Today's spec-decode control pattern.
//
// Both checked vs an INDEPENDENT CPU reference. HARD gate: A's per-(request,pos)
// emitted token == B's == CPU's, BIT-EXACT, AND A's per-step accept-length
// sequence == B's == CPU's, with a VARIABLE n actually exercised (we report the
// accept-length histogram). The completion ring delivers every token exactly once
// (no drops/dupes) under time-staggered admission. Spec-decode output-equivalence
// to greedy decode is also checked (emitted stream == the pure target hash-chain).
//
// Build: nvcc -std=c++17 -arch=sm_100 -O3 -o pde_g8_bench pde_g8_bench.cu -lpthread
#include "pde_g8_mtp.cuh"

#include <cooperative_groups.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <thread>
#include <vector>

using namespace pde;

#define CK(x)                                                     \
  do {                                                            \
    cudaError_t e_ = (x);                                         \
    if (e_ != cudaSuccess) {                                      \
      printf("CUDA_ERR %s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
             cudaGetErrorString(e_));                             \
      std::exit(1);                                               \
    }                                                             \
  } while (0)

static inline float fill_val(uint64_t i, uint64_t salt) {
  uint64_t z = (i + salt) * 0x9E3779B97F4A7C15ull;
  z ^= z >> 29; z *= 0xBF58476D1CE4E5B9ull; z ^= z >> 32;
  return ((float)(z & 0xFFFF) / 32768.0f) - 1.0f;
}

struct ReqSpec {
  unsigned long long request_id;
  int gen_len;
  int arrival;
};

static std::vector<ReqSpec> make_workload(int n_req, int seed, int min_len,
                                          int max_len) {
  std::vector<ReqSpec> w(n_req);
  for (int i = 0; i < n_req; ++i) {
    uint64_t h = (uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull + (uint64_t)seed;
    h ^= h >> 31;
    int len = min_len + (int)(h % (uint64_t)(max_len - min_len + 1));
    w[i].request_id = 0x1000ull + (unsigned long long)i;
    w[i].gen_len = len;
    w[i].arrival = i;
  }
  return w;
}

// ---------------------------------------------------------------------------
// CPU reference. For each request, replay the EXACT MTP loop the device runs:
//   pos = 0; state = seed; while pos < gen_len:
//     dec = decide_step(req, pos, K, gen_len-pos);          // device-identical
//     emit n_emit target tokens (advancing the target chain);
//     record this step's accept_len for EVERY emitted token (telemetry parity);
//     pos += n_emit; state = last emitted token.
// Returns, per request: the emitted-token vector (length gen_len) AND the
// per-step accept-length sequence (variable-length). Ground truth — NEVER a
// device self-compare. Also returns the PURE greedy target stream so we can check
// spec-decode output-equivalence (emitted == greedy).
// ---------------------------------------------------------------------------
struct RefEntry {
  std::vector<unsigned long long> toks;     // emitted tokens by position [gen_len]
  std::vector<int> accept_per_pos;          // step's accept_len, by position [gen_len]
  std::vector<int> step_accept;             // per-step accept-length sequence
  std::vector<unsigned long long> greedy;   // pure target hash-chain [gen_len]
};

static std::map<unsigned long long, RefEntry> cpu_reference(
    const std::vector<ReqSpec>& w, int K, unsigned int anum, unsigned int aden) {
  std::map<unsigned long long, RefEntry> ref;
  for (const auto& r : w) {
    RefEntry e;
    e.toks.resize(r.gen_len);
    e.accept_per_pos.resize(r.gen_len);
    e.greedy.resize(r.gen_len);
    // pure greedy target stream (output-equivalence reference).
    {
      unsigned long long s = g8::seed_for_request(r.request_id);
      for (int pos = 0; pos < r.gen_len; ++pos) {
        unsigned long long tok = g8::target_token(s, pos);
        e.greedy[pos] = tok;
        s = tok;
      }
    }
    // MTP loop (must reproduce the device's variable-accept advance exactly).
    int pos = 0;
    unsigned long long state = g8::seed_for_request(r.request_id);
    while (pos < r.gen_len) {
      int remaining = r.gen_len - pos;
      g8::StepDecision dec = g8::decide_step(r.request_id, pos, K, remaining, anum, aden);
      for (int j = 0; j < dec.n_emit; ++j) {
        unsigned long long tok = g8::target_token(state, pos + j);
        e.toks[pos + j] = tok;
        e.accept_per_pos[pos + j] = dec.n_accept;
        state = tok;
      }
      e.step_accept.push_back(dec.n_accept);
      pos += dec.n_emit;
    }
    ref[r.request_id] = std::move(e);
  }
  return ref;
}

struct RingCheck { bool ok; long total; long drops; long dupes; long mismatch; long missing; };
static RingCheck check_completions(
    const std::vector<g8::Completion>& got,
    const std::map<unsigned long long, RefEntry>& ref) {
  RingCheck c{true, (long)got.size(), 0, 0, 0, 0};
  std::map<unsigned long long, std::vector<int>> seen;
  long expected_total = 0;
  for (const auto& kv : ref) { seen[kv.first].assign(kv.second.toks.size(), 0);
                               expected_total += (long)kv.second.toks.size(); }
  for (const auto& comp : got) {
    auto it = ref.find(comp.request_id);
    if (it == ref.end()) { c.mismatch++; c.ok = false; continue; }
    if (comp.pos < 0 || comp.pos >= (int)it->second.toks.size()) { c.mismatch++; c.ok = false; continue; }
    seen[comp.request_id][comp.pos]++;
    if (seen[comp.request_id][comp.pos] > 1) { c.dupes++; c.ok = false; }
    if (comp.token != it->second.toks[comp.pos]) { c.mismatch++; c.ok = false; }
    // per-token accept_len telemetry must match the reference's accept_per_pos.
    if (comp.accept_len != it->second.accept_per_pos[comp.pos]) { c.mismatch++; c.ok = false; }
  }
  for (const auto& kv : seen)
    for (int v : kv.second) if (v == 0) { c.missing++; c.ok = false; }
  if (c.total != expected_total) c.drops = expected_total - c.total;
  if (c.drops != 0) c.ok = false;
  return c;
}

// Reconstruct, from a drained completion set, each request's emitted-token vector
// (by pos) AND its per-STEP accept-length sequence. The per-step sequence is
// recovered from the completions' (step_idx, accept_len) tags — purely as
// device->host telemetry, never used as a control decision. Returns maps keyed by
// request_id.
struct Recon {
  std::map<unsigned long long, std::vector<unsigned long long>> toks;   // by pos
  std::map<unsigned long long, std::vector<int>> step_accept;           // by step
};
static Recon reconstruct(const std::vector<g8::Completion>& got,
                         const std::map<unsigned long long, RefEntry>& ref) {
  Recon r;
  for (const auto& kv : ref) {
    r.toks[kv.first].assign(kv.second.toks.size(), 0ull);
  }
  // per request, collect step_idx -> accept_len (each step tags all its tokens
  // with the same accept_len; take the first seen per step_idx).
  std::map<unsigned long long, std::map<int,int>> per_req_steps;
  for (const auto& comp : got) {
    auto it = r.toks.find(comp.request_id);
    if (it == r.toks.end()) continue;
    if (comp.pos >= 0 && comp.pos < (int)it->second.size())
      it->second[comp.pos] = comp.token;
    per_req_steps[comp.request_id][comp.step_idx] = comp.accept_len;
  }
  for (const auto& kv : per_req_steps) {
    std::vector<int> seq;
    seq.reserve(kv.second.size());
    for (const auto& sk : kv.second) seq.push_back(sk.second);  // ordered by step_idx
    r.step_accept[kv.first] = std::move(seq);
  }
  return r;
}

static bool toks_bit_equal(
    const std::map<unsigned long long, std::vector<unsigned long long>>& a,
    const std::map<unsigned long long, std::vector<unsigned long long>>& b) {
  if (a.size() != b.size()) return false;
  for (const auto& kv : a) {
    auto it = b.find(kv.first);
    if (it == b.end() || it->second.size() != kv.second.size()) return false;
    for (size_t i = 0; i < kv.second.size(); ++i)
      if (kv.second[i] != it->second[i]) return false;
  }
  return true;
}
static bool toks_equal_ref(
    const std::map<unsigned long long, std::vector<unsigned long long>>& a,
    const std::map<unsigned long long, RefEntry>& ref,
    bool greedy) {
  if (a.size() != ref.size()) return false;
  for (const auto& kv : a) {
    auto it = ref.find(kv.first);
    if (it == ref.end()) return false;
    const auto& rv = greedy ? it->second.greedy : it->second.toks;
    if (kv.second.size() != rv.size()) return false;
    for (size_t i = 0; i < rv.size(); ++i)
      if (kv.second[i] != rv[i]) return false;
  }
  return true;
}
static bool stepacc_equal_ref(
    const std::map<unsigned long long, std::vector<int>>& a,
    const std::map<unsigned long long, RefEntry>& ref) {
  if (a.size() != ref.size()) return false;
  for (const auto& kv : a) {
    auto it = ref.find(kv.first);
    if (it == ref.end()) return false;
    if (kv.second.size() != it->second.step_accept.size()) return false;
    for (size_t i = 0; i < kv.second.size(); ++i)
      if (kv.second[i] != it->second.step_accept[i]) return false;
  }
  return true;
}
static bool stepacc_bit_equal(
    const std::map<unsigned long long, std::vector<int>>& a,
    const std::map<unsigned long long, std::vector<int>>& b) {
  if (a.size() != b.size()) return false;
  for (const auto& kv : a) {
    auto it = b.find(kv.first);
    if (it == b.end() || it->second.size() != kv.second.size()) return false;
    for (size_t i = 0; i < kv.second.size(); ++i)
      if (kv.second[i] != it->second[i]) return false;
  }
  return true;
}

static int occ_blocks_per_sm(const void* kernel, int bt, size_t smem) {
  int b = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, kernel, bt, smem));
  return b;
}
template <typename T>
static void alloc_mapped(int count, T** hptr, T** dptr) {
  CK(cudaHostAlloc((void**)hptr, (size_t)count * sizeof(T), cudaHostAllocMapped));
  CK(cudaHostGetDevicePointer((void**)dptr, (void*)*hptr, 0));
}

// ===========================================================================
// Run (A): device MTP engine. Host PRODUCER enqueues on a stagger; host CONSUMER
// drains completions. The accept-length never crosses to host as control.
// ===========================================================================
struct RunResult {
  std::vector<g8::Completion> comps;
  double wall_us;
  double per_token_us;
};

static RunResult run_engine(const std::vector<ReqSpec>& w, int n_slots,
                            int stagger_every, int sms, int occ, size_t smem,
                            const float* dW, int K, unsigned int anum,
                            unsigned int aden, bool timed) {
  const unsigned int n_req = (unsigned int)w.size();
  unsigned int total_tokens = 0;
  for (const auto& r : w) total_tokens += (unsigned int)r.gen_len;

  g8::RequestDesc *hRqBuf, *dRqBuf;
  unsigned int *hRqHead, *dRqHead, *hRqTail, *dRqTail;
  int *hDone, *dDone;
  const unsigned int rq_cap = 1u << 12;
  alloc_mapped<g8::RequestDesc>(rq_cap, &hRqBuf, &dRqBuf);
  alloc_mapped<unsigned int>(1, &hRqHead, &dRqHead);
  alloc_mapped<unsigned int>(1, &hRqTail, &dRqTail);
  alloc_mapped<int>(1, &hDone, &dDone);

  g8::Completion *hCqBuf, *dCqBuf;
  unsigned int *dCqProd, *hCqReady, *dCqReady;
  const unsigned int cq_cap = total_tokens + 16u;
  alloc_mapped<g8::Completion>(cq_cap, &hCqBuf, &dCqBuf);
  CK(cudaMalloc(&dCqProd, sizeof(unsigned int)));
  alloc_mapped<unsigned int>(cq_cap, &hCqReady, &dCqReady);

  g8::DecodeState st;
  CK(cudaMalloc(&st.occupied, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.req_id, g8::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.pos, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.gen_len, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.hstate, g8::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.step_ctr, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.svec, (size_t)g8::kMaxSlots * g8::kD * sizeof(float)));
  std::vector<float> hsv((size_t)g8::kMaxSlots * g8::kD);
  for (size_t i = 0; i < hsv.size(); ++i) hsv[i] = fill_val(i, 555);

  int* dStop; CK(cudaMalloc(&dStop, 2 * sizeof(int)));

  g8::EngineParams p{};
  p.rq = {dRqBuf, dRqHead, dRqTail, dDone, rq_cap};
  p.cq = {dCqBuf, dCqProd, dCqReady, cq_cap};
  p.st = st; p.W = dW; p.total_requests = n_req; p.total_tokens = total_tokens;
  p.K = K; p.accept_num = anum; p.accept_den = aden; p.engine_stop = dStop;

  cudaStream_t stream; CK(cudaStreamCreate(&stream));
  int grid = sms * occ;
  void* args[] = {&p};

  std::atomic<int> slot_inflight[g8::kMaxSlots];
  for (int i = 0; i < g8::kMaxSlots; ++i) slot_inflight[i].store(0);

  auto launch = [&]() {
    *hRqHead = 0; *hRqTail = 0; *hDone = 0;
    CK(cudaMemsetAsync(dCqProd, 0, sizeof(unsigned int), stream));
    for (unsigned int i = 0; i < cq_cap; ++i) hCqReady[i] = 0u;
    CK(cudaMemsetAsync(dStop, 0, 2 * sizeof(int), stream));
    CK(cudaMemsetAsync(st.occupied, 0, g8::kMaxSlots * sizeof(int), stream));
    CK(cudaMemsetAsync(st.pos, 0, g8::kMaxSlots * sizeof(int), stream));
    CK(cudaMemsetAsync(st.step_ctr, 0, g8::kMaxSlots * sizeof(int), stream));
    CK(cudaMemcpyAsync(st.svec, hsv.data(), hsv.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    CK(cudaStreamSynchronize(stream));
    CK(cudaLaunchCooperativeKernel((void*)g8::kDeviceMtpEngine, grid,
                                   g8::kBlockThreads, args, smem, stream));
  };

  std::vector<g8::Completion> drained;
  auto run_once = [&](std::vector<g8::Completion>* capture) {
    drained.clear();
    for (int i = 0; i < g8::kMaxSlots; ++i) slot_inflight[i].store(0);
    launch();

    std::thread producer([&]() {
      unsigned int next = 0;
      while (next < n_req) {
        int slot = -1;
        for (int s = 0; s < n_slots; ++s)
          if (slot_inflight[s].load(std::memory_order_acquire) == 0) { slot = s; break; }
        if (slot < 0) { std::this_thread::yield(); continue; }
        const ReqSpec& r = w[next];
        slot_inflight[slot].store(1, std::memory_order_release);
        unsigned int tail = *hRqTail;
        g8::RequestDesc rd; rd.request_id = r.request_id; rd.slot = slot;
        rd.gen_len = r.gen_len; rd.pad = 0;
        hRqBuf[tail % rq_cap] = rd;
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<std::atomic<unsigned int>*>(hRqTail)
            ->store(tail + 1u, std::memory_order_release);
        next++;
        if (stagger_every > 0 && (next % stagger_every) == 0)
          std::this_thread::sleep_for(std::chrono::microseconds(20));
      }
      std::atomic_thread_fence(std::memory_order_release);
      reinterpret_cast<std::atomic<int>*>(hDone)->store(1, std::memory_order_release);
    });

    unsigned int drained_cnt = 0;
    std::map<unsigned long long, int> reqlen;
    for (const auto& r : w) reqlen[r.request_id] = r.gen_len;
    std::map<int, int> slot_emitted;
    while (drained_cnt < total_tokens) {
      unsigned int slot_pos = drained_cnt % cq_cap;
      unsigned int rdy = reinterpret_cast<std::atomic<unsigned int>*>(&hCqReady[slot_pos])
                             ->load(std::memory_order_acquire);
      if (rdy != 0u) {
        std::atomic_thread_fence(std::memory_order_acquire);
        g8::Completion c = hCqBuf[slot_pos];
        drained.push_back(c);
        slot_emitted[c.slot]++;
        if (slot_emitted[c.slot] >= reqlen[c.request_id]) {
          slot_emitted[c.slot] = 0;
          slot_inflight[c.slot].store(0, std::memory_order_release);
        }
        drained_cnt++;
      } else {
        std::this_thread::yield();
      }
    }
    producer.join();
    CK(cudaStreamSynchronize(stream));
    if (capture) *capture = drained;
  };

  RunResult res;
  run_once(&res.comps);
  if (timed) {
    const int REPS = 20, WARM = 3;
    for (int i = 0; i < WARM; ++i) run_once(nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPS; ++r) run_once(nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / REPS;
    res.wall_us = us; res.per_token_us = us / total_tokens;
  } else { res.wall_us = 0; res.per_token_us = 0; }

  cudaStreamDestroy(stream);
  cudaFree(st.occupied); cudaFree(st.req_id); cudaFree(st.pos); cudaFree(st.gen_len);
  cudaFree(st.hstate); cudaFree(st.step_ctr); cudaFree(st.svec); cudaFree(dStop);
  cudaFreeHost(hRqBuf); cudaFreeHost(hRqHead); cudaFreeHost(hRqTail); cudaFreeHost(hDone);
  cudaFreeHost(hCqBuf); cudaFree(dCqProd); cudaFreeHost(hCqReady);
  return res;
}

// ===========================================================================
// Run (B): host-orchestrated MTP. Each step the kernel writes the accept-length
// (out_nemit) + emitted tokens; the host d2h's the accept-length, BRANCHES
// (advance pos by n_emit, set hstate = last emitted token, retire finished slots,
// rebuild the active list), h2d's the advanced state, relaunches. The d2h of the
// accept-length control value + the host branch + the h2d is what G8 eliminates.
// ===========================================================================
static RunResult run_host_orch(const std::vector<ReqSpec>& w, int n_slots,
                               int stagger_every, int sms, size_t smem,
                               const float* dW, int K, unsigned int anum,
                               unsigned int aden, bool timed) {
  const int n_req = (int)w.size();
  unsigned int total_tokens = 0;
  for (const auto& r : w) total_tokens += (unsigned int)r.gen_len;

  g8::DecodeState st;
  CK(cudaMalloc(&st.occupied, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.req_id, g8::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.pos, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.gen_len, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.hstate, g8::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.step_ctr, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.svec, (size_t)g8::kMaxSlots * g8::kD * sizeof(float)));
  std::vector<float> hsv_init((size_t)g8::kMaxSlots * g8::kD);
  for (size_t i = 0; i < hsv_init.size(); ++i) hsv_init[i] = fill_val(i, 555);

  int *hActive, *dActive;
  alloc_mapped<int>(g8::kMaxSlots, &hActive, &dActive);
  unsigned long long *hTok, *dTok;
  alloc_mapped<unsigned long long>(g8::kMaxSlots * (g8::kMaxK + 1), &hTok, &dTok);
  int *hNemit, *dNemit, *hNacc, *dNacc;
  alloc_mapped<int>(g8::kMaxSlots, &hNemit, &dNemit);
  alloc_mapped<int>(g8::kMaxSlots, &hNacc, &dNacc);
  // pinned host mirrors of the per-slot state the host advances each step.
  int *hPos, *hOcc, *hGen, *hStepc; unsigned long long *hReqId, *hHstate;
  CK(cudaHostAlloc((void**)&hPos, g8::kMaxSlots*sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hOcc, g8::kMaxSlots*sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hGen, g8::kMaxSlots*sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hStepc, g8::kMaxSlots*sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hReqId, g8::kMaxSlots*sizeof(unsigned long long), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hHstate, g8::kMaxSlots*sizeof(unsigned long long), cudaHostAllocDefault));

  cudaStream_t stream; CK(cudaStreamCreate(&stream));
  int grid = std::min(sms * 4, g8::kMaxSlots);
  if (grid < 1) grid = 1;

  std::vector<g8::Completion> drained;
  auto run_once = [&](std::vector<g8::Completion>* capture) {
    drained.clear();
    CK(cudaMemcpyAsync(st.svec, hsv_init.data(), hsv_init.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    CK(cudaMemsetAsync(st.occupied, 0, g8::kMaxSlots*sizeof(int), stream));
    CK(cudaMemsetAsync(st.pos, 0, g8::kMaxSlots*sizeof(int), stream));
    CK(cudaStreamSynchronize(stream));
    for (int s = 0; s < g8::kMaxSlots; ++s) { hOcc[s]=0; hPos[s]=0; hGen[s]=0; hStepc[s]=0;
                                              hReqId[s]=0; hHstate[s]=0; }

    int next = 0;
    unsigned int emitted = 0;
    int step = 0;
    while (emitted < total_tokens) {
      // ---- HOST ADMISSION + METADATA REBUILD ----
      bool stagger_pause = (stagger_every > 0 && step > 0 && (step % stagger_every) == 0);
      if (!stagger_pause) {
        for (int s = 0; s < n_slots && next < n_req; ++s) {
          if (hOcc[s] == 0) {
            const ReqSpec& r = w[next++];
            hOcc[s] = 1; hReqId[s] = r.request_id; hPos[s] = 0; hGen[s] = r.gen_len;
            hHstate[s] = g8::seed_for_request(r.request_id); hStepc[s] = 0;
          }
        }
      }
      // push per-slot state h2d (metadata + advanced-state write).
      CK(cudaMemcpyAsync(st.occupied, hOcc, g8::kMaxSlots*sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.pos, hPos, g8::kMaxSlots*sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.gen_len, hGen, g8::kMaxSlots*sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.req_id, hReqId, g8::kMaxSlots*sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.hstate, hHstate, g8::kMaxSlots*sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
      // build the active list.
      int active_count = 0;
      for (int s = 0; s < n_slots; ++s)
        if (hOcc[s] && hPos[s] < hGen[s]) hActive[active_count++] = s;
      if (active_count == 0 && next >= n_req) break;
      CK(cudaMemcpyAsync(dActive, hActive, active_count*sizeof(int), cudaMemcpyHostToDevice, stream));

      // ---- RELAUNCH the MTP step kernel ----
      g8::StepParams sp{}; sp.st=st; sp.W=dW; sp.active_slots=dActive;
      sp.out_tok=dTok; sp.out_nemit=dNemit; sp.out_naccept=dNacc;
      sp.active_count=active_count; sp.K=K; sp.accept_num=anum; sp.accept_den=aden;
      int g = std::min(grid, active_count > 0 ? active_count : 1);
      g8::kHostMtpStep<<<g, g8::kBlockThreads, smem, stream>>>(sp);

      // ---- d2h the ACCEPT-LENGTH (control value) + emitted tokens ----
      CK(cudaMemcpyAsync(hNemit, dNemit, active_count*sizeof(int), cudaMemcpyDeviceToHost, stream));
      CK(cudaMemcpyAsync(hNacc, dNacc, active_count*sizeof(int), cudaMemcpyDeviceToHost, stream));
      CK(cudaMemcpyAsync(hTok, dTok, (size_t)active_count*(g8::kMaxK+1)*sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));   // host MUST see accept-length to branch

      // ---- HOST BRANCH on the accept-length: advance pos by n_emit, emit tokens,
      //      set hstate, retire finished slots, bump step counter. ----
      for (int i = 0; i < active_count; ++i) {
        int slot = hActive[i];
        int n_emit = hNemit[i];
        int n_acc = hNacc[i];
        int pos0 = hPos[slot];
        for (int j = 0; j < n_emit; ++j) {
          unsigned long long tok = hTok[(size_t)i*(g8::kMaxK+1)+j];
          g8::Completion c;
          c.request_id = hReqId[slot]; c.token = tok; c.slot = slot;
          c.pos = pos0 + j; c.accept_len = n_acc; c.step_idx = hStepc[slot];
          drained.push_back(c);
          emitted++;
        }
        // the data-dependent advance, done ON THE HOST after the d2h.
        hPos[slot] = pos0 + n_emit;
        if (n_emit > 0) hHstate[slot] = hTok[(size_t)i*(g8::kMaxK+1)+(n_emit-1)];
        hStepc[slot] += 1;
        if (hPos[slot] >= hGen[slot]) hOcc[slot] = 0;   // retire
      }
      step++;
      if (step > (int)total_tokens + n_req + 1024) break;
    }
    if (capture) *capture = drained;
  };

  RunResult res;
  run_once(&res.comps);
  if (timed) {
    const int REPS = 20, WARM = 3;
    for (int i = 0; i < WARM; ++i) run_once(nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPS; ++r) run_once(nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / REPS;
    res.wall_us = us; res.per_token_us = us / total_tokens;
  } else { res.wall_us = 0; res.per_token_us = 0; }

  cudaStreamDestroy(stream);
  cudaFree(st.occupied); cudaFree(st.req_id); cudaFree(st.pos); cudaFree(st.gen_len);
  cudaFree(st.hstate); cudaFree(st.step_ctr); cudaFree(st.svec);
  cudaFreeHost(hActive); cudaFreeHost(hTok); cudaFreeHost(hNemit); cudaFreeHost(hNacc);
  cudaFreeHost(hPos); cudaFreeHost(hOcc); cudaFreeHost(hGen); cudaFreeHost(hStepc);
  cudaFreeHost(hReqId); cudaFreeHost(hHstate);
  return res;
}

// ===========================================================================
// STEADY-STATE per-step benchmark: isolate the per-step MTP-control-round-trip
// win, free of the host arrival stagger. Fixed fully-active batch of B slots,
// each decoded gen_len = LONG so every run takes many MTP steps, no admission
// mid-run. Sweeps draft length K and batch B.
//
//   (A_ss) device engine, all B pre-enqueued + done pre-set: admit all B at step
//          0 then run the MTP loop device-side with ZERO host contact. Device-only
//          CUDA-event timed. per-step = wall / mean_steps (steps vary by accept).
//   (B_ss) host-orchestrated: the host loop runs the SAME MTP until all B finish,
//          d2h'ing the accept-length + branching + h2d'ing every step. Wall-clock
//          timed over the host loop. per-step = wall / mean_steps.
//
// Because the accept-length is data-dependent, #steps differs per request; we
// drive a fixed token budget (B requests x gen_len tokens) and divide wall time
// by the measured mean #steps so A and B are compared per-MTP-step. We report the
// mean accept-length + #steps so the per-step normalization is auditable.
// ===========================================================================
struct SteadyResult {
  double A_perstep_us; double B_perstep_us; double overhead_us;
  double A_pertok_us; double B_pertok_us;
  double mean_accept; double mean_steps; int total_tokens;
};

static SteadyResult run_steady_state(int B, int gen_len, int K, unsigned int anum,
                                     unsigned int aden, int sms, int occ,
                                     size_t smem, const float* dW) {
  std::vector<ReqSpec> w(B);
  for (int i = 0; i < B; ++i) { w[i].request_id = 0x900000ull + i; w[i].gen_len = gen_len; w[i].arrival = i; }
  const unsigned int total_tokens = (unsigned int)B * (unsigned int)gen_len;

  // measure #MTP steps + mean accept-length from the CPU reference (device runs
  // the identical loop, so #steps is deterministic + identical).
  auto ref = cpu_reference(w, K, anum, aden);
  long total_steps = 0, total_accept = 0, total_emit = 0;
  for (auto& kv : ref) {
    total_steps += (long)kv.second.step_accept.size();
    for (int a : kv.second.step_accept) total_accept += a;
    total_emit += (long)kv.second.toks.size();
  }
  double mean_steps_per_req = (double)total_steps / B;
  double mean_accept = total_steps ? (double)total_accept / total_steps : 0.0;

  g8::DecodeState st;
  CK(cudaMalloc(&st.occupied, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.req_id, g8::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.pos, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.gen_len, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.hstate, g8::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.step_ctr, g8::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.svec, (size_t)g8::kMaxSlots * g8::kD * sizeof(float)));
  std::vector<float> hsv((size_t)g8::kMaxSlots * g8::kD);
  for (size_t i = 0; i < hsv.size(); ++i) hsv[i] = fill_val(i, 555);

  cudaStream_t stream; CK(cudaStreamCreate(&stream));

  // ---- (A_ss) device engine, pre-admitted, device-only timed ----
  g8::RequestDesc *hRqBuf, *dRqBuf; unsigned int *hRqHead,*dRqHead,*hRqTail,*dRqTail; int *hDone,*dDone;
  const unsigned int rq_cap = 1u << 12;
  alloc_mapped<g8::RequestDesc>(rq_cap, &hRqBuf, &dRqBuf);
  alloc_mapped<unsigned int>(1, &hRqHead, &dRqHead);
  alloc_mapped<unsigned int>(1, &hRqTail, &dRqTail);
  alloc_mapped<int>(1, &hDone, &dDone);
  g8::Completion *hCqBuf, *dCqBuf; unsigned int *dCqProd,*hCqReady,*dCqReady;
  const unsigned int cq_cap = total_tokens + 16u;
  alloc_mapped<g8::Completion>(cq_cap, &hCqBuf, &dCqBuf);
  CK(cudaMalloc(&dCqProd, sizeof(unsigned int)));
  alloc_mapped<unsigned int>(cq_cap, &hCqReady, &dCqReady);
  int* dStop; CK(cudaMalloc(&dStop, 2 * sizeof(int)));

  g8::EngineParams p{};
  p.rq = {dRqBuf, dRqHead, dRqTail, dDone, rq_cap};
  p.cq = {dCqBuf, dCqProd, dCqReady, cq_cap};
  p.st = st; p.W = dW; p.total_requests = (unsigned int)B; p.total_tokens = total_tokens;
  p.K = K; p.accept_num = anum; p.accept_den = aden; p.engine_stop = dStop;
  int gridA = sms * occ;
  void* args[] = {&p};

  auto launchA = [&]() {
    for (int i = 0; i < B; ++i) { hRqBuf[i].request_id = w[i].request_id; hRqBuf[i].slot = i;
                                  hRqBuf[i].gen_len = gen_len; hRqBuf[i].pad = 0; }
    *hRqHead = 0; *hRqTail = (unsigned int)B; *hDone = 1;
    CK(cudaMemsetAsync(dCqProd, 0, sizeof(unsigned int), stream));
    for (unsigned int i = 0; i < cq_cap; ++i) hCqReady[i] = 0u;
    CK(cudaMemsetAsync(dStop, 0, 2 * sizeof(int), stream));
    CK(cudaMemsetAsync(st.occupied, 0, g8::kMaxSlots * sizeof(int), stream));
    CK(cudaMemsetAsync(st.pos, 0, g8::kMaxSlots * sizeof(int), stream));
    CK(cudaMemsetAsync(st.step_ctr, 0, g8::kMaxSlots * sizeof(int), stream));
    CK(cudaMemcpyAsync(st.svec, hsv.data(), hsv.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    CK(cudaLaunchCooperativeKernel((void*)g8::kDeviceMtpEngine, gridA,
                                   g8::kBlockThreads, args, smem, stream));
  };
  double A_us;
  {
    const int REPS = 30, WARM = 5;
    for (int i = 0; i < WARM; ++i) { launchA(); CK(cudaStreamSynchronize(stream)); }
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double acc = 0;
    for (int r = 0; r < REPS; ++r) {
      CK(cudaEventRecord(e0, stream));
      launchA();
      CK(cudaEventRecord(e1, stream)); CK(cudaEventSynchronize(e1));
      float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); acc += ms * 1000.0;
    }
    A_us = acc / REPS;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }

  // ---- (B_ss) host-orchestrated, host loop, accept-length d2h + branch each step ----
  int *hActive, *dActive; alloc_mapped<int>(g8::kMaxSlots, &hActive, &dActive);
  unsigned long long *hTok, *dTok; alloc_mapped<unsigned long long>(g8::kMaxSlots*(g8::kMaxK+1), &hTok, &dTok);
  int *hNemit, *dNemit, *hNacc, *dNacc;
  alloc_mapped<int>(g8::kMaxSlots, &hNemit, &dNemit);
  alloc_mapped<int>(g8::kMaxSlots, &hNacc, &dNacc);
  int *hPos,*hOcc,*hGen; unsigned long long *hReqId,*hHstate;
  CK(cudaHostAlloc((void**)&hPos, g8::kMaxSlots*sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hOcc, g8::kMaxSlots*sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hGen, g8::kMaxSlots*sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hReqId, g8::kMaxSlots*sizeof(unsigned long long), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hHstate, g8::kMaxSlots*sizeof(unsigned long long), cudaHostAllocDefault));
  int gridB = std::min(sms * 4, B);

  auto runB = [&]() {
    for (int s = 0; s < g8::kMaxSlots; ++s) { hOcc[s]=0; hPos[s]=0; hGen[s]=0; hReqId[s]=0; hHstate[s]=0; }
    for (int i = 0; i < B; ++i) { hOcc[i]=1; hReqId[i]=w[i].request_id; hPos[i]=0; hGen[i]=gen_len;
                                  hHstate[i]=g8::seed_for_request(w[i].request_id); }
    CK(cudaMemcpyAsync(st.svec, hsv.data(), hsv.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    unsigned int emitted = 0; int step = 0;
    while (emitted < total_tokens) {
      CK(cudaMemcpyAsync(st.occupied, hOcc, g8::kMaxSlots*sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.pos, hPos, g8::kMaxSlots*sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.gen_len, hGen, g8::kMaxSlots*sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.req_id, hReqId, g8::kMaxSlots*sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.hstate, hHstate, g8::kMaxSlots*sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
      int active_count = 0;
      for (int s = 0; s < B; ++s) if (hOcc[s] && hPos[s] < hGen[s]) hActive[active_count++] = s;
      if (active_count == 0) break;
      CK(cudaMemcpyAsync(dActive, hActive, active_count*sizeof(int), cudaMemcpyHostToDevice, stream));
      g8::StepParams sp{}; sp.st=st; sp.W=dW; sp.active_slots=dActive;
      sp.out_tok=dTok; sp.out_nemit=dNemit; sp.out_naccept=dNacc;
      sp.active_count=active_count; sp.K=K; sp.accept_num=anum; sp.accept_den=aden;
      int g = std::min(gridB, active_count > 0 ? active_count : 1);
      g8::kHostMtpStep<<<g, g8::kBlockThreads, smem, stream>>>(sp);
      CK(cudaMemcpyAsync(hNemit, dNemit, active_count*sizeof(int), cudaMemcpyDeviceToHost, stream));
      CK(cudaMemcpyAsync(hTok, dTok, (size_t)active_count*(g8::kMaxK+1)*sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));   // host must see accept-length to branch
      for (int i = 0; i < active_count; ++i) {
        int slot = hActive[i]; int n_emit = hNemit[i];
        hPos[slot] += n_emit;
        if (n_emit > 0) hHstate[slot] = hTok[(size_t)i*(g8::kMaxK+1)+(n_emit-1)];
        if (hPos[slot] >= hGen[slot]) hOcc[slot] = 0;
        emitted += (unsigned int)n_emit;
      }
      step++;
      if (step > (int)total_tokens + 1024) break;
    }
  };
  double B_us;
  {
    const int REPS = 30, WARM = 5;
    for (int i = 0; i < WARM; ++i) runB();
    double acc = 0;
    for (int r = 0; r < REPS; ++r) {
      auto t0 = std::chrono::high_resolution_clock::now();
      runB();
      auto t1 = std::chrono::high_resolution_clock::now();
      acc += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    B_us = acc / REPS;
  }

  SteadyResult r;
  // per-step normalization uses the mean #steps PER REQUEST (the engine runs all
  // B in parallel, so wall is dominated by the slowest request ~ max steps; we
  // report per-step against mean steps and also raw per-token).
  r.A_perstep_us = A_us / mean_steps_per_req;
  r.B_perstep_us = B_us / mean_steps_per_req;
  r.overhead_us = r.B_perstep_us - r.A_perstep_us;
  r.A_pertok_us = A_us / total_tokens;
  r.B_pertok_us = B_us / total_tokens;
  r.mean_accept = mean_accept;
  r.mean_steps = mean_steps_per_req;
  r.total_tokens = (int)total_tokens;

  cudaStreamDestroy(stream);
  cudaFree(st.occupied); cudaFree(st.req_id); cudaFree(st.pos); cudaFree(st.gen_len);
  cudaFree(st.hstate); cudaFree(st.step_ctr); cudaFree(st.svec); cudaFree(dStop);
  cudaFreeHost(hRqBuf); cudaFreeHost(hRqHead); cudaFreeHost(hRqTail); cudaFreeHost(hDone);
  cudaFreeHost(hCqBuf); cudaFree(dCqProd); cudaFreeHost(hCqReady);
  cudaFreeHost(hActive); cudaFreeHost(hTok); cudaFreeHost(hNemit); cudaFreeHost(hNacc);
  cudaFreeHost(hPos); cudaFreeHost(hOcc); cudaFreeHost(hGen); cudaFreeHost(hReqId); cudaFreeHost(hHstate);
  return r;
}

// accept-length histogram over a reference (the variable-n distribution proof).
static void accept_histogram(const std::map<unsigned long long, RefEntry>& ref,
                             int K, long hist[/*K+2*/], long* n_steps) {
  for (int i = 0; i <= K + 1; ++i) hist[i] = 0;
  long steps = 0;
  for (const auto& kv : ref)
    for (int a : kv.second.step_accept) { if (a >= 0 && a <= K + 1) hist[a]++; steps++; }
  *n_steps = steps;
}

int main() {
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  unsigned int flags = 0; cudaGetDeviceFlags(&flags);
  printf("device: %s  SMs=%d  cc=%d.%d  coopLaunch=%d  canMapHost=%d\n",
         prop.name, prop.multiProcessorCount, prop.major, prop.minor,
         prop.cooperativeLaunch, prop.canMapHostMemory);
  if (!prop.cooperativeLaunch) { printf("FATAL: no cooperative launch\n"); return 1; }
  if (!prop.canMapHostMemory)  { printf("FATAL: no host-mapped memory\n"); return 1; }
  const int sms = prop.multiProcessorCount;

  std::vector<float> hW((size_t)g8::kH * g8::kD);
  for (size_t i = 0; i < hW.size(); ++i) hW[i] = fill_val(i, 909) * 0.1f;
  float* dW; CK(cudaMalloc(&dW, hW.size() * sizeof(float)));
  CK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(float), cudaMemcpyHostToDevice));

  size_t eng_smem = g8::engine_smem_bytes();
  CK(cudaFuncSetAttribute((void*)g8::kDeviceMtpEngine,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)eng_smem));
  int occ = occ_blocks_per_sm((void*)g8::kDeviceMtpEngine, g8::kBlockThreads, eng_smem);
  if (occ < 1) occ = 1;
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, (void*)g8::kDeviceMtpEngine));
  printf("OCC: kDeviceMtpEngine %d blk/SM (%d resident CTAs) regs=%d smem=%.1fKB\n",
         occ, occ * sms, fa.numRegs, eng_smem / 1024.0);

  // acceptance rate p_acc = anum/aden ~ 0.7 (representative EAGLE-class accept).
  const unsigned int ANUM = 7, ADEN = 10;

  bool all_ok = true;

  // ---- correctness: time-staggered admission, variable active count, slot reuse,
  //      variable accept-length per step. K=4. ----
  struct Scen { int n_req; int n_slots; int stagger; int min_len; int max_len; int K; const char* name; };
  Scen scens[3] = {
    {16,  8,  4, 16, 48, 4, "N16_S8_stag4_K4"},
    {32,  16, 8, 24, 64, 4, "N32_S16_stag8_K4"},
    {8,   8,  0, 64, 96, 6, "N8_S8_nostag_K6"},
  };

  double A_pertok[3]={0}, B_pertok[3]={0}, speedup[3]={0};
  double A_wall[3]={0}, B_wall[3]={0};
  long A_total[3]={0}, B_total[3]={0};
  int corr_pass[3]={0};
  long hist0[64]; long nsteps0 = 0;   // accept histogram of scenario 0
  int hist0_K = scens[0].K;

  for (int s = 0; s < 3; ++s) {
    const Scen& sc = scens[s];
    printf("\n############ SCENARIO %s : N=%d slots=%d stagger=%d len[%d,%d] K=%d pacc=%u/%u ############\n",
           sc.name, sc.n_req, sc.n_slots, sc.stagger, sc.min_len, sc.max_len, sc.K, ANUM, ADEN);
    auto w = make_workload(sc.n_req, 1234 + s, sc.min_len, sc.max_len);
    unsigned int total_tokens = 0; for (auto& r : w) total_tokens += r.gen_len;
    auto ref = cpu_reference(w, sc.K, ANUM, ADEN);

    // accept-length distribution (the variable-n proof).
    long hist[64]; long nsteps = 0; accept_histogram(ref, sc.K, hist, &nsteps);
    long minA = 1<<30, maxA = -1; double sumA = 0;
    for (int a = 0; a <= sc.K + 1; ++a) if (hist[a] > 0) { if (a<minA)minA=a; if(a>maxA)maxA=a; sumA += (double)a*hist[a]; }
    printf("workload: %d requests, %u total tokens, %ld MTP steps; accept-len[min=%ld max=%ld mean=%.3f] hist=",
           sc.n_req, total_tokens, nsteps, minA, maxA, nsteps? sumA/nsteps : 0.0);
    for (int a = 0; a <= sc.K + 1; ++a) printf("%ld:%ld ", (long)a, hist[a]);
    printf("\n");
    bool variable_n = (maxA > minA);
    if (s == 0) { for (int a=0;a<64;a++) hist0[a]=hist[a]; nsteps0=nsteps; }

    RunResult A = run_engine(w, sc.n_slots, sc.stagger, sms, occ, eng_smem, dW, sc.K, ANUM, ADEN, true);
    RunResult B = run_host_orch(w, sc.n_slots, sc.stagger, sms, eng_smem, dW, sc.K, ANUM, ADEN, true);

    RingCheck cA = check_completions(A.comps, ref);
    RingCheck cB = check_completions(B.comps, ref);
    Recon rA = reconstruct(A.comps, ref);
    Recon rB = reconstruct(B.comps, ref);
    bool tok_AB = toks_bit_equal(rA.toks, rB.toks);
    bool tok_AC = toks_equal_ref(rA.toks, ref, false);
    bool tok_BC = toks_equal_ref(rB.toks, ref, false);
    bool acc_AB = stepacc_bit_equal(rA.step_accept, rB.step_accept);
    bool acc_AC = stepacc_equal_ref(rA.step_accept, ref);
    bool acc_BC = stepacc_equal_ref(rB.step_accept, ref);
    bool greedy_equiv = toks_equal_ref(rA.toks, ref, true);  // spec-decode == greedy
    bool ok = cA.ok && cB.ok && tok_AB && tok_AC && tok_BC &&
              acc_AB && acc_AC && acc_BC && greedy_equiv && variable_n;
    all_ok = all_ok && ok;
    corr_pass[s] = ok ? 1 : 0;

    printf("RING (A engine):   total=%ld drops=%ld dupes=%ld mismatch=%ld missing=%ld -> %s\n",
           cA.total, cA.drops, cA.dupes, cA.mismatch, cA.missing, cA.ok?"exactly-once":"BROKEN");
    printf("RING (B hostorch): total=%ld drops=%ld dupes=%ld mismatch=%ld missing=%ld -> %s\n",
           cB.total, cB.drops, cB.dupes, cB.mismatch, cB.missing, cB.ok?"exactly-once":"BROKEN");
    printf("BIT-EXACT tokens:  A==CPU=%d B==CPU=%d A==B=%d | greedy-equiv(A==pure-target)=%d\n",
           (int)tok_AC,(int)tok_BC,(int)tok_AB,(int)greedy_equiv);
    printf("ACCEPT-LEN seq:    A==CPU=%d B==CPU=%d A==B=%d | variable-n exercised=%d -> %s\n",
           (int)acc_AC,(int)acc_BC,(int)acc_AB,(int)variable_n, ok?"PASS":"FAIL");

    double sp = (A.per_token_us > 0) ? B.per_token_us / A.per_token_us : 0.0;
    printf("PERF (per-token us, lower=better):\n");
    printf("  (A) device MTP engine : %.4f us/tok (wall %.1f us, %u tok)\n", A.per_token_us, A.wall_us, total_tokens);
    printf("  (B) host-orchestrated : %.4f us/tok (wall %.1f us) -> A is %.2fx %s\n",
           B.per_token_us, B.wall_us, sp, (A.per_token_us<B.per_token_us)?"FASTER":"slower");
    printf("  per-step control round-trip eliminated (B-A): %.4f us/tok\n", B.per_token_us - A.per_token_us);

    A_pertok[s]=A.per_token_us; B_pertok[s]=B.per_token_us; speedup[s]=sp;
    A_wall[s]=A.wall_us; B_wall[s]=B.wall_us; A_total[s]=cA.total; B_total[s]=cB.total;
  }

  // ---- STEADY-STATE per-step sweep: per-step overhead win + K & batch scaling.
  //      Long gen_len so every run is many MTP steps; fully-active batch. ----
  printf("\n############ STEADY-STATE per-step (fixed batch, no admission mid-run, p_acc=%u/%u) ############\n", ANUM, ADEN);
  struct SS { int B; int gen; int K; const char* name; };
  SS sss[6] = {
    {1,   512, 4, "B1_g512_K4"},     // single-slot latency-bound
    {8,   512, 4, "B8_g512_K4"},
    {32,  512, 4, "B32_g512_K4"},
    {128, 512, 4, "B128_g512_K4"},   // full batch
    {32,  512, 1, "B32_g512_K1"},    // K=1 (no speculation: 1 token/step) — scaling vs K
    {32,  512, 8, "B32_g512_K8"},    // K=8 (deep MTP) — scaling vs K
  };
  double ssA[6]={0}, ssB[6]={0}, ssOv[6]={0}, ssSp[6]={0}, ssMacc[6]={0}, ssMstep[6]={0};
  double ssApt[6]={0}, ssBpt[6]={0};
  for (int i = 0; i < 6; ++i) {
    SteadyResult r = run_steady_state(sss[i].B, sss[i].gen, sss[i].K, ANUM, ADEN, sms, occ, eng_smem, dW);
    ssA[i]=r.A_perstep_us; ssB[i]=r.B_perstep_us; ssOv[i]=r.overhead_us;
    ssSp[i]=(r.A_perstep_us>0)?r.B_perstep_us/r.A_perstep_us:0.0;
    ssMacc[i]=r.mean_accept; ssMstep[i]=r.mean_steps; ssApt[i]=r.A_pertok_us; ssBpt[i]=r.B_pertok_us;
    printf("  %-13s : A %.4f us/step | B %.4f us/step | cut %.4f -> %.2fx | "
           "A %.4f us/tok B %.4f us/tok | mean_accept=%.2f steps/req=%.1f\n",
           sss[i].name, r.A_perstep_us, r.B_perstep_us, r.overhead_us, ssSp[i],
           r.A_pertok_us, r.B_pertok_us, r.mean_accept, r.mean_steps);
  }

  printf("\n=== G8 GATE: %s ===\n", all_ok ? "PASS(correctness)" : "FAIL(correctness)");
  printf("SUMMARY_JSON {\"sm\":%d,\"occ_engine\":%d,\"resident_ctas\":%d,\"regs\":%d,"
         "\"pacc_num\":%u,\"pacc_den\":%u,"
         "\"scen\":[\"%s\",\"%s\",\"%s\"],"
         "\"A_pertok_us\":[%.4f,%.4f,%.4f],\"B_pertok_us\":[%.4f,%.4f,%.4f],"
         "\"speedup\":[%.3f,%.3f,%.3f],\"overhead_cut_us\":[%.4f,%.4f,%.4f],"
         "\"A_wall_us\":[%.1f,%.1f,%.1f],\"B_wall_us\":[%.1f,%.1f,%.1f],"
         "\"A_comps\":[%ld,%ld,%ld],\"B_comps\":[%ld,%ld,%ld],"
         "\"corr_per_scen\":[%d,%d,%d],"
         "\"scen0_K\":%d,\"scen0_accept_hist\":[",
         sms, occ, occ*sms, fa.numRegs, ANUM, ADEN,
         scens[0].name, scens[1].name, scens[2].name,
         A_pertok[0],A_pertok[1],A_pertok[2], B_pertok[0],B_pertok[1],B_pertok[2],
         speedup[0],speedup[1],speedup[2],
         B_pertok[0]-A_pertok[0],B_pertok[1]-A_pertok[1],B_pertok[2]-A_pertok[2],
         A_wall[0],A_wall[1],A_wall[2], B_wall[0],B_wall[1],B_wall[2],
         A_total[0],A_total[1],A_total[2], B_total[0],B_total[1],B_total[2],
         corr_pass[0],corr_pass[1],corr_pass[2], hist0_K);
  for (int a = 0; a <= hist0_K + 1; ++a) printf("%ld%s", hist0[a], (a==hist0_K+1)?"":",");
  printf("],\"scen0_steps\":%ld,"
         "\"ss_name\":[\"B1_g512_K4\",\"B8_g512_K4\",\"B32_g512_K4\",\"B128_g512_K4\",\"B32_g512_K1\",\"B32_g512_K8\"],"
         "\"ss_A_perstep_us\":[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f],"
         "\"ss_B_perstep_us\":[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f],"
         "\"ss_overhead_cut_us\":[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f],"
         "\"ss_speedup\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f],"
         "\"ss_A_pertok_us\":[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f],"
         "\"ss_B_pertok_us\":[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f],"
         "\"ss_mean_accept\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f],"
         "\"ss_steps_per_req\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f],"
         "\"corr_gate\":\"%s\"}\n",
         nsteps0,
         ssA[0],ssA[1],ssA[2],ssA[3],ssA[4],ssA[5],
         ssB[0],ssB[1],ssB[2],ssB[3],ssB[4],ssB[5],
         ssOv[0],ssOv[1],ssOv[2],ssOv[3],ssOv[4],ssOv[5],
         ssSp[0],ssSp[1],ssSp[2],ssSp[3],ssSp[4],ssSp[5],
         ssApt[0],ssApt[1],ssApt[2],ssApt[3],ssApt[4],ssApt[5],
         ssBpt[0],ssBpt[1],ssBpt[2],ssBpt[3],ssBpt[4],ssBpt[5],
         ssMacc[0],ssMacc[1],ssMacc[2],ssMacc[3],ssMacc[4],ssMacc[5],
         ssMstep[0],ssMstep[1],ssMstep[2],ssMstep[3],ssMstep[4],ssMstep[5],
         all_ok ? "PASS" : "FAIL");

  cudaFree(dW);
  return all_ok ? 0 : 2;
}
