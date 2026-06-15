// PDE G9 gate microbench (standalone, sm_100). REAL numbers on GPU0.
//
// G9 = CROSS-STEP PERSISTENCE: ONE persistent cooperative launch runs the decode
// loop DEVICE-SIDE across steps (polls a device-visible request ring, advances a
// device-resident decode state, emits to a device->host completion ring); the
// host only enqueues requests over time + drains completions. Compared to:
//
//   (A) PERSISTENT CROSS-STEP ENGINE : one cudaLaunchCooperativeKernel; a host
//        producer thread enqueues requests into the host-mapped request ring on
//        a stagger, a host consumer drains the completion ring. NO per-step
//        relaunch, NO host metadata rebuild.
//   (B) PER-STEP-RELAUNCH BASELINE   : the SAME synthetic decode step, but the
//        host loop relaunches the step kernel every step, rebuilds the active
//        slot list on the host each step, and reads per-slot state back across
//        the launch boundary (today's pattern).
//
// Both checked against an INDEPENDENT CPU reference: the emitted token stream is a
// deterministic integer hash-chain keyed on (request_id, token_pos). The HARD
// gate: A's per-(request,pos) token == B's == CPU's, BIT-EXACT, AND the request
// ring + completion ring deliver every request and every completion EXACTLY ONCE
// (no drops, no dupes) under a time-staggered admission pattern.
//
// Build: nvcc -std=c++17 -arch=sm_100 -O3 -o pde_g9_bench pde_g9_bench.cu
#include "pde_g9_persist.cuh"

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

// ---------------------------------------------------------------------------
// An admission workload: N requests, each with a request_id, a slot it will
// occupy, a gen_len, and an arrival "tick" (logical order of host enqueue). We
// stagger arrivals so the active count varies over the run (continuous batching).
// Slots are reused as requests finish: we keep <= kMaxSlots in flight by assigning
// each request a slot from a free-list as it is admitted (host-side admission
// policy; identical for A's host producer and B's host loop and the CPU ref).
// ---------------------------------------------------------------------------
struct ReqSpec {
  unsigned long long request_id;
  int gen_len;
  int arrival;     // logical arrival index (smaller = earlier)
};

// Build a deterministic workload.
static std::vector<ReqSpec> make_workload(int n_req, int seed, int min_len,
                                          int max_len) {
  std::vector<ReqSpec> w(n_req);
  for (int i = 0; i < n_req; ++i) {
    uint64_t h = (uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull + (uint64_t)seed;
    h ^= h >> 31;
    int len = min_len + (int)(h % (uint64_t)(max_len - min_len + 1));
    w[i].request_id = 0x1000ull + (unsigned long long)i;  // unique ids
    w[i].gen_len = len;
    w[i].arrival = i;   // arrive in id order; stagger applied at enqueue time
  }
  return w;
}

// ---------------------------------------------------------------------------
// CPU reference: for each request, the exact token stream is
//   s0 = seed_for_request(request_id); token(pos) = (s_{pos+1}) where
//   s_{p+1} = advance_token(s_p, p). We store per (request_id) the vector of
//   gen_len tokens. Ground truth — NEVER a device self-compare.
// ---------------------------------------------------------------------------
static std::map<unsigned long long, std::vector<unsigned long long>>
cpu_reference(const std::vector<ReqSpec>& w) {
  std::map<unsigned long long, std::vector<unsigned long long>> ref;
  for (const auto& r : w) {
    std::vector<unsigned long long> toks(r.gen_len);
    unsigned long long s = g9::seed_for_request(r.request_id);
    for (int pos = 0; pos < r.gen_len; ++pos) {
      unsigned long long tok = g9::advance_token(s, pos);
      toks[pos] = tok;
      s = tok;
    }
    ref[r.request_id] = std::move(toks);
  }
  return ref;
}

// Verify a drained set of completions delivers EXACTLY the reference tokens, each
// exactly once. Returns {bit_exact, total_completions, drops, dupes, mism}.
struct RingCheck { bool ok; long total; long drops; long dupes; long mismatch; long missing; };
static RingCheck check_completions(
    const std::vector<g9::Completion>& got,
    const std::map<unsigned long long, std::vector<unsigned long long>>& ref) {
  RingCheck c{true, (long)got.size(), 0, 0, 0, 0};
  // seen[request_id][pos] count
  std::map<unsigned long long, std::vector<int>> seen;
  long expected_total = 0;
  for (const auto& kv : ref) { seen[kv.first].assign(kv.second.size(), 0);
                               expected_total += (long)kv.second.size(); }
  for (const auto& comp : got) {
    auto it = ref.find(comp.request_id);
    if (it == ref.end()) { c.mismatch++; c.ok = false; continue; }
    if (comp.pos < 0 || comp.pos >= (int)it->second.size()) { c.mismatch++; c.ok = false; continue; }
    seen[comp.request_id][comp.pos]++;
    if (seen[comp.request_id][comp.pos] > 1) { c.dupes++; c.ok = false; }
    if (comp.token != it->second[comp.pos]) { c.mismatch++; c.ok = false; }
  }
  for (const auto& kv : seen)
    for (int v : kv.second) if (v == 0) { c.missing++; c.ok = false; }
  if (c.total != expected_total) { c.drops = expected_total - c.total; }
  if (c.drops != 0) c.ok = false;
  return c;
}

// Compare two token maps for bit-exactness (A vs B device streams).
static bool maps_bit_equal(
    const std::map<unsigned long long, std::vector<unsigned long long>>& a,
    const std::map<unsigned long long, std::vector<unsigned long long>>& b) {
  if (a.size() != b.size()) return false;
  for (const auto& kv : a) {
    auto it = b.find(kv.first);
    if (it == b.end()) return false;
    if (it->second.size() != kv.second.size()) return false;
    for (size_t i = 0; i < kv.second.size(); ++i)
      if (kv.second[i] != it->second[i]) return false;
  }
  return true;
}

// Turn a flat completion vector into a per-request token map (last-writer for a
// pos is irrelevant since we separately assert no dupes).
static std::map<unsigned long long, std::vector<unsigned long long>>
completions_to_map(const std::vector<g9::Completion>& got,
                   const std::map<unsigned long long, std::vector<unsigned long long>>& ref) {
  std::map<unsigned long long, std::vector<unsigned long long>> m;
  for (const auto& kv : ref) m[kv.first].assign(kv.second.size(), 0ull);
  for (const auto& comp : got) {
    auto it = m.find(comp.request_id);
    if (it == m.end()) continue;
    if (comp.pos >= 0 && comp.pos < (int)it->second.size())
      it->second[comp.pos] = comp.token;
  }
  return m;
}

static int occ_blocks_per_sm(const void* kernel, int bt, size_t smem) {
  int b = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, kernel, bt, smem));
  return b;
}

// host-mapped allocation helper: returns (host_ptr, device_ptr).
template <typename T>
static void alloc_mapped(int count, T** hptr, T** dptr) {
  CK(cudaHostAlloc((void**)hptr, (size_t)count * sizeof(T), cudaHostAllocMapped));
  CK(cudaHostGetDevicePointer((void**)dptr, (void*)*hptr, 0));
}

// ===========================================================================
// Run (A): persistent cross-step engine. A host PRODUCER thread enqueues the
// workload into the request ring on a stagger; the MAIN thread launches the
// engine and a host CONSUMER drains the completion ring until total_tokens seen.
// Returns the drained completions + the measured wall time + per-step time.
// ===========================================================================
struct RunResult {
  std::vector<g9::Completion> comps;
  double wall_us;          // total device-busy wall time
  double per_token_us;     // wall / total_tokens
  long steps_or_relaunches;
};

static RunResult run_engine(const std::vector<ReqSpec>& w, int n_slots,
                            int stagger_every, int sms, int occ, size_t smem,
                            const float* dW, bool timed) {
  const unsigned int n_req = (unsigned int)w.size();
  unsigned int total_tokens = 0;
  for (const auto& r : w) total_tokens += (unsigned int)r.gen_len;

  // ---- host-mapped rings ----
  g9::RequestDesc *hRqBuf, *dRqBuf;
  unsigned int *hRqHead, *dRqHead, *hRqTail, *dRqTail;
  int *hDone, *dDone;
  const unsigned int rq_cap = 1u << 12;   // 4096 (> n_req; no wrap in bench)
  alloc_mapped<g9::RequestDesc>(rq_cap, &hRqBuf, &dRqBuf);
  alloc_mapped<unsigned int>(1, &hRqHead, &dRqHead);
  alloc_mapped<unsigned int>(1, &hRqTail, &dRqTail);
  alloc_mapped<int>(1, &hDone, &dDone);
  *hRqHead = 0; *hRqTail = 0; *hDone = 0;

  g9::Completion *hCqBuf, *dCqBuf;
  unsigned int *dCqProd, *hCqReady, *dCqReady;   // prod is DEVICE-only now
  const unsigned int cq_cap = total_tokens + 16u;   // sized to never wrap
  alloc_mapped<g9::Completion>(cq_cap, &hCqBuf, &dCqBuf);
  CK(cudaMalloc(&dCqProd, sizeof(unsigned int)));
  alloc_mapped<unsigned int>(cq_cap, &hCqReady, &dCqReady);
  for (unsigned int i = 0; i < cq_cap; ++i) hCqReady[i] = 0u;

  // ---- device-resident decode state ----
  g9::DecodeState st;
  CK(cudaMalloc(&st.occupied, g9::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.req_id, g9::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.pos, g9::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.gen_len, g9::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.hstate, g9::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.svec, (size_t)g9::kMaxSlots * g9::kD * sizeof(float)));
  CK(cudaMemset(st.occupied, 0, g9::kMaxSlots * sizeof(int)));
  CK(cudaMemset(st.pos, 0, g9::kMaxSlots * sizeof(int)));
  // init svec deterministically (so the float-state evolution is reproducible).
  {
    std::vector<float> hsv((size_t)g9::kMaxSlots * g9::kD);
    for (size_t i = 0; i < hsv.size(); ++i) hsv[i] = fill_val(i, 555);
    CK(cudaMemcpy(st.svec, hsv.data(), hsv.size() * sizeof(float),
                  cudaMemcpyHostToDevice));
  }

  int* dStop;   // [2]: [0]=stop flag, [1]=tokens_emitted progress
  CK(cudaMalloc(&dStop, 2 * sizeof(int)));
  CK(cudaMemset(dStop, 0, 2 * sizeof(int)));

  g9::EngineParams p{};
  p.rq = {dRqBuf, dRqHead, dRqTail, dDone, rq_cap};
  p.cq = {dCqBuf, dCqProd, dCqReady, cq_cap};
  p.st = st;
  p.W = dW;
  p.total_requests = n_req;
  p.total_tokens = total_tokens;
  p.engine_stop = dStop;

  // free-list slot assignment (host admission policy): assign slot i%n_slots in
  // arrival order, BUT only reuse a slot once its prior request has been fully
  // drained. We keep it simple + deterministic: since we drain completions, we
  // track per-slot completion counts and only enqueue a request to a slot when
  // that slot has no in-flight request. The producer below enforces this.
  cudaStream_t stream; CK(cudaStreamCreate(&stream));

  // grid: occupancy-capped, but cooperative launch requires grid <= resident.
  int grid = sms * occ;
  // a modest grid is plenty (kMaxSlots=256 slots); cap to resident.
  void* args[] = {&p};

  // shared device-visible drain bookkeeping for the producer's slot reuse:
  std::atomic<int> slot_inflight[g9::kMaxSlots];
  for (int i = 0; i < g9::kMaxSlots; ++i) slot_inflight[i].store(0);

  auto launch = [&]() {
    *hRqHead = 0; *hRqTail = 0; *hDone = 0;
    CK(cudaMemsetAsync(dCqProd, 0, sizeof(unsigned int), stream));
    for (unsigned int i = 0; i < cq_cap; ++i) hCqReady[i] = 0u;
    CK(cudaMemsetAsync(dStop, 0, 2 * sizeof(int), stream));
    CK(cudaMemsetAsync(st.occupied, 0, g9::kMaxSlots * sizeof(int), stream));
    CK(cudaMemsetAsync(st.pos, 0, g9::kMaxSlots * sizeof(int), stream));
    CK(cudaStreamSynchronize(stream));
    CK(cudaLaunchCooperativeKernel((void*)g9::kPersistentEngine, grid,
                                   g9::kBlockThreads, args, smem, stream));
  };

  std::vector<g9::Completion> drained;

  auto run_once = [&](std::vector<g9::Completion>* capture) {
    drained.clear();
    for (int i = 0; i < g9::kMaxSlots; ++i) slot_inflight[i].store(0);
    launch();

    // ---- host PRODUCER thread: enqueue requests on a stagger, reusing slots ----
    std::atomic<bool> prod_done{false};
    std::thread producer([&]() {
      unsigned int next = 0;     // next request index to enqueue
      // simple slot pool: a slot is free if slot_inflight==0.
      while (next < n_req) {
        // find a free slot
        int slot = -1;
        for (int s = 0; s < n_slots; ++s)
          if (slot_inflight[s].load(std::memory_order_acquire) == 0) { slot = s; break; }
        if (slot < 0) { std::this_thread::yield(); continue; }
        const ReqSpec& r = w[next];
        slot_inflight[slot].store(1, std::memory_order_release);
        // write the descriptor into the ring then publish (release the tail).
        unsigned int tail = *hRqTail;          // single host producer
        g9::RequestDesc rd; rd.request_id = r.request_id; rd.slot = slot;
        rd.gen_len = r.gen_len; rd.pad = 0;
        hRqBuf[tail % rq_cap] = rd;
        std::atomic_thread_fence(std::memory_order_release);
        // publish tail with a system-scope release store (host side).
        reinterpret_cast<std::atomic<unsigned int>*>(hRqTail)
            ->store(tail + 1u, std::memory_order_release);
        next++;
        // stagger: every `stagger_every` enqueues, briefly back off so the active
        // count varies (continuous-batching admission over time).
        if (stagger_every > 0 && (next % stagger_every) == 0)
          std::this_thread::sleep_for(std::chrono::microseconds(20));
      }
      // signal no more requests.
      std::atomic_thread_fence(std::memory_order_release);
      reinterpret_cast<std::atomic<int>*>(hDone)->store(1, std::memory_order_release);
      prod_done.store(true);
    });

    // ---- host CONSUMER (main thread): drain the completion ring until all tokens ----
    unsigned int drained_cnt = 0;
    // also maintain per-slot remaining to free slots for reuse as requests finish.
    std::vector<int> slot_remaining(g9::kMaxSlots, 0);
    std::map<unsigned long long, int> reqlen;
    for (const auto& r : w) reqlen[r.request_id] = r.gen_len;
    std::map<int, int> slot_emitted;   // slot -> #tokens emitted so far (per current req)
    while (drained_cnt < total_tokens) {
      // drain the dense ready-prefix: while the slot at the cursor is stamped
      // ready, consume it (acquire). No shared publish counter to spin on.
      unsigned int slot_pos = drained_cnt % cq_cap;
      unsigned int rdy = reinterpret_cast<std::atomic<unsigned int>*>(&hCqReady[slot_pos])
                             ->load(std::memory_order_acquire);
      if (rdy != 0u) {
        std::atomic_thread_fence(std::memory_order_acquire);
        g9::Completion c = hCqBuf[slot_pos];
        drained.push_back(c);
        // free the slot when this request's last token is drained.
        slot_emitted[c.slot]++;
        if (slot_emitted[c.slot] >= reqlen[c.request_id]) {
          slot_emitted[c.slot] = 0;
          slot_inflight[c.slot].store(0, std::memory_order_release);  // reusable
        }
        drained_cnt++;
      } else {
        std::this_thread::yield();
      }
    }
    producer.join();
    CK(cudaStreamSynchronize(stream));   // engine should have set its stop flag
    if (capture) *capture = drained;
  };

  RunResult res;
  // functional run (captures completions for correctness)
  run_once(&res.comps);

  if (timed) {
    const int REPS = 20, WARM = 3;
    for (int i = 0; i < WARM; ++i) run_once(nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPS; ++r) run_once(nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / REPS;
    res.wall_us = us;
    res.per_token_us = us / total_tokens;
  } else {
    res.wall_us = 0; res.per_token_us = 0;
  }
  res.steps_or_relaunches = 0;

  cudaStreamDestroy(stream);
  cudaFree(st.occupied); cudaFree(st.req_id); cudaFree(st.pos); cudaFree(st.gen_len);
  cudaFree(st.hstate); cudaFree(st.svec); cudaFree(dStop);
  cudaFreeHost(hRqBuf); cudaFreeHost(hRqHead); cudaFreeHost(hRqTail); cudaFreeHost(hDone);
  cudaFreeHost(hCqBuf); cudaFree(dCqProd); cudaFreeHost(hCqReady);
  return res;
}

// ===========================================================================
// Run (B): per-step-relaunch baseline. The host loop, every step: rebuilds the
// active-slot list (admitting newly-arrived requests into free slots, dropping
// finished ones), copies it h2d, relaunches the step kernel, copies the step's
// emissions + advanced state d2h, and advances its host-side bookkeeping. This is
// the per-step launch + metadata rebuild + sync the engine eliminates.
// ===========================================================================
static RunResult run_relaunch(const std::vector<ReqSpec>& w, int n_slots,
                              int stagger_every, int sms, size_t smem,
                              const float* dW, bool timed) {
  const int n_req = (int)w.size();
  unsigned int total_tokens = 0;
  for (const auto& r : w) total_tokens += (unsigned int)r.gen_len;

  g9::DecodeState st;
  CK(cudaMalloc(&st.occupied, g9::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.req_id, g9::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.pos, g9::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.gen_len, g9::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.hstate, g9::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.svec, (size_t)g9::kMaxSlots * g9::kD * sizeof(float)));
  std::vector<float> hsv_init((size_t)g9::kMaxSlots * g9::kD);
  for (size_t i = 0; i < hsv_init.size(); ++i) hsv_init[i] = fill_val(i, 555);

  // host-mapped active list + per-step emission buffer (the h2d/d2h each step).
  int *hActive, *dActive;
  alloc_mapped<int>(g9::kMaxSlots, &hActive, &dActive);
  g9::Completion *hStepOut, *dStepOut;
  alloc_mapped<g9::Completion>(g9::kMaxSlots, &hStepOut, &dStepOut);
  // pinned mirrors of the per-slot state the host rebuilds metadata from.
  int *hPos, *hOcc, *hGen; unsigned long long *hReqId, *hHstate;
  CK(cudaHostAlloc((void**)&hPos, g9::kMaxSlots * sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hOcc, g9::kMaxSlots * sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hGen, g9::kMaxSlots * sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hReqId, g9::kMaxSlots * sizeof(unsigned long long), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hHstate, g9::kMaxSlots * sizeof(unsigned long long), cudaHostAllocDefault));

  cudaStream_t stream; CK(cudaStreamCreate(&stream));
  int grid = std::min(sms * 4, g9::kMaxSlots);
  if (grid < 1) grid = 1;

  std::vector<g9::Completion> drained;

  auto run_once = [&](std::vector<g9::Completion>* capture) {
    drained.clear();
    // reset device + host state
    CK(cudaMemcpyAsync(st.svec, hsv_init.data(), hsv_init.size() * sizeof(float),
                       cudaMemcpyHostToDevice, stream));
    CK(cudaMemsetAsync(st.occupied, 0, g9::kMaxSlots * sizeof(int), stream));
    CK(cudaMemsetAsync(st.pos, 0, g9::kMaxSlots * sizeof(int), stream));
    CK(cudaStreamSynchronize(stream));
    for (int s = 0; s < g9::kMaxSlots; ++s) { hOcc[s] = 0; hPos[s] = 0; hGen[s] = 0;
                                              hReqId[s] = 0; hHstate[s] = 0; }

    // host-side admission bookkeeping (mirrors run_engine's policy).
    int next = 0;                 // next request to admit
    std::vector<int> slot_remaining(g9::kMaxSlots, 0);
    unsigned int emitted = 0;
    int step = 0;
    // arrival pacing: admit up to `arrival_budget` new requests per step early,
    // staggered the same way as the engine producer (so the active-count profile
    // matches). With stagger_every>0 we admit 1 new request every step until a
    // stagger boundary then pause one step.
    while (emitted < total_tokens) {
      // ---- HOST METADATA REBUILD: admit arrivals into free slots ----
      bool stagger_pause = (stagger_every > 0 && step > 0 && (step % stagger_every) == 0);
      if (!stagger_pause) {
        for (int s = 0; s < n_slots && next < n_req; ++s) {
          if (hOcc[s] == 0) {
            const ReqSpec& r = w[next++];
            hOcc[s] = 1; hReqId[s] = r.request_id; hPos[s] = 0; hGen[s] = r.gen_len;
            hHstate[s] = g9::seed_for_request(r.request_id);
          }
        }
      }
      // push the (possibly updated) per-slot state h2d (metadata + state write).
      CK(cudaMemcpyAsync(st.occupied, hOcc, g9::kMaxSlots * sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.pos, hPos, g9::kMaxSlots * sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.gen_len, hGen, g9::kMaxSlots * sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.req_id, hReqId, g9::kMaxSlots * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.hstate, hHstate, g9::kMaxSlots * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
      // ---- build the active list (host metadata) ----
      int active_count = 0;
      for (int s = 0; s < n_slots; ++s)
        if (hOcc[s] && hPos[s] < hGen[s]) hActive[active_count++] = s;
      if (active_count == 0 && next >= n_req) break;   // nothing to do, all done
      CK(cudaMemcpyAsync(dActive, hActive, active_count * sizeof(int), cudaMemcpyHostToDevice, stream));

      // ---- RELAUNCH the step kernel ----
      g9::StepParams sp{}; sp.st = st; sp.W = dW; sp.active_slots = dActive;
      sp.out_step = dStepOut; sp.active_count = active_count;
      int g = std::min(grid, active_count > 0 ? active_count : 1);
      g9::kRelaunchStep<<<g, g9::kBlockThreads, smem, stream>>>(sp);

      // ---- read step emissions + advanced state back (the d2h round-trip) ----
      CK(cudaMemcpyAsync(hStepOut, dStepOut, active_count * sizeof(g9::Completion), cudaMemcpyDeviceToHost, stream));
      CK(cudaMemcpyAsync(hPos, st.pos, g9::kMaxSlots * sizeof(int), cudaMemcpyDeviceToHost, stream));
      CK(cudaMemcpyAsync(hHstate, st.hstate, g9::kMaxSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
      CK(cudaMemcpyAsync(hOcc, st.occupied, g9::kMaxSlots * sizeof(int), cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));

      // ---- drain this step's completions + free finished slots (host) ----
      for (int i = 0; i < active_count; ++i) {
        drained.push_back(hStepOut[i]);
        emitted++;
      }
      step++;
      if (step > (int)total_tokens + n_req + 1024) break;  // safety
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
    res.wall_us = us;
    res.per_token_us = us / total_tokens;
  } else { res.wall_us = 0; res.per_token_us = 0; }
  res.steps_or_relaunches = 0;

  cudaStreamDestroy(stream);
  cudaFree(st.occupied); cudaFree(st.req_id); cudaFree(st.pos); cudaFree(st.gen_len);
  cudaFree(st.hstate); cudaFree(st.svec);
  cudaFreeHost(hActive); cudaFreeHost(hStepOut);
  cudaFreeHost(hPos); cudaFreeHost(hOcc); cudaFreeHost(hGen); cudaFreeHost(hReqId); cudaFreeHost(hHstate);
  return res;
}

// ===========================================================================
// STEADY-STATE per-step benchmark (the CLEAN per-step-overhead isolation).
//
// The staggered scenarios above prove correctness + realistic admission, but
// their wall time mixes in the host-paced arrival stagger (the engine spins
// waiting for the host to enqueue), which contaminates the A-vs-B per-step
// comparison. Here we isolate the actual G9 win — eliminating the per-step
// kernel-launch + host metadata rebuild + state sync — at a FIXED fully-active
// batch of B slots decoded for exactly S steps, no admission mid-run:
//
//   (A_ss) persistent engine, all B requests pre-enqueued + done_flag pre-set:
//          the engine admits all B at step 0 then loops S steps with ZERO host
//          interaction. Timed device-only with CUDA events. per-step = wall/S.
//   (B_ss) S host relaunches of the step kernel over the fixed active list, with
//          the per-step host metadata rebuild (active list) + state d2h/h2d each
//          step (today's pattern). Timed with CUDA events over the host loop.
//
// Bit-exactness is already proven by the staggered scenarios (same kernels); here
// we report per-step time + the eliminated overhead + its scaling in (B, S).
// ===========================================================================
struct SteadyResult { double A_perstep_us; double B_perstep_us; double overhead_us; };

static SteadyResult run_steady_state(int B, int S, int sms, int occ,
                                     size_t smem, const float* dW) {
  // build a fixed batch: B requests, each gen_len = S, slots [0,B).
  std::vector<ReqSpec> w(B);
  for (int i = 0; i < B; ++i) { w[i].request_id = 0x900000ull + i; w[i].gen_len = S; w[i].arrival = i; }
  const unsigned int total_tokens = (unsigned int)B * (unsigned int)S;

  // ---- shared device-resident state ----
  g9::DecodeState st;
  CK(cudaMalloc(&st.occupied, g9::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.req_id, g9::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.pos, g9::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.gen_len, g9::kMaxSlots * sizeof(int)));
  CK(cudaMalloc(&st.hstate, g9::kMaxSlots * sizeof(unsigned long long)));
  CK(cudaMalloc(&st.svec, (size_t)g9::kMaxSlots * g9::kD * sizeof(float)));
  std::vector<float> hsv((size_t)g9::kMaxSlots * g9::kD);
  for (size_t i = 0; i < hsv.size(); ++i) hsv[i] = fill_val(i, 555);

  cudaStream_t stream; CK(cudaStreamCreate(&stream));

  // ============ (A_ss) persistent engine, pre-admitted, device-only timed ============
  g9::RequestDesc *hRqBuf, *dRqBuf; unsigned int *hRqHead,*dRqHead,*hRqTail,*dRqTail; int *hDone,*dDone;
  const unsigned int rq_cap = 1u << 12;
  alloc_mapped<g9::RequestDesc>(rq_cap, &hRqBuf, &dRqBuf);
  alloc_mapped<unsigned int>(1, &hRqHead, &dRqHead);
  alloc_mapped<unsigned int>(1, &hRqTail, &dRqTail);
  alloc_mapped<int>(1, &hDone, &dDone);
  g9::Completion *hCqBuf, *dCqBuf; unsigned int *dCqProd,*hCqReady,*dCqReady;
  const unsigned int cq_cap = total_tokens + 16u;
  alloc_mapped<g9::Completion>(cq_cap, &hCqBuf, &dCqBuf);
  CK(cudaMalloc(&dCqProd, sizeof(unsigned int)));
  alloc_mapped<unsigned int>(cq_cap, &hCqReady, &dCqReady);
  int* dStop; CK(cudaMalloc(&dStop, 2 * sizeof(int)));

  g9::EngineParams p{};
  p.rq = {dRqBuf, dRqHead, dRqTail, dDone, rq_cap};
  p.cq = {dCqBuf, dCqProd, dCqReady, cq_cap};
  p.st = st; p.W = dW; p.total_requests = (unsigned int)B; p.total_tokens = total_tokens;
  p.engine_stop = dStop;
  int gridA = sms * occ;
  void* args[] = {&p};

  auto launchA = [&]() {
    // pre-fill the request ring with all B requests + set done up front, so the
    // engine admits all B at step 0 then decodes S steps with no host contact.
    for (int i = 0; i < B; ++i) { hRqBuf[i].request_id = w[i].request_id; hRqBuf[i].slot = i;
                                  hRqBuf[i].gen_len = S; hRqBuf[i].pad = 0; }
    *hRqHead = 0; *hRqTail = (unsigned int)B; *hDone = 1;
    CK(cudaMemsetAsync(dCqProd, 0, sizeof(unsigned int), stream));
    for (unsigned int i = 0; i < cq_cap; ++i) hCqReady[i] = 0u;
    CK(cudaMemsetAsync(dStop, 0, 2 * sizeof(int), stream));
    CK(cudaMemsetAsync(st.occupied, 0, g9::kMaxSlots * sizeof(int), stream));
    CK(cudaMemsetAsync(st.pos, 0, g9::kMaxSlots * sizeof(int), stream));
    CK(cudaMemcpyAsync(st.svec, hsv.data(), hsv.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    CK(cudaLaunchCooperativeKernel((void*)g9::kPersistentEngine, gridA,
                                   g9::kBlockThreads, args, smem, stream));
  };
  double A_us;
  {
    const int REPS = 30, WARM = 5;
    for (int i = 0; i < WARM; ++i) { launchA(); CK(cudaStreamSynchronize(stream)); }
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    // event-bracket EACH launch+sync (the engine is one launch that runs S steps).
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

  // ============ (B_ss) per-step relaunch, S launches, host metadata each step ============
  int *hActive, *dActive; alloc_mapped<int>(g9::kMaxSlots, &hActive, &dActive);
  g9::Completion *hStepOut, *dStepOut; alloc_mapped<g9::Completion>(g9::kMaxSlots, &hStepOut, &dStepOut);
  int *hPos,*hOcc,*hGen; unsigned long long *hReqId,*hHstate;
  CK(cudaHostAlloc((void**)&hPos, g9::kMaxSlots*sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hOcc, g9::kMaxSlots*sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hGen, g9::kMaxSlots*sizeof(int), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hReqId, g9::kMaxSlots*sizeof(unsigned long long), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&hHstate, g9::kMaxSlots*sizeof(unsigned long long), cudaHostAllocDefault));
  int gridB = std::min(sms * 4, B);

  auto runB = [&]() {
    // init host + device state for a fresh S-step run.
    for (int s = 0; s < g9::kMaxSlots; ++s) { hOcc[s]=0; hPos[s]=0; hGen[s]=0; hReqId[s]=0; hHstate[s]=0; }
    for (int i = 0; i < B; ++i) { hOcc[i]=1; hReqId[i]=w[i].request_id; hPos[i]=0; hGen[i]=S;
                                  hHstate[i]=g9::seed_for_request(w[i].request_id); }
    CK(cudaMemcpyAsync(st.svec, hsv.data(), hsv.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    for (int step = 0; step < S; ++step) {
      // HOST METADATA REBUILD + state write h2d (today's per-step pattern).
      CK(cudaMemcpyAsync(st.occupied, hOcc, g9::kMaxSlots*sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.pos, hPos, g9::kMaxSlots*sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.gen_len, hGen, g9::kMaxSlots*sizeof(int), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.req_id, hReqId, g9::kMaxSlots*sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
      CK(cudaMemcpyAsync(st.hstate, hHstate, g9::kMaxSlots*sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
      int active_count = 0;
      for (int s = 0; s < B; ++s) if (hOcc[s] && hPos[s] < hGen[s]) hActive[active_count++] = s;
      CK(cudaMemcpyAsync(dActive, hActive, active_count*sizeof(int), cudaMemcpyHostToDevice, stream));
      g9::StepParams sp{}; sp.st=st; sp.W=dW; sp.active_slots=dActive; sp.out_step=dStepOut; sp.active_count=active_count;
      int g = std::min(gridB, active_count > 0 ? active_count : 1);
      g9::kRelaunchStep<<<g, g9::kBlockThreads, smem, stream>>>(sp);
      // read step emissions + advanced state back (d2h round-trip).
      CK(cudaMemcpyAsync(hStepOut, dStepOut, active_count*sizeof(g9::Completion), cudaMemcpyDeviceToHost, stream));
      CK(cudaMemcpyAsync(hPos, st.pos, g9::kMaxSlots*sizeof(int), cudaMemcpyDeviceToHost, stream));
      CK(cudaMemcpyAsync(hHstate, st.hstate, g9::kMaxSlots*sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));   // host must see state to rebuild next step
    }
  };
  double B_us;
  {
    const int REPS = 30, WARM = 5;
    for (int i = 0; i < WARM; ++i) runB();
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double acc = 0;
    for (int r = 0; r < REPS; ++r) {
      auto t0 = std::chrono::high_resolution_clock::now();
      runB();
      auto t1 = std::chrono::high_resolution_clock::now();
      acc += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    B_us = acc / REPS;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }

  SteadyResult r;
  r.A_perstep_us = A_us / S;
  r.B_perstep_us = B_us / S;
  r.overhead_us = r.B_perstep_us - r.A_perstep_us;

  cudaStreamDestroy(stream);
  cudaFree(st.occupied); cudaFree(st.req_id); cudaFree(st.pos); cudaFree(st.gen_len);
  cudaFree(st.hstate); cudaFree(st.svec); cudaFree(dStop);
  cudaFreeHost(hRqBuf); cudaFreeHost(hRqHead); cudaFreeHost(hRqTail); cudaFreeHost(hDone);
  cudaFreeHost(hCqBuf); cudaFree(dCqProd); cudaFreeHost(hCqReady);
  cudaFreeHost(hActive); cudaFreeHost(hStepOut);
  cudaFreeHost(hPos); cudaFreeHost(hOcc); cudaFreeHost(hGen); cudaFreeHost(hReqId); cudaFreeHost(hHstate);
  return r;
}

int main() {
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  // host-mapped memory requires this on most setups; harmless if already default.
  unsigned int flags = 0; cudaGetDeviceFlags(&flags);
  printf("device: %s  SMs=%d  cc=%d.%d  coopLaunch=%d  canMapHost=%d\n",
         prop.name, prop.multiProcessorCount, prop.major, prop.minor,
         prop.cooperativeLaunch, prop.canMapHostMemory);
  if (!prop.cooperativeLaunch) { printf("FATAL: no cooperative launch\n"); return 1; }
  if (!prop.canMapHostMemory)  { printf("FATAL: no host-mapped memory\n"); return 1; }

  const int sms = prop.multiProcessorCount;

  // shared GEMV weight (device).
  std::vector<float> hW((size_t)g9::kH * g9::kD);
  for (size_t i = 0; i < hW.size(); ++i) hW[i] = fill_val(i, 909) * 0.1f;
  float* dW; CK(cudaMalloc(&dW, hW.size() * sizeof(float)));
  CK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(float), cudaMemcpyHostToDevice));

  // occupancy of the persistent engine.
  size_t eng_smem = g9::engine_smem_bytes();
  CK(cudaFuncSetAttribute((void*)g9::kPersistentEngine,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)eng_smem));
  int occ = occ_blocks_per_sm((void*)g9::kPersistentEngine, g9::kBlockThreads, eng_smem);
  if (occ < 1) occ = 1;
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, (void*)g9::kPersistentEngine));
  printf("OCC: kPersistentEngine %d blk/SM (%d resident CTAs) regs=%d smem=%.1fKB\n",
         occ, occ * sms, fa.numRegs, eng_smem / 1024.0);

  // ---- correctness workload: time-staggered admission, variable active count,
  //      slot reuse. Modest N so the relaunch baseline (which does N_tokens host
  //      round-trips) finishes inside the SIGKILL budget. ----
  bool all_ok = true;

  struct Scen { int n_req; int n_slots; int stagger; int min_len; int max_len; const char* name; };
  Scen scens[3] = {
    {16,  8,  4, 16, 48, "N16_S8_stag4"},     // light, staggered, slot reuse
    {32,  16, 8, 24, 64, "N32_S16_stag8"},    // medium concurrency
    {8,   8,  0, 64, 96, "N8_S8_nostag"},     // all arrive immediately, long gens
  };

  // JSON accumulators
  double A_pertok[3] = {0}, B_pertok[3] = {0}, speedup[3] = {0};
  double A_wall[3] = {0}, B_wall[3] = {0};
  long A_total[3] = {0}, B_total[3] = {0};
  int corr_pass[3] = {0};

  for (int s = 0; s < 3; ++s) {
    const Scen& sc = scens[s];
    printf("\n############ SCENARIO %s : N=%d slots=%d stagger=%d len[%d,%d] ############\n",
           sc.name, sc.n_req, sc.n_slots, sc.stagger, sc.min_len, sc.max_len);
    auto w = make_workload(sc.n_req, 1234 + s, sc.min_len, sc.max_len);
    unsigned int total_tokens = 0; for (auto& r : w) total_tokens += r.gen_len;
    printf("workload: %d requests, %u total tokens\n", sc.n_req, total_tokens);

    auto ref = cpu_reference(w);

    // ---- (A) persistent engine ----
    RunResult A = run_engine(w, sc.n_slots, sc.stagger, sms, occ, eng_smem, dW, true);
    // ---- (B) per-step relaunch ----
    RunResult B = run_relaunch(w, sc.n_slots, sc.stagger, sms, eng_smem, dW, true);

    // ---- correctness: exactly-once + bit-exact vs CPU, and A==B ----
    RingCheck cA = check_completions(A.comps, ref);
    RingCheck cB = check_completions(B.comps, ref);
    auto mapA = completions_to_map(A.comps, ref);
    auto mapB = completions_to_map(B.comps, ref);
    bool AB = maps_bit_equal(mapA, mapB);
    bool AC = maps_bit_equal(mapA, ref);
    bool BC = maps_bit_equal(mapB, ref);
    bool ok = cA.ok && cB.ok && AB && AC && BC;
    all_ok = all_ok && ok;
    corr_pass[s] = ok ? 1 : 0;

    printf("RING (A engine): total=%ld drops=%ld dupes=%ld mismatch=%ld missing=%ld -> %s\n",
           cA.total, cA.drops, cA.dupes, cA.mismatch, cA.missing, cA.ok ? "exactly-once" : "BROKEN");
    printf("RING (B relaunch): total=%ld drops=%ld dupes=%ld mismatch=%ld missing=%ld -> %s\n",
           cB.total, cB.drops, cB.dupes, cB.mismatch, cB.missing, cB.ok ? "exactly-once" : "BROKEN");
    printf("BIT-EXACT: A==CPU=%d  B==CPU=%d  A==B=%d -> %s\n",
           (int)AC, (int)BC, (int)AB, ok ? "PASS" : "FAIL");

    double sp = (A.per_token_us > 0) ? B.per_token_us / A.per_token_us : 0.0;
    printf("PERF (per-token us, lower=better):\n");
    printf("  (A) persistent engine : %.4f us/tok  (wall %.1f us, %u tok)\n",
           A.per_token_us, A.wall_us, total_tokens);
    printf("  (B) per-step relaunch : %.4f us/tok  (wall %.1f us)  -> A is %.2fx %s\n",
           B.per_token_us, B.wall_us, sp, (A.per_token_us < B.per_token_us) ? "FASTER" : "slower");
    printf("  per-step overhead eliminated (B-A): %.4f us/tok\n",
           B.per_token_us - A.per_token_us);

    A_pertok[s] = A.per_token_us; B_pertok[s] = B.per_token_us; speedup[s] = sp;
    A_wall[s] = A.wall_us; B_wall[s] = B.wall_us;
    A_total[s] = cA.total; B_total[s] = cB.total;
  }

  // ---- STEADY-STATE per-step sweep: isolate the per-step-overhead win, free of
  //      the host arrival stagger. Sweep batch B and step count S. ----
  printf("\n############ STEADY-STATE per-step (fixed fully-active batch, no admission mid-run) ############\n");
  struct SS { int B; int S; const char* name; };
  SS sss[5] = {
    {1,   64,  "B1_S64"},      // single-slot decode (latency-bound)
    {8,   64,  "B8_S64"},
    {32,  64,  "B32_S64"},
    {128, 64,  "B128_S64"},    // full batch
    {32,  256, "B32_S256"},    // longer horizon (amortization vs S)
  };
  double ssA[5]={0}, ssB[5]={0}, ssOv[5]={0}, ssSp[5]={0};
  for (int i = 0; i < 5; ++i) {
    SteadyResult r = run_steady_state(sss[i].B, sss[i].S, sms, occ, eng_smem, dW);
    ssA[i]=r.A_perstep_us; ssB[i]=r.B_perstep_us; ssOv[i]=r.overhead_us;
    ssSp[i] = (r.A_perstep_us > 0) ? r.B_perstep_us / r.A_perstep_us : 0.0;
    printf("  %-9s : A(engine) %.4f us/step | B(relaunch) %.4f us/step | "
           "overhead cut %.4f us/step -> %.2fx\n",
           sss[i].name, r.A_perstep_us, r.B_perstep_us, r.overhead_us, ssSp[i]);
  }

  printf("\n=== G9 GATE: %s ===\n", all_ok ? "PASS(correctness)" : "FAIL(correctness)");
  printf("SUMMARY_JSON {\"sm\":%d,\"occ_engine\":%d,\"resident_ctas\":%d,\"regs\":%d,"
         "\"scen\":[\"%s\",\"%s\",\"%s\"],"
         "\"A_pertok_us\":[%.4f,%.4f,%.4f],\"B_pertok_us\":[%.4f,%.4f,%.4f],"
         "\"speedup\":[%.3f,%.3f,%.3f],\"overhead_cut_us\":[%.4f,%.4f,%.4f],"
         "\"A_wall_us\":[%.1f,%.1f,%.1f],\"B_wall_us\":[%.1f,%.1f,%.1f],"
         "\"A_comps\":[%ld,%ld,%ld],\"B_comps\":[%ld,%ld,%ld],"
         "\"ss_name\":[\"B1_S64\",\"B8_S64\",\"B32_S64\",\"B128_S64\",\"B32_S256\"],"
         "\"ss_A_perstep_us\":[%.4f,%.4f,%.4f,%.4f,%.4f],"
         "\"ss_B_perstep_us\":[%.4f,%.4f,%.4f,%.4f,%.4f],"
         "\"ss_overhead_cut_us\":[%.4f,%.4f,%.4f,%.4f,%.4f],"
         "\"ss_speedup\":[%.3f,%.3f,%.3f,%.3f,%.3f],"
         "\"corr_per_scen\":[%d,%d,%d],\"corr_gate\":\"%s\"}\n",
         sms, occ, occ * sms, fa.numRegs,
         scens[0].name, scens[1].name, scens[2].name,
         A_pertok[0], A_pertok[1], A_pertok[2],
         B_pertok[0], B_pertok[1], B_pertok[2],
         speedup[0], speedup[1], speedup[2],
         B_pertok[0]-A_pertok[0], B_pertok[1]-A_pertok[1], B_pertok[2]-A_pertok[2],
         A_wall[0], A_wall[1], A_wall[2], B_wall[0], B_wall[1], B_wall[2],
         A_total[0], A_total[1], A_total[2], B_total[0], B_total[1], B_total[2],
         ssA[0], ssA[1], ssA[2], ssA[3], ssA[4],
         ssB[0], ssB[1], ssB[2], ssB[3], ssB[4],
         ssOv[0], ssOv[1], ssOv[2], ssOv[3], ssOv[4],
         ssSp[0], ssSp[1], ssSp[2], ssSp[3], ssSp[4],
         corr_pass[0], corr_pass[1], corr_pass[2],
         all_ok ? "PASS" : "FAIL");

  cudaFree(dW);
  return all_ok ? 0 : 2;
}
