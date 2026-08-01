#pragma once
// ggml-moe-residency.hpp — universal MoE expert residency (continuum fork).
//
// PROBLEM: when a MoE's experts exceed VRAM (and RAM), the op-offload path re-fetches the
// router-selected experts from the giant mmap every token. Without a resident copy, every
// token pays full NVMe latency for its whole working set (~0.2 tok/s wall).
//
// DESIGN — three separable concerns, one type each, so any one can change alone:
//
//   ExpertId       an expert's SEMANTIC identity: (weight-tensor name, expert index). Stable
//                  across tokens and IDENTICAL on every MoE — Mixtral, Qwen3-MoE, DeepSeek-V3,
//                  Kimi — because they all name expert weights blk.N.ffn_{gate,up,down}_exps.
//                  This is the cache KEY. Never a raw runtime pointer (which varies per token
//                  and per platform — the bug this replaces).
//
//   ExpertFetcher  HOW an expert's bytes reach a resident slot — the adapter. MmapFaultFetcher
//                  (portable) or DirectReadFetcher (Windows NVMe). This is the PHYSICAL concern
//                  and may legitimately use the address->file mapping. Swappable: an async /
//                  overlapped fetcher drops in here without touching identity or caching.
//
//   ResidencyCache size-classed PINNED host pools + LRU, admitting misses through the fetcher.
//                  Pure caching. Knows nothing about how bytes are fetched or what a model is.
//
// No model-specific constants. Zero cost when budget == 0 (disabled). Included once, into
// ggml-backend.cpp, after ggml-backend-impl.h (for the backend-buffer API it allocates through).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sys/stat.h>   // stat() for plan-file mtime polling (portable)

#ifdef _WIN32
#include <malloc.h>   // _aligned_malloc / _aligned_free for unbuffered (sector-aligned) direct I/O
#endif

namespace ggml_moe {

// FNV-1a over the weight tensor's name — a stable per-tensor id, no allocation.
static inline uint64_t hash_name(const char * s) {
    uint64_t h = 1469598103934665603ull;
    for (; s && *s; ++s) { h ^= (uint8_t) *s; h *= 1099511628211ull; }
    return h;
}

// The universal, model-agnostic identity of one expert.
struct ExpertId {
    uint64_t tensor;   // stable file-key of the expert's bytes (see expert_id_for)
    int32_t  index;    // expert slot (0 when the file-key already encodes position)

    uint64_t key() const {
        return tensor ^ ((uint64_t) (uint32_t) index * 0x9E3779B97F4A7C15ull);
    }
};

// Strip the scheduler-copy decoration ggml adds at an offload split. When the offload source is a
// COPY node (the usual case at a mul_mat_id split), ggml_format_name stamps it "BACKEND#src#C" —
// e.g. "CUDA0#blk.3.ffn_up_exps.weight#0" (ggml-backend.cpp ggml_format_name(copy,"%s#%s#%d",...)).
// The DECORATED name varies per graph (backend prefix + copy index regenerate every forward pass) =>
// hashing it gave EXACTLY 0 cross-token reuse. The middle '#'-delimited segment is the STABLE weight-
// leaf name (blk.N.ffn_*_exps.weight), created once at load, identical every token. No '#' => already
// a clean leaf name. (Root-cause + fix from M5, with vendored-source receipts.)
static inline uint64_t canonical_name_key(const char * name) {
    const char * first = strchr(name, '#');
    if (first == nullptr) { return hash_name(name); }   // already the canonical leaf name
    const char * start = first + 1;
    const char * last  = strrchr(start, '#');
    const size_t len   = last ? (size_t) (last - start) : strlen(start);
    uint64_t h = 1469598103934665603ull;                // FNV-1a over just the canonical segment
    for (size_t i = 0; i < len; i++) { h ^= (uint8_t) start[i]; h *= 1099511628211ull; }
    return h;
}

// Expert identity = canonical weight-leaf name (stable across tokens) + expert index. mmap_src is
// unused now (the per-eval pointer/file-offset are NOT token-stable — that was the bug).
static inline ExpertId expert_id_for(const void * /*mmap_src*/, const char * tensor_name, int32_t index) {
    return ExpertId{ canonical_name_key(tensor_name), index };
}

// Cache keys for one ROUTED expert (layer, expert): a MoE expert is 3 weight tensors (gate/up/down),
// each its own canonical-name key — so pinning one expert pins three cache keys. Names match the
// universal llama.cpp MoE convention the offload seam hashes (blk.N.ffn_{gate,up,down}_exps.weight).
// This is how a controller's (layer,expert) pin_list maps onto ResidencyCache identity.
static inline void expert_pin_keys(uint32_t layer, uint32_t expert, uint64_t out[3]) {
    static const char * mats[3] = { "gate", "up", "down" };
    for (int m = 0; m < 3; m++) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "blk.%u.ffn_%s_exps.weight", layer, mats[m]);
        out[m] = ExpertId{ canonical_name_key(nm), (int32_t) expert }.key();
    }
}

// The v1 actuator plan the controller (continuum-core ServingExpertPager) writes to GGML_MOE_PLAN_FILE:
//   {"version":1,"budget_bytes":<u64>,"window_k":<u32>,"pin_list":[{"layer":0,"expert":7},...]}
// Atomic-rename control file; the cache polls its mtime per token and reloads on change. Minimal
// hand-parse (no JSON dep at the ggml layer): tolerant scan for the three known keys + the pin array.
struct PagerPlan {
    uint64_t budget_bytes = 0;
    uint32_t window_k     = 0;
    std::vector<std::pair<uint32_t,uint32_t>> pins;   // (layer, expert)
    bool     ok           = false;
};
static inline bool parse_pager_plan(const char * text, PagerPlan & plan) {
    if (!text) { return false; }
    auto num_after = [&](const char * key) -> long long {
        const char * p = std::strstr(text, key);
        if (!p) { return -1; }
        p += std::strlen(key);
        while (*p && (*p == '"' || *p == ':' || *p == ' ' || *p == '\t')) { p++; }
        return std::strtoll(p, nullptr, 10);
    };
    long long b = num_after("\"budget_bytes\"");
    long long k = num_after("\"window_k\"");
    if (b >= 0) { plan.budget_bytes = (uint64_t) b; }
    if (k >= 0) { plan.window_k     = (uint32_t) k; }
    // pin_list: scan {"layer":L,"expert":E} objects inside the "pin_list" array
    const char * pl = std::strstr(text, "\"pin_list\"");
    if (pl) {
        const char * p = std::strchr(pl, '[');
        const char * end = p ? std::strchr(p, ']') : nullptr;
        for (const char * q = p; q && end && q < end; ) {
            const char * lq = std::strstr(q, "\"layer\"");
            if (!lq || lq > end) { break; }
            const char * eq = std::strstr(lq, "\"expert\"");
            if (!eq || eq > end) { break; }
            const char * lp = lq + std::strlen("\"layer\""); while (*lp && (*lp==':'||*lp==' '||*lp=='"')) lp++;
            const char * ep = eq + std::strlen("\"expert\""); while (*ep && (*ep==':'||*ep==' '||*ep=='"')) ep++;
            plan.pins.emplace_back((uint32_t) std::strtoul(lp, nullptr, 10), (uint32_t) std::strtoul(ep, nullptr, 10));
            q = ep;
        }
    }
    plan.ok = (b >= 0 || k >= 0 || !plan.pins.empty());
    return plan.ok;
}

// ------------------------------------------------------------------------------------------
// fetch adapter — how expert bytes are pulled into a slot
// ------------------------------------------------------------------------------------------
struct FetchItem {
    void *       dst;    // pinned slot to fill
    const void * src;    // expert's host address (mmap-backed)
    size_t       bytes;  // expert size
};

class ExpertFetcher {
public:
    virtual ~ExpertFetcher() = default;
    virtual const char * name() const = 0;
    // Copy `bytes` for the expert backing host address `src` into `dst`. false => caller falls back.
    virtual bool fetch(void * dst, const void * src, size_t bytes) = 0;
    // Fill a batch. Default = sequential fetch(); adapters that can overlap I/O override for bandwidth.
    virtual void fetch_many(const FetchItem * items, size_t n) {
        for (size_t i = 0; i < n; i++) { fetch(items[i].dst, items[i].src, items[i].bytes); }
    }
};

// Portable default: memcpy faults the mmap page once (NVMe stall), then it is pinned-resident.
// On Windows the BATCH path uses PrefetchVirtualMemory to parallelize: one call issues concurrent
// read-ahead of ALL the selected experts' mmap ranges into the working set (vs per-page demand
// faults that serialize), then the memcpys land on resident pages. Reading THROUGH the mmap (not a
// second file handle) avoids the file-cache contention that pinned the direct-read path at ~190 MB/s
// while that same file was being demand-paged.
class MmapFaultFetcher final : public ExpertFetcher {
public:
    const char * name() const override { return "mmap-prefetch"; }
    bool fetch(void * dst, const void * src, size_t bytes) override {
        memcpy(dst, src, bytes);   // single expert: fault the page(s) once, then it is resident
        return true;
    }
#ifdef _WIN32
    void fetch_many(const FetchItem * items, size_t n) override {
        std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges(n);
        for (size_t i = 0; i < n; i++) {
            ranges[i].VirtualAddress = (PVOID) items[i].src;
            ranges[i].NumberOfBytes  = items[i].bytes;
        }
        // kick off concurrent read-ahead of the whole batch, THEN copy (pages now inbound/resident)
        PrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR) n, ranges.data(), 0);
        for (size_t i = 0; i < n; i++) { memcpy(items[i].dst, items[i].src, items[i].bytes); }
    }
#endif
};

#ifdef _WIN32
// Windows: read expert byte ranges straight from the backing file with UNBUFFERED, OVERLAPPED direct
// I/O — the fast path. Recovers (file, offset) from the mmap address (VirtualQuery -> AllocationBase,
// GetMappedFileNameW -> \Device\.. via \\?\GLOBALROOT). Generic; any mmap-backed MoE.
//
// WHY UNBUFFERED IS THE WHOLE GAME: reads that go through the memory manager (a plain ReadFile, or a
// page fault on the mmap, or PrefetchVirtualMemory) serialize on the file's section object — Windows
// caps concurrent page-ins and pays ~5000 cycles + a VAD lock per 4K fault, which pinned every
// "concurrent" approach at ~190 MB/s while the drive itself does 2.6+ GB/s. FILE_FLAG_NO_BUFFERING
// bypasses the memory manager entirely: DMA straight from NVMe into our buffer, so overlapped reads
// actually run in parallel. It requires sector alignment (offset, length, buffer), so we read an
// aligned superset into an aligned bounce buffer and copy the expert bytes out.
class DirectReadFetcher final : public ExpertFetcher {
    static const uint64_t A = 4096;   // alignment: 4K covers 512e and 4Kn drives
    static const size_t   HPOOL = 16; // handles per file — Windows serializes I/O on ONE file object
                                      // even when overlapped, so concurrency needs DISTINCT handles.

    static const size_t MAXW = 64;    // max sliding-window depth (== fetch_many cap)

    std::mutex mtx;
    std::unordered_map<const void *, std::vector<HANDLE>> base2pool;  // AllocationBase -> handle pool
    size_t rr = 0;                                                    // round-robin cursor

    // PERSISTENT bounce buffers + events, reused across every fetch call. Re-allocating W aligned
    // buffers and W kernel events PER CALL (once per layer-matrix, thousands of times per generation)
    // commits/zero-fills/decommits ~tens of MB each time — that churn is what collapses the live fetch
    // from ~2 GB/s to ~140 MB/s under accumulating memory pressure, even though the raw NVMe does 3 GB/s.
    // Allocated once (lazily), grown only if a bigger expert size-class appears. Access is serialized by
    // the owning ResidencyCache's mutex (prefetch()/get() both hold it), so no extra lock is needed here.
    std::vector<void *>  bbuf_;       // MAXW aligned bounce buffers, each `bcap_` bytes
    std::vector<HANDLE>  bev_;        // MAXW reusable auto-reset events
    size_t               bcap_ = 0;   // current per-buffer capacity (grows to the largest aligned read)
    // per-call scratch, also persistent (reused, never re-allocated per fetch)
    std::vector<OVERLAPPED> ov_;
    std::vector<HANDLE>     hs_, evs_;
    std::vector<size_t>     head_, item_of_, active_;

    // Ensure the persistent pools exist and each buffer holds >= need bytes. One-time create of the
    // events + scratch vectors; buffers grow (free+realloc all) only when a larger size-class is seen.
    void ensure_pool(size_t need) {
        if (bev_.empty()) {
            bev_.assign(MAXW, nullptr);
            for (size_t j = 0; j < MAXW; j++) { bev_[j] = CreateEventW(nullptr, FALSE, FALSE, nullptr); }
            bbuf_.assign(MAXW, nullptr);
            ov_.resize(MAXW); hs_.assign(MAXW, INVALID_HANDLE_VALUE);
            head_.assign(MAXW, 0); item_of_.assign(MAXW, 0);
            active_.reserve(MAXW); evs_.reserve(MAXW);
        }
        if (need > bcap_) {
            for (size_t j = 0; j < MAXW; j++) { if (bbuf_[j]) { _aligned_free(bbuf_[j]); } bbuf_[j] = _aligned_malloc(need, A); }
            bcap_ = need;
        }
    }

public:
    ~DirectReadFetcher() {
        for (void * b : bbuf_) { if (b) { _aligned_free(b); } }
        for (HANDLE e : bev_)  { if (e) { CloseHandle(e); } }
        for (auto & kv : base2pool) { for (HANDLE h : kv.second) { if (h != INVALID_HANDLE_VALUE) { CloseHandle(h); } } }
    }
private:

    // Resolve an mmap host address to (a backing file handle from the pool, byte offset). Handles are
    // unbuffered + overlapped; round-robined so concurrent reads land on different file objects and
    // actually parallelize. INVALID_HANDLE_VALUE on failure (empty pool cached so it is not retried).
    HANDLE resolve(const void * addr, uint64_t & file_off) {
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(addr, &mbi, sizeof(mbi)) == 0 || mbi.AllocationBase == nullptr) {
            return INVALID_HANDLE_VALUE;
        }
        const void * alloc_base = mbi.AllocationBase;
        file_off = (uint64_t) ((const uint8_t *) addr - (const uint8_t *) alloc_base);
        std::lock_guard<std::mutex> lk(mtx);
        auto it = base2pool.find(alloc_base);
        if (it == base2pool.end()) {
            std::vector<HANDLE> pool;
            wchar_t dev[MAX_PATH * 2] = {0};
            if (GetMappedFileNameW(GetCurrentProcess(), (LPVOID) addr, dev, MAX_PATH * 2) != 0) {
                std::wstring path = std::wstring(L"\\\\?\\GLOBALROOT") + dev;
                for (size_t k = 0; k < HPOOL; k++) {
                    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                           nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
                    if (h != INVALID_HANDLE_VALUE) { pool.push_back(h); }
                }
            }
            it = base2pool.emplace(alloc_base, std::move(pool)).first;
        }
        if (it->second.empty()) { return INVALID_HANDLE_VALUE; }
        return it->second[rr++ % it->second.size()];
    }

    // One expert: aligned unbuffered read into a bounce buffer, then copy out the exact bytes.
    bool read_one(const void * src, void * dst, size_t bytes) {
        uint64_t off = 0;
        HANDLE h = resolve(src, off);
        if (h == INVALID_HANDLE_VALUE) { return false; }
        const uint64_t aoff = off & ~(A - 1);
        const size_t   head = (size_t) (off - aoff);
        const size_t   alen = (size_t) (((head + bytes) + A - 1) & ~(A - 1));
        ensure_pool(alen);                       // persistent bounce buffer[0] (reused, not malloc'd here)
        void * bounce = bbuf_[0];
        if (!bounce) { return false; }
        OVERLAPPED ov = {}; ov.hEvent = bev_[0];
        ov.Offset = (DWORD) (aoff & 0xFFFFFFFF); ov.OffsetHigh = (DWORD) (aoff >> 32);
        DWORD got = 0;
        bool ok = ReadFile(h, bounce, (DWORD) alen, &got, &ov) || GetLastError() == ERROR_IO_PENDING;
        if (ok) { ok = GetOverlappedResult(h, &ov, &got, TRUE) != 0; }
        if (ok && got >= head + bytes) { memcpy(dst, (uint8_t *) bounce + head, bytes); }
        else { ok = false; }
        return ok;
    }

public:
    const char * name() const override { return "direct-read-unbuffered"; }

    bool fetch(void * dst, const void * src, size_t bytes) override { return read_one(src, dst, bytes); }

    // Concurrent batch as a SLIDING WINDOW: keep W unbuffered reads in flight and, the instant ANY
    // completes, issue the next — so the NVMe queue stays full instead of draining between waves. The
    // old per-wave barrier reaped all W before issuing more, so each wave waited on its SLOWEST read
    // and a big batch stacked (N/W) worst-case latencies (measured: same 235-expert batch swung
    // 279ms..4850ms). The window holds queue depth constant, killing that variance.
    void fetch_many(const FetchItem * items, size_t n) override {
        if (n == 0) { return; }
        const size_t W   = std::min<size_t>(MAXW, n);           // window depth (<=64 for WaitForMultipleObjects)
        const size_t esz = items[0].bytes;                      // uniform within a size-class pool
        const size_t bsz = (esz + A + (A - 1)) & ~(A - 1);      // aligned bounce size (head slack + expert)
        ensure_pool(bsz);                                       // PERSISTENT bounce buffers + events + scratch (reused, not per-call)

        size_t n_ok = 0, n_rf = 0, n_if = 0, n_pf = 0;          // probe counters
        const auto t_start = std::chrono::steady_clock::now();

        // issue item i into slot j; true iff a read is now in flight (else it was memcpy'd in place)
        auto issue = [&](size_t j, size_t i) -> bool {
            const FetchItem & it = items[i];
            uint64_t off = 0;
            HANDLE h = bbuf_[j] ? resolve(it.src, off) : INVALID_HANDLE_VALUE;
            if (h == INVALID_HANDLE_VALUE) { n_rf++; memcpy(it.dst, it.src, it.bytes); return false; }
            const uint64_t aoff = off & ~(A - 1);
            head_[j] = (size_t) (off - aoff); item_of_[j] = i;
            const size_t alen = (size_t) (((head_[j] + it.bytes) + A - 1) & ~(A - 1));
            ov_[j] = {}; ov_[j].Offset = (DWORD)(aoff & 0xFFFFFFFF); ov_[j].OffsetHigh = (DWORD)(aoff >> 32); ov_[j].hEvent = bev_[j];
            DWORD got = 0;
            if (ReadFile(h, bbuf_[j], (DWORD) alen, &got, &ov_[j]) || GetLastError() == ERROR_IO_PENDING) { hs_[j] = h; return true; }
            n_if++; memcpy(it.dst, it.src, it.bytes); return false;
        };
        auto reap = [&](size_t j) {
            const FetchItem & it = items[item_of_[j]];
            DWORD got = 0;
            if (GetOverlappedResult(hs_[j], &ov_[j], &got, TRUE) && got >= head_[j] + it.bytes) {
                n_ok++; memcpy(it.dst, (uint8_t *) bbuf_[j] + head_[j], it.bytes);
            } else { n_pf++; memcpy(it.dst, it.src, it.bytes); }
        };

        size_t next = 0;
        active_.clear();                                        // slot indices with a read in flight (persistent)
        for (size_t j = 0; j < W; j++) {                        // prime the window
            while (next < n && !issue(j, next)) { next++; }
            if (next < n) { active_.push_back(j); next++; }
        }
        while (!active_.empty()) {                              // slide: wait-any, reap, refill
            evs_.clear();
            for (size_t s : active_) { evs_.push_back(bev_[s]); }
            const DWORD r = WaitForMultipleObjects((DWORD) evs_.size(), evs_.data(), FALSE, INFINITE);
            const size_t wi = (r >= WAIT_OBJECT_0 && r < WAIT_OBJECT_0 + evs_.size()) ? (size_t)(r - WAIT_OBJECT_0) : 0;
            const size_t j = active_[wi];
            reap(j);
            bool refilled = false;
            while (next < n) { if (issue(j, next++)) { refilled = true; break; } }
            if (!refilled) { active_.erase(active_.begin() + wi); }
        }

        if (getenv("GGML_MOE_OFFLOAD_STATS")) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
            const double mbps = ms > 0 ? (n * (double) esz / (1024.0 * 1024.0)) / (ms / 1000.0) : 0;
            fprintf(stderr, "[FETCH] batch=%zu win=%zu unbuf_ok=%zu resolve_fail=%zu issue_fail=%zu reap_fail=%zu | %.0f ms %.0f MB/s\n",
                    n, W, n_ok, n_rf, n_if, n_pf, ms, mbps);
        }
    }
};
#endif // _WIN32

// ------------------------------------------------------------------------------------------
// residency cache — size-classed pinned pools + RECENCY-WINDOW eviction, fed by a fetcher
// ------------------------------------------------------------------------------------------
// Policy is a recency WINDOW at TOKEN granularity, not per-access LRU. Each expert is stamped with the
// token-generation it was last used in; eviction removes the OLDEST generation first (tie-broken by
// access tick). This keeps the last K tokens' expert SETS resident as units — where K = (budget /
// per-token working set) emerges from the budget the governor sets. Measured on K3 trace replay: keeping
// the last decode token's experts => 40.6% of the next token already resident (last 2 => 51%), vs
// per-access LRU ~12% at the same budget. LRU loses because it churns the previous token's set out
// mid-token (its ticks go stale as this token's experts stream in) before the next token reuses it;
// generation-primary ordering protects whole recent tokens. The controller (continuum-core
// ServingExpertPager) sets budget_bytes (and later per-expert precision); this is the mechanism.
class ResidencyCache {
    // one pinned buffer per distinct expert byte-size (gate/up/down can differ), split into slots.
    // Pinned => non-pageable (hits never swap to pagefile) AND DMA-fast H2D (~25 vs ~6 GB/s pageable).
    struct Pool {
        ggml_backend_buffer_t buf = nullptr;
        uint8_t * base = nullptr;
        size_t    slot_size = 0, max_slots = 0, used = 0;
        bool      failed = false;
        std::unordered_map<uint64_t, size_t> key2slot;
        std::vector<uint64_t> slot_key, slot_tick, slot_gen;
    };
    static const size_t MAX_POOLS = 3; // a MoE has at most a few distinct expert byte-sizes

    std::mutex mtx;
    size_t         budget_bytes;
    ExpertFetcher& fetcher;
    uint64_t tick = 0, token_gen = 0, hits = 0, misses = 0;

    // [RUNG-2] controller actuator state (set via the GGML_MOE_PLAN_FILE control file; see PagerPlan).
    std::unordered_set<uint64_t> pinned_;   // expert cache-keys that are evict-EXEMPT (always resident)
    uint32_t     window_k_    = 0;          // recency window depth in tokens (0 = budget-implicit, legacy)
    std::string  plan_path_;                // GGML_MOE_PLAN_FILE (empty => no controller; pure legacy)
    int64_t      plan_mtime_  = 0;          // last-seen mtime; reload on change (atomic-rename by writer)
    double   admit_us = 0.0;   // accumulated fetch time on misses
    size_t   admit_bytes = 0;  // bytes fetched on misses  -> fetch MB/s = admit_bytes / admit_us
    std::unordered_map<size_t, Pool> pools;

    // lazily create this size-class's pinned pool, sharing the budget across up to MAX_POOLS classes
    void ensure_pool(Pool & p, ggml_backend_buffer_type_t host_buft, size_t need) {
        if (p.slot_size != 0 || p.failed) { return; }
        p.slot_size = need;
        const size_t share = budget_bytes / MAX_POOLS;
        p.max_slots = need ? (share / need) : 0;
        if (p.max_slots == 0 || pools.size() > MAX_POOLS) { p.failed = true; return; }
        p.buf = host_buft ? ggml_backend_buft_alloc_buffer(host_buft, p.max_slots * need) : nullptr;
        if (p.buf == nullptr) { p.failed = true; return; }
        p.base = (uint8_t *) ggml_backend_buffer_get_base(p.buf);
        p.slot_key.assign(p.max_slots, 0);
        p.slot_tick.assign(p.max_slots, 0);
        p.slot_gen.assign(p.max_slots, 0);
    }

    // pick a slot: next free one, else evict the OLDEST token-generation (tie-break: oldest access tick),
    // dropping its key mapping. Generation-primary = recency window at token granularity (see class doc).
    size_t reserve_slot(Pool & p) {
        if (p.used < p.max_slots) { return p.used++; }
        // evict OLDEST token-generation (tie: oldest tick) among NON-PINNED slots — pins (the
        // controller's always-hot set, e.g. the shared experts) are evict-exempt guaranteed hits.
        size_t slot = SIZE_MAX;
        uint64_t og = UINT64_MAX, ot = UINT64_MAX;
        const bool have_pins = !pinned_.empty();
        for (size_t i = 0; i < p.max_slots; i++) {
            if (have_pins && pinned_.count(p.slot_key[i])) { continue; }   // never evict a pinned expert
            if (p.slot_gen[i] < og || (p.slot_gen[i] == og && p.slot_tick[i] < ot)) {
                og = p.slot_gen[i]; ot = p.slot_tick[i]; slot = i;
            }
        }
        if (slot == SIZE_MAX) {   // pin_list oversized for this budget: fall back to oldest overall
            og = UINT64_MAX; ot = UINT64_MAX; slot = 0;
            for (size_t i = 0; i < p.max_slots; i++) {
                if (p.slot_gen[i] < og || (p.slot_gen[i] == og && p.slot_tick[i] < ot)) {
                    og = p.slot_gen[i]; ot = p.slot_tick[i]; slot = i;
                }
            }
        }
        p.key2slot.erase(p.slot_key[slot]);
        return slot;
    }

public:
    ResidencyCache(size_t budget, ExpertFetcher & f) : budget_bytes(budget), fetcher(f) {
        if (const char * pf = getenv("GGML_MOE_PLAN_FILE")) { plan_path_ = pf; }
    }

    void        set_budget(size_t b)   { budget_bytes = b; }   // idempotent; apply the env budget once

    // Reload the controller's actuator plan if the control file changed (atomic-rename by the writer, so
    // a size/mtime bump is a complete new plan — never a torn read). Called once per token from
    // advance_generation under the cache mutex. Applies budget_bytes / window_k / pin_list live.
    void reload_plan_locked() {
        if (plan_path_.empty()) { return; }
        struct stat st;
        if (stat(plan_path_.c_str(), &st) != 0) { return; }
        const int64_t mt = (int64_t) st.st_mtime;
        if (mt == plan_mtime_ || st.st_size <= 0) { return; }
        plan_mtime_ = mt;
        FILE * f = fopen(plan_path_.c_str(), "rb");
        if (!f) { return; }
        std::string buf; buf.resize((size_t) st.st_size);
        const size_t got = fread(&buf[0], 1, buf.size(), f);
        fclose(f);
        buf.resize(got);
        PagerPlan plan;
        if (!parse_pager_plan(buf.c_str(), plan)) { return; }
        if (plan.budget_bytes > 0) { budget_bytes = plan.budget_bytes; }
        window_k_ = plan.window_k;
        pinned_.clear();
        for (const auto & pe : plan.pins) {
            uint64_t k3[3]; expert_pin_keys(pe.first, pe.second, k3);
            pinned_.insert(k3[0]); pinned_.insert(k3[1]); pinned_.insert(k3[2]);
        }
        if (getenv("GGML_MOE_OFFLOAD_STATS")) {
            fprintf(stderr, "[PLAN] reloaded: budget=%zuMB window_k=%u pins=%zu (keys=%zu)\n",
                    budget_bytes / (1024*1024), window_k_, plan.pins.size(), pinned_.size());
        }
    }

    void        advance_generation()   { std::lock_guard<std::mutex> lk(mtx); ++token_gen; reload_plan_locked(); }  // once per token (compute call)
    bool        enabled()        const { return budget_bytes > 0; }
    uint64_t    n_hits()         const { return hits; }
    uint64_t    n_misses()       const { return misses; }
    double      hit_rate()       const { const uint64_t t = hits + misses; return t ? (double) hits / t : 0.0; }
    double      fetch_mb_s()     const { return admit_us > 0 ? (admit_bytes / admit_us) : 0.0; } // bytes/us == MB/s
    double      admit_micros()   const { return admit_us; }     // cumulative fetch time (for per-token deltas)
    size_t      admit_bytes_n()  const { return admit_bytes; }  // cumulative bytes fetched
    const char* fetcher_name()   const { return fetcher.name(); }

    // Pinned host pointer holding (expert_size + pad) bytes for `id`: the expert weight then zeroed
    // padding. On miss, fetches expert_size bytes via the adapter into a pinned slot. Returns nullptr
    // if this size-class cannot be cached (caller then streams straight from the mmap).
    uint8_t * get(ggml_backend_buffer_type_t host_buft, ExpertId id,
                  const uint8_t * src, size_t expert_size, size_t pad) {
        std::lock_guard<std::mutex> lk(mtx);
        const size_t need = expert_size + pad;
        Pool & p = pools[need];
        ensure_pool(p, host_buft, need);
        if (p.failed) { return nullptr; }

        const uint64_t key = id.key();   // SEMANTIC identity — stable across tokens, universal across MoEs
        auto it = p.key2slot.find(key);
        if (it != p.key2slot.end()) {
            hits++;
            p.slot_tick[it->second] = ++tick;
            p.slot_gen[it->second]  = token_gen;   // refresh generation on reuse
            return p.base + it->second * p.slot_size;
        }
        misses++;
        const size_t slot = reserve_slot(p);
        uint8_t * dst = p.base + slot * p.slot_size;
        const auto t0 = std::chrono::steady_clock::now();
        fetcher.fetch(dst, src, expert_size);            // adapter: mmap-fault or direct NVMe read
        const auto t1 = std::chrono::steady_clock::now();
        admit_us    += std::chrono::duration<double, std::micro>(t1 - t0).count();
        admit_bytes += expert_size;
        if (pad) { memset(dst + expert_size, 0, pad); }
        p.slot_key[slot]  = key;
        p.slot_tick[slot] = ++tick;
        p.slot_gen[slot]  = token_gen;
        p.key2slot[key]   = slot;
        return dst;
    }

    // Concurrently admit a whole layer's selected experts before they are consumed. Already-resident
    // experts are skipped; misses are reserved then fetched TOGETHER via the adapter's batched path
    // (overlapped I/O => queue depth == batch size => saturated read, vs the QD1 crawl of per-expert
    // get()). After this, get() for the same ids returns RAM hits. This is where the bandwidth lives.
    // Admission cost accrues to admit_us/admit_bytes; hit/miss accounting stays with get() so hit_rate
    // measures reuse, not admission. Contract for host_buft/expert_size/pad matches get().
    void prefetch(ggml_backend_buffer_type_t host_buft, const ExpertId * ids, const uint8_t * const * srcs,
                  size_t n, size_t expert_size, size_t pad) {
        std::lock_guard<std::mutex> lk(mtx);
        const size_t need = expert_size + pad;
        Pool & p = pools[need];
        ensure_pool(p, host_buft, need);
        if (p.failed) { return; }

        std::vector<FetchItem> batch;
        batch.reserve(n);
        size_t n_resident = 0, n_evict = 0;                              // retention probe
        for (size_t i = 0; i < n; i++) {
            const uint64_t key = ids[i].key();
            auto rit = p.key2slot.find(key);
            if (rit != p.key2slot.end()) {                               // reuse! refresh generation
                p.slot_gen[rit->second]  = token_gen;
                p.slot_tick[rit->second] = ++tick;
                n_resident++; continue;
            }
            if (p.used >= p.max_slots) { n_evict++; }                    // this reserve will evict oldest generation
            const size_t slot = reserve_slot(p);
            uint8_t * dst = p.base + slot * p.slot_size;
            p.slot_key[slot]  = key;
            p.slot_tick[slot] = ++tick;
            p.slot_gen[slot]  = token_gen;
            p.key2slot[key]   = slot;
            batch.push_back(FetchItem{ dst, srcs[i], expert_size });
        }
        // RETENTION probe: sel=selected this layer, resident=reused from prior tokens (the signal),
        // fetch=new reads, evict=experts kicked out to fit. resident~0 => no locality/thrashing;
        // evict>0 => cache too small for the working set. pool shows fill vs capacity.
        if (getenv("GGML_MOE_OFFLOAD_STATS")) {
            fprintf(stderr, "[RETAIN] sel=%zu resident=%zu fetch=%zu evict=%zu | pool=%zu/%zu\n",
                    n, n_resident, batch.size(), n_evict, p.used, p.max_slots);
        }
        if (batch.empty()) { return; }
        const auto t0 = std::chrono::steady_clock::now();
        fetcher.fetch_many(batch.data(), batch.size());  // concurrent — the saturated-bandwidth path
        const auto t1 = std::chrono::steady_clock::now();
        admit_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        for (const auto & b : batch) {
            admit_bytes += b.bytes;
            if (pad) { memset((uint8_t *) b.dst + expert_size, 0, pad); }
        }
    }
};

} // namespace ggml_moe
