#pragma once

// Continuum fork: a decode step is a FRAME (CONTINUUM-ENGINE 4.7.8, from bgfx's Context::frame/swap,
// FreeHandle/freeAllHandles, transient buffers and m_cmdPre).
//
// Two frames are in flight: workers RECORD step N+1 while the GPU EXECUTES step N; at the boundary the
// decode thread swaps them. A frame owns, for exactly as long as the GPU may use them:
//   - retained resources: every resident its commands read or write (a checkpoint a restore reads, a
//     buffer a blit writes). Dropping the last outside handle never frees bytes a queued command still
//     touches: the frame releases them when it RETIRES (its GPU completion), the way bgfx frees a
//     destroyed handle only after the frame that used it has rendered;
//   - a scratch arena (token arrays, sampler candidates, mask rows, JSON fragments) reset on retire: no
//     allocator on the hot path;
//   - boundary commands (a host-pool grant, a decode width, an adapter attach) queued by the core and
//     applied in order on the decode thread at the swap, never under a lock held against a running step.
//
// Pure and header-only, proven by tests/test-server-step-frame.cpp without a model.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

// A per-step bump allocator: allocations are pointer bumps into one block, all freed at once on
// reset. Grows by doubling when a step needs more, then stays at that size (measured high water).
class step_arena {
public:
    std::span<std::byte> allocate(size_t bytes, size_t align = alignof(std::max_align_t)) {
        size_t start = (used_ + align - 1) & ~(align - 1);
        if (start + bytes > block_.size()) {
            grow(start + bytes);
            start = (used_ + align - 1) & ~(align - 1);
        }
        used_ = start + bytes;
        high_water_ = std::max(high_water_, used_);
        return { block_.data() + start, bytes };
    }

    template <typename T>
    std::span<T> allocate_array(size_t n) {
        auto raw = allocate(n * sizeof(T), alignof(T));
        return { reinterpret_cast<T *>(raw.data()), n };
    }

    // The step retired: every span it handed out is dead. Blocks outgrown mid-step are freed; the
    // current (largest) block is kept, so the next step allocates nothing.
    void reset() {
        used_ = 0;
        retired_.clear();
    }
    size_t used()       const { return used_; }
    size_t capacity()   const { return block_.size(); }
    size_t high_water() const { return high_water_; }

private:
    // A step that outgrows its block copies nothing: the block is replaced by a doubled one, and the
    // old block stays alive (spans handed out earlier in THIS step remain valid) until reset.
    void grow(size_t need) {
        size_t next = block_.empty() ? 4096 : block_.size();
        while (next < need) {
            next *= 2;
        }
        if (used_ > 0) {
            retired_.push_back(std::move(block_));
        }
        block_.assign(next, std::byte{0});
        used_ = 0;
    }

    std::vector<std::byte>              block_;
    std::vector<std::vector<std::byte>> retired_;
    size_t                              used_       = 0;
    size_t                              high_water_ = 0;
};

// One step's frame.
class step_frame {
public:
    explicit step_frame(uint64_t number) : number_(number) {}

    uint64_t number() const { return number_; }

    // Keep `resource` alive until this frame retires (any shared handle: a host_bytes resident, a
    // buffer a blit writes).
    template <typename T>
    void retain(std::shared_ptr<T> resource) {
        retained_.push_back(std::shared_ptr<const void>(std::move(resource)));
    }
    size_t retained() const { return retained_.size(); }

    step_arena & arena() { return arena_; }

    // Retire: the GPU is done with this frame. Its resources are released and its arena reset, then it
    // is reused for a later step.
    void retire(uint64_t next_number) {
        retained_.clear();
        arena_.reset();
        number_ = next_number;
    }

private:
    uint64_t                                 number_;
    std::vector<std::shared_ptr<const void>> retained_;
    step_arena                               arena_;
};

// Two frames in flight, and the boundary commands for the next one.
class step_pipeline {
public:
    using command = std::function<void()>;

    step_pipeline() : record_(1), execute_(0) {}

    // The frame being recorded (step N+1): workers retain resources and allocate scratch into it.
    step_frame & recording() { return record_; }
    // The frame the GPU is executing (step N).
    step_frame & executing() { return execute_; }

    // Queue a command for the next boundary. Any thread; one short critical section.
    void queue(command c) {
        std::scoped_lock lock(mutex_);
        commands_.push_back(std::move(c));
    }

    // The boundary, on the decode thread, once the GPU has finished the executing frame: that frame
    // retires (its resources released, its arena reset), the boundary commands run in order, and the
    // recorded frame becomes the executing one. Returns the number of commands applied.
    size_t swap() {
        std::vector<command> due;
        {
            std::scoped_lock lock(mutex_);
            due.swap(commands_);
        }
        execute_.retire(record_.number() + 1);
        for (auto & c : due) {
            c();
        }
        std::swap(record_, execute_);
        return due.size();
    }

private:
    std::mutex           mutex_;
    std::vector<command> commands_;
    step_frame           record_;
    step_frame           execute_;
};
