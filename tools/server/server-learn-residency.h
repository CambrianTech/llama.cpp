#pragma once

// Immutable mirrors of core-issued leases, not a physical resource allocator.
// A reference cannot be reused concurrently or released before its allocations leave.
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <vector>

struct server_residency_row {
    std::string device, buffer_type, kind;
    uint64_t bytes = 0;
    bool host = false;
    bool operator==(const server_residency_row & r) const {
        return std::tie(device, buffer_type, kind, bytes, host) == std::tie(r.device, r.buffer_type, r.kind, r.bytes, r.host);
    }
};
struct server_residency_lease {
    std::string kind, lease_id;
    uint64_t bytes = 0;
    bool operator==(const server_residency_lease & r) const {
        return std::tie(kind, lease_id, bytes) == std::tie(r.kind, r.lease_id, r.bytes);
    }
};
struct server_residency {
    std::string id, engine, owner, subject;
    std::vector<server_residency_row> rows;
    std::vector<server_residency_lease> leases;
    bool pending = false, live = false, revoked = false;
    int32_t lora_id = -1;
};
struct server_residency_need {
    std::string id, subject;
    std::vector<server_residency_row> rows;
};
class server_residency_mirror {
    std::string engine;
    mutable std::mutex mutex;
    std::map<std::string, server_residency> records;
    std::set<std::string> retired;
public:
    explicit server_residency_mirror(std::string incarnation) : engine(std::move(incarnation)) {}
    std::string register_residency(const server_residency & r) {
        std::lock_guard<std::mutex> lock(mutex);
        if (r.engine != engine) { return "stale_engine"; }
        if (r.id.empty() || r.owner.empty() || r.subject.empty() || r.rows.empty() || r.leases.empty() ||
                r.pending || r.live || r.revoked || r.lora_id != -1) { return "invalid"; }
        if (retired.count(r.id)) { return "retired"; }
        const auto old = records.find(r.id);
        if (old != records.end()) {
            const auto & a = old->second;
            return a.owner == r.owner && a.subject == r.subject && a.rows == r.rows && a.leases == r.leases
                ? "none" : "residency_mismatch";
        }
        uint64_t required[2] = {}, covered[2] = {};
        std::set<std::tuple<std::string, std::string, std::string>> keys;
        auto add = [](uint64_t & total, uint64_t bytes) {
            if (bytes > uint64_t(INT64_MAX) || total > uint64_t(INT64_MAX) - bytes) { return false; }
            total += bytes;
            return true;
        };
        for (const auto & row : r.rows) {
            if (row.device.empty() || row.buffer_type.empty() ||
                    (row.kind != "weights" && row.kind != "optimizer_static_required" &&
                     row.kind != "optimizer_host_required" && row.kind != "shared_compute_required") ||
                    !keys.emplace(row.device, row.buffer_type, row.kind).second ||
                    !add(required[row.host], row.bytes)) { return "invalid"; }
        }
        std::set<std::string> lease_ids;
        for (const auto & lease : r.leases) {
            if ((lease.kind != "vram" && lease.kind != "ram") || lease.lease_id.empty() ||
                    !lease_ids.insert(lease.lease_id).second || !add(covered[lease.kind == "ram"], lease.bytes)) {
                return "invalid";
            }
            for (const auto & entry : records) {
                for (const auto & used : entry.second.leases) {
                    if (used.lease_id == lease.lease_id) { return "lease_already_bound"; }
                }
            }
        }
        if (required[0] > covered[0] || required[1] > covered[1]) { return "uncovered"; }
        records.emplace(r.id, r);
        return "none";
    }
    std::string claim(const std::string & id, const std::string & subject) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = records.find(id);
        if (it == records.end() || it->second.revoked) { return "unknown_residency"; }
        auto & r = it->second;
        if (r.subject != subject) { return "subject_mismatch"; }
        if (r.pending || r.live) { return "busy"; }
        r.pending = true;
        return "none";
    }
    // Validate the entire measured optimizer + engine workspace plan before the
    // caller can mutate the context. A failed second binding must not claim the first.
    // Live bindings are reused, not charged again; only new claims enter pending.
    std::string claim_plan(const std::vector<server_residency_need> & plan,
                           std::vector<std::string> & acquired,
                           const std::string & previous_workspace = {}) {
        std::lock_guard<std::mutex> lock(mutex);
        acquired.clear();
        if (plan.empty()) { return "invalid"; }
        std::set<std::string> ids;
        std::string owner;
        for (const auto & need : plan) {
            if (!ids.insert(need.id).second || need.rows.empty()) { return "invalid"; }
            const auto it = records.find(need.id);
            if (it == records.end() || it->second.revoked) { return "unknown_residency"; }
            const auto & record = it->second;
            if (!owner.empty() && owner != record.owner) { return "owner_mismatch"; }
            owner = record.owner;
            if (record.subject != need.subject) { return "subject_mismatch"; }
            if (record.pending) { return "busy"; }
            std::set<std::tuple<std::string, std::string, std::string>> keys;
            for (const auto & row : need.rows) {
                if (!keys.emplace(row.device, row.buffer_type, row.kind).second) { return "invalid"; }
                bool covered = false;
                for (const auto & allowed : record.rows) {
                    covered |= allowed.device == row.device && allowed.buffer_type == row.buffer_type &&
                        allowed.kind == row.kind && allowed.host == row.host && row.bytes <= allowed.bytes;
                }
                if (!covered) { return "uncovered"; }
            }
        }
        // Transfer retained workspace custody only as part of this successful whole
        // plan. The replacement covers every old row, not merely the next graph.
        // Nothing is freed: the new physical lease assumes the existing allocation.
        server_residency * previous = nullptr;
        server_residency * replacement = nullptr;
        if (!previous_workspace.empty()) {
            const auto it = records.find(previous_workspace);
            if (it == records.end()) { return "unknown_residency"; }
            previous = &it->second;
            if (previous->subject != "workspace" || previous->owner != owner) { return "owner_mismatch"; }
            if (previous->pending) { return "busy"; }
            for (const auto & need : plan) {
                if (need.subject == "workspace") {
                    if (replacement) { return "invalid"; }
                    replacement = &records.at(need.id);
                }
            }
            if (!replacement || replacement == previous || replacement->live) { return "invalid"; }
            for (const auto & row : previous->rows) {
                bool covered = false;
                for (const auto & allowed : replacement->rows) {
                    covered |= allowed.device == row.device && allowed.buffer_type == row.buffer_type &&
                        allowed.kind == row.kind && allowed.host == row.host && row.bytes <= allowed.bytes;
                }
                if (!covered) { return "uncovered"; }
            }
        }
        if (previous && previous->live) {
            replacement->live = true;
            previous->live = false;
        }
        for (const auto & need : plan) {
            auto & record = records.at(need.id);
            if (!record.live) {
                record.pending = true;
                acquired.push_back(need.id);
            }
        }
        return "none";
    }
    bool allows(const std::string & id, const server_residency_row & need, bool restoring = false) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = records.find(id);
        if (it == records.end() || (it->second.revoked && !restoring) || (!it->second.pending && !it->second.live)) { return false; }
        for (const auto & row : it->second.rows) {
            if (row.device == need.device && row.buffer_type == need.buffer_type && row.kind == need.kind &&
                    row.host == need.host && need.bytes <= row.bytes) { return true; }
        }
        return false;
    }
    void finish(const std::string & id, bool allocated, int32_t lora_id = -1) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = records.find(id);
        if (it == records.end() || !it->second.pending) { return; }
        it->second.pending = false;
        it->second.live = allocated;
        it->second.lora_id = allocated ? lora_id : -1;
    }
    // Called only by the allocation owner after buffers have actually been freed.
    void freed(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = records.find(id);
        if (it != records.end() && !it->second.pending) { it->second.live = false; it->second.lora_id = -1; }
    }
    void freed_lora(int32_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & entry : records) {
            if (entry.second.lora_id == id && !entry.second.pending) {
                entry.second.live = false;
                entry.second.lora_id = -1;
            }
        }
    }
    std::string revoke(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = records.find(id);
        if (it == records.end()) { return "unknown_residency"; }
        it->second.revoked = true;
        return "none";
    }
    std::string release(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = records.find(id);
        if (it == records.end()) { return retired.count(id) ? "none" : "unknown_residency"; }
        if (it->second.pending || it->second.live) { return "busy"; }
        retired.insert(id);
        records.erase(it);
        return "none";
    }
    std::vector<server_residency> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<server_residency> result;
        for (const auto & entry : records) { result.push_back(entry.second); }
        return result;
    }
};
