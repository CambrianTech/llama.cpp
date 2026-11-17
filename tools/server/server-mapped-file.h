#pragma once

// Continuum fork: a file mapped read-only, so its bytes are inserted straight into a sequence (flash -> unified
// memory at the drive's speed, no read into a buffer first). The OS calls live in server-mapped-file.cpp.

#include <cstddef>
#include <cstdint>
#include <string>

class server_mapped_file {
public:
    explicit server_mapped_file(const std::string & path);
    ~server_mapped_file();
    server_mapped_file(const server_mapped_file &)             = delete;
    server_mapped_file & operator=(const server_mapped_file &) = delete;

    bool            ok()   const { return data_ != nullptr; }
    const uint8_t * data() const { return data_; }
    size_t          size() const { return size_; }

private:
    const uint8_t * data_ = nullptr;
    size_t          size_ = 0;
    intptr_t        file_ = -1; // a POSIX descriptor or a Windows file HANDLE
#if defined(_WIN32)
    intptr_t        map_  = -1; // the mapping HANDLE (Windows maps through a second handle)
#endif
};
