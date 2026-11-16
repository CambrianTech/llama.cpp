#include "server-mapped-file.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

server_mapped_file::server_mapped_file(const std::string & path) {
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        return;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart == 0) {
        CloseHandle(f);
        return;
    }
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) {
        CloseHandle(f);
        return;
    }
    void * p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (p == nullptr) {
        CloseHandle(m);
        CloseHandle(f);
        return;
    }
    data_ = static_cast<const uint8_t *>(p);
    size_ = (size_t) sz.QuadPart;
    file_ = (intptr_t) f;
    map_  = (intptr_t) m;
}

server_mapped_file::~server_mapped_file() {
    if (data_ != nullptr) {
        UnmapViewOfFile(data_);
        CloseHandle((HANDLE) map_);
        CloseHandle((HANDLE) file_);
    }
}

#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

server_mapped_file::server_mapped_file(const std::string & path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return;
    }
    void * p = ::mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        return;
    }
    ::madvise(p, (size_t) st.st_size, MADV_SEQUENTIAL | MADV_WILLNEED);
    data_ = static_cast<const uint8_t *>(p);
    size_ = (size_t) st.st_size;
    file_ = fd;
}

server_mapped_file::~server_mapped_file() {
    if (data_ != nullptr) {
        ::munmap(const_cast<uint8_t *>(data_), size_);
        ::close((int) file_);
    }
}
#endif
