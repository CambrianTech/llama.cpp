// The disk tier's only platform-specific lines (server-spill-dir.h): an exclusive file lock and the process
// id. Logic lives in the header; this file holds just the OS calls.
#include "server-spill-dir.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

bool server_owner_lock::try_take(const std::filesystem::path & path, bool create) {
    release();
#if defined(_WIN32)
    // no read/write sharing: a second open of a file another process holds open fails (delete sharing
    // lets the owner rename its own marker)
    HANDLE h = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_DELETE, nullptr,
                           create ? CREATE_NEW : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    handle = (intptr_t) h;
    return true;
#else
    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC | (create ? (O_CREAT | O_EXCL) : 0), 0600);
    if (fd < 0) {
        return false;
    }
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd);
        return false;
    }
    handle = fd;
    return true;
#endif
}

void server_owner_lock::release() {
    if (handle == -1) {
        return;
    }
#if defined(_WIN32)
    CloseHandle((HANDLE) handle);
#else
    ::close((int) handle); // closing releases the flock
#endif
    handle = -1;
}

uint64_t server_process_id() {
#if defined(_WIN32)
    return (uint64_t) GetCurrentProcessId();
#else
    return (uint64_t) ::getpid();
#endif
}
