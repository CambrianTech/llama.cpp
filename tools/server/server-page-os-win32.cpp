#include "server-page-os.h"

#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>

namespace page_os {

long current_pid() {
    return (long) _getpid();
}

bool process_alive(long pid) {
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD) pid);
    if (h == nullptr) {
        return false;
    }
    const bool alive = WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
    CloseHandle(h);
    return alive;
}

bool sync_file(const std::filesystem::path & path) {
    const int fd = _wopen(path.c_str(), _O_RDWR | _O_BINARY);
    if (fd < 0) {
        return false;
    }
    const bool ok = _commit(fd) == 0;
    return _close(fd) == 0 && ok;
}

// Windows has no directory sync; every entry was already made durable by place()'s write-through move.
bool sync_dir(const std::filesystem::path &) {
    return true;
}

// "MOVEFILE_WRITE_THROUGH: the function does not return until the file is actually moved on the disk."
std::error_code place(const std::filesystem::path & from, const std::filesystem::path & to) {
    if (MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return {};
    }
    return std::error_code((int) GetLastError(), std::system_category());
}

}
