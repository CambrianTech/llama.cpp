#include "server-page-os.h"

#include <cerrno>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

namespace page_os {

long current_pid() {
    return (long) getpid();
}

bool process_alive(long pid) {
    return kill((pid_t) pid, 0) == 0 || errno == EPERM;
}

static bool fsync_path(const std::filesystem::path & path, bool full) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
#if defined(__APPLE__)
    // macOS fsync stops at the drive's cache; only F_FULLFSYNC orders the data before what follows
    bool ok = (full && fcntl(fd, F_FULLFSYNC) == 0) || fsync(fd) == 0;
#else
    (void) full;
    bool ok = fsync(fd) == 0;
#endif
    return close(fd) == 0 && ok;
}

bool sync_file(const std::filesystem::path & path) {
    return fsync_path(path, true);
}

bool sync_dir(const std::filesystem::path & dir) {
    return fsync_path(dir, false);
}

std::error_code place(const std::filesystem::path & from, const std::filesystem::path & to) {
    std::error_code ec;
    std::filesystem::rename(from, to, ec);
    return ec;
}

}
