#pragma once

#include <filesystem>
#include <system_error>

// The OS calls page publication needs. One signature each; each OS supplies only its own lines
// (server-page-os-posix.cpp, server-page-os-win32.cpp). The publication logic never branches on the OS.
namespace page_os {

long current_pid();

// whether the process with this pid still runs
bool process_alive(long pid);

// makes a file's contents durable before anything that depends on them
bool sync_file(const std::filesystem::path & path);

// makes a dir's entries durable: every rename into it that place() made
bool sync_dir(const std::filesystem::path & dir);

// moves `from` onto `to`, replacing it; durable once sync_dir of its dir returns
std::error_code place(const std::filesystem::path & from, const std::filesystem::path & to);

}
