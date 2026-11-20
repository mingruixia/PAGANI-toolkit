#pragma once

#include <atomic>
#include <filesystem>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

// The temporary file is on the same volume. Concurrent CLI runs publish whole files.
template<class Writer>
void bc_replace_file(const std::filesystem::path& path, Writer write) {
    static std::atomic<unsigned long long> serial{0};
#ifdef _WIN32
    const auto pid = _getpid();
#else
    const auto pid = getpid();
#endif
    auto tmp = path;
    tmp += ".tmp." + std::to_string(pid) + "." + std::to_string(serial++);
    try {
        write(tmp);
        std::filesystem::rename(tmp, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        throw;
    }
}
