// PAGANI Toolkit - common utilities
// Error handling, filesystem helpers, simple timer and thread helpers.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace pagani {

constexpr const char* kVersion = "2.1.0";

// All recoverable errors are reported through this exception; tool entry
// points catch it, print the message and return a non-zero exit code.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

// Wrap a tool's main body: prints banner, catches exceptions, maps them to
// exit codes (0 ok, 1 error, 2 usage error).
int run_tool(const char* tool_name, int argc, char** argv,
             const std::function<int(int, char**)>& body);

// Thrown for command line usage errors (exit code 2).
class UsageError : public Error {
public:
    explicit UsageError(const std::string& msg) : Error(msg) {}
};

// ---------------------------------------------------------------- paths --
// Remove trailing path separators (keeps a bare drive root such as "C:\").
std::string strip_trailing_separator(const std::string& path);
// Join directory and file name using the native separator.
std::string path_join(const std::string& dir, const std::string& name);
// File name without directory.
std::string path_filename(const std::string& path);
// Path with the last extension removed ("a/b.c.nii" -> "a/b.c").
std::string path_without_extension(const std::string& path);
bool is_directory(const std::string& path);
bool file_exists(const std::string& path);
void ensure_directory(const std::string& path);
std::uint64_t file_size(const std::string& path);

// List regular files in `dir` whose name ends with `suffix` (case-sensitive,
// like PAGANI 2.0). The result is sorted to make processing order
// deterministic across platforms.
std::vector<std::string> list_files_with_suffix(const std::string& dir, const std::string& suffix);

// Same, case-insensitive suffix match.
std::vector<std::string> list_files_with_suffix_ci(const std::string& dir, const std::string& suffix);

bool ends_with(const std::string& s, const std::string& suffix);
bool iequals(const std::string& a, const std::string& b);
std::string to_lower(std::string s);

// ---------------------------------------------------------- argument parsing
// Strict numeric parsing: the whole string must be a valid number.
double parse_double(const std::string& s, const char* what);
long long parse_int(const std::string& s, const char* what);

// ------------------------------------------------------------------ timer --
class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    void reset() { start_ = std::chrono::steady_clock::now(); }
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_;
};

// ---------------------------------------------------------------- threads --
unsigned hardware_threads();
// Run fn(i) for i in [begin, end) on all hardware threads using a static
// interleaved schedule (i = begin + tid, begin + tid + nthreads, ...).
// fn receives (index, thread_id).
void parallel_for(long long begin, long long end,
                  const std::function<void(long long, unsigned)>& fn,
                  unsigned nthreads = 0);

}  // namespace pagani
