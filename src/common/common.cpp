#include "pagani/common.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <thread>

namespace fs = std::filesystem;

namespace pagani {

const char* const kVersion = PAGANI_TOOLKIT_VERSION;


int run_tool(const char* tool_name, int argc, char** argv,
             const std::function<int(int, char**)>& body) {
    std::cout << "PAGANI Toolkit " << kVersion << " - " << tool_name << std::endl;
    try {
        return body(argc, argv);
    } catch (const UsageError& e) {
        std::cerr << "\n[" << tool_name << "] usage error: " << e.what() << std::endl;
        return 2;
    } catch (const Error& e) {
        std::cerr << "\n[" << tool_name << "] ERROR: " << e.what() << std::endl;
        return 1;
    } catch (const std::bad_alloc&) {
        std::cerr << "\n[" << tool_name << "] ERROR: out of host memory." << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n[" << tool_name << "] ERROR: " << e.what() << std::endl;
        return 1;
    }
}

std::string strip_trailing_separator(const std::string& path) {
    std::string p = path;
    while (p.size() > 1 && (p.back() == '\\' || p.back() == '/')) {
        if (p.size() == 3 && p[1] == ':') break;  // "C:\"
        p.pop_back();
    }
    return p;
}

std::string path_join(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    std::string d = strip_trailing_separator(dir);
    char last = d.back();
    if (last == '\\' || last == '/') return d + name;
#ifdef _WIN32
    return d + "\\" + name;
#else
    return d + "/" + name;
#endif
}

std::string path_filename(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string path_without_extension(const std::string& path) {
    auto sep = path.find_last_of("/\\");
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos || (sep != std::string::npos && dot < sep)) return path;
    return path.substr(0, dot);
}

bool is_directory(const std::string& path) {
    std::error_code ec;
    return fs::is_directory(fs::path(path), ec);
}

bool file_exists(const std::string& path) {
    std::error_code ec;
    return fs::is_regular_file(fs::path(path), ec);
}

void ensure_directory(const std::string& path) {
    std::error_code ec;
    if (fs::is_directory(fs::path(path), ec)) return;
    fs::create_directories(fs::path(path), ec);
    if (ec || !fs::is_directory(fs::path(path)))
        throw Error("cannot create directory '" + path + "': " + ec.message());
}

std::uint64_t file_size(const std::string& path) {
    std::error_code ec;
    auto s = fs::file_size(fs::path(path), ec);
    if (ec) throw Error("cannot get size of '" + path + "': " + ec.message());
    return s;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string to_lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool iequals(const std::string& a, const std::string& b) { return to_lower(a) == to_lower(b); }

static std::vector<std::string> list_files_impl(const std::string& dir, const std::string& suffix, bool ci) {
    std::error_code ec;
    fs::path p(dir);
    if (!fs::is_directory(p, ec)) throw Error("cannot open directory '" + dir + "'");
    std::vector<std::string> out;
    const std::string lsuffix = to_lower(suffix);
    for (fs::directory_iterator it(p, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code ec2;
        if (!it->is_regular_file(ec2)) continue;
        std::string name = it->path().filename().string();
        if (name.size() <= suffix.size()) continue;  // need a non-empty stem
        bool match = ci ? ends_with(to_lower(name), lsuffix) : ends_with(name, suffix);
        if (match) out.push_back(name);
    }
    if (ec) throw Error("error while listing directory '" + dir + "': " + ec.message());
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::string> list_files_with_suffix(const std::string& dir, const std::string& suffix) {
    return list_files_impl(dir, suffix, false);
}

std::vector<std::string> list_files_with_suffix_ci(const std::string& dir, const std::string& suffix) {
    return list_files_impl(dir, suffix, true);
}

double parse_double(const std::string& s, const char* what) {
    const char* b = s.c_str();
    char* e = nullptr;
    errno = 0;
    double v = std::strtod(b, &e);
    if (s.empty() || e == b || *e != '\0' || errno == ERANGE)
        throw UsageError(std::string("invalid number for ") + what + ": '" + s + "'");
    return v;
}

long long parse_int(const std::string& s, const char* what) {
    const char* b = s.c_str();
    char* e = nullptr;
    errno = 0;
    long long v = std::strtoll(b, &e, 10);
    if (s.empty() || e == b || *e != '\0' || errno == ERANGE)
        throw UsageError(std::string("invalid integer for ") + what + ": '" + s + "'");
    return v;
}

unsigned hardware_threads() {
    unsigned n = std::thread::hardware_concurrency();
    if (const char* env = std::getenv("PAGANI_THREADS")) {
        long v = std::strtol(env, nullptr, 10);
        if (v > 0) n = static_cast<unsigned>(v);
    }
    return n == 0 ? 1 : n;
}

namespace {

// Persistent worker pool: the graph algorithms call parallel_for thousands of
// times on small ranges (e.g. every power iteration), where creating threads
// for each call costs more than the work itself (especially on Windows).
class ThreadPool {
public:
    explicit ThreadPool(unsigned n) : size_(n) {
        for (unsigned i = 1; i < n; ++i) workers_.emplace_back([this, i] { loop(i); });
    }
    unsigned size() const { return size_; }

    // runs job(t) for t = 0..nt-1, job(0) on the calling thread
    void run(unsigned nt, const std::function<void(unsigned)>& job) {
        std::lock_guard<std::mutex> region(region_);  // one parallel region at a time
        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = &job;
            nt_ = nt;
            pending_ = nt - 1;
            ++generation_;
        }
        cv_.notify_all();
        in_worker() = true;
        job(0);
        in_worker() = false;
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [this] { return pending_ == 0; });
        job_ = nullptr;
    }
    static bool& in_worker() {
        thread_local bool flag = false;
        return flag;
    }

private:
    void loop(unsigned index) {
        in_worker() = true;
        unsigned long long seen = 0;
        for (;;) {
            const std::function<void(unsigned)>* job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return generation_ != seen; });
                seen = generation_;
                if (index >= nt_) continue;
                job = job_;
            }
            (*job)(index);
            std::lock_guard<std::mutex> lk(m_);
            if (--pending_ == 0) done_.notify_one();
        }
    }
    unsigned size_;
    std::vector<std::thread> workers_;
    std::mutex region_, m_;
    std::condition_variable cv_, done_;
    const std::function<void(unsigned)>* job_ = nullptr;
    unsigned nt_ = 0, pending_ = 0;
    unsigned long long generation_ = 0;
};

ThreadPool& pool() {
    // intentionally never destroyed: the workers end with the process
    static ThreadPool* p = new ThreadPool(hardware_threads());
    return *p;
}

}  // namespace

void parallel_for(long long begin, long long end,
                  const std::function<void(long long, unsigned)>& fn, unsigned nthreads) {
    if (end <= begin) return;
    const long long count = end - begin;
    if (nthreads == 0) nthreads = hardware_threads();
    if (static_cast<long long>(nthreads) > count) nthreads = static_cast<unsigned>(count);
    // nested calls (from inside a parallel region) run serially
    if (nthreads <= 1 || ThreadPool::in_worker()) {
        for (long long i = begin; i < end; ++i) fn(i, 0);
        return;
    }
    ThreadPool& tp = pool();
    nthreads = std::min(nthreads, tp.size());
    if (nthreads <= 1) {
        for (long long i = begin; i < end; ++i) fn(i, 0);
        return;
    }
    // static round-robin schedule of small chunks: chunk k goes to thread k % nthreads.
    // Balances uneven work and keeps the assignment (and so any per-thread
    // partial sums) identical from run to run.
    const long long chunk = std::max<long long>(1, count / (static_cast<long long>(nthreads) * 16));
    std::vector<std::exception_ptr> errors(nthreads);
    std::atomic<bool> failed(false);
    tp.run(nthreads, [&](unsigned t) {
        try {
            for (long long b = begin + t * chunk; b < end; b += chunk * nthreads) {
                if (failed.load(std::memory_order_relaxed)) return;
                const long long e = std::min(end, b + chunk);
                for (long long i = b; i < e; ++i) fn(i, t);
            }
        } catch (...) {
            errors[t] = std::current_exception();
            failed = true;
        }
    });
    for (auto& e : errors)
        if (e) std::rethrow_exception(e);
}

}  // namespace pagani
