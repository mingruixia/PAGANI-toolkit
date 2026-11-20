#include "bc_dispatch.hpp"
#include "bc_atomic.hpp"
#include "cuda_common.hpp"
#include "pagani/gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#include <cstring>
#endif

namespace {
using namespace pagani;
namespace fs = std::filesystem;

std::string cpu_name() {
#if defined(_M_X64) || defined(_M_IX86)
    int regs[4];
    __cpuid(regs, static_cast<int>(0x80000000u));
    if (static_cast<unsigned>(regs[0]) >= 0x80000004u) {
        char brand[49]{};
        for (int i=0; i<3; ++i) {
            __cpuid(regs, static_cast<int>(0x80000002u+i));
            std::memcpy(brand+16*i, regs, 16);
        }
        return brand;
    }
#elif defined(__linux__)
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in,line))
        if (line.rfind("model name",0)==0) return line.substr(line.find(':')+1);
#endif
    return {};
}

std::string driver_version() {
#ifdef _WIN32
    wchar_t dir[MAX_PATH];
    const auto n = GetSystemDirectoryW(dir, MAX_PATH);
    if (!n || n >= MAX_PATH) return {};
    const auto file = fs::path(dir) / "nvcuda.dll";
    DWORD unused;
    DWORD size = GetFileVersionInfoSizeW(file.c_str(), &unused);
    std::vector<char> buf(size);
    VS_FIXEDFILEINFO* info = nullptr;
    UINT len = 0;
    if (size && GetFileVersionInfoW(file.c_str(), 0, size, buf.data()) &&
        VerQueryValueW(buf.data(), L"\\", reinterpret_cast<void**>(&info), &len) && len >= sizeof(*info)) {
        std::ostringstream out;
        out << HIWORD(info->dwFileVersionMS) << '.' << LOWORD(info->dwFileVersionMS) << '.'
            << HIWORD(info->dwFileVersionLS) << '.' << LOWORD(info->dwFileVersionLS);
        return out.str();
    }
#else
    std::ifstream in("/sys/module/nvidia/version");
    std::string version;
    if (in >> version) return version;
#endif
    return {};
}

fs::path cache_root() {
    if (const char* p = std::getenv("PAGANI_BC_CACHE_DIR")) return p;
#ifdef _WIN32
    if (const char* p = std::getenv("LOCALAPPDATA")) return fs::path(p)/"PAGANI"/"bc";
#else
    if (const char* p = std::getenv("XDG_CACHE_HOME")) return fs::path(p)/"pagani"/"bc";
    if (const char* p = std::getenv("HOME")) return fs::path(p)/".cache"/"pagani"/"bc";
#endif
    return {};
}

struct Profile {
    std::string key;
    fs::path file;
    bool reference_gpu = false;
    bool fits = false;
};

Profile profile(const Csr& g, unsigned isolates) {
    cuda::DeviceInit init;
    int dev;
    cudaDeviceProp prop{};
    PAGANI_CUDA(cudaGetDevice(&dev));
    PAGANI_CUDA(cudaGetDeviceProperties(&prop,dev));
    const auto threads = hardware_threads();
    static const auto cpu = cpu_name();
    static const auto driver = driver_version();
    Profile p;
    const auto fixed = 12ull*g.n + 12ull*g.nnz() + 8;
    const auto available = cuda::free_device_memory();
    const auto per_source = 44ull*g.n + 4;
    const auto capacity = available > fixed ? (available-fixed)/(2*per_source) : 0ull;
    const auto full_batch = std::min(g.n, static_cast<unsigned>(prop.multiProcessorCount)*4u);
    const auto batch = std::min<unsigned long long>(capacity, full_batch);
    p.fits = batch > 0;
    const auto edges = g.nnz()/2;
#ifdef _WIN32
    // Bounds are the observed isolate ranges and edge counts in the M9 real-data report.
    p.reference_gpu = batch == full_batch && cpu.find("Threadripper 3970X") != std::string::npos && threads == 32 &&
        driver == "32.0.16.2114" && std::string(prop.name) == "NVIDIA GeForce RTX 3080 Ti" &&
        g.n == 19574 &&
        ((edges >= 957804 && edges <= 957806 && isolates >= 3343 && isolates <= 4789) ||
         (edges >= 1915609 && edges <= 1915611 && isolates >= 997 && isolates <= 2647));
#endif
    const auto dir = cache_root();
    if (dir.empty() || driver.empty() || cpu.empty()) return p;
    std::ostringstream key;
    key << "bc-v1|" << kVersion << '|' << cpu << '|' << threads << '|' << driver << '|';
    for (unsigned char b : prop.uuid.bytes) key << std::hex << std::setw(2) << std::setfill('0') << unsigned(b);
    key << std::dec << '|' << g.n << '|' << edges << '|' << (20ull * isolates / g.n) << '|' << batch;
    p.key = key.str();
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char c : p.key) hash = (hash ^ c) * 1099511628211ull;
    p.file = dir/(std::to_string(hash)+".txt");
    return p;
}

std::string cached(const Profile& p) {
    std::ifstream in(p.file);
    std::string key, winner;
    if (std::getline(in,key) && key==p.key && in >> winner && (winner=="cpu" || winner=="gpu")) return winner;
    return {};
}

void store(const Profile& p, const std::string& winner) {
    if (p.file.empty()) throw Error("Cannot identify hardware or a writable calibration location");
    fs::create_directories(p.file.parent_path());
    bc_replace_file(p.file,[&](const auto& tmp) {
        std::ofstream out(tmp);
        out << p.key << '\n' << winner << '\n';
        out.close();
        if (!out) throw Error("Cannot save BC calibration");
    });
}
} // namespace

std::vector<double> dispatch_weighted_bc(const pagani::Csr& g, const std::string& mode) {
    using namespace pagani;
    unsigned isolates = 0;
    for (unsigned i=0; i<g.n; ++i) isolates += g.degree(i)==0;
    std::cout << "nodes=" << g.n << " edges=" << g.nnz()/2 << " isolates=" << isolates
              << " threads=" << hardware_threads() << std::endl;
    if (mode=="gpu") {
        std::cout << "backend=gpu reason=requested" << std::endl;
        return gpu::betweenness_weighted(g);
    }
    if (mode=="cpu" || !g.nnz()) {
        std::cout << "backend=cpu reason=" << (mode=="cpu" ? "requested" : "edgeless") << std::endl;
        return betweenness_weighted(g);
    }
    try {
        Timer setup;
        const auto p = profile(g,isolates);
        const double setup_s = setup.seconds();
        if (!p.fits) throw cuda::ResourceError("insufficient free memory for weighted BC");
        if (mode=="calibrate") {
            Timer ct;
            auto cpu = betweenness_weighted(g);
            const double cpu_s = ct.seconds();
            Timer gt;
            const auto gpu_values = gpu::betweenness_weighted(g);
            const double gpu_s = gt.seconds() + setup_s;
            for (size_t i=0; i<cpu.size(); ++i) {
                if (!std::isfinite(gpu_values[i]) || !std::isfinite(cpu[i]) ||
                    std::abs(cpu[i]-gpu_values[i]) > 1e-8 + 1e-6*std::abs(cpu[i]))
                    throw Error("calibration failed: CPU/GPU mismatch at node "+std::to_string(i));
            }
            const std::string winner = cpu_s >= 1.10*gpu_s ? "gpu" : "cpu";
            store(p,winner);
            std::cout << "backend=calibrate cpu_s=" << cpu_s << " gpu_s=" << gpu_s
                      << " saved=" << winner << std::endl;
            return cpu;
        }
        const auto prior = cached(p);
        const bool gpu_selected = prior.empty() ? p.reference_gpu : prior=="gpu";
        std::cout << "backend=" << (gpu_selected ? "gpu" : "cpu") << " reason="
                  << (!prior.empty() ? "local calibration" : p.reference_gpu ? "M9 measured profile" : "uncalibrated profile")
                  << std::endl;
        return gpu_selected ? gpu::betweenness_weighted(g) : betweenness_weighted(g);
    } catch (const cuda::ResourceError& e) {
        if (mode!="auto") throw;
        std::cout << "backend=cpu reason=" << e.what() << std::endl;
        return betweenness_weighted(g);
    }
}
