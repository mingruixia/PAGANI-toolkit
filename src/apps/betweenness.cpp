// Weighted GUI calls add --backend and --threads. Legacy directory invocations
// keep their original backend and thread settings.
#include "app_util.hpp"
#include "bc_atomic.hpp"
#include <cstdlib>
#include <algorithm>
#ifdef PAGANI_CPU_BACKEND
#include "pagani/metrics.hpp"
#else
#include "pagani/gpu.hpp"
#endif
#if defined(PAGANI_WEIGHTED) && !defined(PAGANI_CPU_BACKEND)
#include "bc_dispatch.hpp"
#endif

using namespace pagani;

int main(int argc, char** argv) {
#ifdef PAGANI_CPU_BACKEND
    constexpr const char* name = "CUBC";
    std::string mode = "cpu";
#else
    constexpr const char* name = kWeighted ? "CUBC_GPU" : "CUBC";
    std::string mode = "gpu";
#endif
    return run_tool(name, argc, argv, [&](int argc, char** argv) {
        if (argc != 2) {
            if (!kWeighted || (argc != 4 && argc != 6) || std::string(argv[2]) != "--backend")
                throw UsageError(std::string(name) + " dir_for_csr [--backend auto|cpu|gpu|calibrate [--threads N]]");
            mode = argv[3];
            if (mode != "auto" && mode != "cpu" && mode != "gpu" && mode != "calibrate")
                throw UsageError("invalid weighted BC backend");
#ifdef PAGANI_CPU_BACKEND
            if (mode != "cpu" && mode != "auto") throw Error("GPU executable is required for " + mode);
#endif
            if (argc == 6) {
                if (std::string(argv[4]) != "--threads") throw UsageError("expected --threads N");
                auto n = parse_int(argv[5], "threads");
                if (n < 0 || n > 1024) throw UsageError("threads must be between 0 and 1024");
                if (n == 0 && !std::getenv("PAGANI_THREADS")) n = std::min(32u, hardware_threads());
                if (n) {
#ifdef _WIN32
                    _putenv_s("PAGANI_THREADS", std::to_string(n).c_str());
#else
                    setenv("PAGANI_THREADS", std::to_string(n).c_str(), 1);
#endif
                }
            }
        }
        const std::string dir = argv[1];
        for (const auto& f : csr_files(dir)) {
            const std::string path = path_join(dir, f);
            std::cout << "\ncalculating betweenness for " << path << " ..." << std::endl;
            Timer t;
            const Csr g = read_csr(path, kWeighted ? CsrWeights::Require : CsrWeights::Ignore);
#ifdef PAGANI_CPU_BACKEND
            if (kWeighted) std::cout << "backend=cpu reason=" << (mode=="auto" ? "CPU-only package" : "requested")
                << " nodes=" << g.n << " edges=" << g.nnz()/2 << " threads=" << hardware_threads() << std::endl;
            const auto bc = kWeighted ? betweenness_weighted(g) : betweenness_binary(g);
#elif defined(PAGANI_WEIGHTED)
            const auto bc = dispatch_weighted_bc(g, mode);
#else
            const auto bc = gpu::betweenness_binary(g);
#endif
            bc_replace_file(path_without_extension(path) + "_bc.nm", [&](const auto& tmp) {
                write_nm(tmp.string(), std::vector<float>(bc.begin(), bc.end()));
            });
            std::cout << "done in " << fmt6(t.seconds()) << " s" << std::endl;
        }
        return 0;
    });
}
