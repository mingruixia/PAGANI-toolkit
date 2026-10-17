// CUBC.exe dir_for_csr
// Betweenness centrality of unweighted networks (Brandes; every ordered
// node pair counted, as in PAGANI 2.0 and BCT betweenness_bin).
// Output: <network>_bc.nm
#include "app_util.hpp"
#ifdef PAGANI_CPU_BACKEND
#include "pagani/metrics.hpp"
#else
#include "pagani/gpu.hpp"
#endif

using namespace pagani;

int main(int argc, char** argv) {
    return run_tool("CUBC", argc, argv, [](int argc, char** argv) {
        if (argc != 2) throw UsageError("CUBC.exe dir_for_csr");
        const std::string dir = argv[1];
        for (const auto& f : csr_files(dir)) {
            const std::string path = path_join(dir, f);
            std::cout << "\ncalculating betweenness for " << path << " ..." << std::endl;
            Timer t;
            Csr g = read_csr(path, CsrWeights::Ignore);
#ifdef PAGANI_CPU_BACKEND
            auto bc = betweenness_binary(g);
#else
            auto bc = gpu::betweenness_binary(g);
#endif
            write_nm(path_without_extension(path) + "_bc.nm", std::vector<float>(bc.begin(), bc.end()));
            std::cout << "done in " << fmt6(t.seconds()) << " s" << std::endl;
        }
        return 0;
    });
}
