// Louvain_Modularity.exe / Newman_Modularity_CPU.exe / CUmodularity.exe
//   tool dir_for_csr [num_of_random_networks]
// Outputs: <network>_modu.nm  module index per node (1..M, 0 = isolated node)
//          <network>_modu.txt Q and number of modules for the network and the random networks
#include <fstream>

#include "app_util.hpp"
#include "pagani/metrics.hpp"
#include "pagani/random_network.hpp"
#ifdef PAGANI_GPU_SPMV
#include "pagani/gpu.hpp"
#endif

#ifndef PAGANI_MODULARITY_TOOL
#define PAGANI_MODULARITY_TOOL "Louvain_Modularity"
#endif

using namespace pagani;

static Partition detect(const Csr& g, std::mt19937_64& rng) {
#if defined(PAGANI_NEWMAN) && defined(PAGANI_GPU_SPMV)
    (void)rng;
    gpu::SpMV spmv(g, kWeighted);
    return newman_eigenvector(g, kWeighted, [&](const std::vector<double>& x, std::vector<double>& y) { spmv.multiply(x, y); });
#elif defined(PAGANI_NEWMAN)
    (void)rng;
    return newman_eigenvector(g, kWeighted);
#else
    return louvain(g, kWeighted, rng);
#endif
}

int main(int argc, char** argv) {
    return run_tool(PAGANI_MODULARITY_TOOL, argc, argv, [](int argc, char** argv) {
        if (argc != 2 && argc != 3)
            throw UsageError(std::string(PAGANI_MODULARITY_TOOL) + " dir_for_csr [num_of_random_networks]\n"
                             "  e.g. " PAGANI_MODULARITY_TOOL " d:\\data\\unweighted 10");
        const std::string dir = argv[1];
        const int nrand = argc == 3 ? parse_random_count(argv[2]) : 0;
        RandomNetworkGenerator gen;
        for (const auto& f : csr_files(dir)) {
            const std::string path = path_join(dir, f);
            const std::string stem = path_without_extension(path);
            std::cout << "\nmodular analysis for " << path << " ..." << std::endl;
            Timer t;
            Csr g = load_network(path);
            Partition p = detect(g, gen.engine());
            std::cout << "Q = " << fmt6(p.q) << ", modules: " << p.num_modules << "  (" << fmt6(t.seconds()) << " s)"
                      << std::endl;
            std::vector<float> lab(p.module.begin(), p.module.end());
            write_nm(stem + "_modu.nm", lab);
            std::ofstream o(stem + "_modu.txt");
            o << "Algorithm : " << PAGANI_MODULARITY_TOOL << (kWeighted ? " (weighted)" : " (unweighted)") << "\n";
            o << "Final Modularity Q = " << fmt6(p.q) << "\n";
            o << "Final number of modules : " << p.num_modules << "\n";
            if (nrand > 0) {
                o << "\n************************* Modularity of Random Networks *************************\n\n";
                double mean = 0;
                for_each_random_network(g, nrand, random_networks_seed(g), [&](int r, const Csr& rg) {
                    Partition pr = detect(rg, gen.engine());
                    mean += pr.q;
                    o << "Random Network " << r + 1 << ":\n";
                    o << "Final Modularity Q = " << fmt6(pr.q) << "\n";
                    o << "Final number of modules : " << pr.num_modules << "\n\n";
                });
                o << "Mean Q of random networks = " << fmt6(mean / nrand) << "\n";
            }
            if (!o) throw Error("cannot write " + stem + "_modu.txt");
        }
        return 0;
    });
}
