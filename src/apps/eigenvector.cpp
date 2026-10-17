// CUEC.exe dir_for_csr
// Eigenvector centrality (leading eigenvector, unit L2 norm, non-negative).
// PAGANI 2.0 wrote the eigenvector multiplied by the eigenvalue (audit E-01).
// Outputs: <network>_ec.nm, <network>_evalue.txt (leading eigenvalue)
#include <fstream>

#include "app_util.hpp"
#include "pagani/metrics.hpp"
#ifndef PAGANI_CPU_BACKEND
#include "pagani/gpu.hpp"
#endif

using namespace pagani;

int main(int argc, char** argv) {
    return run_tool("CUEC", argc, argv, [](int argc, char** argv) {
        if (argc != 2) throw UsageError("CUEC.exe dir_for_csr");
        const std::string dir = argv[1];
        for (const auto& f : csr_files(dir)) {
            const std::string path = path_join(dir, f);
            const std::string stem = path_without_extension(path);
            std::cout << "\ncalculating eigenvector centrality for " << path << " ..." << std::endl;
            Timer t;
            Csr g = load_network(path);
#ifdef PAGANI_CPU_BACKEND
            EigenResult r = eigenvector_centrality(g, kWeighted);
#else
            EigenResult r = gpu::eigenvector_centrality(g, kWeighted);
#endif
            if (!r.converged) std::cout << "warning: power iteration did not converge" << std::endl;
            std::cout << "eigenvalue " << fmt6(r.value) << " after " << r.iterations << " iterations  ("
                      << fmt6(t.seconds()) << " s)" << std::endl;
            write_nm(stem + "_ec.nm", std::vector<float>(r.vector.begin(), r.vector.end()));
            std::ofstream o(stem + "_evalue.txt");
            o << fmt6(r.value) << "\n";
            if (!o) throw Error("cannot write " + stem + "_evalue.txt");
        }
        return 0;
    });
}
