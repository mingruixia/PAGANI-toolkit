// Shared driver for the characteristic path length tools
// (CUBFW_Lp.exe on GPU, BFS_MulCPU.exe on CPU).
//   tool dir_for_csr num_of_random_networks flags
//   flags: any combination of g (global Lp), n (nodal efficiency),
//          l (lambda = Lp / Lp_random); "b" is accepted as "gn" (PAGANI 1.x)
// Outputs: <network>_eff.nm, <network>_Lp.txt, <network>_lambda.txt
#pragma once

#include <fstream>
#include <functional>
#include <limits>

#include "app_util.hpp"
#include "pagani/metrics.hpp"
#include "pagani/random_network.hpp"

namespace pagani {

using EfficiencyFn = std::function<EfficiencyResult(const Csr&)>;

inline int lp_main(const char* tool, int argc, char** argv, const EfficiencyFn& compute) {
    return run_tool(tool, argc, argv, [&](int argc, char** argv) {
        if (argc != 4)
            throw UsageError(std::string(tool) + " dir_for_csr num_of_random_networks flags(g|n|l combination)\n"
                             "  e.g. " + tool + " d:\\data\\unweighted 10 gnl");
        const std::string dir = argv[1];
        const int nrand = parse_random_count(argv[2]);
        std::string fl = argv[3];
        if (fl == "b") fl = "gn";
        auto flags = parse_flags(fl, "gnl", "output flags");
        const bool want_g = flags.count('g'), want_n = flags.count('n'), want_l = flags.count('l');
        if (want_l && nrand == 0) throw UsageError("lambda (l) requires at least one random network");

        RandomNetworkGenerator gen;
        for (const auto& f : csr_files(dir)) {
            const std::string path = path_join(dir, f);
            const std::string stem = path_without_extension(path);
            std::cout << "\ncalculating Lp for " << path << " ..." << std::endl;
            Timer t;
            Csr g = load_network(path);
            EfficiencyResult r = compute(g);
            std::cout << "average Lp: " << fmt6(r.lp) << "   global efficiency: " << fmt6(r.global_efficiency)
                      << "  (" << fmt6(t.seconds()) << " s)" << std::endl;
            if (r.isolated) std::cout << "isolated nodes: " << r.isolated << std::endl;
            if (!r.connected) std::cout << "note: the network is not connected (Lp computed as harmonic mean)" << std::endl;
            if (want_n) write_nm(stem + "_eff.nm", r.nodal);

            std::vector<double> rand_lp;
            if (nrand > 0 && (want_g || want_l)) {
                std::cout << "calculating Lp for " << nrand << " random network(s) ..." << std::endl;
                for_each_random_network(g, nrand, gen.engine()(),
                                        [&](int, const Csr& rg) { rand_lp.push_back(compute(rg).lp); });
            }
            if (want_g) {
                std::ofstream o(stem + "_Lp.txt");
                o << fmt6(r.lp) << "\n";
                for (double v : rand_lp) o << fmt6(v) << "\n";
                if (!o) throw Error("cannot write " + stem + "_Lp.txt");
            }
            if (want_l) {
                double mean = 0;
                for (double v : rand_lp) mean += v;
                mean /= rand_lp.size();
                double lambda = r.lp / mean;
                std::ofstream o(stem + "_lambda.txt");
                o << fmt6(lambda) << "\n";
                if (!o) throw Error("cannot write " + stem + "_lambda.txt");
                std::cout << "lambda = " << fmt6(lambda) << std::endl;
            }
        }
        return 0;
    });
}

}  // namespace pagani
