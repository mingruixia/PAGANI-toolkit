// LocalEfficiency dir_for_csr num_of_random_networks flags(g|n|r)
#include <fstream>
#include <limits>
#include <numeric>

#include "app_util.hpp"
#include "pagani/metrics.hpp"
#include "pagani/random_network.hpp"

using namespace pagani;

int main(int argc, char** argv) {
    return run_tool("LocalEfficiency", argc, argv, [](int argc, char** argv) {
        const bool normalize_weights = kWeighted && argc == 5 &&
            std::string(argv[4]) == "--normalize-weights";
        if (argc != 4 && !normalize_weights) {
            const std::string usage = "LocalEfficiency dir_for_csr num_of_random_networks flags (combination of g, n, r)";
            throw UsageError(kWeighted ? usage + " [--normalize-weights]" : usage);
        }
        const auto calculate = [normalize_weights](const Csr& g, std::vector<float>& nodal) {
            return kWeighted ? local_efficiency_weighted(g, nodal, normalize_weights)
                             : local_efficiency_binary(g, nodal);
        };
        const std::string dir = argv[1];
        const int nrand = parse_random_count(argv[2]);
        const auto flags = parse_flags(argv[3], "gnr", "output flags");
        const bool want_g = flags.count('g');
        const bool want_n = flags.count('n');
        const bool want_r = flags.count('r');
        if (want_r && nrand == 0)
            throw UsageError("normalization (r) requires at least one random network");

        if (kWeighted)
            std::cout << "Weighted local efficiency: BCT local=2, "
                      << (normalize_weights ? "weights divided by the network maximum" : "original weights")
                      << std::endl;

        for (const auto& file : csr_files(dir)) {
            const auto path = path_join(dir, file);
            const auto stem = path_without_extension(path);
            const Csr g = load_network(path);
            check_symmetric(g, path);

            Timer timer;
            std::vector<float> nodal;
            const double observed = calculate(g, nodal);
            std::cout << path << ": mean local efficiency = " << fmt6(observed)
                      << " (" << fmt6(timer.seconds()) << " s)" << std::endl;

            std::vector<double> random;
            if (nrand > 0 && (want_g || want_r)) {
                std::cout << "calculating local efficiency for " << nrand
                          << " random network(s) ..." << std::endl;
                std::vector<float> scratch;
                for_each_random_network(g, nrand, random_networks_seed(g),
                    [&](int, const Csr& rg) {
                        random.push_back(calculate(rg, scratch));
                    });
            }
            if (want_n) write_nm(stem + "_leff.nm", nodal);
            if (want_g) {
                const auto output = stem + "_leff.txt";
                std::ofstream out(output);
                out << fmt6(observed) << '\n';
                for (double value : random) out << fmt6(value) << '\n';
                if (!out) throw Error("cannot write " + output);
            }
            if (want_r) {
                const double mean = std::accumulate(random.begin(), random.end(), 0.0) / random.size();
                const double ratio = mean > 0.0 ? observed / mean : std::numeric_limits<double>::quiet_NaN();
                const auto output = stem + "_leff_norm.txt";
                std::ofstream out(output);
                out << fmt6(ratio) << '\n';
                if (!out) throw Error("cannot write " + output);
                if (mean == 0.0)
                    std::cout << "Warning: random-network mean is zero; normalized local efficiency is undefined."
                              << std::endl;
            }
        }
        return 0;
    });
}
