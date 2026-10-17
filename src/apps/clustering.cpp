// Cp.exe   (unweighted)  dir_for_csr num_of_random_networks flags
// Cp.exe   (weighted)    dir_for_csr num_of_random_networks [type] flags
//   flags: any combination of g (global Cp), n (nodal Cp), k (gamma = Cp / Cp_random)
//   type (weighted only): 1 = Barrat et al. 2004, 2 = Onnela et al. 2005 (default)
// Outputs: <network>_cp.nm, <network>_cp.txt, <network>_gamma.txt
#include <fstream>
#include <limits>

#include "app_util.hpp"
#include "pagani/metrics.hpp"
#include "pagani/random_network.hpp"

using namespace pagani;

int main(int argc, char** argv) {
    return run_tool("Cp", argc, argv, [](int argc, char** argv) {
        ClusteringType type = kWeighted ? ClusteringType::Onnela : ClusteringType::Binary;
        std::string flag_arg;
        if (!kWeighted) {
            if (argc != 4)
                throw UsageError("Cp.exe dir_for_csr num_of_random_networks flags(g|n|k combination)\n"
                                 "  e.g. Cp.exe d:\\data\\unweighted 10 gnk");
            flag_arg = argv[3];
        } else {
            if (argc == 4) {
                flag_arg = argv[3];
            } else if (argc == 5) {
                std::string t = argv[3];
                if (t == "1") type = ClusteringType::Barrat;
                else if (t == "2") type = ClusteringType::Onnela;
                else throw UsageError("Cp type must be 1 (Barrat et al. 2004) or 2 (Onnela et al. 2005)");
                flag_arg = argv[4];
            } else {
                throw UsageError("Cp.exe dir_for_csr num_of_random_networks [type 1|2] flags(g|n|k combination)\n"
                                 "  e.g. Cp.exe d:\\data\\weighted 10 2 gnk");
            }
        }
        const std::string dir = argv[1];
        const int nrand = parse_random_count(argv[2]);
        auto flags = parse_flags(flag_arg, "gnk", "output flags");
        const bool want_g = flags.count('g'), want_n = flags.count('n'), want_k = flags.count('k');
        if (want_k && nrand == 0) throw UsageError("gamma (k) requires at least one random network");
        if (kWeighted)
            std::cout << "clustering coefficient: " << (type == ClusteringType::Barrat ? "Barrat" : "Onnela") << std::endl;

        RandomNetworkGenerator gen;
        for (const auto& f : csr_files(dir)) {
            const std::string path = path_join(dir, f);
            const std::string stem = path_without_extension(path);
            std::cout << "\ncalculating Cp for " << path << " ..." << std::endl;
            Timer t;
            Csr g = load_network(path);
            std::vector<float> nodal;
            const double cp = clustering(g, type, nodal);
            std::cout << "average Cp: " << fmt6(cp) << "  (" << fmt6(t.seconds()) << " s)" << std::endl;
            if (want_n) write_nm(stem + "_cp.nm", nodal);

            std::vector<double> rand_cp;
            if (nrand > 0 && (want_g || want_k)) {
                std::cout << "calculating Cp for " << nrand << " random network(s) ..." << std::endl;
                std::vector<float> tmp;
                for (int r = 0; r < nrand; ++r) rand_cp.push_back(clustering(gen.rewire(g), type, tmp));
            }
            if (want_g) {
                std::ofstream o(stem + "_cp.txt");
                o << fmt6(cp) << "\n";
                for (double v : rand_cp) o << fmt6(v) << "\n";
                if (!o) throw Error("cannot write " + stem + "_cp.txt");
            }
            if (want_k) {
                double mean = 0;
                for (double v : rand_cp) mean += v;
                mean /= rand_cp.size();
                double gamma = mean > 0 ? cp / mean : std::numeric_limits<double>::quiet_NaN();
                std::ofstream o(stem + "_gamma.txt");
                o << fmt6(gamma) << "\n";
                if (!o) throw Error("cannot write " + stem + "_gamma.txt");
                std::cout << "gamma = " << fmt6(gamma) << std::endl;
            }
        }
        return 0;
    });
}
