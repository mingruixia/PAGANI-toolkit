// Degree.exe dir_for_csr
// Unweighted: degree centrality.  Weighted: nodal strength.
// Output: <network>_deg.nm
#include "app_util.hpp"
#include "pagani/metrics.hpp"

using namespace pagani;

int main(int argc, char** argv) {
    return run_tool("Degree", argc, argv, [](int argc, char** argv) {
        if (argc != 2)
            throw UsageError("Degree.exe dir_for_csr\n  e.g. Degree.exe d:\\data\\unweighted");
        const std::string dir = argv[1];
        for (const auto& f : csr_files(dir)) {
            const std::string path = path_join(dir, f);
            std::cout << "\ncalculating degree for " << path << " ..." << std::endl;
            Timer t;
            Csr g = load_network(path);
            auto d = degree_centrality(g, kWeighted);
            const std::string out = path_without_extension(path) + "_deg.nm";
            write_nm(out, d);
            std::cout << "saved " << out << "  (" << fmt6(t.seconds()) << " s)" << std::endl;
        }
        return 0;
    });
}
