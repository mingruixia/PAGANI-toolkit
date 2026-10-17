// Maslov.exe network.csr
// Writes one degree-preserving random network as <network>_Maslov.csr
#include "app_util.hpp"
#include "pagani/random_network.hpp"

using namespace pagani;

int main(int argc, char** argv) {
    return run_tool("Maslov", argc, argv, [](int argc, char** argv) {
        if (argc != 2) throw UsageError("Maslov.exe network.csr");
        const std::string path = argv[1];
        Csr g = read_csr(path, CsrWeights::Auto);
        RandomNetworkGenerator gen;
        Csr r = gen.rewire(g);
        const std::string out = path_without_extension(path) + "_Maslov.csr";
        write_csr(out, r, g.weighted());
        std::cout << "saved " << out << std::endl;
        return 0;
    });
}
