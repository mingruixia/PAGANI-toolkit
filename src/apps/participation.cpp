// PC_CPU.exe dir_for_csr
// Participation coefficient; requires <network>_modu.nm from a modularity tool.
// Outputs: <network>_pc.nm, <network>_pc_normalized.nm
//   (PAGANI 2.0 named the latter "<network>_pc.nm_Normalized.nm")
#include "app_util.hpp"
#include "pagani/metrics.hpp"

using namespace pagani;

int main(int argc, char** argv) {
    return run_tool("PC_CPU", argc, argv, [](int argc, char** argv) {
        if (argc != 2) throw UsageError("PC_CPU.exe dir_for_csr\n  (run a modularity tool first)");
        const std::string dir = argv[1];
        int missing = 0;
        for (const auto& f : csr_files(dir)) {
            const std::string path = path_join(dir, f);
            const std::string stem = path_without_extension(path);
            if (!file_exists(stem + "_modu.nm")) {
                std::cout << "skip " << path << ": " << stem << "_modu.nm not found (run modularity first)" << std::endl;
                ++missing;
                continue;
            }
            std::cout << "\ncalculating participation coefficient for " << path << " ..." << std::endl;
            Csr g = load_network(path);
            auto modules = read_nm(stem + "_modu.nm");
            std::vector<float> pc, pcn;
            participation_coefficient(g, kWeighted, modules, pc, pcn);
            write_nm(stem + "_pc.nm", pc);
            write_nm(stem + "_pc_normalized.nm", pcn);
        }
        if (missing) throw Error(std::to_string(missing) + " network(s) skipped because module files are missing");
        return 0;
    });
}
