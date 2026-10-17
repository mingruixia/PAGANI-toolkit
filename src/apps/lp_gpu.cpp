// CUBFW_Lp.exe dir_for_csr num_of_random_networks flags(g|n|l)
// GPU blocked Floyd-Warshall; unweighted networks sparser than 1% use the
// multi-threaded CPU BFS instead (as in PAGANI 2.0).
#include "lp_main.hpp"
#include "pagani/gpu.hpp"

int main(int argc, char** argv) {
    return pagani::lp_main("CUBFW_Lp", argc, argv, [](const pagani::Csr& g) {
        const double density = g.n > 1 ? static_cast<double>(g.nnz()) / (static_cast<double>(g.n) * (g.n - 1.0)) : 0.0;
        if (!kWeighted && density < 0.01) return pagani::efficiency(g, false);
        return pagani::gpu::efficiency_fw(g, kWeighted);
    });
}
