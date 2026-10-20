// CUBFW_Lp.exe dir_for_csr num_of_random_networks flags(g|n|l)
// Weighted networks: GPU blocked Floyd-Warshall (distance = 1/weight).
// Unweighted networks: GPU bit-parallel multi-source breadth-first search
// (multi-threaded CPU BFS if the network does not fit in GPU memory).
#include "lp_main.hpp"
#include "pagani/gpu.hpp"

int main(int argc, char** argv) {
    return pagani::lp_main("CUBFW_Lp", argc, argv, [](const pagani::Csr& g) {
        if (kWeighted) return pagani::gpu::efficiency_fw(g, true);
        if (pagani::gpu::efficiency_bfs_fits(g)) return pagani::gpu::efficiency_bfs(g);
        std::cout << "network does not fit in GPU memory: breadth-first search on the CPU" << std::endl;
        return pagani::efficiency(g, false);
    });
}
