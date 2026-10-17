// BFS_MulCPU.exe - characteristic path length / nodal efficiency on CPU
// (multi-threaded BFS; Dijkstra with distance = 1/weight for weighted networks).
#include "lp_main.hpp"

int main(int argc, char** argv) {
    return pagani::lp_main("BFS_MulCPU", argc, argv,
                           [](const pagani::Csr& g) { return pagani::efficiency(g, kWeighted); });
}
