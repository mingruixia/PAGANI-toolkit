// PAGANI Toolkit - GPU graph algorithms.
#pragma once

#include <vector>

#include "pagani/graph.hpp"
#include "pagani/metrics.hpp"

namespace pagani {
namespace gpu {

// All-pairs shortest paths with a blocked Floyd-Warshall on the GPU; returns
// the same quantities as pagani::efficiency().  Distances are 1 (unweighted)
// or 1/w (weighted).  The distance matrix is kept on the GPU when it fits,
// otherwise it is streamed tile by tile from host memory.
EfficiencyResult efficiency_fw(const Csr& g, bool weighted);

// Path length / efficiency of an unweighted network with a bit-parallel
// multi-source BFS on the GPU (no distance matrix); same results as
// pagani::efficiency(g, false).  efficiency_bfs_fits() tells whether the
// network fits in GPU memory.
EfficiencyResult efficiency_bfs(const Csr& g);
bool efficiency_bfs_fits(const Csr& g);

// Bytes of host/device memory the Floyd-Warshall needs for n nodes.
std::size_t fw_matrix_bytes(std::uint32_t n);

// Brandes betweenness (unweighted), ordered-pair convention (see metrics.hpp).
std::vector<double> betweenness_binary(const Csr& g);

// Eigenvector centrality by power iteration on (A + I) with cuSPARSE.
EigenResult eigenvector_centrality(const Csr& g, bool weighted, double tol = 1e-6, int max_iter = 10000);

// y = A x on the GPU for Newman's method (full-size vectors).
class SpMV {
public:
    SpMV(const Csr& g, bool weighted);
    ~SpMV();
    SpMV(const SpMV&) = delete;
    SpMV& operator=(const SpMV&) = delete;
    void multiply(const std::vector<double>& x, std::vector<double>& y);
private:
    struct Impl;
    Impl* impl_;
};

}  // namespace gpu
}  // namespace pagani
