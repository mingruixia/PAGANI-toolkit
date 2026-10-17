// PAGANI Toolkit - CPU implementations of graph metrics.
//
// Conventions (kept from PAGANI 2.0 unless stated):
//  * networks are undirected, stored symmetrically in CSR
//  * mean values are averaged over all N nodes (nodes with degree < 2 have Cp = 0)
//  * characteristic path length Lp = 1 / global efficiency (harmonic mean),
//    so disconnected pairs contribute 0 and are allowed
//  * weighted path lengths use distance = 1 / weight
#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "pagani/graph.hpp"

namespace pagani {

// ------------------------------------------------------------------ degree
// Unweighted: number of neighbours.  Weighted: strength (sum of weights).
std::vector<float> degree_centrality(const Csr& g, bool weighted);

// --------------------------------------------------------------- clustering
enum class ClusteringType {
    Binary,   // Watts & Strogatz
    Barrat,   // Barrat et al. 2004 (PAGANI weighted type 1)
    Onnela,   // Onnela et al. 2005 (PAGANI weighted type 2), weights scaled by max weight
};
// Returns mean clustering over all nodes; nodal values in `nodal` (size N).
double clustering(const Csr& g, ClusteringType type, std::vector<float>& nodal);

// --------------------------------------------------------------- efficiency
struct EfficiencyResult {
    double lp = 0.0;                 // 1 / global efficiency (inf if no edges)
    double global_efficiency = 0.0;
    std::vector<float> nodal;        // nodal efficiency, sum_j 1/d_ij / (N-1)
    bool connected = true;
    std::uint32_t isolated = 0;      // nodes with no neighbour
};
// BFS for unweighted networks, Dijkstra (distance = 1/w) for weighted ones.
EfficiencyResult efficiency(const Csr& g, bool weighted, bool want_nodal = true);
// Assemble an EfficiencyResult from per-node sums of inverse distances.
EfficiencyResult efficiency_from_inverse_sums(const std::vector<double>& inv_sum, bool connected,
                                              std::uint32_t isolated);

// -------------------------------------------------------------- betweenness
// Brandes' algorithm on the unweighted network.  Every ordered pair (s,t) is
// counted, i.e. values are twice the undirected-pair convention (as in BCT
// betweenness_bin and PAGANI 2.0).
std::vector<double> betweenness_binary(const Csr& g);

// ----------------------------------------------------- eigenvector centrality
// Leading eigenvector (unit L2 norm, non-negative) and eigenvalue.
struct EigenResult {
    std::vector<double> vector;
    double value = 0.0;
    int iterations = 0;
    bool converged = false;
};
EigenResult eigenvector_centrality(const Csr& g, bool weighted, double tol = 1e-6, int max_iter = 10000);

// ------------------------------------------------------------- modularity
struct Partition {
    std::vector<int> module;   // 1-based module id per node; 0 for isolated nodes
    int num_modules = 0;
    double q = 0.0;
};
// Newman modularity Q of a partition (module ids as above; isolated nodes are
// ignored since they have no edges).
double modularity_q(const Csr& g, bool weighted, const std::vector<int>& module);

// Louvain method (Blondel et al. 2008).  Node visiting order is shuffled with
// `rng` in every pass.
Partition louvain(const Csr& g, bool weighted, std::mt19937_64& rng);

// Newman's leading-eigenvector method with fine tuning (Newman 2006).
// `spmv` (optional) computes y = A x for the full network; it lets callers
// provide a GPU implementation.  The default uses the CPU.
using SpmvFn = std::function<void(const std::vector<double>& x, std::vector<double>& y)>;
Partition newman_eigenvector(const Csr& g, bool weighted, const SpmvFn& spmv = nullptr);

// ------------------------------------------------ participation coefficient
// PC_i = 1 - sum_m (k_im / k_i)^2 (strengths for weighted networks).
// Normalised PC divides by (M-1)/M where M is the number of modules
// (unchanged from PAGANI 2.0).
void participation_coefficient(const Csr& g, bool weighted, const std::vector<float>& module,
                               std::vector<float>& pc, std::vector<float>& pc_normalized);

}  // namespace pagani
