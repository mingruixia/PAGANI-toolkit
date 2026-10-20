// PAGANI Toolkit - voxel-wise functional network construction.
//
// A correlation backend (CPU or CUDA) produces the strictly upper triangle of
// the N x N correlation matrix block by block.  The host side turns it into
// thresholded networks:
//
//   * correlation threshold t : edges with r > t
//   * sparsity s (%)          : exactly K = round-half-up(M * s / 100) edges
//                               with the largest r, M = N (N - 1) / 2; ties at
//                               the K-th value are broken by (i, j) order.
//
// The sparsity threshold is found exactly from a histogram of all values, so no
// N x N matrix is needed (high-resolution networks).  A histogram of a sample
// of rows gives a provisional threshold, so that the histogram and the
// candidate edges are collected in one pass; a second (extraction) pass is
// needed only if the provisional threshold was too high.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "pagani/graph.hpp"

namespace pagani {

enum class CorrType { Pearson, Spearman, Kendall };
const char* corr_name(CorrType t);

struct WeightedEdge {
    std::uint32_t i, j;  // i < j
    float r;
};

// Histogram over [-1, 1] used to locate sparsity thresholds.
constexpr int kHistBins = 1 << 16;
inline int hist_bin(float v) {
    float x = (v + 1.0f) * (0.5f * kHistBins);
    int b = static_cast<int>(x);
    return b < 0 ? 0 : (b >= kHistBins ? kHistBins - 1 : b);
}
inline float hist_bin_low(int b) { return -1.0f + 2.0f * static_cast<float>(b) / kHistBins; }

// Something that can enumerate the strict upper triangle of a correlation matrix.
class EdgeSource {
public:
    virtual ~EdgeSource() = default;
    virtual std::uint32_t size() const = 0;  // N
    // Histogram (kHistBins) of all N(N-1)/2 values.
    virtual std::vector<std::uint64_t> histogram() = 0;
    // All pairs with r >= threshold, in any order.
    virtual std::vector<WeightedEdge> extract(float threshold) = 0;
    // Histogram (kHistBins) of the values of some rows spread over the matrix
    // (empty if not supported).
    virtual std::vector<std::uint64_t> sample_histogram() { return {}; }
    // histogram() and extract(threshold) in one pass.  Returns false (edges
    // cleared) if more than max_edges pairs have r >= threshold.
    virtual bool histogram_and_extract(float threshold, std::uint64_t max_edges, std::vector<std::uint64_t>& hist,
                                       std::vector<WeightedEdge>& edges) {
        hist = histogram();
        edges = extract(threshold);
        if (edges.size() <= max_edges) return true;
        std::vector<WeightedEdge>().swap(edges);
        return false;
    }
};

// Rows used by sample_histogram(): about 2048 rows evenly spread over [0, n).
std::vector<std::uint32_t> sample_rows(std::uint32_t n);

// Correlation backend.  `series` is voxel-major: series[v * L + t].
class CorrelationBackend : public EdgeSource {
public:
    // Upload/prepare one subject.  Returns the number of voxels with a
    // constant time series (their correlations are defined as 0).
    virtual std::uint32_t load(const std::vector<float>& series, std::uint32_t n, std::uint32_t l, CorrType type) = 0;

    enum class Accumulate { Assign, Add, AddFisherZ };
    // Write (or accumulate) the packed strict upper triangle, row major:
    // index(i, j) = i*N - i(i+1)/2 + (j - i - 1).
    virtual void packed(float* out, Accumulate mode) = 0;
    virtual std::string description() const = 0;
};

std::unique_ptr<CorrelationBackend> make_cpu_backend();
// Implemented in the CUDA library; throws if CUDA is unavailable.
std::unique_ptr<CorrelationBackend> make_cuda_backend();

// EdgeSource over an already computed packed matrix (group averages).
class PackedSource : public EdgeSource {
public:
    PackedSource(const float* packed, std::uint32_t n) : p_(packed), n_(n) {}
    std::uint32_t size() const override { return n_; }
    std::vector<std::uint64_t> histogram() override;
    std::vector<WeightedEdge> extract(float threshold) override;
private:
    const float* p_;
    std::uint32_t n_;
};

inline std::uint64_t packed_count(std::uint32_t n) { return static_cast<std::uint64_t>(n) * (n - 1) / 2; }

// --------------------------------------------------------------- thresholds
struct Threshold {
    bool sparsity = false;  // false: correlation threshold
    double value = 0.0;     // r, or sparsity in percent
};

struct ThresholdedNetwork {
    Threshold spec;
    double r = 0.0;          // correlation threshold actually applied / K-th value
    double sparsity = 0.0;   // percentage of possible edges present
    Csr graph;               // symmetric, rows sorted, V = r
};

// Number of edges for a sparsity (PAGANI 2.0 rounding: Clength = floor(2*M*s/100),
// rounded up to an even number).
std::uint64_t sparsity_edge_count(std::uint32_t n, double sparsity_percent);

// Build networks for all thresholds from one source (2 passes at most).
// `emit` is called once per threshold, in the order given.
void build_networks(EdgeSource& src, const std::vector<Threshold>& thresholds, bool keep_weights,
                    const std::function<void(ThresholdedNetwork&)>& emit);

// File name fragment used by PAGANI: "_spa%.3f%%_cor%.3f"
std::string threshold_tag(double sparsity, double r);

// Fisher z with clamping (|r| < 1).
float fisher_z(float r);
float inverse_fisher_z(float z);

// ----------------------------------------------------------- preprocessing
// Replace each series by its ranks (average ranks for ties).
void rank_transform(std::vector<float>& series, std::uint32_t n, std::uint32_t l);
// Centre and scale each series to unit L2 norm; constant series become 0.
// Returns the number of constant series.
std::uint32_t standardize(std::vector<float>& series, std::uint32_t n, std::uint32_t l);

// --------------------------------------------------------------- .cormat
// Legacy whole-matrix file: int32 count, float32 packed[count].  If the count
// exceeds INT32_MAX the header is int32 -1 followed by uint64 count.
void write_cormat(const std::string& path, const float* packed, std::uint64_t count);

}  // namespace pagani
