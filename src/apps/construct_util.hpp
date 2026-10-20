// Helpers for the network construction tools.
#pragma once

#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "app_util.hpp"
#include "pagani/construct.hpp"
#include "pagani/nifti.hpp"

namespace pagani {

inline std::unique_ptr<CorrelationBackend> make_backend() {
#ifdef PAGANI_CPU_BACKEND
    return make_cpu_backend();
#else
    return make_cuda_backend();
#endif
}

// 4D images to be processed: *.nii in dir, excluding the mask itself and the
// mask names PAGANI 2.0 skipped.
inline std::vector<std::string> bold_files(const std::string& dir, const std::string& mask_path) {
    auto files = list_files_with_suffix_ci(dir, ".nii");
    const std::string mask_name = to_lower(path_filename(mask_path));
    std::vector<std::string> out;
    for (const auto& f : files) {
        std::string lf = to_lower(f);
        if (lf == "mask.nii" || lf == "grey_mask.nii" || lf == "1mm_grey.nii" || lf == "1mm_grey_mask.nii") continue;
        std::error_code ec;
        if (lf == mask_name && std::filesystem::equivalent(path_join(dir, f), mask_path, ec)) continue;
        out.push_back(f);
    }
    std::cout << out.size() << " image file(s) to be processed in " << dir << std::endl;
    return out;
}

struct SubjectData {
    std::vector<float> series;  // voxel-major n x l
    std::uint32_t n = 0, l = 0;
};

// Returns false (with a message) when the image is not usable.
inline bool load_subject(const std::string& path, const Mask& mask, SubjectData& out) {
    NiftiHeader h = read_nifti_header(path);
    if (h.nx() != mask.header.nx() || h.ny() != mask.header.ny() || h.nz() != mask.header.nz())
        throw Error("'" + path + "' (" + std::to_string(h.nx()) + "x" + std::to_string(h.ny()) + "x" +
                    std::to_string(h.nz()) + ") and the mask (" + std::to_string(mask.header.nx()) + "x" +
                    std::to_string(mask.header.ny()) + "x" + std::to_string(mask.header.nz()) +
                    ") have different dimensions");
    const std::int64_t l = h.volumes();
    if (l < 3) {
        std::cout << "skip " << path << ": not a 4D time series (" << l << " volume(s))" << std::endl;
        return false;
    }
    std::vector<float> tm = read_nifti_masked(path, mask.indices);
    const size_t n = mask.indices.size();
    out.n = static_cast<std::uint32_t>(n);
    out.l = static_cast<std::uint32_t>(l);
    out.series.assign(n * static_cast<size_t>(l), 0.0f);
    for (std::int64_t t = 0; t < l; ++t)
        for (size_t v = 0; v < n; ++v) out.series[v * l + t] = tm[t * n + v];
    for (float x : out.series)
        if (!std::isfinite(x)) throw Error("'" + path + "' contains NaN or Inf values inside the mask");
    std::cout << "time points: " << l << ", voxels: " << n << std::endl;
    return true;
}

inline std::vector<Threshold> parse_thresholds(const std::string& type, char** begin, char** end) {
    std::vector<Threshold> th;
    const char t = static_cast<char>(std::tolower(static_cast<unsigned char>(type.empty() ? '?' : type[0])));
    if (type.size() != 1 || (t != 'r' && t != 's'))
        throw UsageError("threshold type must be r (correlation) or s (sparsity in %)");
    for (char** p = begin; p != end; ++p) {
        double v = parse_double(*p, "threshold");
        if (t == 'r' && !(v > -1.0 && v < 1.0)) throw UsageError("correlation thresholds must be in (-1, 1)");
        if (t == 's' && !(v > 0.0 && v <= 100.0)) throw UsageError("sparsity thresholds must be in (0, 100] (percent)");
        if (t == 'r' && kWeighted && v < 0.0)
            throw UsageError("weighted networks need a correlation threshold >= 0 (edge weights are r and must be positive)");
        th.push_back({t == 's', v});
    }
    if (th.empty()) throw UsageError("at least one threshold value is required");
    return th;
}

inline void report_flat(std::uint32_t flat, std::uint32_t n) {
    if (flat)
        std::cout << "warning: " << flat << " of " << n
                  << " voxels have a constant time series; their correlations are set to 0 (consider a tighter mask)"
                  << std::endl;
}

inline void write_network(const std::string& stem, const ThresholdedNetwork& net, const std::string& suffix,
                          bool weighted) {
    // PAGANI naming: requested sparsity in sparsity mode, achieved sparsity in r mode
    const std::string path = stem + threshold_tag(net.spec.sparsity ? net.spec.value : net.sparsity, net.r) + suffix;
    std::cout << (net.spec.sparsity ? "sparsity " : "r > ") << net.spec.value << " -> " << net.graph.nnz() / 2
              << " edges (sparsity " << fmt6(net.sparsity) << "%, r = " << fmt6(net.r) << ")  " << path << std::endl;
    if (weighted && net.graph.nnz() > 0 && net.r <= 0.0)
        std::cout << "warning: this weighted network contains edges with r <= 0; the weighted metric tools "
                     "require positive weights (use a lower sparsity)" << std::endl;
    write_csr(path, net.graph, weighted);
}

}  // namespace pagani
