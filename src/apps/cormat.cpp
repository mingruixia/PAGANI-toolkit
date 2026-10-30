// CUCorMat.exe Dir_for_BOLD mask.nii mask_threshold to_average to_save_cormat threshold_type thresholds...
//   to_average : n  individual networks only
//                yn / yf  group network only (average of r / of Fisher z)
//                bn / bf  both individual and group networks
//   to_save_cormat : y / n  (whole correlation matrix, .cormat)
//   threshold_type : r (correlation, edges with r > t) or s (sparsity in %)
//   e.g.  CUCorMat.exe d:\BOLD d:\mask.nii 0.5 n n r 0.2 0.25 0.3
//   Images (one 4D image per subject) and the mask may be .nii or .nii.gz.
// Outputs (in Dir_for_BOLD\unweighted or \weighted):
//   <subject>_<N>_spa<sparsity>%_cor<r>.csr        (unweighted)
//   <subject>_<N>_spa<sparsity>%_cor<r>_weighted.csr  (weighted)
//   ave_<count>_<N>_spa..                           (group network)
#include <cctype>

#include "construct_util.hpp"

using namespace pagani;

int main(int argc, char** argv) {
    return run_tool("CUCorMat", argc, argv, [](int argc, char** argv) {
        if (argc < 8)
            throw UsageError("CUCorMat.exe Dir_for_BOLD mask.nii mask_threshold to_average(n|yn|yf|bn|bf) "
                             "to_save_cormat(y|n) threshold_type(r|s) threshold(s)...\n"
                             "  e.g. CUCorMat.exe d:\\BOLD d:\\mask.nii 0.5 n n r 0.2 0.25 0.3");
        const std::string dir = strip_trailing_separator(argv[1]);
        const std::string mask_path = argv[2];
        const double mask_thr = parse_double(argv[3], "mask threshold");
        const std::string avg = to_lower(argv[4]);
        if (avg != "n" && avg != "yn" && avg != "yf" && avg != "bn" && avg != "bf")
            throw UsageError("to_average must be one of n, yn, yf, bn, bf (got '" + std::string(argv[4]) + "')");
        const std::string save = to_lower(argv[5]);
        if (save != "y" && save != "n") throw UsageError("to_save_cormat must be y or n");
        const auto thresholds = parse_thresholds(argv[6], argv + 7, argv + argc);
        const bool individual = avg[0] == 'n' || avg[0] == 'b';
        const bool group = avg[0] == 'y' || avg[0] == 'b';
        const bool fisher = group && avg[1] == 'f';
        const bool save_cormat = save == "y";

        Mask mask = load_mask(mask_path, mask_thr);
        const std::uint32_t n = static_cast<std::uint32_t>(mask.indices.size());
        const std::uint64_t m = packed_count(n);
        std::cout << "mask: " << n << " voxels >= " << mask_thr << "  (" << m << " voxel pairs)" << std::endl;
        const std::string outdir = path_join(dir, kWeighted ? "weighted" : "unweighted");
        ensure_directory(outdir);
        const std::string suffix = kWeighted ? "_weighted.csr" : ".csr";
        auto files = bold_files(dir, mask_path);
        if (files.empty()) throw Error("no .nii or .nii.gz image found in " + dir);

        std::vector<float> acc, packed;
        if (group || save_cormat) {
            std::cout << "host memory for the whole correlation matrix: " << (m * 4 >> 20) << " MB"
                      << (group && save_cormat && individual ? " (x2)" : "") << std::endl;
        }
        if (group) acc.assign(m, 0.0f);
        auto backend = make_backend();
        int used = 0;
        for (const auto& f : files) {
            const std::string path = path_join(dir, f);
            std::cout << "\ncalculating correlation for " << path << " ..." << std::endl;
            Timer t;
            SubjectData s;
            if (!load_subject(path, mask, s)) continue;
            report_flat(backend->load(s.series, s.n, s.l, CorrType::Pearson), s.n);
            ++used;
            const std::string name = nifti_stem(f) + "_" + std::to_string(n);
            if (save_cormat && individual) {
                packed.assign(m, 0.0f);
                backend->packed(packed.data(), CorrelationBackend::Accumulate::Assign);
                write_cormat(path_join(dir, name + ".cormat"), packed.data(), m);
                if (group) {
                    for (std::uint64_t k = 0; k < m; ++k) acc[k] += fisher ? fisher_z(packed[k]) : packed[k];
                }
                PackedSource src(packed.data(), n);
                build_networks(src, thresholds, kWeighted, [&](ThresholdedNetwork& net) {
                    write_network(path_join(outdir, name), net, suffix, kWeighted);
                });
                std::vector<float>().swap(packed);
            } else {
                if (group)
                    backend->packed(acc.data(), fisher ? CorrelationBackend::Accumulate::AddFisherZ
                                                       : CorrelationBackend::Accumulate::Add);
                if (individual)
                    build_networks(*backend, thresholds, kWeighted, [&](ThresholdedNetwork& net) {
                        write_network(path_join(outdir, name), net, suffix, kWeighted);
                    });
            }
            std::cout << "done in " << fmt6(t.seconds()) << " s" << std::endl;
        }
        if (used == 0) throw Error("no usable 4D image found");
        if (group) {
            std::cout << "\ncomputing group network from " << used << " subject(s) ("
                      << (fisher ? "Fisher z average" : "average of r") << ") ..." << std::endl;
            const float inv = 1.0f / used;
            parallel_for(0, static_cast<long long>((m + 65535) / 65536), [&](long long c, unsigned) {
                std::uint64_t b = static_cast<std::uint64_t>(c) * 65536, e = std::min<std::uint64_t>(m, b + 65536);
                for (std::uint64_t k = b; k < e; ++k) acc[k] = fisher ? inverse_fisher_z(acc[k] * inv) : acc[k] * inv;
            });
            const std::string name = "ave_" + std::to_string(used) + "_" + std::to_string(n);
            if (save_cormat) write_cormat(path_join(dir, name + ".cormat"), acc.data(), m);
            PackedSource src(acc.data(), n);
            build_networks(src, thresholds, kWeighted, [&](ThresholdedNetwork& net) {
                write_network(path_join(outdir, name), net, suffix, kWeighted);
            });
        }
        return 0;
    });
}
