// High-resolution network construction (no whole matrix in host memory).
//
// CUCorMatHr.exe Dir_for_BOLD mask.nii mask_threshold to_average(n) threshold_type coefficient thresholds...
//   coefficient: p (Pearson), s (Spearman), ps / sp (both)
//   e.g. CUCorMatHr.exe d:\BOLD d:\mask.nii 0.2 n s p 0.1
//   Images (one 4D image per subject) and the mask may be .nii or .nii.gz.
//   output: Dir\weighted\<subject>_<N>_pearson_spa..%_cor.._weighted.csr  (or \unweighted\..._unweighted.csr)
//
// CuCorMat-sca-kendall.exe Dir_for_BOLD mask.nii mask_threshold to_average(n) threshold_type thresholds...
//   output: Dir\weightedKendall\<subject>_<N>_kendall_spa..%_cor.._weighted.csr
#include <cctype>

#include "construct_util.hpp"

using namespace pagani;

#ifdef PAGANI_KENDALL
static const char* kTool = "CuCorMat-sca-kendall";
#else
static const char* kTool = "CUCorMatHr";
#endif

int main(int argc, char** argv) {
    return run_tool(kTool, argc, argv, [](int argc, char** argv) {
#ifdef PAGANI_KENDALL
        const int first_thr = 6;
        if (argc < 7)
            throw UsageError("CuCorMat-sca-kendall.exe Dir_for_BOLD mask.nii mask_threshold n threshold_type(r|s) threshold(s)...\n"
                             "  e.g. CuCorMat-sca-kendall.exe d:\\BOLD d:\\mask.nii 0.2 n s 1.5 1 0.5");
        std::vector<CorrType> types{CorrType::Kendall};
        const std::string subdir = "weightedKendall";
#else
        const int first_thr = 7;
        if (argc < 8)
            throw UsageError("CUCorMatHr.exe Dir_for_BOLD mask.nii mask_threshold n threshold_type(r|s) "
                             "coefficient(p|s|ps) threshold(s)...\n"
                             "  e.g. CUCorMatHr.exe d:\\BOLD d:\\mask.nii 0.2 n s p 0.1");
        const std::string coef = to_lower(argv[6]);
        std::vector<CorrType> types;
        if (coef == "p") types = {CorrType::Pearson};
        else if (coef == "s") types = {CorrType::Spearman};
        else if (coef == "ps" || coef == "sp") types = {CorrType::Pearson, CorrType::Spearman};
        else throw UsageError("coefficient must be p (Pearson), s (Spearman) or ps (both)");
        const std::string subdir = kWeighted ? "weighted" : "unweighted";
#endif
        const std::string dir = strip_trailing_separator(argv[1]);
        const std::string mask_path = argv[2];
        const double mask_thr = parse_double(argv[3], "mask threshold");
        if (to_lower(argv[4]) != "n")
            throw UsageError("group averaging is not available for high-resolution construction; "
                             "to_average must be n (use CUCorMat.exe for group networks)");
        const auto thresholds = parse_thresholds(argv[5], argv + first_thr, argv + argc);

        Mask mask = load_mask(mask_path, mask_thr);
        const std::uint32_t n = static_cast<std::uint32_t>(mask.indices.size());
        std::cout << "mask: " << n << " voxels >= " << mask_thr << std::endl;
        const std::string outdir = path_join(dir, subdir);
        ensure_directory(outdir);
        const std::string suffix = kWeighted ? "_weighted.csr" : "_unweighted.csr";
        auto files = bold_files(dir, mask_path);
        if (files.empty()) throw Error("no .nii or .nii.gz image found in " + dir);
        auto backend = make_backend();
        int used = 0;
        for (const auto& f : files) {
            const std::string path = path_join(dir, f);
            std::cout << "\ncalculating correlation for " << path << " ..." << std::endl;
            SubjectData s;
            if (!load_subject(path, mask, s)) continue;
            ++used;
            for (CorrType type : types) {
                Timer t;
                std::cout << "computing " << corr_name(type) << " correlation" << std::endl;
                // each coefficient starts from the original data (audit C-02)
                report_flat(backend->load(s.series, s.n, s.l, type), s.n);
                const std::string stem =
                    path_join(outdir, nifti_stem(f) + "_" + std::to_string(n) + "_" + corr_name(type));
                build_networks(*backend, thresholds, kWeighted,
                               [&](ThresholdedNetwork& net) { write_network(stem, net, suffix, kWeighted); });
                std::cout << "done in " << fmt6(t.seconds()) << " s" << std::endl;
            }
        }
        if (used == 0) throw Error("no usable 4D image found");
        return 0;
    });
}
