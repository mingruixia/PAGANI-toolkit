// ConvertNII.exe dir mask.nii [mask_threshold(default 0.5)]
// Converts every nodal metric file (*.nm) in dir into a 3D NIfTI image
// (<file>.nii) using the mask that was used to build the networks.
#include "app_util.hpp"
#include "pagani/nifti.hpp"

using namespace pagani;

int main(int argc, char** argv) {
    return run_tool("ConvertNII", argc, argv, [](int argc, char** argv) {
        if (argc != 3 && argc != 4)
            throw UsageError("ConvertNII.exe dir_with_nm_files mask.nii [mask_threshold]\n"
                             "  e.g. ConvertNII.exe d:\\data\\unweighted d:\\data\\mask.nii 0.2");
        const std::string dir = argv[1];
        const double thr = argc == 4 ? parse_double(argv[3], "mask threshold") : 0.5;
        Mask mask = load_mask(argv[2], thr);
        std::cout << "mask: " << mask.indices.size() << " voxels >= " << thr << std::endl;
        auto files = list_files_with_suffix(dir, ".nm");
        std::cout << files.size() << " .nm file(s) to be converted" << std::endl;
        int mismatched = 0;
        std::vector<float> vol(static_cast<size_t>(mask.total_voxels));
        for (const auto& f : files) {
            const std::string path = path_join(dir, f);
            auto v = read_nm(path);
            if (v.size() != mask.indices.size()) {
                std::cout << "skip " << f << ": " << v.size() << " nodes but the mask has " << mask.indices.size()
                          << " voxels (different mask or threshold?)" << std::endl;
                ++mismatched;
                continue;
            }
            std::fill(vol.begin(), vol.end(), 0.0f);
            for (size_t i = 0; i < v.size(); ++i) vol[static_cast<size_t>(mask.indices[i])] = v[i];
            const std::string out = path_without_extension(path) + ".nii";
            write_nifti_float3d(out, mask.header, vol);
            std::cout << "converted " << out << std::endl;
        }
        if (mismatched) throw Error(std::to_string(mismatched) + " file(s) did not match the mask");
        return 0;
    });
}
