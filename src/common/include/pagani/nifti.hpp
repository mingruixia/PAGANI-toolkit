// PAGANI Toolkit - minimal NIfTI-1 (.nii single file) reader/writer.
//
// Fixes over PAGANI 2.0 (audit N-01/N-02):
//  * data type taken from `datatype` (not `bitpix`); uint8/int8/int16/uint16/
//    int32/uint32/float32/float64 supported
//  * `vox_offset` honoured (header extensions allowed)
//  * `scl_slope`/`scl_inter` applied
//  * byte-swapped (big-endian) files supported
//  * unsupported files are rejected with an error instead of producing
//    uninitialised data
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace pagani {

constexpr int kNiftiHeaderSize = 348;
constexpr int kNiftiVoxOffset = 352;  // header + 4-byte extension flag

struct NiftiHeader {
    std::array<unsigned char, kNiftiHeaderSize> raw{};  // native byte order
    bool swapped = false;   // file is in non-native byte order
    std::array<std::int16_t, 8> dim{};
    std::int16_t datatype = 0;
    std::int16_t bitpix = 0;
    float vox_offset = 352.0f;
    float scl_slope = 0.0f;
    float scl_inter = 0.0f;

    std::int64_t nx() const { return dim[1]; }
    std::int64_t ny() const { return dim[2]; }
    std::int64_t nz() const { return dim[3]; }
    std::int64_t voxels_per_volume() const { return nx() * ny() * nz(); }
    // Number of volumes (product of dim[4..7] for dim[0] >= 4, else 1).
    std::int64_t volumes() const;
    bool has_scaling() const {
        return scl_slope != 0.0f && (scl_slope != 1.0f || scl_inter != 0.0f);
    }
};

// Read and validate the header of a .nii file.
NiftiHeader read_nifti_header(const std::string& path);

// Read all volumes as float (after scaling). Layout: volume-major,
// i.e. value of voxel k in volume t is data[t * voxels_per_volume + k].
std::vector<float> read_nifti_float(const std::string& path, NiftiHeader* hdr_out = nullptr);

// Read only the voxels listed in `indices` (linear voxel indices) for every
// volume.  Output layout: out[t * indices.size() + i].  Streams one volume at
// a time so that very large 4D files do not need to fit in memory twice.
std::vector<float> read_nifti_masked(const std::string& path, const std::vector<std::int64_t>& indices,
                                     NiftiHeader* hdr_out = nullptr);

// Mask: voxels whose (scaled) value >= threshold, in NIfTI voxel order.
struct Mask {
    NiftiHeader header;
    std::vector<std::int64_t> indices;  // linear indices of in-mask voxels
    std::int64_t total_voxels = 0;
};
Mask load_mask(const std::string& path, double threshold);

// Write a single 3D float32 volume using `templ` for geometry.  The header
// is sanitised: datatype/bitpix float32, vox_offset 352, no scaling, no
// extensions, dim[0]=3.
void write_nifti_float3d(const std::string& path, const NiftiHeader& templ, const std::vector<float>& volume);

}  // namespace pagani
