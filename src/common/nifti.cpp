#include "pagani/nifti.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#include <zlib.h>

#include "pagani/common.hpp"

namespace pagani {

namespace {

// Offsets in the NIfTI-1 header.
constexpr int kOffSizeofHdr = 0;
constexpr int kOffDim = 40;
constexpr int kOffDatatype = 70;
constexpr int kOffBitpix = 72;
constexpr int kOffVoxOffset = 108;
constexpr int kOffSclSlope = 112;
constexpr int kOffSclInter = 116;
constexpr int kOffCalMax = 124;
constexpr int kOffCalMin = 128;
constexpr int kOffMagic = 344;

template <typename T>
T get(const unsigned char* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}
template <typename T>
void put(unsigned char* p, T v) {
    std::memcpy(p, &v, sizeof(T));
}

void swap_bytes(unsigned char* p, int n) { std::reverse(p, p + n); }

// Swap every multi-byte field of a NIfTI-1 header in place.
void swap_header(unsigned char* h) {
    struct Field { int off, size, count; };
    static const Field fields[] = {
        {0, 4, 1},     // sizeof_hdr
        {32, 4, 1},    // extents
        {36, 2, 1},    // session_error
        {40, 2, 8},    // dim
        {56, 4, 3},    // intent_p1..3
        {68, 2, 1},    // intent_code
        {70, 2, 1},    // datatype
        {72, 2, 1},    // bitpix
        {74, 2, 1},    // slice_start
        {76, 4, 8},    // pixdim
        {108, 4, 1},   // vox_offset
        {112, 4, 1},   // scl_slope
        {116, 4, 1},   // scl_inter
        {120, 2, 1},   // slice_end
        {124, 4, 1},   // cal_max
        {128, 4, 1},   // cal_min
        {132, 4, 1},   // slice_duration
        {136, 4, 1},   // toffset
        {140, 4, 1},   // glmax
        {144, 4, 1},   // glmin
        {252, 2, 1},   // qform_code
        {254, 2, 1},   // sform_code
        {256, 4, 6},   // quatern_b..qoffset_z
        {280, 4, 12},  // srow_x/y/z
    };
    for (const auto& f : fields)
        for (int i = 0; i < f.count; ++i) swap_bytes(h + f.off + i * f.size, f.size);
}

int datatype_size(int dt) {
    switch (dt) {
        case 2: case 256: return 1;     // uint8, int8
        case 4: case 512: return 2;     // int16, uint16
        case 8: case 768: case 16: return 4;  // int32, uint32, float32
        case 64: return 8;              // float64
        default: return 0;
    }
}

const char* datatype_name(int dt) {
    switch (dt) {
        case 2: return "uint8"; case 256: return "int8";
        case 4: return "int16"; case 512: return "uint16";
        case 8: return "int32"; case 768: return "uint32";
        case 16: return "float32"; case 64: return "float64";
        case 32: return "complex64"; case 128: return "rgb24";
        default: return "unknown";
    }
}

// Convert `n` raw voxels starting at `src` into floats.
void convert_to_float(const unsigned char* src, std::int64_t n, int dt, bool swapped, float* dst) {
    const int sz = datatype_size(dt);
    unsigned char tmp[8];
    for (std::int64_t i = 0; i < n; ++i) {
        const unsigned char* p = src + i * sz;
        if (swapped && sz > 1) {
            std::memcpy(tmp, p, sz);
            swap_bytes(tmp, sz);
            p = tmp;
        }
        switch (dt) {
            case 2: dst[i] = static_cast<float>(*p); break;
            case 256: dst[i] = static_cast<float>(static_cast<std::int8_t>(*p)); break;
            case 4: dst[i] = static_cast<float>(get<std::int16_t>(p)); break;
            case 512: dst[i] = static_cast<float>(get<std::uint16_t>(p)); break;
            case 8: dst[i] = static_cast<float>(get<std::int32_t>(p)); break;
            case 768: dst[i] = static_cast<float>(get<std::uint32_t>(p)); break;
            case 16: dst[i] = get<float>(p); break;
            case 64: dst[i] = static_cast<float>(get<double>(p)); break;
            default: break;
        }
    }
}

void apply_scaling(const NiftiHeader& h, float* v, std::int64_t n) {
    if (!h.has_scaling()) return;
    for (std::int64_t i = 0; i < n; ++i) v[i] = v[i] * h.scl_slope + h.scl_inter;
}

// Sequential reader of a .nii or a gzip-compressed .nii.gz image; the format
// is recognised by the gzip magic bytes, not by the file name.  A gzip stream
// can only be read forwards, which is all the readers below need.
class ImageStream {
public:
    explicit ImageStream(const std::string& path) : path_(path) {
        unsigned char magic[2] = {0, 0};
        {
            std::ifstream probe(path, std::ios::binary);
            if (!probe) throw Error("cannot open NIfTI file '" + path + "'");
            probe.read(reinterpret_cast<char*>(magic), 2);
        }
        if (magic[0] == 0x1f && magic[1] == 0x8b) {
            gz_ = gzopen(path.c_str(), "rb");
            if (!gz_) throw Error("cannot open NIfTI file '" + path + "'");
            gzbuffer(gz_, 1u << 20);
        } else {
            f_.open(path, std::ios::binary);
            if (!f_) throw Error("cannot open NIfTI file '" + path + "'");
        }
    }
    ~ImageStream() {
        if (gz_) gzclose(gz_);
    }
    ImageStream(const ImageStream&) = delete;
    ImageStream& operator=(const ImageStream&) = delete;

    bool compressed() const { return gz_ != nullptr; }

    // Reads up to n bytes; returns the number read (less only at the end of the data).
    std::uint64_t read_some(void* dst, std::uint64_t n) {
        char* p = static_cast<char*>(dst);
        if (!gz_) {
            f_.read(p, static_cast<std::streamsize>(n));
            return static_cast<std::uint64_t>(f_.gcount());
        }
        std::uint64_t done = 0;
        while (done < n) {
            const unsigned chunk = static_cast<unsigned>(std::min<std::uint64_t>(n - done, 1u << 30));
            const int got = gzread(gz_, p + done, chunk);
            if (got < 0) {
                int err = 0;
                throw Error("'" + path_ + "' is not a valid gzip file or is corrupt (" + gzerror(gz_, &err) + ")");
            }
            if (got == 0) break;
            done += static_cast<std::uint64_t>(got);
        }
        return done;
    }

    // Reads exactly n bytes.
    void read(void* dst, std::uint64_t n) {
        if (read_some(dst, n) != n)
            throw Error(gz_ ? "'" + path_ + "' is truncated (the compressed data ends before the image data)"
                            : "read error in '" + path_ + "'");
    }

    // Skips n bytes forwards.
    void skip(std::uint64_t n) {
        if (n == 0) return;
        if (!gz_) {
            f_.seekg(static_cast<std::streamoff>(n), std::ios::cur);
            return;
        }
        std::vector<char> buf(static_cast<size_t>(std::min<std::uint64_t>(n, 1u << 16)));
        while (n > 0) {
            const std::uint64_t m = std::min<std::uint64_t>(n, buf.size());
            read(buf.data(), m);
            n -= m;
        }
    }

private:
    std::string path_;
    std::ifstream f_;
    gzFile gz_ = nullptr;
};

// Reads and validates the header; leaves `s` positioned at the image data.
NiftiHeader read_header(ImageStream& s, const std::string& path) {
    NiftiHeader h;
    if (s.read_some(h.raw.data(), kNiftiHeaderSize) != kNiftiHeaderSize)
        throw Error("'" + path + "' is too short to be a NIfTI file");

    std::int32_t sizeof_hdr = get<std::int32_t>(h.raw.data() + kOffSizeofHdr);
    if (sizeof_hdr != kNiftiHeaderSize) {
        unsigned char tmp[4];
        std::memcpy(tmp, h.raw.data(), 4);
        swap_bytes(tmp, 4);
        if (get<std::int32_t>(tmp) == kNiftiHeaderSize) {
            h.swapped = true;
            swap_header(h.raw.data());
        } else if (sizeof_hdr == 540 || get<std::int32_t>(tmp) == 540) {
            throw Error("'" + path + "' is a NIfTI-2 file, which is not supported (please save as NIfTI-1)");
        } else {
            throw Error("'" + path + "' is not a valid NIfTI-1 file (sizeof_hdr = " + std::to_string(sizeof_hdr) + ")");
        }
    }
    const char* magic = reinterpret_cast<const char*>(h.raw.data() + kOffMagic);
    if (std::strncmp(magic, "n+1", 3) != 0) {
        if (std::strncmp(magic, "ni1", 3) == 0)
            throw Error("'" + path + "' is a .hdr/.img pair header; please convert to a single .nii file");
        // Analyze 7.5 files without magic: data starts at vox_offset 0 in a separate .img
        throw Error("'" + path + "' is not a single-file NIfTI-1 (.nii) image");
    }
    for (int i = 0; i < 8; ++i) h.dim[i] = get<std::int16_t>(h.raw.data() + kOffDim + 2 * i);
    h.datatype = get<std::int16_t>(h.raw.data() + kOffDatatype);
    h.bitpix = get<std::int16_t>(h.raw.data() + kOffBitpix);
    h.vox_offset = get<float>(h.raw.data() + kOffVoxOffset);
    h.scl_slope = get<float>(h.raw.data() + kOffSclSlope);
    h.scl_inter = get<float>(h.raw.data() + kOffSclInter);
    if (!std::isfinite(h.scl_slope)) h.scl_slope = 0.0f;
    if (!std::isfinite(h.scl_inter)) h.scl_inter = 0.0f;

    if (h.dim[0] < 1 || h.dim[0] > 7) throw Error("'" + path + "': invalid dim[0] = " + std::to_string(h.dim[0]));
    for (int i = 1; i <= 3; ++i)
        if (h.dim[i] < 1) throw Error("'" + path + "': invalid image dimension dim[" + std::to_string(i) + "]");
    if (datatype_size(h.datatype) == 0)
        throw Error("'" + path + "': unsupported NIfTI datatype " + std::to_string(h.datatype) + " (" +
                    datatype_name(h.datatype) + ")");
    if (h.vox_offset < kNiftiVoxOffset - 4 || std::floor(h.vox_offset) != h.vox_offset)
        throw Error("'" + path + "': invalid vox_offset " + std::to_string(h.vox_offset));
    // Check the file actually contains the declared data (the uncompressed size
    // of a .nii.gz is only known after decompressing it: checked while reading).
    std::uint64_t need = static_cast<std::uint64_t>(h.vox_offset) +
                         static_cast<std::uint64_t>(h.voxels_per_volume()) * h.volumes() * datatype_size(h.datatype);
    if (!s.compressed() && file_size(path) < need)
        throw Error("'" + path + "' is truncated: expected at least " + std::to_string(need) + " bytes");
    s.skip(static_cast<std::uint64_t>(h.vox_offset) - kNiftiHeaderSize);
    return h;
}

}  // namespace

std::int64_t NiftiHeader::volumes() const {
    std::int64_t v = 1;
    for (int i = 4; i <= dim[0] && i <= 7; ++i) v *= std::max<std::int16_t>(dim[i], 1);
    return v;
}

NiftiHeader read_nifti_header(const std::string& path) {
    ImageStream s(path);
    return read_header(s, path);
}

std::vector<float> read_nifti_float(const std::string& path, NiftiHeader* hdr_out) {
    ImageStream f(path);
    NiftiHeader h = read_header(f, path);
    const std::int64_t n = h.voxels_per_volume() * h.volumes();
    const int sz = datatype_size(h.datatype);
    std::vector<float> out(static_cast<size_t>(n));
    const std::int64_t chunk = 1 << 22;
    std::vector<unsigned char> buf(static_cast<size_t>(std::min(n, chunk) * sz));
    for (std::int64_t off = 0; off < n; off += chunk) {
        std::int64_t m = std::min(chunk, n - off);
        f.read(buf.data(), static_cast<std::uint64_t>(m * sz));
        convert_to_float(buf.data(), m, h.datatype, h.swapped, out.data() + off);
    }
    apply_scaling(h, out.data(), n);
    if (hdr_out) *hdr_out = h;
    return out;
}

std::vector<float> read_nifti_masked(const std::string& path, const std::vector<std::int64_t>& indices,
                                     NiftiHeader* hdr_out) {
    ImageStream f(path);
    NiftiHeader h = read_header(f, path);
    const std::int64_t nvox = h.voxels_per_volume();
    const std::int64_t nvol = h.volumes();
    const int sz = datatype_size(h.datatype);
    for (auto k : indices)
        if (k < 0 || k >= nvox) throw Error("'" + path + "': mask index out of range (mask and data dimensions differ?)");
    std::vector<unsigned char> raw(static_cast<size_t>(nvox * sz));
    std::vector<float> vol(static_cast<size_t>(nvox));
    std::vector<float> out(static_cast<size_t>(nvol) * indices.size());
    for (std::int64_t t = 0; t < nvol; ++t) {
        f.read(raw.data(), static_cast<std::uint64_t>(nvox * sz));
        convert_to_float(raw.data(), nvox, h.datatype, h.swapped, vol.data());
        float* dst = out.data() + t * indices.size();
        for (size_t i = 0; i < indices.size(); ++i) dst[i] = vol[static_cast<size_t>(indices[i])];
    }
    apply_scaling(h, out.data(), static_cast<std::int64_t>(out.size()));
    if (hdr_out) *hdr_out = h;
    return out;
}

std::string nifti_stem(const std::string& path) {
    if (ends_with(to_lower(path), ".nii.gz")) return path.substr(0, path.size() - 7);
    return path_without_extension(path);
}

Mask load_mask(const std::string& path, double threshold) {
    Mask m;
    std::vector<float> v = read_nifti_float(path, &m.header);
    m.total_voxels = m.header.voxels_per_volume();
    if (m.header.volumes() != 1)
        throw Error("mask '" + path + "' has " + std::to_string(m.header.volumes()) + " volumes; a 3D mask is required");
    for (std::int64_t k = 0; k < m.total_voxels; ++k)
        if (v[static_cast<size_t>(k)] >= threshold) m.indices.push_back(k);
    if (m.indices.empty())
        throw Error("mask '" + path + "' contains no voxel >= threshold " + std::to_string(threshold));
    return m;
}

void write_nifti_float3d(const std::string& path, const NiftiHeader& templ, const std::vector<float>& volume) {
    if (static_cast<std::int64_t>(volume.size()) != templ.voxels_per_volume())
        throw Error("internal error: output volume size does not match template header");
    std::array<unsigned char, kNiftiVoxOffset> h{};
    std::memcpy(h.data(), templ.raw.data(), kNiftiHeaderSize);  // native order
    put<std::int32_t>(h.data() + kOffSizeofHdr, kNiftiHeaderSize);
    put<std::int16_t>(h.data() + kOffDim, 3);
    for (int i = 4; i < 8; ++i) put<std::int16_t>(h.data() + kOffDim + 2 * i, 1);
    put<std::int16_t>(h.data() + kOffDatatype, 16);
    put<std::int16_t>(h.data() + kOffBitpix, 32);
    put<float>(h.data() + kOffVoxOffset, static_cast<float>(kNiftiVoxOffset));
    put<float>(h.data() + kOffSclSlope, 1.0f);
    put<float>(h.data() + kOffSclInter, 0.0f);
    float vmin = 0, vmax = 0;
    for (size_t i = 0; i < volume.size(); ++i) {
        if (i == 0 || volume[i] < vmin) vmin = volume[i];
        if (i == 0 || volume[i] > vmax) vmax = volume[i];
    }
    put<float>(h.data() + kOffCalMax, vmax);
    put<float>(h.data() + kOffCalMin, vmin);
    std::memcpy(h.data() + kOffMagic, "n+1\0", 4);
    // bytes 348..351: extension flag = 0 (already zero)
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw Error("cannot create '" + path + "'");
    f.write(reinterpret_cast<const char*>(h.data()), kNiftiVoxOffset);
    f.write(reinterpret_cast<const char*>(volume.data()), static_cast<std::streamsize>(volume.size() * sizeof(float)));
    if (!f) throw Error("write error on '" + path + "'");
}

}  // namespace pagani
