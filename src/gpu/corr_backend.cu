// CUDA correlation backend (Pearson / Spearman via cuBLAS SGEMM, Kendall tau-b
// via SGEMM on pairwise sign vectors).
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "cuda_common.hpp"
#include "kernels.cuh"
#include "pagani/construct.hpp"

namespace pagani {
namespace {

using cuda::DeviceBuffer;
using cuda::PinnedBuffer;

struct DevEdge {
    unsigned int i, j;
    float r;
};
static_assert(sizeof(DevEdge) == sizeof(WeightedEdge), "edge layout");

// C[a + b*B] = value(i0 + a, j0 + b); valid pairs: i < j < n
__global__ void hist_kernel(const float* __restrict__ C, int B, long long i0, long long j0, long long n,
                            unsigned int* __restrict__ hist) {
    const long long total = static_cast<long long>(B) * B;
    for (long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long a = idx % B, b = idx / B;
        const long long i = i0 + a, j = j0 + b;
        if (i < j && j < n) {
            float v = C[idx];
            float x = (v + 1.0f) * (0.5f * kHistBins);
            int bin = static_cast<int>(x);
            bin = bin < 0 ? 0 : (bin >= kHistBins ? kHistBins - 1 : bin);
            atomicAdd(&hist[bin], 1u);
        }
    }
}

__global__ void extract_kernel(const float* __restrict__ C, int B, long long i0, long long j0, long long n,
                               float thr, DevEdge* __restrict__ out, unsigned long long* __restrict__ count) {
    const long long total = static_cast<long long>(B) * B;
    for (long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long a = idx % B, b = idx / B;
        const long long i = i0 + a, j = j0 + b;
        if (i < j && j < n) {
            float v = C[idx];
            if (v >= thr) {
                unsigned long long p = atomicAdd(count, 1ull);
                out[p] = DevEdge{static_cast<unsigned int>(i), static_cast<unsigned int>(j), v};
            }
        }
    }
}

// --- Kendall ---------------------------------------------------------------
// number of tied time-point pairs for each voxel (raw: voxel-major, n0 x L)
__global__ void kendall_ties_kernel(const float* __restrict__ raw, int n0, int L, float* __restrict__ ties) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n0; v += gridDim.x * blockDim.x) {
        const float* x = raw + static_cast<size_t>(v) * L;
        unsigned long long t = 0;
        for (int a = 0; a < L; ++a)
            for (int b = a + 1; b < L; ++b) t += (x[a] == x[b]);
        ties[v] = static_cast<float>(t);
    }
}

// S[p + v*Pc] = sign(x_v[a_p] - x_v[b_p]) for pairs p0..p0+Pc-1 (0 beyond P)
__global__ void kendall_sign_kernel(const float* __restrict__ raw, int L, int nb, const uint2* __restrict__ pairs,
                                    long long p0, int Pc, long long P, float* __restrict__ S) {
    const long long total = static_cast<long long>(Pc) * nb;
    for (long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long p = idx % Pc, v = idx / Pc;
        const long long gp = p0 + p;
        float s = 0.0f;
        if (gp < P) {
            const uint2 ab = pairs[gp];
            const float* x = raw + v * L;
            float d = x[ab.x] - x[ab.y];
            s = (d > 0.0f) ? 1.0f : ((d < 0.0f) ? -1.0f : 0.0f);
        }
        S[idx] = s;
    }
}

__global__ void kendall_normalize_kernel(float* __restrict__ C, int B, const float* __restrict__ ti,
                                         const float* __restrict__ tj, double n0) {
    const long long total = static_cast<long long>(B) * B;
    for (long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long a = idx % B, b = idx / B;
        double d = (n0 - ti[a]) * (n0 - tj[b]);
        C[idx] = d > 0.0 ? static_cast<float>(C[idx] / sqrt(d)) : 0.0f;
    }
}

class CudaCorrBackend : public CorrelationBackend {
public:
    CudaCorrBackend() = default;

    std::string description() const override { return "CUDA"; }
    std::uint32_t size() const override { return n_; }

    std::uint32_t load(const std::vector<float>& series, std::uint32_t n, std::uint32_t l, CorrType type) override {
        n_ = n;
        l_ = l;
        type_ = type;
        cached1_ = cached2_ = -1;  // device slices of the previous subject are stale
        B_ = choose_block(n);
        nb_ = (n + B_ - 1) / B_;
        const size_t n0 = static_cast<size_t>(nb_) * B_;
        host_.assign(n0 * l, 0.0f);
        std::copy(series.begin(), series.end(), host_.begin());
        std::uint32_t flat = 0;
        if (type == CorrType::Kendall) {
            flat = 0;
        } else {
            std::vector<float> tmp(series);
            if (type == CorrType::Spearman) rank_transform(tmp, n, l);
            flat = standardize(tmp, n, l);
            std::copy(tmp.begin(), tmp.end(), host_.begin());
        }
        C_.resize(static_cast<size_t>(B_) * B_);
        // keep the whole data matrix on the GPU when it fits comfortably
        const size_t bytes = n0 * l * sizeof(float);
        const size_t reserve = C_.size() * sizeof(float) * 3 + (256ull << 20) + extra_bytes();
        resident_ = bytes + reserve < cuda::free_device_memory();
        if (resident_) {
            data_.resize(n0 * l);
            PAGANI_CUDA(cudaMemcpy(data_.get(), host_.data(), bytes, cudaMemcpyHostToDevice));
            slice1_.release();
            slice2_.release();
        } else {
            data_.release();
            slice1_.resize(static_cast<size_t>(B_) * l);
            slice2_.resize(static_cast<size_t>(B_) * l);
            std::cout << "note: data (" << (bytes >> 20) << " MB) does not fit in GPU memory, streaming blocks" << std::endl;
        }
        if (type == CorrType::Kendall) {
            ties_.resize(n0);
            if (resident_) {
                kendall_ties_kernel<<<cuda::grid_for(n0, 128), 128>>>(data_.get(), static_cast<int>(n0), l, ties_.get());
            } else {
                for (int bi = 0; bi < nb_; ++bi) {
                    const float* s = slice(bi, 0);
                    kendall_ties_kernel<<<cuda::grid_for(B_, 128), 128>>>(s, B_, l, ties_.get() + static_cast<size_t>(bi) * B_);
                }
            }
            PAGANI_KERNEL_CHECK();
            std::vector<float> t(n0);
            PAGANI_CUDA(cudaMemcpy(t.data(), ties_.get(), n0 * sizeof(float), cudaMemcpyDeviceToHost));
            const double all = static_cast<double>(l) * (l - 1) / 2.0;
            for (std::uint32_t v = 0; v < n; ++v) flat += (t[v] >= all);
            // pair table and sign buffers
            P_ = static_cast<long long>(l) * (l - 1) / 2;
            std::vector<uint2> pairs(static_cast<size_t>(P_));
            size_t k = 0;
            for (unsigned a = 0; a < l; ++a)
                for (unsigned b = a + 1; b < l; ++b) pairs[k++] = make_uint2(a, b);
            pairs_.resize(pairs.size());
            PAGANI_CUDA(cudaMemcpy(pairs_.get(), pairs.data(), pairs.size() * sizeof(uint2), cudaMemcpyHostToDevice));
            Pc_ = static_cast<int>(std::min<long long>((P_ + 15) / 16 * 16, std::max<long long>(1024, (256ll << 20) / (4ll * B_) / 16 * 16)));
            S1_.resize(static_cast<size_t>(Pc_) * B_);
            S2_.resize(static_cast<size_t>(Pc_) * B_);
        }
        return flat;
    }

    std::vector<std::uint64_t> histogram() override {
        DeviceBuffer<unsigned int> dh(kHistBins);
        std::vector<std::uint64_t> hist(kHistBins, 0);
        std::vector<unsigned int> part(kHistBins);
        for_each_block([&](int bi, int bj) {
            PAGANI_CUDA(cudaMemset(dh.get(), 0, kHistBins * sizeof(unsigned int)));
            hist_kernel<<<cuda::grid_for(C_.size()), 256>>>(C_.get(), B_, 1ll * bi * B_, 1ll * bj * B_, n_, dh.get());
            PAGANI_KERNEL_CHECK();
            PAGANI_CUDA(cudaMemcpy(part.data(), dh.get(), kHistBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            for (int b = 0; b < kHistBins; ++b) hist[b] += part[b];
        });
        return hist;
    }

    std::vector<WeightedEdge> extract(float threshold) override {
        DeviceBuffer<DevEdge> de(C_.size());
        DeviceBuffer<unsigned long long> dc(1);
        std::vector<WeightedEdge> out;
        for_each_block([&](int bi, int bj) {
            PAGANI_CUDA(cudaMemset(dc.get(), 0, sizeof(unsigned long long)));
            extract_kernel<<<cuda::grid_for(C_.size()), 256>>>(C_.get(), B_, 1ll * bi * B_, 1ll * bj * B_, n_, threshold,
                                                                de.get(), dc.get());
            PAGANI_KERNEL_CHECK();
            unsigned long long cnt = 0;
            PAGANI_CUDA(cudaMemcpy(&cnt, dc.get(), sizeof(cnt), cudaMemcpyDeviceToHost));
            if (cnt) {
                size_t old = out.size();
                out.resize(old + cnt);
                PAGANI_CUDA(cudaMemcpy(out.data() + old, de.get(), cnt * sizeof(DevEdge), cudaMemcpyDeviceToHost));
            }
        });
        return out;
    }

    void packed(float* out, Accumulate mode) override {
        PinnedBuffer<float> hc(C_.size());
        for_each_block([&](int bi, int bj) {
            PAGANI_CUDA(cudaMemcpy(hc.get(), C_.get(), C_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            const float* c = hc.get();
            const long long i0 = 1ll * bi * B_, j0 = 1ll * bj * B_;
            const long long n = n_;
            parallel_for(0, B_, [&](long long a, unsigned) {
                const long long i = i0 + a;
                if (i >= n) return;
                const std::uint64_t base = static_cast<std::uint64_t>(i) * n - static_cast<std::uint64_t>(i) * (i + 1) / 2;
                const long long jstart = std::max(j0, i + 1);
                for (long long j = jstart; j < std::min<long long>(j0 + B_, n); ++j) {
                    float r = c[a + (j - j0) * B_];
                    float& o = out[base + (j - i - 1)];
                    if (mode == Accumulate::Assign) o = r;
                    else if (mode == Accumulate::Add) o += r;
                    else o += fisher_z(r);
                }
            });
        });
    }

private:
    // block edge: multiple of the GEMM tile (96)
    static int choose_block(std::uint32_t n) {
        const int t = gpu::kTile;
        int b = 42 * t;  // 4032
        if (const char* env = std::getenv("PAGANI_BLOCK")) b = std::max(t, std::atoi(env) / t * t);
        int need = static_cast<int>((n + t - 1) / t * t);
        return std::min(b, std::max(t, need));
    }

    size_t extra_bytes() const {
        // edge buffer used by extract() + Kendall sign buffers
        size_t e = C_.size() * sizeof(DevEdge);
        if (type_ == CorrType::Kendall) e += 2ull * (256ll << 20);
        return e;
    }

    // device pointer to the data of voxel block b (slot 0 or 1 when streaming)
    const float* slice(int b, int slot) {
        const size_t off = static_cast<size_t>(b) * B_ * l_;
        if (resident_) return data_.get() + off;
        DeviceBuffer<float>& s = slot == 0 ? slice1_ : slice2_;
        int& cached = slot == 0 ? cached1_ : cached2_;
        if (cached != b) {
            PAGANI_CUDA(cudaMemcpy(s.get(), host_.data() + off, static_cast<size_t>(B_) * l_ * sizeof(float),
                                   cudaMemcpyHostToDevice));
            cached = b;
        }
        return s.get();
    }

    void compute_block(int bi, int bj) {
        const float* A = slice(bi, 0);
        const float* Bm = slice(bj, 1);
        if (type_ != CorrType::Kendall) {
            gpu::gemm_tn(B_, B_, l_, A, l_, Bm, l_, 0.0f, C_.get(), B_);
            return;
        }
        for (long long p0 = 0; p0 < P_; p0 += Pc_) {
            kendall_sign_kernel<<<cuda::grid_for(static_cast<size_t>(Pc_) * B_), 256>>>(A, l_, B_, pairs_.get(), p0, Pc_, P_, S1_.get());
            kendall_sign_kernel<<<cuda::grid_for(static_cast<size_t>(Pc_) * B_), 256>>>(Bm, l_, B_, pairs_.get(), p0, Pc_, P_, S2_.get());
            PAGANI_KERNEL_CHECK();
            gpu::gemm_tn(B_, B_, Pc_, S1_.get(), Pc_, S2_.get(), Pc_, p0 == 0 ? 0.0f : 1.0f, C_.get(), B_);
        }
        const double n0 = static_cast<double>(P_);
        kendall_normalize_kernel<<<cuda::grid_for(C_.size()), 256>>>(C_.get(), B_, ties_.get() + 1ll * bi * B_,
                                                                      ties_.get() + 1ll * bj * B_, n0);
        PAGANI_KERNEL_CHECK();
    }

    template <typename F>
    void for_each_block(F&& f) {
        cached1_ = cached2_ = -1;
        const long long total = 1ll * nb_ * (nb_ + 1) / 2;
        long long done = 0;
        Timer t;
        double last = 0;
        for (int bi = 0; bi < nb_; ++bi)
            for (int bj = bi; bj < nb_; ++bj) {
                compute_block(bi, bj);
                f(bi, bj);
                ++done;
                if (total > 8 && t.seconds() - last > 10.0) {
                    last = t.seconds();
                    std::cout << "  " << (100 * done / total) << "% of correlation blocks" << std::endl;
                }
            }
        PAGANI_CUDA(cudaDeviceSynchronize());
    }

    cuda::DeviceInit init_;
    std::uint32_t n_ = 0, l_ = 0;
    CorrType type_ = CorrType::Pearson;
    int B_ = 4096, nb_ = 0;
    bool resident_ = true;
    std::vector<float> host_;
    DeviceBuffer<float> data_, slice1_, slice2_, C_;
    int cached1_ = -1, cached2_ = -1;
    // Kendall
    DeviceBuffer<float> ties_, S1_, S2_;
    DeviceBuffer<uint2> pairs_;
    long long P_ = 0;
    int Pc_ = 0;
};

}  // namespace

std::unique_ptr<CorrelationBackend> make_cuda_backend() { return std::make_unique<CudaCorrBackend>(); }

}  // namespace pagani
