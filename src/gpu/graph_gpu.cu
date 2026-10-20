// GPU betweenness, eigenvector centrality and SpMV (cuSPARSE generic API).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "cuda_common.hpp"
#include "kernels.cuh"
#include "pagani/gpu.hpp"

namespace pagani {
namespace gpu {
namespace {

using cuda::DeviceBuffer;

// ---------------------------------------------------------------- betweenness
// One BFS source per CUDA block; ws_* are per-block workspaces of n entries.
__global__ void brandes_kernel(const unsigned* __restrict__ R, const unsigned* __restrict__ C, int n,
                               int* ws_dist, double* ws_sigma, double* ws_delta, double* bc,
                               unsigned int* next_source) {
    int* dist = ws_dist + static_cast<long long>(blockIdx.x) * n;
    double* sigma = ws_sigma + static_cast<long long>(blockIdx.x) * n;
    double* delta = ws_delta + static_cast<long long>(blockIdx.x) * n;
    __shared__ int s_src;
    __shared__ int s_more;
    while (true) {
        if (threadIdx.x == 0) s_src = static_cast<int>(atomicAdd(next_source, 1u));
        __syncthreads();
        const int s = s_src;
        if (s >= n) return;
        for (int v = threadIdx.x; v < n; v += blockDim.x) { dist[v] = -1; sigma[v] = 0.0; delta[v] = 0.0; }
        __syncthreads();
        if (threadIdx.x == 0) { dist[s] = 0; sigma[s] = 1.0; }
        __syncthreads();
        int d = 0;
        while (true) {
            if (threadIdx.x == 0) s_more = 0;
            __syncthreads();
            // expand: unvisited neighbours of level d get level d + 1 (identical
            // values written concurrently, so plain stores are enough)
            bool found = false;
            for (int v = threadIdx.x; v < n; v += blockDim.x) {
                if (dist[v] != d) continue;
                for (unsigned k = R[v]; k < R[v + 1]; ++k) {
                    const int w = static_cast<int>(C[k]);
                    if (dist[w] == -1) { dist[w] = d + 1; found = true; }
                }
            }
            if (found) s_more = 1;
            __syncthreads();
            if (!s_more) break;
            // count shortest paths by pulling from level d (no atomics)
            for (int w = threadIdx.x; w < n; w += blockDim.x) {
                if (dist[w] != d + 1) continue;
                double acc = 0.0;
                for (unsigned k = R[w]; k < R[w + 1]; ++k) {
                    const int u = static_cast<int>(C[k]);
                    if (dist[u] == d) acc += sigma[u];
                }
                sigma[w] = acc;
            }
            ++d;
            __syncthreads();
        }
        // d is now the last level reached; dependencies from the deepest level up
        for (int lvl = d - 1; lvl >= 1; --lvl) {
            for (int v = threadIdx.x; v < n; v += blockDim.x) {
                if (dist[v] != lvl) continue;
                double acc = 0.0;
                const double sv = sigma[v];
                for (unsigned k = R[v]; k < R[v + 1]; ++k) {
                    const int w = static_cast<int>(C[k]);
                    if (dist[w] == lvl + 1) acc += sv / sigma[w] * (1.0 + delta[w]);
                }
                delta[v] = acc;
            }
            __syncthreads();
        }
        for (int v = threadIdx.x; v < n; v += blockDim.x)
            if (v != s && dist[v] > 0 && delta[v] != 0.0) atomicAdd(&bc[v], delta[v]);
        __syncthreads();
    }
}

// Same algorithm with the BFS levels in shared memory (16 bit, so n < 65535 and
// 2n bytes must fit in shared memory): the many level tests of the neighbour
// scans hit shared memory, only sigma/delta of matching nodes are read from
// global memory.  sigma and delta are written when a node is reached, so the
// global workspaces need no initialisation per source.
constexpr unsigned short kUnseen = 0xFFFF;

__global__ void brandes_smem_kernel(const unsigned* __restrict__ R, const unsigned* __restrict__ C, int n,
                                    double* ws_sigma, double* ws_delta, double* bc, unsigned int* next_source) {
    extern __shared__ unsigned short dist[];
    double* sigma = ws_sigma + static_cast<long long>(blockIdx.x) * n;
    double* delta = ws_delta + static_cast<long long>(blockIdx.x) * n;
    __shared__ int s_src;
    __shared__ int s_more;
    while (true) {
        if (threadIdx.x == 0) s_src = static_cast<int>(atomicAdd(next_source, 1u));
        __syncthreads();
        const int s = s_src;
        if (s >= n) return;
        for (int v = threadIdx.x; v < n; v += blockDim.x) dist[v] = kUnseen;
        __syncthreads();
        if (threadIdx.x == 0) { dist[s] = 0; sigma[s] = 1.0; delta[s] = 0.0; }
        __syncthreads();
        int d = 0;
        while (true) {
            if (threadIdx.x == 0) s_more = 0;
            __syncthreads();
            bool found = false;
            for (int v = threadIdx.x; v < n; v += blockDim.x) {
                if (dist[v] != d) continue;
                for (unsigned k = R[v]; k < R[v + 1]; ++k) {
                    const int w = static_cast<int>(C[k]);
                    if (dist[w] == kUnseen) { dist[w] = static_cast<unsigned short>(d + 1); found = true; }
                }
            }
            if (found) s_more = 1;
            __syncthreads();
            if (!s_more) break;
            for (int w = threadIdx.x; w < n; w += blockDim.x) {
                if (dist[w] != d + 1) continue;
                double acc = 0.0;
                for (unsigned k = R[w]; k < R[w + 1]; ++k) {
                    const int u = static_cast<int>(C[k]);
                    if (dist[u] == d) acc += sigma[u];
                }
                sigma[w] = acc;
                delta[w] = 0.0;
            }
            ++d;
            __syncthreads();
        }
        for (int lvl = d - 1; lvl >= 1; --lvl) {
            for (int v = threadIdx.x; v < n; v += blockDim.x) {
                if (dist[v] != lvl) continue;
                double acc = 0.0;
                const double sv = sigma[v];
                for (unsigned k = R[v]; k < R[v + 1]; ++k) {
                    const int w = static_cast<int>(C[k]);
                    if (dist[w] == lvl + 1) acc += sv / sigma[w] * (1.0 + delta[w]);
                }
                delta[v] = acc;
            }
            __syncthreads();
        }
        for (int v = threadIdx.x; v < n; v += blockDim.x)
            if (v != s && dist[v] != kUnseen && delta[v] != 0.0) atomicAdd(&bc[v], delta[v]);
        __syncthreads();
    }
}

// --------------------------------------------------------- eigenvector helpers
__global__ void add_scale_kernel(const float* y, const float* x, float* xn, int n, float inv_norm, float* diff_part) {
    __shared__ float smax[256];
    float m = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = (y[i] + x[i]) * inv_norm;  // (A + I) x, normalised
        m = fmaxf(m, fabsf(v - x[i]));
        xn[i] = v;
    }
    smax[threadIdx.x] = m;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + w]);
        __syncthreads();
    }
    if (threadIdx.x == 0) diff_part[blockIdx.x] = smax[0];
}

__global__ void plus_x_sq_kernel(const float* y, const float* x, int n, double* part) {
    __shared__ double ss[256];
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double v = static_cast<double>(y[i]) + x[i];
        s += v * v;
    }
    ss[threadIdx.x] = s;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) ss[threadIdx.x] += ss[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = ss[0];
}

}  // namespace

std::vector<double> betweenness_binary(const Csr& g) {
    cuda::DeviceInit init;
    const int n = static_cast<int>(g.n);
    std::vector<double> out(n, 0.0);
    if (n == 0) return out;
    DeviceBuffer<unsigned> dR(g.R.size()), dC(std::max<size_t>(1, g.C.size()));
    PAGANI_CUDA(cudaMemcpy(dR.get(), g.R.data(), g.R.size() * 4, cudaMemcpyHostToDevice));
    if (!g.C.empty()) PAGANI_CUDA(cudaMemcpy(dC.get(), g.C.data(), g.C.size() * 4, cudaMemcpyHostToDevice));
    cudaDeviceProp prop{};
    int dev = 0;
    PAGANI_CUDA(cudaGetDevice(&dev));
    PAGANI_CUDA(cudaGetDeviceProperties(&prop, dev));
    const size_t smem = static_cast<size_t>(n) * sizeof(unsigned short);
    bool shared = n < 65535 && smem + 64 <= prop.sharedMemPerBlockOptin;
    if (const char* env = std::getenv("PAGANI_BC_SHARED")) shared = shared && std::atoi(env) != 0;  // test knob
    const int threads = 512;
    size_t blocks;
    if (shared) {
        if (smem > 48 * 1024)
            PAGANI_CUDA(cudaFuncSetAttribute(brandes_smem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(smem)));
        int per_sm = 0;
        PAGANI_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, brandes_smem_kernel, threads, smem));
        blocks = static_cast<size_t>(prop.multiProcessorCount) * std::max(1, per_sm);
    } else {
        blocks = static_cast<size_t>(prop.multiProcessorCount) * 4;
    }
    const size_t per_block = static_cast<size_t>(n) * ((shared ? 0 : sizeof(int)) + 2 * sizeof(double));
    blocks = std::min(blocks, static_cast<size_t>(n));
    blocks = std::min(blocks, std::max<size_t>(1, cuda::free_device_memory() / 2 / per_block));
    DeviceBuffer<int> dist(shared ? 1 : blocks * n);
    DeviceBuffer<double> sigma(blocks * n), delta(blocks * n), bc(n);
    DeviceBuffer<unsigned int> next(1);
    PAGANI_CUDA(cudaMemset(bc.get(), 0, n * sizeof(double)));
    PAGANI_CUDA(cudaMemset(next.get(), 0, sizeof(unsigned int)));
    if (shared)
        brandes_smem_kernel<<<static_cast<unsigned>(blocks), threads, smem>>>(dR.get(), dC.get(), n, sigma.get(),
                                                                               delta.get(), bc.get(), next.get());
    else
        brandes_kernel<<<static_cast<unsigned>(blocks), threads>>>(dR.get(), dC.get(), n, dist.get(), sigma.get(),
                                                                   delta.get(), bc.get(), next.get());
    PAGANI_KERNEL_CHECK();
    PAGANI_CUDA(cudaDeviceSynchronize());
    PAGANI_CUDA(cudaMemcpy(out.data(), bc.get(), n * sizeof(double), cudaMemcpyDeviceToHost));
    return out;
}

EigenResult eigenvector_centrality(const Csr& g, bool weighted, double tol, int max_iter) {
    cuda::DeviceInit init;
    const int n = static_cast<int>(g.n);
    EigenResult r;
    r.vector.assign(n, 0.0);
    if (n == 0) return r;
    DeviceBuffer<unsigned> dR(n + 1), dC(std::max<size_t>(1, g.C.size()));
    DeviceBuffer<float> dV(std::max<size_t>(1, g.C.size())), x(n), y(n), xn(n);
    std::vector<float> V = weighted ? g.V : std::vector<float>(g.C.size(), 1.0f);
    PAGANI_CUDA(cudaMemcpy(dR.get(), g.R.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
    if (!g.C.empty()) {
        PAGANI_CUDA(cudaMemcpy(dC.get(), g.C.data(), g.C.size() * 4, cudaMemcpyHostToDevice));
        PAGANI_CUDA(cudaMemcpy(dV.get(), V.data(), V.size() * 4, cudaMemcpyHostToDevice));
    }
    std::vector<float> x0(n, static_cast<float>(1.0 / std::sqrt(static_cast<double>(n))));
    PAGANI_CUDA(cudaMemcpy(x.get(), x0.data(), n * 4, cudaMemcpyHostToDevice));
    const unsigned grid = std::min<unsigned>(1024, cuda::grid_for(n));
    DeviceBuffer<double> sq(grid);
    DeviceBuffer<float> dmax(grid);
    std::vector<double> hsq(grid);
    std::vector<float> hmax(grid);
    for (int it = 1; it <= max_iter; ++it) {
        csr_spmv(n, dR.get(), dC.get(), dV.get(), x.get(), y.get());
        plus_x_sq_kernel<<<grid, 256>>>(y.get(), x.get(), n, sq.get());
        PAGANI_KERNEL_CHECK();
        PAGANI_CUDA(cudaMemcpy(hsq.data(), sq.get(), grid * sizeof(double), cudaMemcpyDeviceToHost));
        double norm = 0;
        for (double v : hsq) norm += v;
        norm = std::sqrt(norm);
        if (norm == 0.0) break;
        add_scale_kernel<<<grid, 256>>>(y.get(), x.get(), xn.get(), n, static_cast<float>(1.0 / norm), dmax.get());
        PAGANI_KERNEL_CHECK();
        PAGANI_CUDA(cudaMemcpy(hmax.data(), dmax.get(), grid * sizeof(float), cudaMemcpyDeviceToHost));
        float diff = *std::max_element(hmax.begin(), hmax.end());
        x.swap(xn);
        r.iterations = it;
        if (diff < tol) { r.converged = true; break; }
    }
    std::vector<float> hx(n);
    PAGANI_CUDA(cudaMemcpy(hx.data(), x.get(), n * 4, cudaMemcpyDeviceToHost));
    double nrm = 0;
    for (int i = 0; i < n; ++i) nrm += static_cast<double>(hx[i]) * hx[i];
    nrm = std::sqrt(nrm);
    for (int i = 0; i < n; ++i) r.vector[i] = nrm > 0 ? hx[i] / nrm : 0.0;
    double num = 0;
    for (std::uint32_t i = 0; i < g.n; ++i) {
        double s = 0;
        for (std::uint32_t k = g.R[i]; k < g.R[i + 1]; ++k) s += (weighted ? g.V[k] : 1.0) * r.vector[g.C[k]];
        num += r.vector[i] * s;
    }
    r.value = num;
    return r;
}

// ------------------------------------------------------------------- SpMV
struct SpMV::Impl {
    cuda::DeviceInit init;
    int n = 0;
    DeviceBuffer<unsigned> R, C;
    DeviceBuffer<double> V, x, y;
};

SpMV::SpMV(const Csr& g, bool weighted) : impl_(new Impl) {
    Impl& m = *impl_;
    m.n = static_cast<int>(g.n);
    m.R.resize(g.n + 1);
    m.C.resize(std::max<size_t>(1, g.C.size()));
    m.V.resize(std::max<size_t>(1, g.C.size()));
    m.x.resize(std::max<size_t>(1, g.n));
    m.y.resize(std::max<size_t>(1, g.n));
    std::vector<double> V(g.C.size());
    for (size_t k = 0; k < g.C.size(); ++k) V[k] = weighted ? g.V[k] : 1.0;
    PAGANI_CUDA(cudaMemcpy(m.R.get(), g.R.data(), g.R.size() * 4, cudaMemcpyHostToDevice));
    if (!g.C.empty()) {
        PAGANI_CUDA(cudaMemcpy(m.C.get(), g.C.data(), g.C.size() * 4, cudaMemcpyHostToDevice));
        PAGANI_CUDA(cudaMemcpy(m.V.get(), V.data(), V.size() * 8, cudaMemcpyHostToDevice));
    }
}

SpMV::~SpMV() { delete impl_; }

void SpMV::multiply(const std::vector<double>& x, std::vector<double>& y) {
    Impl& m = *impl_;
    y.resize(m.n);
    PAGANI_CUDA(cudaMemcpy(m.x.get(), x.data(), m.n * 8, cudaMemcpyHostToDevice));
    csr_spmv(m.n, m.R.get(), m.C.get(), m.V.get(), m.x.get(), m.y.get());
    PAGANI_CUDA(cudaMemcpy(y.data(), m.y.get(), m.n * 8, cudaMemcpyDeviceToHost));
}

}  // namespace gpu
}  // namespace pagani
