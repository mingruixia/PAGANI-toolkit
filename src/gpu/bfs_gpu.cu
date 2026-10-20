// PAGANI Toolkit - path length / efficiency of unweighted networks on the GPU
// with a bit-parallel multi-source breadth-first search (MS-BFS; Then et al.,
// PVLDB 2014): 64*W sources are searched at once, one bit per source.
//
// Level d of a batch: for every node v,
//   new(v) = OR_{u in N(v)} frontier(u)  AND NOT visited(v)
// i.e. the sources that reach v for the first time at distance d.  The network
// is undirected, so d(s, v) = d(v, s) and the inverse-distance sum of node v
// is accumulated at v itself: inv(v) += popcount(new(v)) / d.  No distance
// matrix is needed (memory ~ edges + 24*W bytes per node).
#include <algorithm>
#include <iostream>
#include <vector>

#include "cuda_common.hpp"
#include "pagani/gpu.hpp"

namespace pagani {
namespace gpu {
namespace {

using cuda::DeviceBuffer;
using u64 = unsigned long long;

constexpr int W = 8;  // 64-bit words per node: 512 sources per batch (W divides 32)

// visited/frontier of node v for the batch starting at source s0: the bit of
// source v itself, and all bits of the padding sources >= n set as visited
__global__ void msbfs_init(int n, long long s0, u64* frontier, u64* visited) {
    const long long total = static_cast<long long>(n) * W;
    for (long long t = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; t < total;
         t += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long v = t / W;
        const int w = static_cast<int>(t % W);
        const long long first = s0 + 64ll * w;  // source of bit 0 of word w
        u64 pad = 0;
        if (first + 64 > n) pad = first >= n ? ~0ull : (~0ull << (n - first));
        u64 self = 0;
        if (v >= first && v < first + 64) self = 1ull << (v - first);
        frontier[t] = self;
        visited[t] = self | pad;
    }
}

__global__ void msbfs_level(const unsigned* __restrict__ R, const unsigned* __restrict__ C, int n,
                            const u64* __restrict__ frontier, u64* __restrict__ next, u64* __restrict__ visited,
                            double* inv, u64* reach, int d, int* more) {
    // the range is padded to whole warps so that every lane takes part in the shuffles
    const long long total = (static_cast<long long>(n) * W + 31) / 32 * 32;
    for (long long t = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; t < total;
         t += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long v = t / W;
        const int w = static_cast<int>(t % W);
        u64 fresh = 0;
        if (v < n) {
            const u64 vis = visited[t];
            if (vis != ~0ull) {
                u64 acc = 0;
                for (unsigned k = R[v], e = R[v + 1]; k < e; ++k) {
                    acc |= frontier[static_cast<long long>(C[k]) * W + w];
                    if ((acc | vis) == ~0ull) break;
                }
                fresh = acc & ~vis;
                if (fresh) visited[t] = vis | fresh;
            }
            next[t] = fresh;
        }
        int c = __popcll(fresh);
        for (int o = W / 2; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffffu, c, o, W);
        if (w == 0 && v < n && c) {  // one thread per node: no atomics, fixed summation order
            inv[v] += static_cast<double>(c) / d;
            reach[v] += static_cast<u64>(c);
        }
        if (fresh) *more = 1;
    }
}

}  // namespace

bool efficiency_bfs_fits(const Csr& g) {
    cuda::init_device();
    const std::size_t need = (g.R.size() + g.C.size()) * 4 + static_cast<std::size_t>(g.n) * (3 * W * 8 + 16);
    return need + (256ull << 20) < cuda::free_device_memory();
}

EfficiencyResult efficiency_bfs(const Csr& g) {
    cuda::init_device();
    const int n = static_cast<int>(g.n);
    std::uint32_t isolated = 0;
    for (std::uint32_t i = 0; i < g.n; ++i) isolated += (g.R[i + 1] == g.R[i]);
    std::vector<double> inv(n, 0.0);
    if (n < 2) return efficiency_from_inverse_sums(inv, true, isolated);

    DeviceBuffer<unsigned> dR(g.R.size()), dC(std::max<size_t>(1, g.C.size()));
    PAGANI_CUDA(cudaMemcpy(dR.get(), g.R.data(), g.R.size() * 4, cudaMemcpyHostToDevice));
    if (!g.C.empty()) PAGANI_CUDA(cudaMemcpy(dC.get(), g.C.data(), g.C.size() * 4, cudaMemcpyHostToDevice));
    const std::size_t words = static_cast<std::size_t>(n) * W;
    DeviceBuffer<u64> frontier(words), next(words), visited(words), dreach(n);
    DeviceBuffer<double> dinv(n);
    DeviceBuffer<int> dmore(1);
    PAGANI_CUDA(cudaMemset(dinv.get(), 0, n * sizeof(double)));
    PAGANI_CUDA(cudaMemset(dreach.get(), 0, n * sizeof(u64)));

    const unsigned grid = cuda::grid_for(words, 256, 8192);
    u64* fr = frontier.get();
    u64* nx = next.get();
    for (long long s0 = 0; s0 < n; s0 += 64ll * W) {
        msbfs_init<<<grid, 256>>>(n, s0, fr, visited.get());
        PAGANI_KERNEL_CHECK();
        for (int d = 1;; ++d) {
            PAGANI_CUDA(cudaMemsetAsync(dmore.get(), 0, sizeof(int)));
            msbfs_level<<<grid, 256>>>(dR.get(), dC.get(), n, fr, nx, visited.get(), dinv.get(), dreach.get(), d,
                                       dmore.get());
            PAGANI_KERNEL_CHECK();
            int more = 0;
            PAGANI_CUDA(cudaMemcpy(&more, dmore.get(), sizeof(int), cudaMemcpyDeviceToHost));
            if (!more) break;
            std::swap(fr, nx);
        }
    }
    std::vector<u64> reach(n);
    PAGANI_CUDA(cudaMemcpy(inv.data(), dinv.get(), n * sizeof(double), cudaMemcpyDeviceToHost));
    PAGANI_CUDA(cudaMemcpy(reach.data(), dreach.get(), n * sizeof(u64), cudaMemcpyDeviceToHost));
    bool connected = true;
    for (int i = 0; i < n; ++i)
        if (reach[i] + 1 < static_cast<u64>(n)) connected = false;
    return efficiency_from_inverse_sums(inv, connected, isolated);
}

}  // namespace gpu
}  // namespace pagani
