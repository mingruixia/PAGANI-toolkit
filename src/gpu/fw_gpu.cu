// Blocked Floyd-Warshall on the GPU (replaces the PAGANI 2.0 CUBFW kernels).
//
// The distance matrix is processed in 96 x 96 tiles: phase 1 closes the
// diagonal tile in shared memory, phases 2 and 3 are min-plus products
// C = min(C, A (x) B) computed with a register-blocked kernel.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "cuda_common.hpp"
#include "pagani/gpu.hpp"

namespace pagani {
namespace gpu {
namespace {

using cuda::DeviceBuffer;

constexpr int TILE = 96;   // CUDA block tile
constexpr int TPB = 16;    // threads per tile edge
constexpr int RPT = TILE / TPB;  // 6 elements per thread per dimension
constexpr int KCHUNK = 16;

// C = min(C, A (x) B); all sizes multiples of TILE; row-major with leading dims.
__global__ void minplus_kernel(float* C, int ldc, const float* A, int lda, const float* B, int ldb, int K) {
    __shared__ float As[TILE][KCHUNK + 1];
    __shared__ float Bs[KCHUNK][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * TPB + tx;
    const long long r0 = static_cast<long long>(blockIdx.y) * TILE;
    const long long c0 = static_cast<long long>(blockIdx.x) * TILE;
    float acc[RPT][RPT];
#pragma unroll
    for (int a = 0; a < RPT; ++a)
#pragma unroll
        for (int b = 0; b < RPT; ++b) acc[a][b] = C[(r0 + ty + TPB * a) * ldc + c0 + tx + TPB * b];
    for (int kk = 0; kk < K; kk += KCHUNK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < (TILE * KCHUNK) / (TPB * TPB); ++e) {
            int idx = tid + TPB * TPB * e;
            int row = idx / KCHUNK, col = idx % KCHUNK;
            As[row][col] = A[(r0 + row) * lda + kk + col];
            int brow = idx / TILE, bcol = idx % TILE;
            Bs[brow][bcol] = B[static_cast<long long>(kk + brow) * ldb + c0 + bcol];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KCHUNK; ++k) {
            float av[RPT], bv[RPT];
#pragma unroll
            for (int a = 0; a < RPT; ++a) av[a] = As[ty + TPB * a][k];
#pragma unroll
            for (int b = 0; b < RPT; ++b) bv[b] = Bs[k][tx + TPB * b];
#pragma unroll
            for (int a = 0; a < RPT; ++a)
#pragma unroll
                for (int b = 0; b < RPT; ++b) acc[a][b] = fminf(acc[a][b], av[a] + bv[b]);
        }
    }
#pragma unroll
    for (int a = 0; a < RPT; ++a)
#pragma unroll
        for (int b = 0; b < RPT; ++b) C[(r0 + ty + TPB * a) * ldc + c0 + tx + TPB * b] = acc[a][b];
}

// Close one TILE x TILE diagonal tile in shared memory (diagonal must be 0).
__global__ void fw_tile_kernel(float* D, int ld) {
    __shared__ float s[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < RPT; ++b) s[ty + TPB * a][tx + TPB * b] = D[static_cast<long long>(ty + TPB * a) * ld + tx + TPB * b];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        float rowk[RPT], colk[RPT];
        for (int b = 0; b < RPT; ++b) rowk[b] = s[k][tx + TPB * b];
        for (int a = 0; a < RPT; ++a) colk[a] = s[ty + TPB * a][k];
        __syncthreads();
        for (int a = 0; a < RPT; ++a)
            for (int b = 0; b < RPT; ++b) {
                float v = colk[a] + rowk[b];
                if (v < s[ty + TPB * a][tx + TPB * b]) s[ty + TPB * a][tx + TPB * b] = v;
            }
        __syncthreads();
    }
    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < RPT; ++b) D[static_cast<long long>(ty + TPB * a) * ld + tx + TPB * b] = s[ty + TPB * a][tx + TPB * b];
}

__global__ void init_kernel(float* D, long long n0) {
    const long long total = n0 * n0;
    const float inf = __int_as_float(0x7f800000);
    for (long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x)
        D[idx] = (idx / n0 == idx % n0) ? 0.0f : inf;
}

// D (row-major, ld) = edge lengths for rows [row0, row0 + rows) and columns [col0, col0+cols)
__global__ void scatter_kernel(float* D, long long ld, const unsigned* R, const unsigned* C, const float* V,
                               int n, long long row0, long long rows, long long col0, long long cols, int weighted) {
    for (long long i = row0 + blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < row0 + rows && i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        for (unsigned k = R[i]; k < R[i + 1]; ++k) {
            long long j = C[k];
            if (j >= col0 && j < col0 + cols) D[(i - row0) * ld + (j - col0)] = weighted ? 1.0f / V[k] : 1.0f;
        }
}

// inv[i] += sum over valid j != i of 1/D[i][j]; reach[i] += number of finite entries
__global__ void rowsum_kernel(const float* D, long long ld, long long row0, long long col0, int rows, int cols, int n,
                              double* inv, unsigned long long* reach) {
    const int r = blockIdx.x;
    const long long i = row0 + r;
    if (r >= rows || i >= n) return;
    double s = 0.0;
    unsigned long long c = 0;
    for (int b = threadIdx.x; b < cols; b += blockDim.x) {
        long long j = col0 + b;
        if (j >= n || j == i) continue;
        float d = D[static_cast<long long>(r) * ld + b];
        if (isfinite(d)) { s += 1.0 / d; ++c; }
    }
    __shared__ double ss[256];
    __shared__ unsigned long long sc[256];
    ss[threadIdx.x] = s;
    sc[threadIdx.x] = c;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) { ss[threadIdx.x] += ss[threadIdx.x + w]; sc[threadIdx.x] += sc[threadIdx.x + w]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { inv[i] += ss[0]; reach[i] += sc[0]; }
}

void minplus(float* C, int ldc, const float* A, int lda, const float* B, int ldb, int M, int N, int K) {
    dim3 grid(N / TILE, M / TILE), block(TPB, TPB);
    minplus_kernel<<<grid, block>>>(C, ldc, A, lda, B, ldb, K);
    PAGANI_KERNEL_CHECK();
}

// Floyd-Warshall on an s x s device matrix (s multiple of TILE)
void fw_device(float* D, int ld, int s) {
    const int nb = s / TILE;
    for (int kb = 0; kb < nb; ++kb) {
        float* dkk = D + static_cast<long long>(kb) * TILE * ld + kb * TILE;
        float* row = D + static_cast<long long>(kb) * TILE * ld;   // TILE x s
        float* col = D + kb * TILE;                                 // s x TILE
        fw_tile_kernel<<<1, dim3(TPB, TPB)>>>(dkk, ld);
        PAGANI_KERNEL_CHECK();
        if (nb == 1) continue;
        minplus(row, ld, dkk, ld, row, ld, TILE, s, TILE);   // phase 2: row panel
        minplus(col, ld, col, ld, dkk, ld, s, TILE, TILE);   // phase 2: column panel
        minplus(D, ld, col, ld, row, ld, s, s, TILE);         // phase 3
    }
}

std::size_t round_up(std::size_t n, std::size_t m) { return (n + m - 1) / m * m; }

}  // namespace

std::size_t fw_matrix_bytes(std::uint32_t n) {
    std::size_t n0 = round_up(n, TILE);
    return n0 * n0 * sizeof(float);
}

EfficiencyResult efficiency_fw(const Csr& g, bool weighted) {
    cuda::init_device();
    const int n = static_cast<int>(g.n);
    std::uint32_t isolated = 0;
    for (std::uint32_t i = 0; i < g.n; ++i) isolated += (g.R[i + 1] == g.R[i]);
    if (weighted)
        for (float w : g.V)
            if (!(w > 0.0f)) throw Error("weighted path length requires positive edge weights");

    DeviceBuffer<unsigned> dR(g.R.size()), dC(std::max<size_t>(1, g.C.size()));
    DeviceBuffer<float> dV(weighted ? std::max<size_t>(1, g.V.size()) : 1);
    PAGANI_CUDA(cudaMemcpy(dR.get(), g.R.data(), g.R.size() * 4, cudaMemcpyHostToDevice));
    if (!g.C.empty()) PAGANI_CUDA(cudaMemcpy(dC.get(), g.C.data(), g.C.size() * 4, cudaMemcpyHostToDevice));
    if (weighted && !g.V.empty()) PAGANI_CUDA(cudaMemcpy(dV.get(), g.V.data(), g.V.size() * 4, cudaMemcpyHostToDevice));

    DeviceBuffer<double> dinv(n);
    DeviceBuffer<unsigned long long> dreach(n);
    PAGANI_CUDA(cudaMemset(dinv.get(), 0, n * sizeof(double)));
    PAGANI_CUDA(cudaMemset(dreach.get(), 0, n * sizeof(unsigned long long)));

    const std::size_t n0r = round_up(n, TILE);
    const std::size_t resident_bytes = n0r * n0r * sizeof(float);
    if (resident_bytes + (512ull << 20) < cuda::free_device_memory()) {
        DeviceBuffer<float> D(n0r * n0r);
        init_kernel<<<cuda::grid_for(n0r * n0r), 256>>>(D.get(), n0r);
        scatter_kernel<<<cuda::grid_for(n), 256>>>(D.get(), n0r, dR.get(), dC.get(), dV.get(), n, 0, n0r, 0, n0r, weighted);
        PAGANI_KERNEL_CHECK();
        fw_device(D.get(), static_cast<int>(n0r), static_cast<int>(n0r));
        rowsum_kernel<<<n, 256>>>(D.get(), n0r, 0, 0, n, static_cast<int>(n0r), n, dinv.get(), dreach.get());
        PAGANI_KERNEL_CHECK();
    } else {
        // tiles of S x S streamed from host memory
        const int S = 96 * 32;
        const std::size_t n0 = round_up(n, S);
        const int nb = static_cast<int>(n0 / S);
        const std::size_t tile_elems = static_cast<std::size_t>(S) * S;
        std::cout << "distance matrix (" << ((n0 * n0 * 4) >> 20) << " MB) does not fit in GPU memory: streaming "
                  << nb * nb << " tiles from host memory" << std::endl;
        std::vector<float> host;
        try {
            host.resize(n0 * n0);
        } catch (const std::bad_alloc&) {
            throw Error("not enough host memory for the distance matrix (" + std::to_string((n0 * n0 * 4) >> 20) +
                        " MB); use BFS_MulCPU.exe (CPU, no distance matrix) instead");
        }
        DeviceBuffer<float> Tkk(tile_elems), Ta(tile_elems), Tb(tile_elems), Tc(tile_elems);
        auto tile = [&](int bi, int bj) { return host.data() + (static_cast<std::size_t>(bi) * nb + bj) * tile_elems; };
        auto up = [&](float* d, int bi, int bj) {
            PAGANI_CUDA(cudaMemcpy(d, tile(bi, bj), tile_elems * 4, cudaMemcpyHostToDevice));
        };
        auto down = [&](const float* d, int bi, int bj) {
            PAGANI_CUDA(cudaMemcpy(tile(bi, bj), d, tile_elems * 4, cudaMemcpyDeviceToHost));
        };
        // initialise tiles on the device
        for (int bi = 0; bi < nb; ++bi)
            for (int bj = 0; bj < nb; ++bj) {
                const float inf = std::numeric_limits<float>::infinity();
                std::vector<float> t(tile_elems, inf);
                if (bi == bj)
                    for (int a = 0; a < S; ++a) t[static_cast<std::size_t>(a) * S + a] = 0.0f;
                PAGANI_CUDA(cudaMemcpy(Ta.get(), t.data(), tile_elems * 4, cudaMemcpyHostToDevice));
                scatter_kernel<<<cuda::grid_for(S), 256>>>(Ta.get(), S, dR.get(), dC.get(), dV.get(), n,
                                                           1ll * bi * S, S, 1ll * bj * S, S, weighted);
                PAGANI_KERNEL_CHECK();
                down(Ta.get(), bi, bj);
            }
        Timer timer;
        for (int kb = 0; kb < nb; ++kb) {
            up(Tkk.get(), kb, kb);
            fw_device(Tkk.get(), S, S);
            down(Tkk.get(), kb, kb);
            for (int j = 0; j < nb; ++j) {
                if (j == kb) continue;
                up(Ta.get(), kb, j);
                minplus(Ta.get(), S, Tkk.get(), S, Ta.get(), S, S, S, S);
                down(Ta.get(), kb, j);
                up(Ta.get(), j, kb);
                minplus(Ta.get(), S, Ta.get(), S, Tkk.get(), S, S, S, S);
                down(Ta.get(), j, kb);
            }
            for (int i = 0; i < nb; ++i) {
                if (i == kb) continue;
                up(Ta.get(), i, kb);
                for (int j = 0; j < nb; ++j) {
                    if (j == kb) continue;
                    up(Tb.get(), kb, j);
                    up(Tc.get(), i, j);
                    minplus(Tc.get(), S, Ta.get(), S, Tb.get(), S, S, S, S);
                    down(Tc.get(), i, j);
                }
            }
            std::cout << "  Floyd-Warshall round " << kb + 1 << "/" << nb << " (" << static_cast<long long>(timer.seconds()) << " s)"
                      << std::endl;
        }
        for (int bi = 0; bi < nb; ++bi)
            for (int bj = 0; bj < nb; ++bj) {
                up(Ta.get(), bi, bj);
                rowsum_kernel<<<S, 256>>>(Ta.get(), S, 1ll * bi * S, 1ll * bj * S, S, S, n, dinv.get(), dreach.get());
                PAGANI_KERNEL_CHECK();
            }
    }
    PAGANI_CUDA(cudaDeviceSynchronize());
    std::vector<double> inv(n);
    std::vector<unsigned long long> reach(n);
    PAGANI_CUDA(cudaMemcpy(inv.data(), dinv.get(), n * sizeof(double), cudaMemcpyDeviceToHost));
    PAGANI_CUDA(cudaMemcpy(reach.data(), dreach.get(), n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    bool connected = true;
    for (int i = 0; i < n; ++i)
        if (reach[i] + 1 < static_cast<unsigned long long>(n)) connected = false;
    return efficiency_from_inverse_sums(inv, connected, isolated);
}

}  // namespace gpu
}  // namespace pagani
