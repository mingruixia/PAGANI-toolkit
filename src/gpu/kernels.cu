#include "cuda_common.hpp"
#include "kernels.cuh"

namespace pagani {
namespace gpu {

__global__ void gemm_tn_kernel(int K, const float* __restrict__ A, int lda, const float* __restrict__ B, int ldb,
                               float beta, float* __restrict__ C, int ldc) {
    __shared__ float As[kChunk][kTile + 1];
    __shared__ float Bs[kChunk][kTile + 1];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * kTpb + tx;
    const long long a0 = static_cast<long long>(blockIdx.x) * kTile;   // rows of C (columns of A)
    const long long b0 = static_cast<long long>(blockIdx.y) * kTile;   // columns of C (columns of B)
    float acc[kRpt][kRpt];
#pragma unroll
    for (int i = 0; i < kRpt; ++i)
#pragma unroll
        for (int j = 0; j < kRpt; ++j) acc[i][j] = 0.0f;

    for (int kk = 0; kk < K; kk += kChunk) {
        __syncthreads();
        // stage A(kk:kk+16, a0:a0+96) and B(kk:kk+16, b0:b0+96); element e -> (k = e % 16, col = e / 16)
#pragma unroll
        for (int e = 0; e < (kTile * kChunk) / (kTpb * kTpb); ++e) {
            const int idx = tid + kTpb * kTpb * e;
            const int kc = idx % kChunk, col = idx / kChunk;
            const int k = kk + kc;
            As[kc][col] = k < K ? A[(a0 + col) * lda + k] : 0.0f;
            Bs[kc][col] = k < K ? B[(b0 + col) * ldb + k] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kc = 0; kc < kChunk; ++kc) {
            float av[kRpt], bv[kRpt];
#pragma unroll
            for (int i = 0; i < kRpt; ++i) av[i] = As[kc][tx + kTpb * i];
#pragma unroll
            for (int j = 0; j < kRpt; ++j) bv[j] = Bs[kc][ty + kTpb * j];
#pragma unroll
            for (int i = 0; i < kRpt; ++i)
#pragma unroll
                for (int j = 0; j < kRpt; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < kRpt; ++i)
#pragma unroll
        for (int j = 0; j < kRpt; ++j) {
            float* c = C + (a0 + tx + kTpb * i) + (b0 + ty + kTpb * j) * static_cast<long long>(ldc);
            *c = beta == 0.0f ? acc[i][j] : beta * *c + acc[i][j];
        }
}

void gemm_tn(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    if (M % kTile || N % kTile) throw Error("internal error: GEMM size not a multiple of the tile size");
    dim3 grid(M / kTile, N / kTile), block(kTpb, kTpb);
    gemm_tn_kernel<<<grid, block>>>(K, A, lda, B, ldb, beta, C, ldc);
    PAGANI_KERNEL_CHECK();
}

template <typename T>
__global__ void spmv_kernel(int n, const unsigned* __restrict__ R, const unsigned* __restrict__ C,
                            const T* __restrict__ V, const T* __restrict__ x, T* __restrict__ y) {
    const int lane = threadIdx.x & 31;
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const long long nwarps = (static_cast<long long>(gridDim.x) * blockDim.x) >> 5;
    for (long long row = warp; row < n; row += nwarps) {
        T s = 0;
        for (unsigned k = R[row] + lane; k < R[row + 1]; k += 32) s += V[k] * x[C[k]];
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0) y[row] = s;
    }
}

void csr_spmv(int n, const unsigned* R, const unsigned* C, const float* V, const float* x, float* y) {
    spmv_kernel<float><<<cuda::grid_for(static_cast<size_t>(n) * 32, 256, 8192), 256>>>(n, R, C, V, x, y);
    PAGANI_KERNEL_CHECK();
}

void csr_spmv(int n, const unsigned* R, const unsigned* C, const double* V, const double* x, double* y) {
    spmv_kernel<double><<<cuda::grid_for(static_cast<size_t>(n) * 32, 256, 8192), 256>>>(n, R, C, V, x, y);
    PAGANI_KERNEL_CHECK();
}

}  // namespace gpu
}  // namespace pagani
