// Dense and sparse kernels used by the PAGANI GPU tools.
//
// PAGANI 2.1 does not depend on cuBLAS/cuSPARSE: their Windows DLLs are close
// to 1 GB for CUDA 12, which would make the portable package impractical.
// The kernels below are simple register-blocked implementations; only the
// CUDA runtime (linked statically) is required.
#pragma once

#include <cuda_runtime.h>

namespace pagani {
namespace gpu {

constexpr int kTile = 96;    // output tile edge per CUDA block
constexpr int kTpb = 16;     // threads per tile edge (16 x 16 threads)
constexpr int kRpt = kTile / kTpb;  // 6 x 6 outputs per thread
constexpr int kChunk = 16;   // K-dimension chunk staged in shared memory

// C(a, b) = beta * C(a, b) + sum_k A[k + a*lda] * B[k + b*ldb]
// with C stored column-major: C[a + b*ldc].  M and N must be multiples of
// kTile; K is arbitrary.  (A^T B for "K-major" operands, as cublasSgemm(T, N).)
__global__ void gemm_tn_kernel(int K, const float* __restrict__ A, int lda, const float* __restrict__ B, int ldb,
                               float beta, float* __restrict__ C, int ldc);

void gemm_tn(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);

// y = A x for a CSR matrix (one warp per row).
void csr_spmv(int n, const unsigned* R, const unsigned* C, const float* V, const float* x, float* y);
void csr_spmv(int n, const unsigned* R, const unsigned* C, const double* V, const double* x, double* y);

}  // namespace gpu
}  // namespace pagani
