// CUDA helpers: error checking that throws pagani::Error, device info.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <string>

#include "pagani/common.hpp"

namespace pagani {
namespace cuda {

[[noreturn]] void fail(const char* what, const char* file, int line, const std::string& detail);

#define PAGANI_CUDA(call)                                                                             \
    do {                                                                                              \
        cudaError_t e_ = (call);                                                                      \
        if (e_ != cudaSuccess) ::pagani::cuda::fail(#call, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    } while (0)

#define PAGANI_CUBLAS(call)                                                                                 \
    do {                                                                                                    \
        cublasStatus_t s_ = (call);                                                                         \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                    \
            ::pagani::cuda::fail(#call, __FILE__, __LINE__, "cuBLAS status " + std::to_string(static_cast<int>(s_))); \
    } while (0)

#define PAGANI_CUSPARSE(call)                                                                       \
    do {                                                                                            \
        cusparseStatus_t s_ = (call);                                                               \
        if (s_ != CUSPARSE_STATUS_SUCCESS)                                                          \
            ::pagani::cuda::fail(#call, __FILE__, __LINE__, cusparseGetErrorString(s_));            \
    } while (0)

#define PAGANI_KERNEL_CHECK() PAGANI_CUDA(cudaGetLastError())

// Select the device (PAGANI_GPU environment variable, default 0), print its
// name once and return free/total memory in bytes.
void init_device(size_t* free_bytes = nullptr, size_t* total_bytes = nullptr);
size_t free_device_memory();

// RAII device buffer.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { resize(n); }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void resize(size_t n) {
        if (n == n_) return;
        release();
        if (n) {
            cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T));
            if (e != cudaSuccess)
                throw Error("GPU memory allocation of " + std::to_string((n * sizeof(T)) >> 20) +
                            " MB failed (" + cudaGetErrorString(e) + "). Free GPU memory: " +
                            std::to_string(free_device_memory() >> 20) + " MB");
        }
        n_ = n;
    }
    void release() {
        if (p_) cudaFree(p_);
        p_ = nullptr;
        n_ = 0;
    }
    T* get() const { return p_; }
    size_t size() const { return n_; }
    void swap(DeviceBuffer& o) noexcept {
        T* p = p_; p_ = o.p_; o.p_ = p;
        size_t n = n_; n_ = o.n_; o.n_ = n;
    }

private:
    T* p_ = nullptr;
    size_t n_ = 0;
};

// RAII pinned host buffer.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t n) { resize(n); }
    ~PinnedBuffer() { if (p_) cudaFreeHost(p_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    void resize(size_t n) {
        if (n == n_) return;
        if (p_) cudaFreeHost(p_);
        p_ = nullptr;
        if (n) PAGANI_CUDA(cudaMallocHost(reinterpret_cast<void**>(&p_), n * sizeof(T)));
        n_ = n;
    }
    T* get() const { return p_; }
    size_t size() const { return n_; }

private:
    T* p_ = nullptr;
    size_t n_ = 0;
};

// Declare as the first member of GPU classes so the device is selected (and a
// clear message printed when no GPU is present) before any CUDA library call.
struct DeviceInit {
    DeviceInit() { init_device(); }
};

class CublasHandle {
public:
    CublasHandle() { PAGANI_CUBLAS(cublasCreate(&h_)); }
    ~CublasHandle() { cublasDestroy(h_); }
    cublasHandle_t get() const { return h_; }
private:
    cublasHandle_t h_{};
};

class CusparseHandle {
public:
    CusparseHandle() { PAGANI_CUSPARSE(cusparseCreate(&h_)); }
    ~CusparseHandle() { cusparseDestroy(h_); }
    cusparseHandle_t get() const { return h_; }
private:
    cusparseHandle_t h_{};
};

inline unsigned grid_for(size_t n, unsigned threads = 256, unsigned max_blocks = 4096) {
    size_t b = (n + threads - 1) / threads;
    return static_cast<unsigned>(b < 1 ? 1 : (b > max_blocks ? max_blocks : b));
}

}  // namespace cuda
}  // namespace pagani
