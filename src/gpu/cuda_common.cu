#include "cuda_common.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace pagani {
namespace cuda {

void fail(const char* what, const char* file, int line, const std::string& detail) {
    std::string f = file;
    auto p = f.find_last_of("/\\");
    if (p != std::string::npos) f = f.substr(p + 1);
    throw Error(std::string("CUDA error: ") + detail + "\n  at " + f + ":" + std::to_string(line) + "  (" + what + ")");
}

size_t free_device_memory() {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return 0;
    // test knob: pretend the GPU has less memory so that the streaming code
    // paths for very large data can be validated on small inputs
    if (const char* env = std::getenv("PAGANI_GPU_MEM_LIMIT_MB")) {
        const long long mb = std::atoll(env);
        if (mb > 0) f = std::min(f, static_cast<size_t>(mb) << 20);
    }
    return f;
}

void init_device(size_t* free_bytes, size_t* total_bytes) {
    static bool done = false;
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0)
        throw Error(std::string("no usable NVIDIA GPU found (") + cudaGetErrorString(e) +
                    "). Please install a recent NVIDIA driver; the GPU tools require CUDA 12 compatible drivers (>= 527.41 on Windows).");
    int dev = 0;
    if (const char* env = std::getenv("PAGANI_GPU")) dev = std::atoi(env);
    if (dev < 0 || dev >= count) throw Error("PAGANI_GPU=" + std::to_string(dev) + " but only " + std::to_string(count) + " GPU(s) present");
    PAGANI_CUDA(cudaSetDevice(dev));
    size_t f = 0, t = 0;
    PAGANI_CUDA(cudaMemGetInfo(&f, &t));
    if (!done) {
        cudaDeviceProp prop{};
        PAGANI_CUDA(cudaGetDeviceProperties(&prop, dev));
        std::cout << "GPU " << dev << ": " << prop.name << " (compute " << prop.major << "." << prop.minor << ", "
                  << (t >> 20) << " MB, " << (f >> 20) << " MB free)" << std::endl;
        done = true;
    }
    if (free_bytes) *free_bytes = f;
    if (total_bytes) *total_bytes = t;
}

}  // namespace cuda
}  // namespace pagani
