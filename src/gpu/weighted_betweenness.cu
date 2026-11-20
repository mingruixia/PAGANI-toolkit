// Weighted Brandes on symmetric CSR. Each block owns one source and its scratch.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <math_constants.h>

#include "cuda_common.hpp"
#include "pagani/gpu.hpp"

namespace pagani::gpu {
namespace {

using Bits = unsigned long long;
using cuda::DeviceBuffer;

struct Graph {
    const unsigned* rows;
    const unsigned* cols;
    const double* length;
    unsigned n;
};

__device__ bool shortest_arc(double from, double length, double to) {
    double candidate = from + length;
    return from < to && fabs(candidate - to) <=
        kWeightedBetweennessRelativeTolerance * fmax(candidate, to);
}

__device__ double add_log_counts(double a, double b) {
    if (a == -CUDART_INF) return b;
    if (b == -CUDART_INF) return a;
    return fmax(a, b) + log1p(exp(-fabs(a - b)));
}

template<bool wide_rows>
__global__ void weighted_sources(Graph g, unsigned first, Bits* distances, double* paths,
                                 double* dependencies, int* indices, int* error) {
    const unsigned source = first + blockIdx.x;
    const size_t base = size_t(blockIdx.x) * g.n;
    Bits* distance = distances + base;
    double* log_paths = paths + base;
    double* delta = dependencies + base;
    int* queue = indices + size_t(blockIdx.x) * (size_t(5) * g.n + 1);
    int* next = queue + g.n;
    int* tag = next + g.n;
    int* predecessors = tag + g.n;
    int* levels = predecessors + g.n;
    __shared__ int tail;
    constexpr int lanes = wide_rows ? 32 : 1;
    const int lane = threadIdx.x % lanes;
    for (unsigned v = threadIdx.x; v < g.n; v += blockDim.x) {
        distance[v] = __double_as_longlong(v == source ? 0.0 : CUDART_INF);
        delta[v] = 0.0;
        tag[v] = 0;
    }
    if (threadIdx.x == 0) queue[0] = source;
    __syncthreads();

    // Positive double bit patterns have the same ordering as unsigned integers.
    // Re-enqueue every improved label, at most once in the next frontier.
    int count = 1;
    for (int round = 1; count; ++round) {
        if (threadIdx.x == 0) tail = 0;
        __syncthreads();
        // Coalesce long rows across a warp. A read-only snapshot from the
        // previous round filters losing relaxations before issuing atomics.
        if constexpr (wide_rows) {
            for (unsigned v = threadIdx.x; v < g.n; v += blockDim.x)
                delta[v] = __longlong_as_double(distance[v]);
            __syncthreads();
        }
        for (int q = threadIdx.x / lanes; q < count; q += blockDim.x / lanes) {
            unsigned u = queue[q];
            double du;
            if constexpr (wide_rows) du = delta[u];
            else du = __longlong_as_double(atomicCAS(distance + u, 0ULL, 0ULL));
            for (size_t k = size_t(g.rows[u]) + lane; k < g.rows[u + 1]; k += lanes) {
                unsigned v = g.cols[k];
                double proposed = du + g.length[k];
                if constexpr (wide_rows) {
                    if (proposed >= delta[v]) continue;
                }
                Bits candidate = __double_as_longlong(proposed);
                if (atomicMin(distance + v, candidate) > candidate &&
                    atomicExch(tag + v, round) != round)
                    next[atomicAdd(&tail, 1)] = v;
            }
        }
        __syncthreads();
        count = tail;
        int* tmp = queue; queue = next; next = tmp;
        __syncthreads();
    }

    for (unsigned v = threadIdx.x / lanes; v < g.n; v += blockDim.x / lanes) {
        if (wide_rows && lane == 0) delta[v] = 0.0;
        const double dv = __longlong_as_double(distance[v]);
        if (dv == CUDART_INF || v == source) continue;
        int incoming = 0;
        for (size_t k = size_t(g.rows[v]) + lane; k < g.rows[v + 1]; k += lanes)
            if (shortest_arc(__longlong_as_double(distance[g.cols[k]]), g.length[k], dv))
                ++incoming;
        if constexpr (wide_rows)
            for (int offset = 16; offset; offset /= 2)
                incoming += __shfl_down_sync(0xffffffff, incoming, offset);
        if (lane == 0) {
            predecessors[v] = incoming;
            if (incoming == 0) atomicExch(error, 1);
        }
    }
    if (threadIdx.x == 0) {
        queue[0] = source;
        levels[0] = 0;
        tail = 1;
    }
    __syncthreads();

    // Topologically order the shortest-path DAG. Pull counts from CSR rows;
    // each destination enters the next level after its final predecessor.
    int begin = 0, end = 1, depth = 0;
    while (begin < end) {
        for (int q = begin + threadIdx.x / lanes; q < end; q += blockDim.x / lanes) {
            unsigned u = queue[q];
            double du = __longlong_as_double(distance[u]);
            double total = (u == source && lane == 0) ? 0.0 : -CUDART_INF;
            for (size_t k = size_t(g.rows[u]) + lane; k < g.rows[u + 1]; k += lanes) {
                unsigned v = g.cols[k];
                double dv = __longlong_as_double(distance[v]);
                if (shortest_arc(dv, g.length[k], du))
                    total = add_log_counts(total, log_paths[v]);
                else if (shortest_arc(du, g.length[k], dv) &&
                         atomicSub(predecessors + v, 1) == 1)
                    queue[atomicAdd(&tail, 1)] = v;
            }
            if constexpr (wide_rows)
                for (int offset = 16; offset; offset /= 2)
                    total = add_log_counts(total, __shfl_down_sync(0xffffffff, total, offset));
            if (lane == 0) log_paths[u] = total;
        }
        ++depth;
        if (threadIdx.x == 0) levels[depth] = end;
        __syncthreads();
        begin = end;
        end = tail;
        __syncthreads();
    }

    for (int level = depth - 1; level >= 1; --level) {
        for (int q = levels[level] + threadIdx.x / lanes; q < levels[level + 1]; q += blockDim.x / lanes) {
            unsigned u = queue[q];
            double du = __longlong_as_double(distance[u]);
            double sum = 0.0;
            for (size_t k = size_t(g.rows[u]) + lane; k < g.rows[u + 1]; k += lanes) {
                unsigned v = g.cols[k];
                if (shortest_arc(du, g.length[k], __longlong_as_double(distance[v])))
                    sum += exp(log_paths[u] - log_paths[v]) * (1.0 + delta[v]);
            }
            if constexpr (wide_rows)
                for (int offset = 16; offset; offset /= 2)
                    sum += __shfl_down_sync(0xffffffff, sum, offset);
            if (lane == 0) delta[u] = sum;
        }
        __syncthreads();
    }
}

__global__ void sum_sources(const double* delta, double* out, unsigned n, unsigned count) {
    for (size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x; v < n;
         v += size_t(gridDim.x) * blockDim.x) {
        double value = out[v];
        for (unsigned s = 0; s < count; ++s) value += delta[size_t(s) * n + v];
        out[v] = value;
    }
}

}  // namespace

std::vector<double> betweenness_weighted(const Csr& g) {
    const auto lengths = weighted_betweenness_lengths(g);
    std::vector<double> result(g.n, 0.0);
    if (lengths.empty()) return result;

    cuda::DeviceInit init;
    int device;
    cudaDeviceProp prop{};
    PAGANI_CUDA(cudaGetDevice(&device));
    PAGANI_CUDA(cudaGetDeviceProperties(&prop, device));
    DeviceBuffer<unsigned> rows(g.R.size()), cols(g.C.size());
    DeviceBuffer<double> weights(lengths.size()), out(g.n);
    DeviceBuffer<int> error(1);
    const size_t n = g.n;
    const size_t per_source = n * (sizeof(Bits) + 2 * sizeof(double) + 5 * sizeof(int)) + sizeof(int);
    const size_t capacity = cuda::free_device_memory() / 2 / per_source;
    if (!capacity) throw cuda::ResourceError("insufficient GPU memory for one weighted betweenness source");
    const unsigned batch = static_cast<unsigned>(std::min({capacity, n, size_t(prop.multiProcessorCount) * 4}));
    DeviceBuffer<Bits> distances(batch * n);
    DeviceBuffer<double> paths(batch * n), delta(batch * n);
    DeviceBuffer<int> indices(batch * (5 * n + 1));
    const size_t peak = (rows.size() + cols.size()) * sizeof(unsigned) +
        (weights.size() + out.size()) * sizeof(double) + sizeof(int) + batch * per_source;
    std::cout << "Weighted BC CUDA buffer peak: " << peak << " bytes" << std::endl;
    PAGANI_CUDA(cudaMemcpy(rows.get(), g.R.data(), rows.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    PAGANI_CUDA(cudaMemcpy(cols.get(), g.C.data(), cols.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    PAGANI_CUDA(cudaMemcpy(weights.get(), lengths.data(), lengths.size() * sizeof(double), cudaMemcpyHostToDevice));
    PAGANI_CUDA(cudaMemset(error.get(), 0, sizeof(int)));
    PAGANI_CUDA(cudaMemset(out.get(), 0, n * sizeof(double)));
    Graph graph{rows.get(), cols.get(), weights.get(), g.n};
    const bool wide_rows = g.nnz() / n >= 32;
    for (unsigned first = 0; first < g.n; first += batch) {
        unsigned count = std::min(batch, g.n - first);
        if (wide_rows)
            weighted_sources<true><<<count, 128>>>(graph, first, distances.get(), paths.get(), delta.get(), indices.get(), error.get());
        else
            weighted_sources<false><<<count, 128>>>(graph, first, distances.get(), paths.get(), delta.get(), indices.get(), error.get());
        PAGANI_KERNEL_CHECK();
        sum_sources<<<cuda::grid_for(n), 256>>>(delta.get(), out.get(), g.n, count);
        PAGANI_KERNEL_CHECK();
    }
    int failed;
    PAGANI_CUDA(cudaMemcpy(&failed, error.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (failed) throw Error("weighted betweenness: edge lengths exceed the precision of the shortest-path distances");
    PAGANI_CUDA(cudaMemcpy(result.data(), out.get(), n * sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

}  // namespace pagani::gpu
