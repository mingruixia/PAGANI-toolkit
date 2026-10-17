#include "pagani/random_network.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pagani/common.hpp"

namespace pagani {

namespace {

std::uint64_t initial_seed() {
    if (const char* env = std::getenv("PAGANI_SEED")) {
        char* end = nullptr;
        unsigned long long v = std::strtoull(env, &end, 10);
        if (end != env) return v;
    }
    std::random_device rd;
    return (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
}

inline std::uint64_t edge_key(std::uint32_t a, std::uint32_t b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32) | b;
}

struct KeyHash {
    size_t operator()(std::uint64_t k) const noexcept {
        k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33;
        return static_cast<size_t>(k);
    }
};

}  // namespace

RandomNetworkGenerator::RandomNetworkGenerator() : rng_(initial_seed()) {}

Csr RandomNetworkGenerator::rewire(const Csr& src, double swaps_per_edge) {
    const bool weighted = src.weighted();
    // Edge list with u < v.
    struct E { std::uint32_t u, v; float w; };
    std::vector<E> edges;
    edges.reserve(src.nnz() / 2);
    for (std::uint32_t i = 0; i < src.n; ++i)
        for (std::uint32_t k = src.R[i]; k < src.R[i + 1]; ++k)
            if (src.C[k] > i) edges.push_back({i, src.C[k], weighted ? src.V[k] : 0.0f});
    if (edges.size() * 2 != src.nnz())
        throw Error("random network generation: input network is not symmetric");
    const std::uint64_t m = edges.size();

    if (m >= 2) {
        std::unordered_set<std::uint64_t, KeyHash> present;
        present.reserve(static_cast<size_t>(m * 2));
        for (const auto& e : edges) present.insert(edge_key(e.u, e.v));

        std::uniform_int_distribution<std::uint64_t> pick(0, m - 1);
        std::bernoulli_distribution coin(0.5);
        const std::uint64_t attempts = static_cast<std::uint64_t>(swaps_per_edge * static_cast<double>(m));
        for (std::uint64_t t = 0; t < attempts; ++t) {
            std::uint64_t a = pick(rng_), b = pick(rng_);
            if (a == b) continue;
            E& e1 = edges[a];
            E& e2 = edges[b];
            std::uint32_t v1 = e1.u, v2 = e1.v, v3 = e2.u, v4 = e2.v;
            if (v1 == v3 || v1 == v4 || v2 == v3 || v2 == v4) continue;
            // 1-2, 3-4  ->  1-3, 2-4   or   1-4, 2-3
            std::uint32_t x1, y1, x2, y2;
            if (coin(rng_)) { x1 = v1; y1 = v3; x2 = v2; y2 = v4; }
            else            { x1 = v1; y1 = v4; x2 = v2; y2 = v3; }
            std::uint64_t k1 = edge_key(x1, y1), k2 = edge_key(x2, y2);
            if (present.count(k1) || present.count(k2)) continue;
            present.erase(edge_key(v1, v2));
            present.erase(edge_key(v3, v4));
            present.insert(k1);
            present.insert(k2);
            // weights travel with the edge objects
            e1.u = std::min(x1, y1); e1.v = std::max(x1, y1);
            e2.u = std::min(x2, y2); e2.v = std::max(x2, y2);
        }
    }

    // Rebuild CSR (degrees are unchanged, so R is identical).
    Csr out;
    out.n = src.n;
    out.R = src.R;
    out.C.assign(src.nnz(), 0);
    if (weighted) { out.V.assign(src.nnz(), 0.0f); out.has_weights_flag = true; }
    std::vector<std::uint32_t> fill(src.R.begin(), src.R.end() - 1);
    for (const auto& e : edges) {
        std::uint32_t pu = fill[e.u]++, pv = fill[e.v]++;
        out.C[pu] = e.v;
        out.C[pv] = e.u;
        if (weighted) { out.V[pu] = e.w; out.V[pv] = e.w; }
    }
    // sort each row
    std::vector<std::pair<std::uint32_t, float>> tmp;
    for (std::uint32_t i = 0; i < out.n; ++i) {
        std::uint32_t b = out.R[i], e = out.R[i + 1];
        if (!weighted) {
            std::sort(out.C.begin() + b, out.C.begin() + e);
        } else {
            tmp.clear();
            for (std::uint32_t k = b; k < e; ++k) tmp.emplace_back(out.C[k], out.V[k]);
            std::sort(tmp.begin(), tmp.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            for (std::uint32_t k = b; k < e; ++k) { out.C[k] = tmp[k - b].first; out.V[k] = tmp[k - b].second; }
        }
    }
    return out;
}

}  // namespace pagani
