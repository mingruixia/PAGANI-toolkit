#include "pagani/random_network.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <exception>
#include <future>
#include <thread>
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

// Open-addressing hash set of edge keys (linear probing, tombstones); about
// 4x faster than std::unordered_set for the millions of membership tests of a
// rewiring, with identical results.
class EdgeSet {
public:
    explicit EdgeSet(std::uint64_t n) {
        std::uint64_t c = 16;
        while (c < 2 * n + 16) c <<= 1;
        t_.assign(c, kEmpty);
        mask_ = c - 1;
    }
    bool contains(std::uint64_t k) const {
        for (std::uint64_t i = hash(k) & mask_;; i = (i + 1) & mask_) {
            if (t_[i] == k) return true;
            if (t_[i] == kEmpty) return false;
        }
    }
    void insert(std::uint64_t k) {  // k must not be present
        for (std::uint64_t i = hash(k) & mask_;; i = (i + 1) & mask_)
            if (t_[i] == kEmpty || t_[i] == kTomb) { t_[i] = k; return; }
    }
    void erase(std::uint64_t k) {
        for (std::uint64_t i = hash(k) & mask_;; i = (i + 1) & mask_) {
            if (t_[i] == k) { t_[i] = kTomb; return; }
            if (t_[i] == kEmpty) return;
        }
    }
private:
    // edge keys are (a << 32 | b) with a < b < 2^32 - 1, so these never occur
    static constexpr std::uint64_t kEmpty = ~0ULL, kTomb = ~0ULL - 1;
    static std::uint64_t hash(std::uint64_t k) {
        k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33;
        return k;
    }
    std::vector<std::uint64_t> t_;
    std::uint64_t mask_ = 0;
};

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

}  // namespace

RandomNetworkGenerator::RandomNetworkGenerator() : rng_(initial_seed()) {}

std::uint64_t random_networks_seed(const Csr& g) {
    static const std::uint64_t run = initial_seed();  // PAGANI_SEED, or random once per process
    // topology only (weights travel with the edges and do not change the swaps)
    std::uint64_t h = splitmix64(g.n);
    auto mix = [&h](std::uint64_t v) { h = splitmix64(h ^ (v + 0x632be59bd9b4e019ULL + (h << 6) + (h >> 2))); };
    for (auto r : g.R) mix(r);
    for (auto c : g.C) mix(c);
    return splitmix64(run ^ h);
}

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
        EdgeSet present(m);
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
            if (present.contains(k1) || present.contains(k2)) continue;
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

std::uint64_t random_network_seed(std::uint64_t base_seed, int k) {
    return splitmix64(base_seed ^ splitmix64(static_cast<std::uint64_t>(k)));
}

void for_each_random_network(const Csr& g, int count, std::uint64_t base_seed,
                             const std::function<void(int, const Csr&)>& use) {
    if (count <= 0) return;
    // Networks generated at once: one per hardware thread, limited so that the
    // working memory of a batch (edge list, hash set, result) stays below ~3 GB.
    const std::uint64_t m = g.nnz() / 2;
    const std::uint64_t per = g.nnz() * (g.weighted() ? 8 : 4) + m * 48 + 4ull * g.R.size() + 4096;
    const std::uint64_t cap = std::max<std::uint64_t>(1, (3ull << 30) / per);
    const int batch = static_cast<int>(std::min<std::uint64_t>({hardware_threads(), cap,
                                                                 static_cast<std::uint64_t>(count)}));
    // Plain threads rather than parallel_for: the pool runs one parallel region
    // at a time, and the next batch is generated while `use` (which may itself
    // use the pool or the GPU) processes the current one.
    auto generate = [&g, base_seed](int first, int n) {
        std::vector<Csr> out(static_cast<size_t>(n));
        std::vector<std::exception_ptr> err(static_cast<size_t>(n));
        std::vector<std::thread> th;
        for (int i = 0; i < n; ++i)
            th.emplace_back([&, i] {
                try {
                    RandomNetworkGenerator gen(random_network_seed(base_seed, first + i));
                    out[static_cast<size_t>(i)] = gen.rewire(g);
                } catch (...) {
                    err[static_cast<size_t>(i)] = std::current_exception();
                }
            });
        for (auto& t : th) t.join();
        for (auto& e : err)
            if (e) std::rethrow_exception(e);
        return out;
    };
    std::vector<Csr> cur = generate(0, batch);
    for (int first = 0; first < count; first += batch) {
        const int next_first = first + batch;
        std::future<std::vector<Csr>> next;
        if (next_first < count)
            next = std::async(std::launch::async, generate, next_first, std::min(batch, count - next_first));
        try {
            for (size_t i = 0; i < cur.size(); ++i) use(first + static_cast<int>(i), cur[i]);
        } catch (...) {
            if (next.valid()) next.wait();  // do not leave the generator running on `g`
            throw;
        }
        std::vector<Csr>().swap(cur);
        if (next.valid()) cur = next.get();
    }
}

}  // namespace pagani
