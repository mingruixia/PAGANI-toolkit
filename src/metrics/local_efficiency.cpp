#include "pagani/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>

#include "pagani/common.hpp"

namespace pagani {
namespace {

struct Neighbourhood {
    // A zero entry means the vertex is outside the current neighbourhood.
    std::vector<std::uint32_t> local_index;
    std::vector<std::size_t> offsets;
    std::vector<std::uint32_t> edges;
    std::vector<std::uint32_t> visited;
    std::vector<std::uint32_t> queue;
    std::vector<double> edge_lengths;
    std::vector<double> incident_roots;
    std::vector<double> distances;
    using Candidate = std::pair<double, std::uint32_t>;
    std::vector<Candidate> frontier;

    void build(const Csr& g, std::uint32_t center, const std::vector<double>* lengths) {
        if (local_index.empty()) local_index.resize(g.n);
        const auto begin = g.R[center];
        const auto k = g.degree(center);
        for (std::uint32_t i = 0; i < k; ++i)
            local_index[g.C[begin + i]] = i + 1;

        edges.clear();
        edge_lengths.clear();
        if (lengths) {
            incident_roots.resize(k);
            for (std::uint32_t i = 0; i < k; ++i)
                incident_roots[i] = 1.0 / (*lengths)[begin + i];
        }
        offsets.resize(static_cast<std::size_t>(k) + 1);
        for (std::uint32_t i = 0; i < k; ++i) {
            offsets[i] = edges.size();
            const auto v = g.C[begin + i];
            for (auto pos = g.R[v]; pos < g.R[v + 1]; ++pos) {
                const auto index = local_index[g.C[pos]];
                if (index == 0) continue;
                edges.push_back(index - 1);
                if (lengths) edge_lengths.push_back((*lengths)[pos]);
            }
        }
        offsets[k] = edges.size();
        for (std::uint32_t i = 0; i < k; ++i)
            local_index[g.C[begin + i]] = 0;
    }

    double binary_efficiency(std::uint32_t k) {
        if (edges.empty()) return 0.0;
        const double pairs = static_cast<double>(k) * (k - 1.0);
        if (edges.size() == pairs) return 1.0;

        visited.assign(k, 0);
        queue.resize(k);
        double sum = 0.0;
        for (std::uint32_t source = 0; source < k - 1; ++source) {
            if (offsets[source] == offsets[source + 1]) continue;
            const auto mark = source + 1;
            visited[source] = mark;
            queue[0] = source;
            std::size_t head = 0, tail = 1;
            unsigned distance = 1;
            while (head < tail && tail < k) {
                const std::size_t level_end = tail;
                unsigned reached = 0;
                while (head < level_end && tail < k) {
                    const auto v = queue[head++];
                    for (auto pos = offsets[v]; pos < offsets[v + 1]; ++pos) {
                        const auto w = edges[pos];
                        if (visited[w] == mark) continue;
                        visited[w] = mark;
                        queue[tail++] = w;
                        // Visit through all vertices, but sum each unordered pair once.
                        reached += w > source;
                    }
                }
                sum += static_cast<double>(reached) / distance;
                ++distance;
            }
        }
        return 2.0 * sum / pairs;
    }

    double weighted_efficiency(std::uint32_t k) {
        if (edges.empty()) return 0.0;
        const auto compare = std::greater<Candidate>{};
        double sum = 0.0;
        for (std::uint32_t source = 0; source < k - 1; ++source) {
            if (offsets[source] == offsets[source + 1]) continue;
            distances.assign(k, std::numeric_limits<double>::infinity());
            distances[source] = 0.0;
            frontier.clear();
            frontier.emplace_back(0.0, source);
            while (!frontier.empty()) {
                std::pop_heap(frontier.begin(), frontier.end(), compare);
                const auto [distance, node] = frontier.back();
                frontier.pop_back();
                if (distance > distances[node]) continue;
                // Symmetric weighted distances allow each pair to be counted once.
                if (node > source)
                    sum += incident_roots[source] * incident_roots[node] / distance;
                for (auto pos = offsets[node]; pos < offsets[node + 1]; ++pos) {
                    const auto target = edges[pos];
                    const double next = distance + edge_lengths[pos];
                    if (next >= distances[target]) continue;
                    distances[target] = next;
                    frontier.emplace_back(next, target);
                    std::push_heap(frontier.begin(), frontier.end(), compare);
                }
            }
        }
        return 2.0 * sum / (static_cast<double>(k) * (k - 1.0));
    }
};

double compute_local_efficiency(const Csr& g, std::vector<float>& nodal,
                                const std::vector<double>* lengths) {
    const auto threads = std::min<unsigned>(hardware_threads(), g.n);
    std::vector<Neighbourhood> workspaces(threads);
    std::vector<double> values(g.n, 0.0);
    parallel_for(0, g.n, [&](long long index, unsigned tid) {
        const auto node = static_cast<std::uint32_t>(index);
        const auto k = g.degree(node);
        if (k < 2) return;
        auto& work = workspaces[tid];
        work.build(g, node, lengths);
        values[node] = lengths ? work.weighted_efficiency(k) : work.binary_efficiency(k);
    }, threads);

    nodal.resize(values.size());
    std::transform(values.begin(), values.end(), nodal.begin(),
                   [](double value) { return static_cast<float>(value); });
    // Sum in node order for reproducibility across thread counts.
    return g.n ? std::accumulate(values.begin(), values.end(), 0.0) / g.n : 0.0;
}

}  // namespace

double local_efficiency_binary(const Csr& g, std::vector<float>& nodal) {
    return compute_local_efficiency(g, nodal, nullptr);
}

double local_efficiency_weighted(const Csr& g, std::vector<float>& nodal, bool normalize_weights) {
    if (!g.weighted()) throw Error("weighted local efficiency requires a weighted network");
    for (float weight : g.V) {
        if (!(weight > 0.0f && std::isfinite(weight)))
            throw Error("weighted local efficiency requires positive finite edge weights");
    }
    const double scale = normalize_weights && !g.V.empty()
        ? *std::max_element(g.V.begin(), g.V.end()) : 1.0;
    std::vector<double> lengths(g.V.size());
    std::transform(g.V.begin(), g.V.end(), lengths.begin(), [scale](float weight) {
        return 1.0 / std::cbrt(static_cast<double>(weight) / scale);
    });
    return compute_local_efficiency(g, nodal, &lengths);
}

}  // namespace pagani
