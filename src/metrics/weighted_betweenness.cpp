#include "pagani/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>

#include "pagani/common.hpp"

namespace pagani {
namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();

double log_add(double a, double b) {
    if (a == -infinity) return b;
    const double high = std::max(a, b);
    const double low = std::min(a, b);
    return high + std::log1p(std::exp(low - high));
}

bool shortest_arc(double from, double length, double to) {
    if (from >= to) return false;
    const double candidate = from + length;
    return std::abs(candidate - to) <=
        kWeightedBetweennessRelativeTolerance * std::max(candidate, to);
}

struct SourceSearch {
    using Candidate = std::pair<double, std::uint32_t>;
    std::vector<double> distance;
    std::vector<double> log_paths;
    std::vector<double> dependency;
    std::vector<std::uint32_t> order;
    std::vector<Candidate> heap;

    void find_distances(const Csr& g, const std::vector<double>& lengths, std::uint32_t source) {
        const auto compare = std::greater<Candidate>{};
        distance.assign(g.n, infinity);
        order.clear();
        heap.clear();
        distance[source] = 0.0;
        heap.emplace_back(0.0, source);
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), compare);
            const auto [current, node] = heap.back();
            heap.pop_back();
            if (current != distance[node]) continue;
            order.push_back(node);
            for (auto pos = g.R[node]; pos < g.R[node + 1]; ++pos) {
                const auto neighbour = g.C[pos];
                const double candidate = current + lengths[pos];
                if (candidate >= distance[neighbour]) continue;
                distance[neighbour] = candidate;
                heap.emplace_back(candidate, neighbour);
                std::push_heap(heap.begin(), heap.end(), compare);
            }
        }
    }

    void accumulate(const Csr& g, const std::vector<double>& lengths,
                    std::uint32_t source, std::vector<double>& centrality) {
        if (g.degree(source) == 0) return;
        find_distances(g, lengths, source);

        // Count after all distances settle, using the same arc test in both
        // passes. Log counts avoid overflow on long chains of branching paths.
        log_paths.assign(g.n, -infinity);
        log_paths[source] = 0.0;
        for (const auto node : order) {
            if (log_paths[node] == -infinity)
                throw Error("weighted betweenness: edge lengths exceed the precision of the shortest-path distances");
            for (auto pos = g.R[node]; pos < g.R[node + 1]; ++pos) {
                const auto neighbour = g.C[pos];
                if (shortest_arc(distance[node], lengths[pos], distance[neighbour]))
                    log_paths[neighbour] = log_add(log_paths[neighbour], log_paths[node]);
            }
        }

        dependency.assign(g.n, 0.0);
        for (auto it = order.rbegin(); it != order.rend() && *it != source; ++it) {
            const auto node = *it;
            double contribution = 0.0;
            for (auto pos = g.R[node]; pos < g.R[node + 1]; ++pos) {
                const auto neighbour = g.C[pos];
                if (!shortest_arc(distance[node], lengths[pos], distance[neighbour])) continue;
                const double fraction = std::exp(log_paths[node] - log_paths[neighbour]);
                contribution += fraction * (1.0 + dependency[neighbour]);
            }
            dependency[node] = contribution;
            centrality[node] += contribution;
        }
    }
};

}  // namespace

std::vector<double> weighted_betweenness_lengths(const Csr& g) {
    if (!g.weighted()) throw Error("weighted betweenness requires a weighted network");
    std::vector<double> lengths;
    lengths.reserve(g.nnz());
    for (float weight : g.V) {
        if (!(weight > 0.0f && std::isfinite(weight)))
            throw Error("weighted betweenness requires positive finite edge weights");
        lengths.push_back(1.0 / static_cast<double>(weight));
    }
    check_symmetric(g, "weighted betweenness");
    return lengths;
}

std::vector<double> betweenness_weighted(const Csr& g) {
    auto lengths = weighted_betweenness_lengths(g);
    std::vector<double> centrality(g.n, 0.0);
    if (lengths.empty()) return centrality;

    const auto threads = std::min<unsigned>(hardware_threads(), g.n);
    std::vector<SourceSearch> work(threads);
    std::vector<std::vector<double>> partial(threads, std::vector<double>(g.n, 0.0));
    parallel_for(0, g.n, [&](long long source, unsigned thread) {
        work[thread].accumulate(g, lengths, static_cast<std::uint32_t>(source), partial[thread]);
    }, threads);
    for (const auto& values : partial)
        for (std::uint32_t node = 0; node < g.n; ++node)
            centrality[node] += values[node];
    return centrality;
}

}  // namespace pagani
