// PAGANI Toolkit - degree-preserving random networks (Maslov & Sneppen 2002).
//
// Fixes over PAGANI 2.0 (audit M-01/M-02): the rewired network is actually
// written to the output and successive random networks differ (64-bit Mersenne
// Twister streams); set the environment variable PAGANI_SEED to an integer to
// make results reproducible.
#pragma once

#include <cstdint>
#include <functional>
#include <random>

#include "pagani/graph.hpp"

namespace pagani {

class RandomNetworkGenerator {
public:
    // Seed from PAGANI_SEED if set, otherwise from std::random_device.
    RandomNetworkGenerator();
    explicit RandomNetworkGenerator(std::uint64_t seed) : rng_(seed) {}

    // Degree-preserving rewiring of the undirected network `src`.
    // `swaps_per_edge` rewiring attempts are made per edge (PAGANI 2.0 used 2).
    // For weighted networks each weight travels with its edge, so the weight
    // distribution and the degree sequence are both preserved (strengths are
    // not).  Rows of the result are sorted.
    Csr rewire(const Csr& src, double swaps_per_edge = 2.0);

    std::mt19937_64& engine() { return rng_; }

private:
    std::mt19937_64 rng_;
};

// Generates `count` random networks of `g` (several at once on all CPU threads,
// the next batch while the current one is being used) and calls use(k, net)
// for k = 0..count-1 in order, on the calling thread.  Network k is
// RandomNetworkGenerator(random_network_seed(base_seed, k)).rewire(g), so the
// results do not depend on the number of threads.
std::uint64_t random_network_seed(std::uint64_t base_seed, int k);
void for_each_random_network(const Csr& g, int count, std::uint64_t base_seed,
                             const std::function<void(int, const Csr&)>& use);

}  // namespace pagani
