// PAGANI Toolkit - degree-preserving random networks (Maslov & Sneppen 2002).
//
// Fixes over PAGANI 2.0 (audit M-01/M-02): the rewired network is actually
// written to the output; every call draws from one 64-bit Mersenne Twister
// stream so successive random networks differ; set the environment variable
// PAGANI_SEED to an integer to make results reproducible.
#pragma once

#include <cstdint>
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

}  // namespace pagani
