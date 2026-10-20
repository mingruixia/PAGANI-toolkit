// Helpers shared by the command-line tools.
#pragma once

#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pagani/common.hpp"
#include "pagani/graph.hpp"

#ifdef PAGANI_WEIGHTED
constexpr bool kWeighted = true;
#else
constexpr bool kWeighted = false;
#endif

namespace pagani {

// .csr networks in a directory (PAGANI 2.0 behaviour: every file ending in
// ".csr"), sorted by name.
inline std::vector<std::string> csr_files(const std::string& dir) {
    auto files = list_files_with_suffix(dir, ".csr");
    std::cout << files.size() << " network file(s) to be processed in " << dir << std::endl;
    if (files.empty()) std::cout << "Warning: no .csr file found." << std::endl;
    return files;
}

// Weighted tools need positive, finite edge weights (weights are correlation
// coefficients; distances are 1/w, clustering and modularity assume w > 0).
inline Csr load_network(const std::string& path) {
    Csr g = read_csr(path, kWeighted ? CsrWeights::Require : CsrWeights::Ignore);
    if (kWeighted) {
        std::uint64_t bad = 0;
        for (float w : g.V) bad += !(w > 0.0f && w < std::numeric_limits<float>::infinity());
        if (bad)
            throw Error("'" + path + "' has " + std::to_string(bad / 2) +
                        " edge(s) with a weight <= 0 (r <= 0); weighted network metrics require positive weights. "
                        "Build weighted networks with a positive correlation threshold or a lower sparsity, "
                        "or analyse the network as unweighted.");
    }
    return g;
}

// Output flags such as "gnl": every character must be in `allowed`.
inline std::set<char> parse_flags(const std::string& s, const std::string& allowed, const char* what) {
    if (s.empty()) throw UsageError(std::string("empty ") + what);
    std::set<char> out;
    for (char c : s) {
        if (allowed.find(c) == std::string::npos)
            throw UsageError(std::string("invalid ") + what + " '" + s + "' (allowed letters: " + allowed + ")");
        out.insert(c);
    }
    return out;
}

inline int parse_random_count(const std::string& s) {
    long long v = parse_int(s, "number of random networks");
    if (v < 0 || v > 100000) throw UsageError("number of random networks must be between 0 and 100000");
    return static_cast<int>(v);
}

inline std::string fmt6(double v) {
    std::ostringstream o;
    o << std::setprecision(6) << v;
    return o.str();
}

}  // namespace pagani
