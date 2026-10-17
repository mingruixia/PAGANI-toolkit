#include "pagani/graph.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>

#include "pagani/common.hpp"

namespace pagani {

namespace {
template <typename T>
void read_exact(std::ifstream& f, T* p, std::uint64_t count, const std::string& path) {
    if (count == 0) return;
    f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(count * sizeof(T)));
    if (static_cast<std::uint64_t>(f.gcount()) != count * sizeof(T))
        throw Error("'" + path + "' is truncated or not a PAGANI .csr file");
}
}  // namespace

Csr read_csr(const std::string& path, CsrWeights w) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error("cannot open network file '" + path + "'");
    const std::uint64_t fsize = file_size(path);
    Csr g;
    std::int32_t rlength = 0;
    read_exact(f, &rlength, 1, path);
    if (rlength < 2 || static_cast<std::uint64_t>(rlength) * 4 + 8 > fsize)
        throw Error("'" + path + "' is not a valid .csr file (bad row count)");
    g.n = static_cast<std::uint32_t>(rlength - 1);
    g.R.resize(rlength);
    read_exact(f, g.R.data(), g.R.size(), path);
    std::uint32_t clength = 0;
    read_exact(f, &clength, 1, path);
    if (clength != g.R[g.n] || g.R[0] != 0)
        throw Error("'" + path + "' is not a valid .csr file (edge count mismatch)");
    const std::uint64_t base = 4ull + 4ull * rlength + 4ull + 4ull * clength;
    if (base > fsize) throw Error("'" + path + "' is truncated");
    g.C.resize(clength);
    read_exact(f, g.C.data(), g.C.size(), path);

    bool has_v = fsize >= base + 4;
    if (has_v) {
        if (fsize != base + 4 + 4ull * clength)
            throw Error("'" + path + "' has an unexpected size for a weighted .csr file");
    } else if (fsize != base) {
        throw Error("'" + path + "' has trailing data");
    }
    g.has_weights_flag = has_v && w != CsrWeights::Ignore;
    if (w == CsrWeights::Require && !has_v)
        throw Error("'" + path + "' is an unweighted network but a weighted network is required "
                    "(use the tools in exefiles\\ for unweighted networks)");
    if (has_v && w != CsrWeights::Ignore) {
        std::uint32_t vlength = 0;
        read_exact(f, &vlength, 1, path);
        if (vlength != clength) throw Error("'" + path + "': weight count differs from edge count");
        g.V.resize(vlength);
        read_exact(f, g.V.data(), g.V.size(), path);
    }

    // Validation.  Rows are sorted if necessary (all PAGANI algorithms that
    // intersect neighbour lists rely on sorted rows).
    std::vector<std::pair<std::uint32_t, float>> tmp;
    for (std::uint32_t i = 0; i < g.n; ++i) {
        if (g.R[i + 1] < g.R[i] || g.R[i + 1] > clength) throw Error("'" + path + "': row offsets are not monotonic");
        bool sorted = true;
        for (std::uint32_t k = g.R[i]; k < g.R[i + 1]; ++k) {
            std::uint32_t j = g.C[k];
            if (j >= g.n) throw Error("'" + path + "': column index out of range");
            if (j == i) throw Error("'" + path + "': self loop found at node " + std::to_string(i));
            if (k > g.R[i] && g.C[k - 1] >= j) sorted = false;
        }
        if (!sorted) {
            tmp.clear();
            for (std::uint32_t k = g.R[i]; k < g.R[i + 1]; ++k)
                tmp.emplace_back(g.C[k], g.V.empty() ? 0.0f : g.V[k]);
            std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (size_t t = 0; t < tmp.size(); ++t) {
                if (t > 0 && tmp[t].first == tmp[t - 1].first)
                    throw Error("'" + path + "': duplicate edge in row " + std::to_string(i));
                g.C[g.R[i] + t] = tmp[t].first;
                if (!g.V.empty()) g.V[g.R[i] + t] = tmp[t].second;
            }
        }
    }
    return g;
}

void write_csr(const std::string& path, const Csr& g, bool write_weights) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw Error("cannot create '" + path + "'");
    if (g.n >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        throw Error("network too large for the .csr format");
    std::int32_t rlength = static_cast<std::int32_t>(g.n + 1);
    f.write(reinterpret_cast<const char*>(&rlength), 4);
    f.write(reinterpret_cast<const char*>(g.R.data()), static_cast<std::streamsize>(4ull * g.R.size()));
    std::uint32_t clength = static_cast<std::uint32_t>(g.C.size());
    f.write(reinterpret_cast<const char*>(&clength), 4);
    f.write(reinterpret_cast<const char*>(g.C.data()), static_cast<std::streamsize>(4ull * g.C.size()));
    if (write_weights) {
        if (g.V.size() != g.C.size()) throw Error("internal error: weights missing when writing '" + path + "'");
        f.write(reinterpret_cast<const char*>(&clength), 4);
        f.write(reinterpret_cast<const char*>(g.V.data()), static_cast<std::streamsize>(4ull * g.V.size()));
    }
    if (!f) throw Error("write error on '" + path + "'");
}

void check_symmetric(const Csr& g, const std::string& what) {
    for (std::uint32_t i = 0; i < g.n; ++i)
        for (std::uint32_t k = g.R[i]; k < g.R[i + 1]; ++k) {
            std::uint32_t j = g.C[k];
            auto b = g.C.begin() + g.R[j], e = g.C.begin() + g.R[j + 1];
            auto it = std::lower_bound(b, e, i);
            if (it == e || *it != i) throw Error(what + ": network is not symmetric (undirected)");
            if (!g.V.empty() && g.V[it - g.C.begin()] != g.V[k])
                throw Error(what + ": edge weights are not symmetric");
        }
}

void write_nm(const std::string& path, const std::vector<float>& values) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw Error("cannot create '" + path + "'");
    std::int32_t n = static_cast<std::int32_t>(values.size());
    f.write(reinterpret_cast<const char*>(&n), 4);
    f.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(4ull * values.size()));
    if (!f) throw Error("write error on '" + path + "'");
}

std::vector<float> read_nm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error("cannot open '" + path + "'");
    std::int32_t n = 0;
    read_exact(f, &n, 1, path);
    if (n < 0 || file_size(path) != 4ull + 4ull * static_cast<std::uint64_t>(n))
        throw Error("'" + path + "' is not a valid .nm file");
    std::vector<float> v(n);
    read_exact(f, v.data(), v.size(), path);
    return v;
}

}  // namespace pagani
