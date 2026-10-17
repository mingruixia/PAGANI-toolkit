#include "pagani/construct.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>

#include "pagani/common.hpp"

namespace pagani {

const char* corr_name(CorrType t) {
    switch (t) {
        case CorrType::Pearson: return "pearson";
        case CorrType::Spearman: return "spearman";
        case CorrType::Kendall: return "kendall";
    }
    return "?";
}

float fisher_z(float r) {
    const float lim = 0.9999999f;
    if (r > lim) r = lim;
    if (r < -lim) r = -lim;
    return 0.5f * std::log((1.0f + r) / (1.0f - r));
}

float inverse_fisher_z(float z) { return std::tanh(z); }

std::string threshold_tag(double sparsity, double r) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "_spa%.3f%%_cor%.3f", sparsity, r);
    return buf;
}

std::uint64_t sparsity_edge_count(std::uint32_t n, double s) {
    const std::uint64_t m = packed_count(n);
    long double c = 2.0L * static_cast<long double>(m) * s / 100.0L;
    std::uint64_t clength = static_cast<std::uint64_t>(std::floor(c + 1e-9L));
    clength += clength % 2;
    std::uint64_t k = clength / 2;
    return std::min(k, m);
}

// ------------------------------------------------------------ preprocessing
void rank_transform(std::vector<float>& series, std::uint32_t n, std::uint32_t l) {
    parallel_for(0, n, [&](long long v, unsigned) {
        float* x = series.data() + static_cast<size_t>(v) * l;
        std::vector<std::uint32_t> idx(l);
        std::iota(idx.begin(), idx.end(), 0u);
        std::stable_sort(idx.begin(), idx.end(), [&](std::uint32_t a, std::uint32_t b) { return x[a] < x[b]; });
        std::vector<float> rank(l);
        for (std::uint32_t a = 0; a < l;) {
            std::uint32_t b = a;
            while (b + 1 < l && x[idx[b + 1]] == x[idx[a]]) ++b;
            float avg = 0.5f * (static_cast<float>(a) + static_cast<float>(b)) + 1.0f;
            for (std::uint32_t c = a; c <= b; ++c) rank[idx[c]] = avg;
            a = b + 1;
        }
        std::copy(rank.begin(), rank.end(), x);
    });
}

std::uint32_t standardize(std::vector<float>& series, std::uint32_t n, std::uint32_t l) {
    std::vector<char> flat(n, 0);
    parallel_for(0, n, [&](long long v, unsigned) {
        float* x = series.data() + static_cast<size_t>(v) * l;
        double mean = 0;
        for (std::uint32_t t = 0; t < l; ++t) mean += x[t];
        mean /= l;
        double ss = 0;
        for (std::uint32_t t = 0; t < l; ++t) ss += (x[t] - mean) * (x[t] - mean);
        double nrm = std::sqrt(ss);
        // relative test: series that are constant up to float rounding
        if (!(nrm > 1e-6 * std::max(1.0, std::fabs(mean)) * std::sqrt(static_cast<double>(l)))) {
            std::fill(x, x + l, 0.0f);
            flat[v] = 1;
            return;
        }
        for (std::uint32_t t = 0; t < l; ++t) x[t] = static_cast<float>((x[t] - mean) / nrm);
    });
    return static_cast<std::uint32_t>(std::count(flat.begin(), flat.end(), 1));
}

// ------------------------------------------------------------ packed source
std::vector<std::uint64_t> PackedSource::histogram() {
    const unsigned nt = hardware_threads();
    std::vector<std::vector<std::uint64_t>> h(nt, std::vector<std::uint64_t>(kHistBins, 0));
    parallel_for(0, n_, [&](long long i, unsigned tid) {
        const std::uint64_t base = static_cast<std::uint64_t>(i) * n_ - static_cast<std::uint64_t>(i) * (i + 1) / 2;
        for (std::uint32_t j = static_cast<std::uint32_t>(i) + 1; j < n_; ++j) ++h[tid][hist_bin(p_[base + (j - i - 1)])];
    });
    for (unsigned t = 1; t < nt; ++t)
        for (int b = 0; b < kHistBins; ++b) h[0][b] += h[t][b];
    return h[0];
}

std::vector<WeightedEdge> PackedSource::extract(float threshold) {
    const unsigned nt = hardware_threads();
    std::vector<std::vector<WeightedEdge>> parts(nt);
    parallel_for(0, n_, [&](long long i, unsigned tid) {
        const std::uint64_t base = static_cast<std::uint64_t>(i) * n_ - static_cast<std::uint64_t>(i) * (i + 1) / 2;
        for (std::uint32_t j = static_cast<std::uint32_t>(i) + 1; j < n_; ++j) {
            float r = p_[base + (j - i - 1)];
            if (r >= threshold) parts[tid].push_back({static_cast<std::uint32_t>(i), j, r});
        }
    });
    std::vector<WeightedEdge> out;
    size_t total = 0;
    for (auto& p : parts) total += p.size();
    out.reserve(total);
    for (auto& p : parts) { out.insert(out.end(), p.begin(), p.end()); std::vector<WeightedEdge>().swap(p); }
    return out;
}

// --------------------------------------------------------- network building
namespace {

Csr edges_to_csr(std::uint32_t n, const WeightedEdge* e, std::uint64_t count, bool keep_weights) {
    if (2 * count > std::numeric_limits<std::uint32_t>::max())
        throw Error("network has too many edges for the .csr format (" + std::to_string(count) +
                    "); use a higher threshold");
    Csr g;
    g.n = n;
    g.R.assign(n + 1, 0);
    for (std::uint64_t k = 0; k < count; ++k) { ++g.R[e[k].i + 1]; ++g.R[e[k].j + 1]; }
    for (std::uint32_t i = 0; i < n; ++i) g.R[i + 1] += g.R[i];
    g.C.resize(2 * count);
    if (keep_weights) { g.V.resize(2 * count); g.has_weights_flag = true; }
    std::vector<std::uint32_t> fill(g.R.begin(), g.R.end() - 1);
    for (std::uint64_t k = 0; k < count; ++k) {
        std::uint32_t a = fill[e[k].i]++, b = fill[e[k].j]++;
        g.C[a] = e[k].j;
        g.C[b] = e[k].i;
        if (keep_weights) { g.V[a] = e[k].r; g.V[b] = e[k].r; }
    }
    parallel_for(0, n, [&](long long i, unsigned) {
        const std::uint32_t b = g.R[i], en = g.R[i + 1];
        if (en - b < 2) return;
        if (!keep_weights) { std::sort(g.C.begin() + b, g.C.begin() + en); return; }
        std::vector<std::pair<std::uint32_t, float>> tmp(en - b);
        for (std::uint32_t k = b; k < en; ++k) tmp[k - b] = {g.C[k], g.V[k]};
        std::sort(tmp.begin(), tmp.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        for (std::uint32_t k = b; k < en; ++k) { g.C[k] = tmp[k - b].first; g.V[k] = tmp[k - b].second; }
    });
    return g;
}

}  // namespace

void build_networks(EdgeSource& src, const std::vector<Threshold>& thresholds, bool keep_weights,
                    const std::function<void(ThresholdedNetwork&)>& emit) {
    const std::uint32_t n = src.size();
    const std::uint64_t m = packed_count(n);
    if (n < 2) throw Error("fewer than two voxels in the mask");

    float lowest = std::numeric_limits<float>::infinity();
    bool need_hist = false;
    for (const auto& t : thresholds) {
        if (t.sparsity) {
            if (sparsity_edge_count(n, t.value) > 0) need_hist = true;
        } else {
            lowest = std::min(lowest, static_cast<float>(t.value));
        }
    }
    if (need_hist) {
        auto hist = src.histogram();
        std::uint64_t total = 0;
        for (auto c : hist) total += c;
        if (total != m) throw Error("internal error: histogram count " + std::to_string(total) + " != " + std::to_string(m));
        for (const auto& t : thresholds) {
            if (!t.sparsity) continue;
            const std::uint64_t k = sparsity_edge_count(n, t.value);
            if (k == 0) continue;
            std::uint64_t cum = 0;
            int b = kHistBins - 1;
            for (; b >= 0; --b) { cum += hist[b]; if (cum >= k) break; }
            if (b < 0) b = 0;
            // one bin of margin protects against rounding differences in binning
            lowest = std::min(lowest, b > 0 ? hist_bin_low(b - 1) : -2.0f);
        }
    }
    std::vector<WeightedEdge> edges;
    if (lowest < std::numeric_limits<float>::infinity()) {
        edges = src.extract(lowest);
        // descending r, ties by (i, j)
        std::sort(edges.begin(), edges.end(), [](const WeightedEdge& a, const WeightedEdge& b) {
            if (a.r != b.r) return a.r > b.r;
            if (a.i != b.i) return a.i < b.i;
            return a.j < b.j;
        });
        for (auto& e : edges) if (e.r > 1.0f) e.r = 1.0f;  // float rounding
    }

    for (const auto& t : thresholds) {
        ThresholdedNetwork net;
        net.spec = t;
        std::uint64_t count = 0;
        if (t.sparsity) {
            count = sparsity_edge_count(n, t.value);
            if (count > edges.size())
                throw Error("internal error: not enough candidate edges for sparsity " + std::to_string(t.value));
            net.r = count > 0 ? edges[count - 1].r : 1.0;
        } else {
            // edges with r > t form a prefix of the sorted list
            auto it = std::partition_point(edges.begin(), edges.end(),
                                           [&](const WeightedEdge& e) { return e.r > static_cast<float>(t.value); });
            count = static_cast<std::uint64_t>(it - edges.begin());
            net.r = t.value;
        }
        net.sparsity = m ? 100.0 * static_cast<double>(count) / static_cast<double>(m) : 0.0;
        net.graph = edges_to_csr(n, edges.data(), count, keep_weights);
        emit(net);
    }
}

// ------------------------------------------------------------- .cormat file
void write_cormat(const std::string& path, const float* packed, std::uint64_t count) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw Error("cannot create '" + path + "'");
    if (count <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        std::int32_t c = static_cast<std::int32_t>(count);
        f.write(reinterpret_cast<const char*>(&c), 4);
    } else {
        std::int32_t flag = -1;
        f.write(reinterpret_cast<const char*>(&flag), 4);
        f.write(reinterpret_cast<const char*>(&count), 8);
    }
    const std::uint64_t chunk = 1ull << 26;
    for (std::uint64_t off = 0; off < count; off += chunk)
        f.write(reinterpret_cast<const char*>(packed + off),
                static_cast<std::streamsize>(std::min(chunk, count - off) * sizeof(float)));
    if (!f) throw Error("write error on '" + path + "'");
}

// ================================================================ CPU backend
namespace {

class CpuBackend : public CorrelationBackend {
public:
    std::uint32_t load(const std::vector<float>& series, std::uint32_t n, std::uint32_t l, CorrType type) override {
        n_ = n; l_ = l; type_ = type;
        data_ = series;
        if (type == CorrType::Kendall) {
            ties_.assign(n, 0);
            std::uint32_t flat = 0;
            for (std::uint32_t v = 0; v < n; ++v) {
                const float* x = data_.data() + static_cast<size_t>(v) * l;
                std::uint64_t t = 0;
                for (std::uint32_t a = 0; a < l; ++a)
                    for (std::uint32_t b = a + 1; b < l; ++b) t += (x[a] == x[b]);
                ties_[v] = t;
                flat += (t == static_cast<std::uint64_t>(l) * (l - 1) / 2);
            }
            return flat;
        }
        if (type == CorrType::Spearman) rank_transform(data_, n, l);
        return standardize(data_, n, l);
    }

    std::uint32_t size() const override { return n_; }

    float value(std::uint32_t i, std::uint32_t j) const {
        const float* x = data_.data() + static_cast<size_t>(i) * l_;
        const float* y = data_.data() + static_cast<size_t>(j) * l_;
        if (type_ != CorrType::Kendall) {
            double s = 0;
            for (std::uint32_t t = 0; t < l_; ++t) s += static_cast<double>(x[t]) * y[t];
            return static_cast<float>(s);
        }
        long long s = 0;
        for (std::uint32_t a = 0; a < l_; ++a)
            for (std::uint32_t b = a + 1; b < l_; ++b) {
                int sx = (x[a] > x[b]) - (x[a] < x[b]);
                int sy = (y[a] > y[b]) - (y[a] < y[b]);
                s += sx * sy;
            }
        const double n0 = static_cast<double>(l_) * (l_ - 1) / 2.0;
        const double d = (n0 - ties_[i]) * (n0 - ties_[j]);
        return d > 0 ? static_cast<float>(s / std::sqrt(d)) : 0.0f;
    }

    void packed(float* out, Accumulate mode) override {
        parallel_for(0, n_, [&](long long i, unsigned) {
            const std::uint64_t base = static_cast<std::uint64_t>(i) * n_ - static_cast<std::uint64_t>(i) * (i + 1) / 2;
            for (std::uint32_t j = static_cast<std::uint32_t>(i) + 1; j < n_; ++j) {
                float r = value(static_cast<std::uint32_t>(i), j);
                float& o = out[base + (j - i - 1)];
                if (mode == Accumulate::Assign) o = r;
                else if (mode == Accumulate::Add) o += r;
                else o += fisher_z(r);
            }
        });
    }

    std::vector<std::uint64_t> histogram() override {
        const unsigned nt = hardware_threads();
        std::vector<std::vector<std::uint64_t>> h(nt, std::vector<std::uint64_t>(kHistBins, 0));
        parallel_for(0, n_, [&](long long i, unsigned tid) {
            for (std::uint32_t j = static_cast<std::uint32_t>(i) + 1; j < n_; ++j)
                ++h[tid][hist_bin(value(static_cast<std::uint32_t>(i), j))];
        });
        for (unsigned t = 1; t < nt; ++t)
            for (int b = 0; b < kHistBins; ++b) h[0][b] += h[t][b];
        return h[0];
    }

    std::vector<WeightedEdge> extract(float threshold) override {
        std::vector<WeightedEdge> out;
        std::mutex mu;
        parallel_for(0, n_, [&](long long i, unsigned) {
            std::vector<WeightedEdge> local;
            for (std::uint32_t j = static_cast<std::uint32_t>(i) + 1; j < n_; ++j) {
                float r = value(static_cast<std::uint32_t>(i), j);
                if (r >= threshold) local.push_back({static_cast<std::uint32_t>(i), j, r});
            }
            std::lock_guard<std::mutex> lk(mu);
            out.insert(out.end(), local.begin(), local.end());
        });
        return out;
    }

    std::string description() const override { return "CPU (reference implementation)"; }

private:
    std::uint32_t n_ = 0, l_ = 0;
    CorrType type_ = CorrType::Pearson;
    std::vector<float> data_;
    std::vector<std::uint64_t> ties_;
};

}  // namespace

std::unique_ptr<CorrelationBackend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

}  // namespace pagani
