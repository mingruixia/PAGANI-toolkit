#include "pagani/construct.hpp"

#include <algorithm>
#include <atomic>
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

std::vector<std::uint32_t> sample_rows(std::uint32_t n) {
    const std::uint32_t s = std::min<std::uint32_t>(n, 2048);
    std::vector<std::uint32_t> rows(s);
    for (std::uint32_t k = 0; k < s; ++k) rows[k] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(k) * n / s);
    return rows;
}

namespace {

// Lower edge of the bin one below the bin where the cumulative count from the
// top reaches `want` (-2 = everything).
float threshold_from_histogram(const std::vector<std::uint64_t>& hist, double want) {
    double cum = 0;
    int b = kHistBins - 1;
    for (; b >= 0; --b) { cum += static_cast<double>(hist[b]); if (cum >= want) break; }
    // one bin of margin protects against rounding differences in binning
    return b > 0 ? hist_bin_low(b - 1) : -2.0f;
}

}  // namespace

void build_networks(EdgeSource& src, const std::vector<Threshold>& thresholds, bool keep_weights,
                    const std::function<void(ThresholdedNetwork&)>& emit) {
    const std::uint32_t n = src.size();
    const std::uint64_t m = packed_count(n);
    if (n < 2) throw Error("fewer than two voxels in the mask");

    float lowest = std::numeric_limits<float>::infinity();  // lowest correlation threshold
    std::uint64_t kmax = 0;                                   // largest sparsity edge count
    for (const auto& t : thresholds) {
        if (t.sparsity) kmax = std::max(kmax, sparsity_edge_count(n, t.value));
        else lowest = std::min(lowest, static_cast<float>(t.value));
    }
    std::vector<WeightedEdge> edges;
    if (kmax > 0) {
        // provisional threshold from a sample: ~15 % more edges than needed
        float guess = std::numeric_limits<float>::quiet_NaN();
        std::uint64_t cap = std::numeric_limits<std::uint64_t>::max();
        const auto sample = src.sample_histogram();
        if (!sample.empty()) {
            double total = 0;
            for (auto c : sample) total += static_cast<double>(c);
            const float g = threshold_from_histogram(sample, 1.15 * total * static_cast<double>(kmax) / m + 50.0);
            if (g > -1.5f) {
                if (g < lowest) {  // the sparsity thresholds need the most edges
                    guess = g;
                    cap = 3 * kmax + (1u << 20);  // a bad guess must not exhaust memory
                } else {
                    guess = lowest;
                }
            }
        }
        std::vector<std::uint64_t> hist;
        bool have = false;
        if (guess == guess) have = src.histogram_and_extract(guess, cap, hist, edges);
        else hist = src.histogram();
        std::uint64_t total = 0;
        for (auto c : hist) total += c;
        if (total != m) throw Error("internal error: histogram count " + std::to_string(total) + " != " + std::to_string(m));
        for (const auto& t : thresholds)
            if (t.sparsity && sparsity_edge_count(n, t.value) > 0)
                lowest = std::min(lowest, threshold_from_histogram(hist, static_cast<double>(sparsity_edge_count(n, t.value))));
        if (have && guess <= lowest) {
            edges.erase(std::remove_if(edges.begin(), edges.end(), [&](const WeightedEdge& e) { return e.r < lowest; }),
                        edges.end());
        } else {
            if (have) std::cout << "  (sampled threshold too high: second pass)" << std::endl;
            std::vector<WeightedEdge>().swap(edges);
            edges = src.extract(lowest);
        }
    } else if (lowest < std::numeric_limits<float>::infinity()) {
        edges = src.extract(lowest);
    }
    // descending r, ties by (i, j)
    std::sort(edges.begin(), edges.end(), [](const WeightedEdge& a, const WeightedEdge& b) {
        if (a.r != b.r) return a.r > b.r;
        if (a.i != b.i) return a.i < b.i;
        return a.j < b.j;
    });
    for (auto& e : edges) if (e.r > 1.0f) e.r = 1.0f;  // float rounding

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

    std::vector<std::uint64_t> sample_histogram() override {
        if (type_ == CorrType::Kendall) return {};
        const auto rows = sample_rows(n_);
        const unsigned nt = hardware_threads();
        std::vector<std::vector<std::uint64_t>> h(nt, std::vector<std::uint64_t>(kHistBins, 0));
        parallel_for(0, static_cast<long long>(rows.size()), [&](long long k, unsigned tid) {
            for (std::uint32_t j = 0; j < n_; ++j)
                if (j != rows[k]) ++h[tid][hist_bin(value(rows[k], j))];
        });
        for (unsigned t = 1; t < nt; ++t)
            for (int b = 0; b < kHistBins; ++b) h[0][b] += h[t][b];
        return h[0];
    }

    bool histogram_and_extract(float threshold, std::uint64_t max_edges, std::vector<std::uint64_t>& hist,
                               std::vector<WeightedEdge>& edges) override {
        const unsigned nt = hardware_threads();
        std::vector<std::vector<std::uint64_t>> h(nt, std::vector<std::uint64_t>(kHistBins, 0));
        std::vector<std::vector<WeightedEdge>> parts(nt);
        std::atomic<std::uint64_t> count(0);
        parallel_for(0, n_, [&](long long i, unsigned tid) {
            for (std::uint32_t j = static_cast<std::uint32_t>(i) + 1; j < n_; ++j) {
                const float r = value(static_cast<std::uint32_t>(i), j);
                ++h[tid][hist_bin(r)];
                if (r >= threshold && ++count <= max_edges) parts[tid].push_back({static_cast<std::uint32_t>(i), j, r});
            }
        });
        for (unsigned t = 1; t < nt; ++t)
            for (int b = 0; b < kHistBins; ++b) h[0][b] += h[t][b];
        hist = std::move(h[0]);
        edges.clear();
        if (count > max_edges) return false;
        for (auto& p : parts) edges.insert(edges.end(), p.begin(), p.end());
        return true;
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
