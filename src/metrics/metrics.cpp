#include "pagani/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <unordered_map>

#include "pagani/common.hpp"

namespace pagani {

// ======================================================================= degree
std::vector<float> degree_centrality(const Csr& g, bool weighted) {
    std::vector<float> d(g.n, 0.0f);
    for (std::uint32_t i = 0; i < g.n; ++i) {
        if (!weighted) {
            d[i] = static_cast<float>(g.R[i + 1] - g.R[i]);
        } else {
            double s = 0;
            for (std::uint32_t k = g.R[i]; k < g.R[i + 1]; ++k) s += g.V[k];
            d[i] = static_cast<float>(s);
        }
    }
    return d;
}

// =================================================================== clustering
double clustering(const Csr& g, ClusteringType type, std::vector<float>& nodal) {
    const std::uint32_t n = g.n;
    nodal.assign(n, 0.0f);
    if (type != ClusteringType::Binary && !g.weighted())
        throw Error("weighted clustering coefficient requires a weighted network");
    double wmax = 0.0;
    if (type == ClusteringType::Onnela) {
        for (float w : g.V) wmax = std::max(wmax, static_cast<double>(w));
        if (wmax <= 0.0) wmax = 1.0;
    }
    std::vector<double> partial(hardware_threads(), 0.0);
    parallel_for(0, n, [&](long long ii, unsigned tid) {
        const std::uint32_t i = static_cast<std::uint32_t>(ii);
        const std::uint32_t bi = g.R[i], ei = g.R[i + 1];
        const std::uint32_t ki = ei - bi;
        if (ki < 2) { nodal[i] = 0.0f; return; }
        double acc = 0.0;
        // For each neighbour j (position a in row i), intersect the
        // neighbours of i after position a with the neighbours of j: every
        // triangle (i, j, h) with h after j in N(i) is counted exactly once.
        for (std::uint32_t a = bi; a < ei; ++a) {
            const std::uint32_t j = g.C[a];
            std::uint32_t p = a + 1, q = g.R[j];
            const std::uint32_t pe = ei, qe = g.R[j + 1];
            while (p < pe && q < qe) {
                const std::uint32_t hp = g.C[p], hq = g.C[q];
                if (hp == hq) {
                    switch (type) {
                        case ClusteringType::Binary: acc += 1.0; break;
                        case ClusteringType::Barrat: acc += static_cast<double>(g.V[a]) + g.V[p]; break;
                        case ClusteringType::Onnela:
                            acc += std::cbrt((static_cast<double>(g.V[a]) / wmax) * (g.V[p] / wmax) * (g.V[q] / wmax));
                            break;
                    }
                    ++p; ++q;
                } else if (hp < hq) {
                    ++p;
                } else {
                    ++q;
                }
            }
        }
        double c = 0.0;
        if (type == ClusteringType::Barrat) {
            double s = 0;
            for (std::uint32_t a = bi; a < ei; ++a) s += g.V[a];
            c = s > 0 ? acc / (s * (ki - 1.0)) : 0.0;
        } else {
            c = 2.0 * acc / (static_cast<double>(ki) * (ki - 1.0));
        }
        nodal[i] = static_cast<float>(c);
        partial[tid] += c;
    });
    double sum = std::accumulate(partial.begin(), partial.end(), 0.0);
    return n ? sum / n : 0.0;
}

// =================================================================== efficiency
EfficiencyResult efficiency_from_inverse_sums(const std::vector<double>& inv_sum, bool connected,
                                              std::uint32_t isolated) {
    EfficiencyResult r;
    const size_t n = inv_sum.size();
    r.connected = connected;
    r.isolated = isolated;
    r.nodal.assign(n, 0.0f);
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        total += inv_sum[i];
        r.nodal[i] = n > 1 ? static_cast<float>(inv_sum[i] / (n - 1.0)) : 0.0f;
    }
    r.global_efficiency = n > 1 ? total / (static_cast<double>(n) * (n - 1.0)) : 0.0;
    r.lp = r.global_efficiency > 0 ? 1.0 / r.global_efficiency : std::numeric_limits<double>::infinity();
    return r;
}

EfficiencyResult efficiency(const Csr& g, bool weighted, bool /*want_nodal*/) {
    const std::uint32_t n = g.n;
    if (weighted) {
        if (!g.weighted()) throw Error("weighted path length requires a weighted network");
        for (float w : g.V)
            if (!(w > 0.0f)) throw Error("weighted path length requires positive edge weights");
    }
    std::vector<double> inv_sum(n, 0.0);
    std::vector<char> reach_all(hardware_threads(), 1);
    std::uint32_t isolated = 0;
    for (std::uint32_t i = 0; i < n; ++i) isolated += (g.R[i + 1] == g.R[i]);

    if (!weighted) {
        parallel_for(0, n, [&](long long s, unsigned tid) {
            thread_local std::vector<int> dist;
            thread_local std::vector<std::uint32_t> queue;
            dist.assign(n, -1);
            queue.resize(n);
            size_t head = 0, tail = 0;
            dist[s] = 0;
            queue[tail++] = static_cast<std::uint32_t>(s);
            double acc = 0.0;
            while (head < tail) {
                std::uint32_t v = queue[head++];
                for (std::uint32_t k = g.R[v]; k < g.R[v + 1]; ++k) {
                    std::uint32_t w = g.C[k];
                    if (dist[w] < 0) {
                        dist[w] = dist[v] + 1;
                        acc += 1.0 / dist[w];
                        queue[tail++] = w;
                    }
                }
            }
            if (tail < n) reach_all[tid] = 0;
            inv_sum[s] = acc;
        });
    } else {
        parallel_for(0, n, [&](long long s, unsigned tid) {
            thread_local std::vector<double> dist;
            dist.assign(n, std::numeric_limits<double>::infinity());
            using QE = std::pair<double, std::uint32_t>;
            std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
            dist[s] = 0.0;
            pq.push({0.0, static_cast<std::uint32_t>(s)});
            double acc = 0.0;
            size_t settled = 0;
            while (!pq.empty()) {
                auto [d, v] = pq.top();
                pq.pop();
                if (d > dist[v]) continue;
                ++settled;
                if (v != s) acc += 1.0 / d;
                for (std::uint32_t k = g.R[v]; k < g.R[v + 1]; ++k) {
                    std::uint32_t w = g.C[k];
                    double nd = d + 1.0 / static_cast<double>(g.V[k]);
                    if (nd < dist[w]) {
                        dist[w] = nd;
                        pq.push({nd, w});
                    }
                }
            }
            if (settled < n) reach_all[tid] = 0;
            inv_sum[s] = acc;
        });
    }
    bool connected = std::all_of(reach_all.begin(), reach_all.end(), [](char c) { return c != 0; });
    return efficiency_from_inverse_sums(inv_sum, connected, isolated);
}

// ================================================================== betweenness
std::vector<double> betweenness_binary(const Csr& g) {
    const std::uint32_t n = g.n;
    const unsigned nt = hardware_threads();
    std::vector<std::vector<double>> acc(nt, std::vector<double>(n, 0.0));
    parallel_for(0, n, [&](long long s, unsigned tid) {
        thread_local std::vector<int> dist;
        thread_local std::vector<double> sigma, delta;
        thread_local std::vector<std::uint32_t> order;
        dist.assign(n, -1);
        sigma.assign(n, 0.0);
        delta.assign(n, 0.0);
        order.resize(n);
        size_t head = 0, tail = 0;
        dist[s] = 0;
        sigma[s] = 1.0;
        order[tail++] = static_cast<std::uint32_t>(s);
        while (head < tail) {
            std::uint32_t v = order[head++];
            for (std::uint32_t k = g.R[v]; k < g.R[v + 1]; ++k) {
                std::uint32_t w = g.C[k];
                if (dist[w] < 0) { dist[w] = dist[v] + 1; order[tail++] = w; }
                if (dist[w] == dist[v] + 1) sigma[w] += sigma[v];
            }
        }
        // back-propagation in reverse BFS order; predecessors of w are the
        // neighbours v with dist[v] == dist[w] - 1
        for (size_t idx = tail; idx-- > 1;) {
            std::uint32_t w = order[idx];
            for (std::uint32_t k = g.R[w]; k < g.R[w + 1]; ++k) {
                std::uint32_t v = g.C[k];
                if (dist[v] == dist[w] - 1) delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w]);
            }
            acc[tid][w] += delta[w];
        }
    });
    std::vector<double> bc(n, 0.0);
    for (const auto& a : acc)
        for (std::uint32_t i = 0; i < n; ++i) bc[i] += a[i];
    return bc;
}

// ======================================================== eigenvector centrality
EigenResult eigenvector_centrality(const Csr& g, bool weighted, double tol, int max_iter) {
    const std::uint32_t n = g.n;
    EigenResult r;
    r.vector.assign(n, n ? 1.0 / std::sqrt(static_cast<double>(n)) : 0.0);
    if (n == 0) return r;
    std::vector<double> y(n);
    // Iterate on (A + I): same eigenvectors, avoids oscillation on bipartite
    // components, converges to the non-negative Perron vector.
    for (int it = 1; it <= max_iter; ++it) {
        parallel_for(0, n, [&](long long i, unsigned) {
            double s = r.vector[i];
            for (std::uint32_t k = g.R[i]; k < g.R[i + 1]; ++k)
                s += (weighted ? g.V[k] : 1.0) * r.vector[g.C[k]];
            y[i] = s;
        });
        double norm = 0.0;
        for (double v : y) norm += v * v;
        norm = std::sqrt(norm);
        if (norm == 0.0) break;
        double diff = 0.0;
        for (std::uint32_t i = 0; i < n; ++i) {
            y[i] /= norm;
            diff = std::max(diff, std::fabs(y[i] - r.vector[i]));
        }
        r.vector.swap(y);
        r.iterations = it;
        if (diff < tol) { r.converged = true; break; }
    }
    // eigenvalue of A via Rayleigh quotient
    double num = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::uint32_t k = g.R[i]; k < g.R[i + 1]; ++k) s += (weighted ? g.V[k] : 1.0) * r.vector[g.C[k]];
        num += r.vector[i] * s;
    }
    r.value = num;
    return r;
}

// ==================================================================== modularity
double modularity_q(const Csr& g, bool weighted, const std::vector<int>& module) {
    double two_m = 0.0;
    std::unordered_map<int, double> tot;   // sum of degrees per module
    double in = 0.0;                       // sum of A_ij within modules (both directions)
    for (std::uint32_t i = 0; i < g.n; ++i) {
        double ki = 0.0;
        for (std::uint32_t k = g.R[i]; k < g.R[i + 1]; ++k) {
            double w = weighted ? g.V[k] : 1.0;
            ki += w;
            if (module[i] == module[g.C[k]]) in += w;
        }
        two_m += ki;
        tot[module[i]] += ki;
    }
    if (two_m <= 0.0) return 0.0;
    double q = in / two_m;
    for (const auto& kv : tot) q -= (kv.second / two_m) * (kv.second / two_m);
    return q;
}

namespace {

// Relabel arbitrary community ids into 1..M (order of first appearance),
// isolated nodes get 0.
int relabel(const Csr& g, std::vector<int>& module) {
    std::unordered_map<int, int> map;
    for (std::uint32_t i = 0; i < g.n; ++i) {
        if (g.R[i + 1] == g.R[i]) { module[i] = 0; continue; }
        auto it = map.find(module[i]);
        if (it == map.end()) it = map.emplace(module[i], static_cast<int>(map.size()) + 1).first;
        module[i] = it->second;
    }
    return static_cast<int>(map.size());
}

// Weighted graph in adjacency-list form used by Louvain levels.
struct WGraph {
    std::vector<std::uint64_t> off;
    std::vector<std::uint32_t> nbr;
    std::vector<double> w;
    std::vector<double> self;   // self-loop weight (counted once in A_ii)
    size_t size() const { return self.size(); }
};

}  // namespace

Partition louvain(const Csr& g, bool weighted, std::mt19937_64& rng) {
    const double tiny = 1e-12;
    // level-0 graph
    WGraph cur;
    cur.off.assign(g.n + 1, 0);
    for (std::uint32_t i = 0; i <= g.n; ++i) cur.off[i] = g.R[i];
    cur.nbr.assign(g.C.begin(), g.C.end());
    cur.w.resize(g.C.size());
    for (size_t k = 0; k < g.C.size(); ++k) cur.w[k] = weighted ? g.V[k] : 1.0;
    cur.self.assign(g.n, 0.0);

    std::vector<int> node_comm(g.n);
    std::iota(node_comm.begin(), node_comm.end(), 0);

    double two_m = 0.0;
    for (double x : cur.w) two_m += x;
    if (two_m <= 0.0) {
        Partition p;
        p.module.assign(g.n, 0);
        return p;
    }

    while (true) {
        const size_t n = cur.size();
        std::vector<double> k(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            double s = 2.0 * cur.self[i];
            for (auto e = cur.off[i]; e < cur.off[i + 1]; ++e) s += cur.w[e];
            k[i] = s;
        }
        std::vector<int> comm(n);
        std::iota(comm.begin(), comm.end(), 0);
        std::vector<double> tot(k);  // sum of degrees in each community
        std::vector<double> neigh_w(n, 0.0);
        std::vector<int> neigh_list;
        neigh_list.reserve(64);
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        bool improved_level = false;
        bool moved = true;
        while (moved) {
            moved = false;
            std::shuffle(order.begin(), order.end(), rng);
            for (size_t idx = 0; idx < n; ++idx) {
                const size_t i = order[idx];
                if (k[i] == 0.0) continue;
                const int ci = comm[i];
                neigh_list.clear();
                for (auto e = cur.off[i]; e < cur.off[i + 1]; ++e) {
                    int c = comm[cur.nbr[e]];
                    if (neigh_w[c] == 0.0) neigh_list.push_back(c);
                    neigh_w[c] += cur.w[e];
                }
                // remove i from its community
                tot[ci] -= k[i];
                const double k_i_in_old = neigh_w[ci];
                int best = ci;
                double best_gain = k_i_in_old - tot[ci] * k[i] / two_m;
                for (int c : neigh_list) {
                    double gain = neigh_w[c] - tot[c] * k[i] / two_m;
                    if (gain > best_gain + tiny) { best_gain = gain; best = c; }
                }
                tot[best] += k[i];
                if (best != ci) { comm[i] = best; moved = true; improved_level = true; }
                for (int c : neigh_list) neigh_w[c] = 0.0;
                neigh_w[ci] = 0.0;
            }
        }
        // renumber communities of this level
        std::vector<int> renum(n, -1);
        int nc = 0;
        for (size_t i = 0; i < n; ++i) {
            if (renum[comm[i]] < 0) renum[comm[i]] = nc++;
            comm[i] = renum[comm[i]];
        }
        for (auto& c : node_comm) c = comm[c];
        if (!improved_level || static_cast<size_t>(nc) == n) break;

        // aggregate
        WGraph next;
        next.self.assign(nc, 0.0);
        std::vector<std::map<std::uint32_t, double>> rows(nc);
        for (size_t i = 0; i < n; ++i) {
            const int ci = comm[i];
            next.self[ci] += cur.self[i];
            for (auto e = cur.off[i]; e < cur.off[i + 1]; ++e) {
                const int cj = comm[cur.nbr[e]];
                if (cj == ci) next.self[ci] += 0.5 * cur.w[e];  // each internal edge seen twice
                else rows[ci][cj] += cur.w[e];
            }
        }
        next.off.assign(nc + 1, 0);
        for (int c = 0; c < nc; ++c) next.off[c + 1] = next.off[c] + rows[c].size();
        next.nbr.reserve(next.off[nc]);
        next.w.reserve(next.off[nc]);
        for (int c = 0; c < nc; ++c)
            for (const auto& kv : rows[c]) { next.nbr.push_back(kv.first); next.w.push_back(kv.second); }
        cur = std::move(next);
    }

    Partition p;
    p.module = node_comm;
    p.num_modules = relabel(g, p.module);
    p.q = modularity_q(g, weighted, p.module);
    return p;
}

// ---------------------------------------------------------- Newman eigenvector
namespace {

struct NewmanCtx {
    const Csr& g;
    bool weighted;
    std::vector<double> k;   // degrees / strengths
    double two_m = 0.0;
    SpmvFn spmv;
};

double wgt(const NewmanCtx& c, std::uint32_t e) { return c.weighted ? c.g.V[e] : 1.0; }

// y = B^(g) x restricted to `members` (x and y are indexed by position in
// members).  `inmask[node]` gives position+1 for members, 0 otherwise.
void apply_bg(const NewmanCtx& c, const std::vector<std::uint32_t>& members, const std::vector<int>& pos,
              const std::vector<double>& diag, const std::vector<double>& x, std::vector<double>& y) {
    const size_t ng = members.size();
    double kx = 0.0;
    for (size_t a = 0; a < ng; ++a) kx += c.k[members[a]] * x[a];
    parallel_for(0, static_cast<long long>(ng), [&](long long a, unsigned) {
        const std::uint32_t i = members[a];
        double s = 0.0;
        for (std::uint32_t e = c.g.R[i]; e < c.g.R[i + 1]; ++e) {
            int b = pos[c.g.C[e]];
            if (b > 0) s += wgt(c, e) * x[b - 1];
        }
        s -= c.k[i] * kx / c.two_m;
        s -= diag[a] * x[a];
        y[a] = s;
    });
}

// Split `members`; returns true and fills `side` (+1/-1) if a split increases Q.
bool newman_split(const NewmanCtx& c, const std::vector<std::uint32_t>& members, std::vector<int>& pos,
                  std::vector<signed char>& side) {
    const size_t ng = members.size();
    if (ng < 2) return false;
    for (size_t a = 0; a < ng; ++a) pos[members[a]] = static_cast<int>(a) + 1;

    // diag correction: sum_l in g B_il
    double kg = 0.0;
    for (auto i : members) kg += c.k[i];
    std::vector<double> diag(ng);
    double bound = 0.0;
    for (size_t a = 0; a < ng; ++a) {
        const std::uint32_t i = members[a];
        double kin = 0.0;
        for (std::uint32_t e = c.g.R[i]; e < c.g.R[i + 1]; ++e)
            if (pos[c.g.C[e]] > 0) kin += wgt(c, e);
        diag[a] = kin - c.k[i] * kg / c.two_m;
        bound = std::max(bound, 2.0 * (kin + c.k[i] * kg / c.two_m));
    }
    // power iteration on B + shift*I (shift >= spectral radius) -> largest
    // algebraic eigenvalue of B^(g)
    const double shift = bound + 1e-9;
    std::vector<double> x(ng), y(ng);
    for (size_t a = 0; a < ng; ++a) x[a] = 1.0 + 0.1 * std::sin(1.0 + a);  // deterministic, not orthogonal
    double nrm = 0;
    for (double v : x) nrm += v * v;
    nrm = std::sqrt(nrm);
    for (double& v : x) v /= nrm;
    double lambda = 0.0, prev = std::numeric_limits<double>::infinity();
    for (int it = 0; it < 20000; ++it) {
        apply_bg(c, members, pos, diag, x, y);
        double rq = 0.0;
        for (size_t a = 0; a < ng; ++a) { rq += x[a] * y[a]; y[a] += shift * x[a]; }
        double ny = 0.0;
        for (double v : y) ny += v * v;
        ny = std::sqrt(ny);
        if (ny == 0) break;
        double diff = 0.0;
        for (size_t a = 0; a < ng; ++a) { double v = y[a] / ny; diff = std::max(diff, std::fabs(v - x[a])); x[a] = v; }
        lambda = rq;
        if (diff < 1e-9 || std::fabs(rq - prev) < 1e-12 * std::max(1.0, std::fabs(rq))) break;
        prev = rq;
    }
    bool ok = false;
    if (lambda > 1e-8) {
        side.assign(ng, 1);
        for (size_t a = 0; a < ng; ++a) side[a] = x[a] >= 0 ? 1 : -1;
        // s^T B s and fine tuning (single Kernighan-Lin style sweep, as in Newman 2006)
        std::vector<double> s(ng), bs(ng);
        for (size_t a = 0; a < ng; ++a) s[a] = side[a];
        apply_bg(c, members, pos, diag, s, bs);
        double f = 0.0;
        for (size_t a = 0; a < ng; ++a) f += s[a] * bs[a];
        // B_aa = -k_i^2/2m - diag   (A_ii = 0)
        std::vector<double> baa(ng);
        for (size_t a = 0; a < ng; ++a) baa[a] = -c.k[members[a]] * c.k[members[a]] / c.two_m - diag[a];
        double best_f = f;
        std::vector<double> best_s = s;
        std::vector<char> moved(ng, 0);
        for (size_t step = 0; step < ng; ++step) {
            // choose unmoved node with the largest increase
            double best_delta = -std::numeric_limits<double>::infinity();
            size_t pick = ng;
            for (size_t a = 0; a < ng; ++a) {
                if (moved[a]) continue;
                double delta = -4.0 * s[a] * bs[a] + 4.0 * baa[a];
                if (delta > best_delta) { best_delta = delta; pick = a; }
            }
            if (pick == ng) break;
            // flip and update bs = B s
            const double old = s[pick];
            s[pick] = -old;
            moved[pick] = 1;
            f += best_delta;
            const std::uint32_t i = members[pick];
            const double coef = -2.0 * old;  // change of s[pick]
            for (std::uint32_t e = c.g.R[i]; e < c.g.R[i + 1]; ++e) {
                int b = pos[c.g.C[e]];
                if (b > 0) bs[b - 1] += coef * wgt(c, e);
            }
            const double ki = c.k[i];
            for (size_t a = 0; a < ng; ++a) bs[a] += coef * (-c.k[members[a]] * ki / c.two_m);
            bs[pick] += coef * (-diag[pick]);
            if (f > best_f + 1e-12) { best_f = f; best_s = s; }
        }
        int np = 0, nn = 0;
        for (size_t a = 0; a < ng; ++a) { side[a] = best_s[a] > 0 ? 1 : -1; (side[a] > 0 ? np : nn)++; }
        ok = best_f / (2.0 * c.two_m) > 1e-10 && np > 0 && nn > 0;
    }
    for (auto i : members) pos[i] = 0;
    return ok;
}

}  // namespace

Partition newman_eigenvector(const Csr& g, bool weighted, const SpmvFn& spmv) {
    NewmanCtx c{g, weighted, std::vector<double>(g.n, 0.0), 0.0, spmv};
    for (std::uint32_t i = 0; i < g.n; ++i) {
        for (std::uint32_t e = g.R[i]; e < g.R[i + 1]; ++e) c.k[i] += wgt(c, e);
        c.two_m += c.k[i];
    }
    Partition p;
    p.module.assign(g.n, 0);
    if (c.two_m <= 0.0) return p;
    std::vector<int> pos(g.n, 0);
    std::vector<std::vector<std::uint32_t>> todo;
    std::vector<std::uint32_t> all;
    for (std::uint32_t i = 0; i < g.n; ++i)
        if (c.k[i] > 0) all.push_back(i);
    todo.push_back(all);
    int next_id = 1;
    std::vector<signed char> side;
    while (!todo.empty()) {
        auto members = std::move(todo.back());
        todo.pop_back();
        if (newman_split(c, members, pos, side)) {
            std::vector<std::uint32_t> a, b;
            for (size_t t = 0; t < members.size(); ++t) (side[t] > 0 ? a : b).push_back(members[t]);
            todo.push_back(std::move(b));
            todo.push_back(std::move(a));
        } else {
            for (auto i : members) p.module[i] = next_id;
            ++next_id;
        }
    }
    p.num_modules = relabel(g, p.module);
    p.q = modularity_q(g, weighted, p.module);
    return p;
}

// ===================================================== participation coefficient
void participation_coefficient(const Csr& g, bool weighted, const std::vector<float>& module,
                               std::vector<float>& pc, std::vector<float>& pc_normalized) {
    if (module.size() != g.n) throw Error("module file has a different number of nodes than the network");
    std::map<long long, int> ids;
    for (std::uint32_t i = 0; i < g.n; ++i) {
        long long m = std::llround(module[i]);
        if (g.R[i + 1] > g.R[i]) ids.emplace(m, 0);
    }
    int nm = 0;
    for (auto& kv : ids) kv.second = nm++;
    pc.assign(g.n, 0.0f);
    std::vector<double> within(nm, 0.0);
    for (std::uint32_t i = 0; i < g.n; ++i) {
        double ki = 0.0;
        std::fill(within.begin(), within.end(), 0.0);
        for (std::uint32_t e = g.R[i]; e < g.R[i + 1]; ++e) {
            double w = weighted ? g.V[e] : 1.0;
            ki += w;
            within[ids.at(std::llround(module[g.C[e]]))] += w;
        }
        if (ki == 0.0) { pc[i] = 0.0f; continue; }
        double s = 0.0;
        for (double x : within) s += (x / ki) * (x / ki);
        pc[i] = static_cast<float>(1.0 - s);
    }
    pc_normalized = pc;
    if (nm > 1) {
        const double mx = (nm - 1.0) / nm;
        for (auto& v : pc_normalized) v = static_cast<float>(v / mx);
    }
}

}  // namespace pagani
