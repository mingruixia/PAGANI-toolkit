// PAGANI Toolkit - network (CSR) file I/O and nodal-metric (.nm) files.
//
// CSR file layout (unchanged from PAGANI 2.0, little endian):
//   int32   Rlength (= N + 1)
//   uint32  R[Rlength]        row offsets
//   uint32  Clength
//   uint32  C[Clength]        column indices (both directions stored)
// weighted networks additionally:
//   uint32  Vlength (= Clength)
//   float32 V[Vlength]
//
// .nm file layout: int32 N, float32 values[N].
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pagani {

struct Csr {
    std::uint32_t n = 0;               // number of nodes
    std::vector<std::uint32_t> R;      // n + 1
    std::vector<std::uint32_t> C;      // R[n]
    std::vector<float> V;              // empty for unweighted networks
    bool weighted() const { return has_weights_flag && V.size() == C.size(); }
    bool has_weights_flag = false;     // file contained a weight block (and it was loaded)
    std::uint64_t nnz() const { return C.size(); }
    std::uint32_t degree(std::uint32_t i) const { return R[i + 1] - R[i]; }
};

enum class CsrWeights { Auto, Require, Ignore };

// Read and validate a CSR file.  Validation checks monotonic row offsets,
// column range, absence of self loops and that each row is sorted.
//   Auto:    read weights if the file contains them
//   Require: error if the file has no weights
//   Ignore:  do not load weights even if present
Csr read_csr(const std::string& path, CsrWeights w = CsrWeights::Auto);
void write_csr(const std::string& path, const Csr& g, bool write_weights);

// Check structural symmetry (j in N(i) <=> i in N(j)); throws on failure.
void check_symmetric(const Csr& g, const std::string& what);

void write_nm(const std::string& path, const std::vector<float>& values);
std::vector<float> read_nm(const std::string& path);

}  // namespace pagani
