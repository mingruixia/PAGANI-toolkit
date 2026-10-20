# Changelog

## Unreleased

- Weighted tools stop with a clear message when a network contains edges with
  weight ≤ 0 (r ≤ 0). Previously clustering, modularity, eigenvector centrality
  and strength were computed silently from such networks and were meaningless;
  only the path-length tools refused them. Weighted construction rejects
  negative correlation thresholds and warns when a sparsity threshold reaches
  r ≤ 0; the GUI checks this before running.
- Kendall networks with data larger than GPU memory: every subject after the
  first could reuse part of the previous subject's data on the GPU (streaming
  mode only).

### Speed

- Random networks (lambda, gamma, Q of random networks) are generated several
  at a time on all CPU threads, the next batch while the current one is
  analysed, with a faster edge set (a single rewiring is about 2× faster with
  identical output; on a 19,574-node network with 2 million edges this was
  about 80 % of the run time of a 100-random-network analysis). Random network
  *k* now has its own seed derived from `PAGANI_SEED` and *k*, so results with
  a fixed `PAGANI_SEED` do not depend on the number of threads but differ from
  those of 2.1.0 (another sample of random networks; `Maslov` is unchanged).
- `CUBFW_Lp`, unweighted networks: path length and nodal efficiency are
  computed on the GPU with a bit-parallel breadth-first search from 512 sources
  at a time (Then et al. 2014), for every density. 2.1.0 used Floyd–Warshall
  (density ≥ 1 %) or the CPU (density < 1 %). Results are identical up to
  floating-point rounding; no N × N distance matrix is needed.
- Newman modularity: the Kernighan–Lin fine tuning needs one pass over the
  group per moved node instead of two plus a copy (about 5× faster; whole tool
  about 2× faster on a 15,000-node network). The gains of all nodes are now
  computed without accumulated rounding error, so for unweighted networks,
  where many nodes have exactly equal gains, a tie can be resolved differently
  than in 2.1.0 (on test networks Q was equal or slightly higher). Weighted
  results are unchanged.
- `CUBC`: the BFS levels of each source are kept in GPU shared memory
  (networks with up to about 24,000 nodes on GTX 10-series GPUs and about
  50,000 on RTX 20/30/40-series GPUs; larger networks use the previous
  kernel). Results are unchanged.
- Sparsity thresholds: a provisional threshold is estimated from about 2,048
  sampled rows of the correlation matrix, so that the exact histogram and the
  candidate edges are collected in one pass over the matrix instead of two (a
  second pass is made only if the estimate was too high). Results are
  unchanged.
- Kendall networks: the sign vectors are multiplied as 8-bit integers with the
  GPU's 4-way integer dot product (exact) instead of 32-bit floats.

## 2.1.0 — 2026-10-17 (maintenance release)

PAGANI 2.1 is a maintenance release of PAGANI 2.0 (HBM 2018). It adds no new
analyses: the same tools, command lines and file formats are kept, so existing
scripts keep working. All programs were rewritten on a shared, tested code base
to fix the calculation errors and robustness problems found in a full review of
the 2.0 sources (see [docs/CODE_AUDIT.md](docs/CODE_AUDIT.md) for details; the
IDs below refer to it).

### Results that change compared with 2.0

Please read this section if you compare new results with results obtained with
PAGANI 2.0 or earlier.

| Affected results | 2.0 behaviour | 2.1 | ID |
|---|---|---|---|
| Networks built in the **GUI with a correlation threshold** | the threshold was always passed as a *sparsity* (a correlation threshold of 0.3 built a 0.3 % sparsity network) | the threshold type selected in the GUI is used | G-01 |
| **Weighted** Lp, lambda and nodal efficiency run from the GUI | computed by the unweighted program (weights ignored) | weighted programs are used | G-02 |
| **Weighted group networks with Fisher averaging** (`bf`) | the averaged z values were not transformed back to r | averaged z is transformed back with tanh | C-01 |
| **High-resolution "ps"** (Pearson + Spearman) | the "Pearson" networks were computed from rank-transformed data, i.e. were Spearman networks | Pearson from the original data | C-02 |
| Everything that uses **random networks** (lambda, gamma, sigma, Q of random networks) | random networks created within the same second were identical (and the unweighted rewiring in the 2018 sources did not return its result) | every random network is an independent Maslov–Sneppen rewiring (2 × edges swap attempts, degree sequence preserved); set `PAGANI_SEED` to make runs reproducible | M-01, M-02 |
| **Small-world sigma** | lambda and gamma files were paired by a few characters of the file name (wrong pairs for similar subject names or more than 127 files) | paired by the full network name | S-01 |
| **Eigenvector centrality** | eigenvector multiplied by the eigenvalue, so values were scaled differently for every network | unit-length (L2) eigenvector; the eigenvalue is written to `<network>_evalue.txt` | E-01 |
| BOLD images or masks with **scaling factors, extensions or integer types** | `scl_slope/scl_inter` ignored, a 352-byte header was assumed, int16/uint8 images were read as garbage | all common NIfTI-1 data types, scaling, extensions and byte order are handled; unsupported files stop with an error | N-01, N-02 |
| **Weighted vs unweighted edge sets** | weighted networks used `r ≥ t − 1e-6`, unweighted `r > t` | both use `r > t` | C-03 |
| **Sparsity thresholds** | edge count could deviate from the requested sparsity when correlation values were tied | exactly the K strongest edges, K = ⌈⌊2·M·s/100⌋ / 2⌉ for M node pairs as in the 2.0 source code (the released 2.0 executables used ⌊M·s/100⌋, which can differ by one edge); ties broken by node index | — |
| **Weighted clustering coefficient (Onnela)** | weights were divided by the largest of the *first N* edge weights instead of the largest weight of the network, so Cp was too large by a subject-dependent factor (on the real-data test: 3–14 %) | weights divided by the network's maximum weight (Onnela et al. 2005, BCT `clustering_coef_wu`) | CP-03 |
| Voxels with **constant time series** inside the mask | produced NaN correlations that were silently treated as 0 | correlations are 0 and a warning reports how many voxels are affected | C-05 |
| `to_average` given as a single letter (e.g. `y`) | silently produced an empty group network | rejected with a message | C-06 |
| Participation coefficient with modules numbered from 0 | out-of-range write | modules 1…M, isolated nodes 0, both handled | P-01 |

Results that are **not** expected to change (beyond floating-point rounding):
individual Pearson networks built from the command line, degree, unweighted
clustering coefficient, path length/efficiency (Lp is the harmonic mean,
`1 / global efficiency`, as in 2.0; weighted distance = 1 / weight), Barrat
weighted clustering, betweenness (every ordered node pair, as in 2.0 and
BCT `betweenness_bin`), modularity.

Use `tests/compare_results.py old_dir new_dir` to compare a 2.0 result folder
with a 2.1 result folder file by file. A comparison of 2.0 and 2.1 on real
data is summarised in [docs/VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md).

### Other changes in behaviour

- Participation coefficient: the normalised output is now named
  `<network>_pc_normalized.nm` (2.0: `<network>_pc.nm_Normalized.nm`).
- Group networks are named `ave_<number of subjects>_<number of nodes>_…`,
  and the group correlation matrix `ave_<subjects>_<nodes>.cormat`.
- `CUCorMatHr` accepts only `to_average = n` (it never supported group networks
  and silently ignored other values).
- The Kendall tool writes to `weightedKendall\` and computes Kendall's tau-b
  exactly (ties are handled).
- The `BNA_time_log` file is no longer written; run times are printed instead.
- `BC_CPU` (an unmaintained CPU duplicate of `CUBC`) was removed; `CUBC`
  computes betweenness of unweighted networks as before.
- Newman modularity: the leading eigenvector of each split is computed with
  the Lanczos method (fully converged) instead of a power iteration that often
  stopped at its iteration limit, and the Kernighan–Lin fine tuning is repeated
  until it no longer improves the split (Newman 2006). On the real-data test
  the unweighted GPU tool of 2.0 failed on every network (NaN eigenvalue, a
  single module, Q = −∞); 2.1 found partitions with higher Q than 2.0 for
  weighted networks. As in 2.0, modules have at least two nodes.
- Multi-threaded tools reuse a thread pool, which removes a large overhead on
  Windows (most visible for the Newman tools).
- Every program returns a non-zero exit code on failure, prints the reason, and
  never waits for a key press, so scripts and the GUI stop at the first error.
- Memory needs are checked: when the data do not fit in GPU memory the
  correlation and path-length programs automatically process them in blocks
  streamed from host memory instead of crashing.
- Paths containing spaces are supported.
- The application icon was replaced by a neutral network icon.

### GUI

- Runs the steps itself (no console window): live log, progress, current step,
  **Stop** button, stops at the first failing step.
- Settings are checked before running (missing files, threshold ranges,
  metrics that need other metrics, options not available for weighted networks)
  with explanations; dependent options are enabled automatically.
- High-resolution network construction (`CUCorMatHr`, PAGANI 2.0) is available
  in the GUI.
- Saved scripts quote all paths, stop on the first error and contain the GUI
  settings, so *Load Script* restores every option. Scripts saved by PAGANI
  1.x/2.0 can still be loaded.
- Scales with the Windows display scaling (high-DPI screens).
- The tools are located relative to `PAGANI.exe`, so the GUI works when started
  from a shortcut or another folder; settings are remembered between sessions.
- Fixed: nodal efficiency alone was not converted to NIfTI (G-03), script
  loading mixed up threshold and Cp types (G-04, G-05) and other GUI issues
  (G-06…G-11).

### Build and distribution

- One CMake build for all tools and the GUI (MSVC 2022, CUDA 12, Qt 5.15 or 6);
  continuous integration on GitHub Actions builds the Windows package and runs
  the tests.
- GPU programs are built for Pascal to Hopper GPUs (sm_61–sm_90, plus PTX for
  newer GPUs) and need only the NVIDIA driver (CUDA 12 compatible, ≥ 527.41
  on Windows); no CUDA DLLs are shipped. `PAGANI_GPU=<index>` selects the GPU.
- Multi-threaded CPU programs use all cores (`PAGANI_THREADS` limits the number of threads).
- Test suite comparing every tool with reference implementations
  (numpy/scipy/networkx); `tests/run_validation.bat` runs it against a package
  on a GPU machine.
- The repository no longer contains binaries; the PAGANI 2.0 sources are kept
  under `legacy/` for reference.
