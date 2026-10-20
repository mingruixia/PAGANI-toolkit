# PAGANI 2.1 / 2.2 validation report

Date: 2026-10-17 · build: develop @ `ff0c91a` (the `dev-build` package of the
CI) · computer: Windows 11 workstation with an NVIDIA GeForce RTX 3080 Ti
(12 GB, driver 617.42). The computer was in normal use during the
runs, so the run times below are indicative only.

## 1. Test suite on the GPU

`tests/run_validation.bat` with the Windows package (`PAGANI_TEST_GPU=1`):
**68 passed, 2 skipped** (the two GUI end-to-end tests need a build directory;
they pass in CI). This includes the multi-block and forced streaming paths of
the correlation tools and of `CUBFW_Lp` (`test_gpu_scale.py`), and the GPU
betweenness, eigenvector and Newman tools.

## 2. Real data: PAGANI 2.0 vs 2.1

Data: 8 resting-state fMRI runs (500 volumes each) preprocessed with fMRIPrep
(MNI152NLin2009cAsym, 2 mm), resampled to 4 mm, nuisance regression (6 motion
parameters and their derivatives, white matter, CSF, quadratic trend) and 0.01–0.08 Hz band-pass
(`tests/realdata/prepare_fmriprep.py`). Group grey-matter mask ≥ 0.2:
**19 574 nodes**. Networks at 1 % sparsity (1 915 610 edges).

Two comparisons were made (`tests/realdata/run_comparison.py`,
`tests/realdata/metrics_same_networks.py`):

1. each version builds its own networks and computes all metrics;
2. the 2.0 metric tools are run on the networks built by 2.1, so that metric
   differences are not mixed with construction differences.

### Network construction

| | result |
|---|---|
| 2.0 unweighted `CUCorMat` | **wrote no networks** on this GPU, with any options: the released binary reads the mask and exits without output or error message (it is an older CUDA 7.5 build, and 2.0 does not check CUDA errors, C-04) |
| 2.0 weighted `CUCorMat` vs 2.1 | identical edge sets and weights except one edge per network (2.0 binary keeps ⌊M·s/100⌋ = 1 915 609 edges, 2.1 and the 2.0 source code 1 915 610) |
| 2.0 group network with Fisher averaging (`bf`) | **failed** (exit code 1, audit C-01); 2.1 succeeds |
| high-resolution Pearson (`CUCorMatHr`) | both succeed |

### Metrics on identical networks (2.0 tools on 2.1 networks)

| metric | unweighted | weighted |
|---|---|---|
| Lp (global) | identical | identical |
| nodal efficiency | max. difference 4 × 10⁻⁵ (2.0 single precision) | max. difference 5 × 10⁻⁵ |
| clustering coefficient | identical | **2.0 3–14 % too large** (Onnela weights normalised by a wrong maximum, CP-03); nodal values perfectly correlated (r > 0.99999999) |
| degree / strength | identical | identical |
| betweenness | relative difference < 10⁻⁶ (rounding) | — |
| eigenvector centrality | r > 0.99999999; 2.0 values multiplied by the eigenvalue (E-01) | same |
| leading eigenvalue | identical | identical |
| Newman modularity (GPU) | **2.0 fails on all 8 networks** (NaN eigenvalue → one module, Q = −∞) | see below |
| participation coefficient | 2.0: 0 everywhere (follows from the failed modularity) | follows the module partition |

Weighted Newman modularity Q (higher is better; same networks):

| network | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| PAGANI 2.0 | 0.5948 | 0.6195 | 0.5413 | 0.5670 | 0.5447 | 0.5958 | 0.5239 | 0.5084 |
| PAGANI 2.1 | **0.5999** | **0.6209** | **0.5549** | **0.5720** | **0.5546** | **0.6108** | **0.5240** | **0.5103** |

Unweighted Newman modularity Q of 2.1: 0.594, 0.614, 0.550, 0.561, 0.548,
0.610, 0.522, 0.502 (2.0: failed). For comparison, Louvain gives 0.571 and
0.610 on networks 5 and 6.

### Run times (8 networks, seconds)

| step | 2.0 | 2.1 |
|---|---|---|
| construction, unweighted (8 subjects) | — (no output) | 6.0 |
| construction, weighted | 31.2 | 6.8 |
| group network `bf` | failed | 9.3 |
| Lp + nodal efficiency, unweighted | 58.4 | 16.2 |
| Lp + nodal efficiency, weighted | 59.9 | 17.4 |
| clustering, unweighted / weighted | 2.0 / 4.9 | 2.3 / 7.5 |
| eigenvector centrality, weighted (own networks) | 109.3 | 1.1 |
| betweenness, unweighted | 126.9 | 141 |
| Newman (GPU), unweighted | failed | 205* |
| Newman (GPU), weighted | 33.2 | 145 |
| Kendall networks (8 subjects) | — | 116.3 |

\* measured while the computer was heavily loaded. The Newman tools of 2.1
are slower than 2.0 because the fine tuning is run to convergence; they find
partitions with higher modularity (table above).

## 3. Conclusion

All tools of 2.1 agree with the reference implementations of the test suite
on the GPU. On real data 2.1 reproduces every 2.0 result that 2.0 computed
correctly (path length, efficiency, binary clustering, degree, betweenness,
eigenvector centrality up to its scaling, edge sets), and differs exactly where
2.0 had a defect (unweighted construction, Fisher group averaging, Onnela
clustering, eigenvector scaling, Newman modularity).

## 4. PAGANI 2.2.0 vs 2.1.0

Date: 2026-10-20 · build: develop @ `7dd653d` (2.2.0 `dev-build` package) ·
same computer and data (8 subjects, 19,574 voxels, 1 % sparsity).

**GPU test suite:** 80 passed, 2 skipped (GUI end-to-end tests, which pass in CI).

**Real data, file by file** (`tests/realdata/run_comparison.py --extra`):

| Results | Agreement with 2.1.0 |
|---|---|
| Networks: individual, group (Fisher), high-resolution Pearson, Kendall | identical edge sets and weights (max. difference 0) |
| Lp, nodal efficiency, Cp, degree, betweenness, eigenvector centrality, PC (unweighted and weighted) | identical (max. difference 0) |
| Newman modularity, weighted | identical partitions |
| Newman modularity, unweighted | 6 of 8 networks: same Q; 2 networks: higher Q (0.5937 → 0.5941, 0.60975 → 0.60977), different partitions because nodes with exactly equal gains are now assigned without rounding noise; PC follows the partition |

Run times for the 8 networks (seconds):

| Step | 2.1.0 | 2.2.0 |
|---|---|---|
| Lp + nodal efficiency, unweighted | 40.7 | 2.1 |
| Betweenness | 253.6 | 78.4 |
| Newman (GPU), unweighted / weighted | 400.7 / 140.9 | 68.6 / 68.8 |
| Kendall construction | 114.1 | ~33 |

The analyses with random networks were timed with `tests/realdata/benchmark.py`
on one network (CHANGELOG, 2.2.0 "Speed").
