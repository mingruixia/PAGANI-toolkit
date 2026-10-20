# PAGANI Toolkit 2.1 — User Guide

PAGANI (**PA**rallel **G**raph-theoretical **AN**alysIs) Toolkit builds
voxel-wise functional brain networks from resting-state fMRI and computes their
graph-theoretical metrics, using the GPU for the expensive steps. For the
methods and benchmarks see the PAGANI paper (Human Brain Mapping, 2018;
`docs/PAGANI_manuscript_revised.pdf` in the package).

Contents: [1 Installation](#1-installation) · [2 Preparing data](#2-preparing-data) ·
[3 Using the GUI](#3-using-the-gui) · [4 Command-line tools](#4-command-line-tools) ·
[5 Output files](#5-output-files) · [6 Memory and run time](#6-memory-and-run-time) ·
[7 Environment variables](#7-environment-variables) · [8 Troubleshooting](#8-troubleshooting)

---

## 1. Installation

Requirements

- Windows 10/11, 64-bit
- An NVIDIA GPU of the Pascal generation (GeForce GTX 10xx) or newer, with a
  current driver (CUDA 12 compatible, version 527.41 or later). Only network
  construction, Lp (`CUBFW_Lp`), betweenness, eigenvector centrality and the
  Newman modularity tool use the GPU; every other tool runs on the CPU.
- Enough host memory for the networks you build (see section 6).

Unzip `PAGANI-Toolkit-2.1.0-win64.zip` anywhere (no installation, no CUDA
toolkit needed) and start `PAGANI.exe`. The folder layout must be kept:

```
PAGANI.exe                 GUI
exefiles\                  tools for unweighted (binary) networks
exefiles_weighted\         tools for weighted networks
docs\                      this guide, the validation guide, the paper
```

## 2. Preparing data

- **One preprocessed 4D NIfTI-1 image per subject** (`.nii`, uncompressed),
  all in one folder — the *working directory*. Typical preprocessing:
  slice timing, realignment, normalisation to a template, nuisance regression,
  band-pass filtering. All images must have the same dimensions and number of
  time points ≥ 3. Every `.nii` file in the working directory is treated as a
  subject, except the mask file.
- **A mask** with the same x/y/z dimensions, e.g. a grey-matter probability map
  resampled to the functional resolution. Voxels whose mask value is greater
  than or equal to the *mask threshold* are the network nodes (for a binary
  mask use a threshold such as 0.5).
- Supported data types: uint8, int8, int16, uint16, int32, uint32, float32,
  float64, either byte order; `scl_slope/scl_inter` scaling and header
  extensions are handled.
- The number of nodes grows quickly with resolution: a grey-matter mask has
  roughly 10–20 thousand voxels at 4 mm, 25–50 thousand at 3 mm and
  90–150 thousand at 2 mm, depending on the mask threshold. Large networks need
  sparse thresholds and, at the highest resolutions, the high-resolution
  construction (section 6).
- Time series must not contain NaN or Inf inside the mask.

Voxels with a constant time series inside the mask get zero correlations; the
construction tools report how many there are.

## 3. Using the GUI

The window has four groups of settings on the left and the log on the right.

**Data** — working directory, mask file, mask threshold and network type
(unweighted or weighted). The network type selects `exefiles` or
`exefiles_weighted` for every step.

**Network Construction** (uncheck the group to analyse networks built earlier;
they are read from `<working directory>\unweighted` or `\weighted`)

- *Standard* (`CUCorMat`): individual networks; optionally a group-level
  network (average of the correlation matrices, either of r or of Fisher z) and
  the whole correlation matrix (`.cormat`).
- *High resolution* (`CUCorMatHr`): never keeps the whole correlation matrix in
  memory, for very large networks; individual Pearson networks only.
- *Threshold*: correlation (edges with r > t; values in (−1, 1)) or sparsity
  (percentage of all node pairs that become edges; values in (0, 100]). Several
  values separated by spaces build one network per value.

**Global Metrics** — characteristic path length Lp, clustering coefficient Cp
(Onnela or Barrat for weighted networks), gamma, lambda, small-world sigma,
modularity (Louvain, Newman on GPU, Newman on CPU) and the number of random
networks used by gamma/lambda/sigma and the modularity of random networks.

**Nodal Metrics** — nodal efficiency, nodal clustering coefficient, degree
(strength for weighted networks), betweenness (unweighted only), eigenvector
centrality and participation coefficient. Nodal metrics are written as `.nm`
files and converted to NIfTI images with the mask.

Options that need other results are enabled automatically (e.g. sigma enables
gamma and lambda, participation enables modularity); hovering over an option
shows what it computes and which files it writes.

**Run** checks the settings and runs the steps one after another in the
background; the log shows the output of every program and the progress bar the
current step. **Stop** terminates the running step. If a step fails, the run
stops and the reason is shown.

**Save Script…** writes the steps as a Windows batch file that can be run on
another computer or later from a command prompt (all paths are quoted; the
script stops at the first error). **Load Script…** restores all settings from a
saved script; scripts of PAGANI 1.x/2.0 can also be loaded.

## 4. Command-line tools

All tools print their usage when started without arguments, return 0 on
success and a non-zero code on failure. The tools in `exefiles` read unweighted
networks (`*.csr`), those in `exefiles_weighted` weighted networks
(`*_weighted.csr`). A tool that takes a directory processes every network file
in it.

### 4.1 Network construction

```
CUCorMat.exe  BOLD_dir mask.nii mask_threshold to_average to_save_cormat threshold_type thresholds...
```

| argument | values |
|---|---|
| `to_average` | `n` individual networks · `yn`/`yf` group network only (average of r / of Fisher z) · `bn`/`bf` individual and group networks |
| `to_save_cormat` | `y` also writes the correlation matrix (`<subject>_<N>.cormat`, group: `ave_<subjects>_<N>.cormat`) · `n` |
| `threshold_type` | `r` correlation (edges with r > t) · `s` sparsity in % |

Example: `exefiles\CUCorMat.exe D:\study\bold D:\study\grey.nii 0.2 bn n s 1 5 10`

```
CUCorMatHr.exe  BOLD_dir mask.nii mask_threshold n threshold_type coefficient thresholds...
```

`coefficient`: `p` Pearson, `s` Spearman, `ps` both. Networks are named
`<subject>_<N>_pearson_…` / `_spearman_…`.

```
exefiles_weighted\CuCorMat-sca-kendall.exe  BOLD_dir mask.nii mask_threshold n threshold_type thresholds...
```

Kendall's tau-b networks (weighted), written to `BOLD_dir\weightedKendall`.

Networks are written to `BOLD_dir\unweighted` (tools in `exefiles`) or
`BOLD_dir\weighted` (tools in `exefiles_weighted`) as

```
<subject>_<N>_spa<sparsity>%_cor<r>.csr            unweighted
<subject>_<N>_spa<sparsity>%_cor<r>_weighted.csr   weighted
ave_<subjects>_<N>_spa…                            group network
```

`N` is the number of nodes, `spa` the sparsity (the requested value for
sparsity thresholds, the resulting value for correlation thresholds) and `cor`
the correlation threshold (the requested value, or the weakest edge's r for
sparsity thresholds).

### 4.2 Network metrics

| tool | arguments | outputs per network |
|---|---|---|
| `CUBFW_Lp` (GPU) / `BFS_MulCPU` (CPU) | `dir n_random flags` — flags: `g` global Lp, `n` nodal efficiency, `l` lambda | `_Lp.txt` (Lp of the network, then of each random network), `_eff.nm`, `_lambda.txt` |
| `Cp` (unweighted) | `dir n_random flags` — `g` global, `n` nodal, `k` gamma | `_cp.txt`, `_cp.nm`, `_gamma.txt` |
| `Cp` (weighted) | `dir n_random [type] flags` — type `1` Barrat, `2` Onnela (default) | as above |
| `SmallWorldProperty` | `dir` (after Lp with `l` and Cp with `k`) | `_smallWorld.txt` (sigma = gamma / lambda) |
| `Louvain_Modularity`, `Newman_Modularity_CPU`, `CUmodularity` / `CUModularity` (GPU Newman) | `dir [n_random]` | `_modu.nm` (module of each node: 1…M, 0 = isolated), `_modu.txt` (Q and number of modules of the network and of each random network) |
| `PC_CPU` | `dir` (after a modularity tool) | `_pc.nm`, `_pc_normalized.nm` |
| `Degree` | `dir` | `_deg.nm` (degree / strength) |
| `CUBC` (unweighted) | `dir` | `_bc.nm` |
| `CUEC` | `dir` | `_ec.nm`, `_evalue.txt` |
| `ConvertNII` | `dir mask.nii [mask_threshold]` | `<file>.nii` for every `.nm` file in `dir` |
| `Maslov` | `network.csr` | `<network>_Maslov.csr` (one random network) |

Definitions

- Lp = 1 / global efficiency (harmonic mean of the shortest path lengths, so
  disconnected node pairs are allowed); nodal efficiency = mean of 1/d(i,j)
  over all other nodes. Weighted distance = 1 / weight.
- Cp is the mean of the nodal clustering coefficients over all nodes (nodes
  with degree < 2 count as 0).
- gamma = Cp / mean(Cp of random networks); lambda = Lp / mean(Lp of random
  networks); sigma = gamma / lambda. Random networks preserve the degree
  sequence (Maslov & Sneppen 2002); for weighted networks the weights move with
  the edges.
- Betweenness counts every ordered pair of nodes (as BCT `betweenness_bin`).
- Eigenvector centrality is the leading eigenvector, non-negative, with unit
  length.
- Participation coefficient PC = 1 − Σ_m (k_im / k_i)²; the normalised value is
  divided by its maximum (M − 1) / M.

Example script (unweighted, sparsity 5 % and 10 %, 10 random networks):

```bat
set P=C:\PAGANI\exefiles
set D=D:\study\bold
"%P%\CUCorMat.exe" "%D%" "D:\study\grey.nii" 0.2 n n s 5 10 || exit /b 1
"%P%\CUBFW_Lp.exe" "%D%\unweighted" 10 gnl || exit /b 1
"%P%\Cp.exe" "%D%\unweighted" 10 gnk || exit /b 1
"%P%\SmallWorldProperty.exe" "%D%\unweighted" || exit /b 1
"%P%\Louvain_Modularity.exe" "%D%\unweighted" 10 || exit /b 1
"%P%\PC_CPU.exe" "%D%\unweighted" || exit /b 1
"%P%\Degree.exe" "%D%\unweighted" || exit /b 1
"%P%\ConvertNII.exe" "%D%\unweighted" "D:\study\grey.nii" 0.2 || exit /b 1
```

## 5. Output files

All binary files are little-endian.

| file | layout |
|---|---|
| `.csr` | network in compressed sparse row format: int32 `R_length` (= N+1), uint32 `R[R_length]` (row offsets), uint32 `C_length`, uint32 `C[C_length]` (column indices, each edge stored in both directions); weighted files add uint32 `V_length`, float32 `V[V_length]` |
| `.nm` | nodal metric: int32 N, float32 values[N] in mask order (x fastest, then y, then z) |
| `.cormat` | upper triangle of the correlation matrix, row by row: int32 count, float32 values[count]; when count does not fit in int32 the first field is −1 followed by a uint64 count |
| `.nii` | 3D float32 NIfTI-1 image of a nodal metric, with the geometry of the mask |
| `.txt` | plain-text numbers (first line: the network itself) |

MATLAB example for reading a `.nm` file:

```matlab
fid = fopen('sub01_..._deg.nm'); n = fread(fid, 1, 'int32'); v = fread(fid, n, 'float32'); fclose(fid);
```

## 6. Memory and run time

- Network size: an unweighted network with E edges needs about 8·E bytes on
  disk and in memory (weighted: 12·E). A 1 % sparsity network of 50 000 nodes
  has 12.5 million edges (~100 MB).
- `CUCorMat` keeps the selected edges in memory; with `to_save_cormat = y` it
  also writes N(N−1)/2 floats (50 000 nodes: 5 GB). For very large N use
  `CUCorMatHr` with sparse thresholds.
- When the BOLD data do not fit in GPU memory, the construction tools stream
  them block by block (slower but no limit apart from host memory).
- `CUBFW_Lp` computes weighted networks with a GPU all-pairs algorithm
  (Floyd–Warshall) that needs N² × 4 bytes (resident on the GPU when possible,
  otherwise streamed from host memory). Unweighted networks use a GPU
  breadth-first search of 512 sources at a time, which needs no distance
  matrix (CPU breadth-first search if the network does not fit in GPU memory).
  `BFS_MulCPU` always uses the CPU.
- Random networks multiply the cost of Lp, Cp and modularity by (1 + n_random).

## 7. Environment variables

| variable | effect |
|---|---|
| `PAGANI_SEED` | seed for random networks and Louvain (reproducible results) |
| `PAGANI_THREADS` | maximum number of CPU threads |
| `PAGANI_GPU` | index of the GPU to use (default 0) |
| `PAGANI_BLOCK` | block size of the correlation tools (multiple of 96, default 4032) |

## 8. Troubleshooting

| message | what to do |
|---|---|
| *no usable NVIDIA GPU found* | install the current NVIDIA driver; the GPU tools require a CUDA 12 compatible driver |
| *… and the mask (…) have different dimensions* | resample the mask to the functional images |
| *unsupported NIfTI datatype*, *compressed .nii.gz files are not supported*, *.hdr/.img pair* | convert the image to a single uncompressed float32 or int16 `.nii` file (e.g. with SPM, FSL or nibabel) |
| *contains NaN or Inf values inside the mask* | use a mask that excludes voxels without signal, or replace NaN by 0 and exclude those voxels from the mask |
| *not enough host memory for the distance matrix* | use `BFS_MulCPU` for Lp, or fewer nodes |
| *… voxels have a constant time series* | check the mask/brain coverage; these nodes are isolated |
| *skip …: …_modu.nm not found* | run a modularity tool before `PC_CPU` |

Please report problems at <https://github.com/mingruixia/PAGANI-toolkit/issues>.
