# PAGANI Toolkit

**PA**rallel **G**raph-theoretical **AN**alys**I**s toolkit — GPU-accelerated
construction and graph-theoretical analysis of voxel-wise (high-resolution)
functional brain networks.

[中文说明](README_zh.md) · [User guide](docs/USER_GUIDE.md) · [Changelog](CHANGELOG.md) · [Validation](docs/VALIDATION.md) · [Validation report](docs/VALIDATION_REPORT.md)

PAGANI builds a network for every subject from resting-state fMRI, with every
grey-matter voxel as a node and thresholded correlations between voxel time
series as edges, and computes global and nodal network metrics:

- **Construction** — Pearson, Spearman or Kendall correlation; correlation or
  sparsity thresholds; individual and group-level (r or Fisher-z averaged)
  networks; unweighted and weighted.
- **Global metrics** — characteristic path length, clustering coefficient,
  small-world gamma / lambda / sigma (Maslov–Sneppen random networks),
  modularity (Louvain, Newman).
- **Nodal metrics** — degree/strength, nodal efficiency, clustering coefficient,
  betweenness, eigenvector centrality, participation coefficient — written as
  NIfTI images.

A Qt GUI runs the whole pipeline; every step is also a command-line program for
scripts and servers.

## Download

Get `PAGANI-Toolkit-<version>-win64.zip` from the
[Releases](https://github.com/mingruixia/PAGANI-toolkit/releases) page, unzip
and start `PAGANI.exe`. Requirements: 64-bit Windows 10/11 and an NVIDIA GPU
(GTX 10xx or newer, compute capability ≥ 6.1) with a current driver; no CUDA
installation is needed.

On Linux servers and clusters use `PAGANI-Toolkit-<version>-linux-x64.tar.gz`:
the same command-line tools (no GUI), for x86-64 Linux with glibc 2.28 or newer
(RHEL/Rocky/Alma 8, Ubuntu 20.04, Debian 11 and newer) and an NVIDIA driver
≥ 525.60.13; `example_pipeline.sh` in the package shows a complete analysis.
See the [user guide](docs/USER_GUIDE.md).

Version 2.1 is a maintenance release of 2.0 that fixes several calculation
errors; version 2.2 adds further fixes and makes the analyses with random
networks and several GPU tools much faster — please read the
[changelog](CHANGELOG.md) if you compare with earlier results.

## Building from source

Requirements: CMake ≥ 3.21, a C++17 compiler (MSVC 2022 on Windows), the CUDA
toolkit 12.x for the GPU tools and Qt 5.15 or 6 for the GUI.

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=C:\Qt\5.15.2\msvc2019_64
cmake --build build
ctest --test-dir build --output-on-failure
powershell -File packaging\make_package.ps1 -BuildDir build -OutDir dist
```

On Linux the release package is built in the `nvidia/cuda:12.6.3-devel-rockylinux8`
container (see `packaging/linux/setup_build_env.sh` and the `linux-cuda` job of
`.github/workflows/build.yml`):

```bash
source packaging/linux/setup_build_env.sh
cmake -S . -B build -G Ninja -DPAGANI_BUILD_GUI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build && ctest --test-dir build --output-on-failure
packaging/make_package.sh build dist
```

Options: `-DPAGANI_BUILD_CUDA=OFF` (CPU tools only, also on Linux/macOS),
`-DPAGANI_BUILD_GUI=OFF`, `-DPAGANI_BUILD_TESTS=OFF`. The tests need Python
with `pip install -r tests/requirements.txt`.

Repository layout:

```
src/common     NIfTI, CSR network files, random networks, utilities
src/metrics    graph metrics (CPU)
src/construct  correlation and thresholding engine
src/gpu        CUDA kernels and GPU back ends
src/apps       the command-line tools
src/gui        the Qt GUI
tests          reference implementations and tests
legacy         PAGANI 2.0 sources (2018), for reference only
docs           user guide, validation, code review and development plan
```

## Citation

If you use PAGANI, please cite:

Du H, Xia M, Zhao K, Liao X, Yang H, Wang Y, He Y. PAGANI Toolkit: Parallel
graph-theoretical analysis package for brain network big data.
*Human Brain Mapping* 2018. https://doi.org/10.1002/hbm.23996
