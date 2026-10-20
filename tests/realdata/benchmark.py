"""Time the GPU/CPU tools of a PAGANI package on synthetic and real data.

    python benchmark.py --pkg <PAGANI folder> --work <scratch dir> [--nets <dir with real .csr networks>]

Writes <work>/benchmark.txt. Synthetic data: 4D NIfTI with N voxels and L
time points (default 60000 x 1200) to measure the correlation engine.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402

import pagani_ref as R  # noqa: E402


def exe(pkg, weighted, name):
    return str(Path(pkg) / ("exefiles_weighted" if weighted else "exefiles") / (name + (".exe" if os.name == "nt" else "")))


def timed(cmd, log, env=None):
    t0 = time.perf_counter()
    cp = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env={**os.environ, **(env or {})})
    dt = time.perf_counter() - t0
    log.write(f"\n$ {' '.join(map(str, cmd))}\n{cp.stdout[-3000:]}{cp.stderr[-2000:]}\n[{dt:.2f} s, exit {cp.returncode}]\n")
    log.flush()
    return dt, cp


def construct_benchmarks(a, work, rng, log, note):
    """Correlation engine on synthetic data."""
    syn = work / "syn"
    if syn.exists():
        shutil.rmtree(syn)
    syn.mkdir()
    nx, ny = 60, 50
    nz = (a.n + nx * ny - 1) // (nx * ny)
    vol = rng.standard_normal((nx, ny, nz, a.l), dtype=np.float32)
    vol += rng.standard_normal((1, 1, 1, a.l), dtype=np.float32)  # some shared signal
    R.save_nifti(syn / "s1.nii", vol)
    del vol
    mask = work / "syn_mask.nii"
    flat = np.zeros(nx * ny * nz, np.float32)
    flat[: a.n] = 1
    R.save_nifti(mask, flat.reshape((nx, ny, nz), order="F"))
    for label, args in [("CUCorMat N=%d L=%d, r > 0.9 (1 pass)" % (a.n, a.l), ["n", "n", "r", 0.9]),
                        ("CUCorMat N=%d L=%d, sparsity 0.1%% (2 passes)" % (a.n, a.l), ["n", "n", "s", 0.1])]:
        dt, cp = timed([exe(a.pkg, False, "CUCorMat"), syn, mask, 0.5, *args], log)
        m_ = re.search(r"done in ([0-9.e+-]+) s", cp.stdout)
        note(label, dt, f"(tool: {m_.group(1) if m_ else '?'} s)")
        shutil.rmtree(syn / "unweighted", ignore_errors=True)
    dt, cp = timed([exe(a.pkg, False, "CUCorMatHr"), syn, mask, 0.5, "n", "s", "s", 0.1], log)
    note("CUCorMatHr Spearman sparsity 0.1%", dt)
    shutil.rmtree(syn, ignore_errors=True)
    # Kendall: smaller data (cost ~ N^2 L^2)
    kd = work / "kendall"
    shutil.rmtree(kd, ignore_errors=True)
    kd.mkdir()
    nk, lk = 20000, 150
    kz = (nk + nx * ny - 1) // (nx * ny)
    vol = rng.standard_normal((nx, ny, kz, lk), dtype=np.float32)
    vol += rng.standard_normal((1, 1, 1, lk), dtype=np.float32)
    R.save_nifti(kd / "k1.nii", vol)
    del vol
    flat = np.zeros(nx * ny * kz, np.float32)
    flat[:nk] = 1
    R.save_nifti(work / "kendall_mask.nii", flat.reshape((nx, ny, kz), order="F"))
    dt, cp = timed([exe(a.pkg, True, "CuCorMat-sca-kendall"), kd, work / "kendall_mask.nii", 0.5, "n", "s", 0.1], log)
    note(f"Kendall N={nk} L={lk}, sparsity 0.1%", dt, "" if cp.returncode == 0 else f"exit {cp.returncode}")
    shutil.rmtree(kd, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--nets", help="folder with a real unweighted and weighted network (*.csr, *_weighted.csr)")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--l", type=int, default=1200)
    ap.add_argument("--fw-sizes", type=int, nargs="*", default=[20000, 40000])
    ap.add_argument("--skip-construct", action="store_true")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out = open(work / "benchmark.txt", "w")
    log = open(work / "benchmark_log.txt", "w")
    res = []

    def note(name, dt, extra=""):
        res.append((name, dt, extra))
        out.write(f"{name:55s} {dt:9.2f} s  {extra}\n")
        out.flush()

    out.write(f"{platform.platform()}  {platform.processor()}  cpus={os.cpu_count()}\n")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        out.write(f"GPU: {smi}\n\n")
    except OSError:
        pass

    rng = np.random.default_rng(0)
    if not a.skip_construct:
        construct_benchmarks(a, work, rng, log, note)

    # ---- path length on a large synthetic weighted network (Floyd-Warshall scaling)
    import scipy.sparse as sp
    for nn in a.fw_sizes:
        d = work / f"fw{nn}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        e = int(0.0052 * nn * nn)  # ~1 % density after symmetrisation
        i = rng.integers(0, nn, e)
        j = rng.integers(0, nn, e)
        w = rng.uniform(0.2, 0.9, e).astype(np.float32)
        A = sp.coo_matrix((w, (i, j)), shape=(nn, nn)).tocsr()
        A = A.maximum(A.T)
        R.write_csr(d / "syn_weighted.csr", A, True)
        R.write_csr(d / "syn.csr", A, False)
        for weighted in (True, False):
            dd = d / ("w" if weighted else "u")
            dd.mkdir()
            shutil.move(str(d / ("syn_weighted.csr" if weighted else "syn.csr")), dd)
            kind = "weighted" if weighted else "unweighted"
            dt, cp = timed([exe(a.pkg, weighted, "CUBFW_Lp"), dd, 0, "gn"], log)
            note(f"CUBFW_Lp synthetic {kind} n={nn} (1% density)", dt, "" if cp.returncode == 0 else f"exit {cp.returncode}")
            if not weighted and nn <= 20000:
                dt, cp = timed([exe(a.pkg, False, "CUBC"), dd], log)
                note(f"CUBC synthetic n={nn} (1% density)", dt, "" if cp.returncode == 0 else f"exit {cp.returncode}")
        shutil.rmtree(d, ignore_errors=True)

    # ---- graph metrics on real networks (one network each)
    if a.nets:
        for weighted in (False, True):
            files = sorted(p for p in Path(a.nets).glob("*_weighted.csr" if weighted else "*.csr")
                           if weighted or not p.name.endswith("_weighted.csr"))
            if not files:
                continue
            d = work / ("w" if weighted else "u")
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir()
            shutil.copy(files[0], d / files[0].name)
            kind = "weighted" if weighted else "unweighted"
            steps = [("Lp + efficiency (CUBFW_Lp)", "CUBFW_Lp", [d, 0, "gn"]),
                     ("Lp + efficiency (BFS_MulCPU)", "BFS_MulCPU", [d, 0, "gn"]),
                     ("Lp, 1 random network (CUBFW_Lp)", "CUBFW_Lp", [d, 1, "g"]),
                     ("Cp", "Cp", [d, 0, "gn"]),
                     ("Cp + 5 random networks", "Cp", [d, 5, "g"]),
                     ("Eigenvector centrality (CUEC)", "CUEC", [d]),
                     ("Louvain", "Louvain_Modularity", [d]),
                     ("Newman GPU", "CUModularity" if weighted else "CUmodularity", [d]),
                     ("Participation coefficient", "PC_CPU", [d])]
            steps += [("Lp + 10 random networks (CUBFW_Lp)", "CUBFW_Lp", [d, 10, "g"]),
                      ("Cp + 20 random networks", "Cp", [d, 20, "g"]),
                      ("Louvain + 10 random networks", "Louvain_Modularity", [d, 10])]
            if not weighted:
                steps.insert(5, ("Betweenness (CUBC)", "CUBC", [d]))
                steps.insert(6, ("Betweenness (CUBC, global-memory kernel)", "CUBC", [d], {"PAGANI_BC_SHARED": "0"}))
            for name, tool, args, *env in steps:
                dt, cp = timed([exe(a.pkg, weighted, tool), *args], log, env[0] if env else None)
                note(f"{kind:10s} {name}", dt, "" if cp.returncode == 0 else f"exit {cp.returncode}")
    out.write("\ndone\n")


if __name__ == "__main__":
    main()
