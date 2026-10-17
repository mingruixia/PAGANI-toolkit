"""Run the same analyses with two PAGANI packages (e.g. 2.0 and 2.1) on real data,
time every step and compare the results file by file.

    python run_comparison.py --data <dir with 4D .nii> --mask <mask.nii> --out <results dir>
                             --pkg 2.0=<PAGANI 2.0 folder> --pkg 2.1=<PAGANI 2.1 folder>

Every package writes into <data>/unweighted and <data>/weighted; after each
package these folders are moved to <out>/<version>/. A summary with run times
is written to <out>/timing.csv and the comparisons to <out>/compare_*.csv.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def exe(pkg, weighted, name):
    p = Path(pkg) / ("exefiles_weighted" if weighted else "exefiles") / (name + (".exe" if os.name == "nt" else ""))
    return str(p)


def steps(data, mask, thr, sparsity, weighted, version):
    net = str(Path(data) / ("weighted" if weighted else "unweighted"))
    s = [("construct", "CUCorMat", [data, mask, thr, "n", "n", "s", sparsity]),
         ("Lp+eff", "CUBFW_Lp", [net, "0", "gn"]),
         ("Cp", "Cp", [net, "0", "2", "gn"] if weighted else [net, "0", "gn"]),
         ("degree", "Degree", [net]),
         ("EC", "CUEC", [net])]
    if not weighted:
        s.append(("BC", "CUBC", [net]))
    s.append(("Newman (GPU)", "CUModularity" if weighted else "CUmodularity", [net, "0"]))
    s.append(("PC", "PC_CPU", [net]))
    return s


def run(cmd, log, cwd, timeout):
    t0 = time.perf_counter()
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        fh.flush()
        try:
            cp = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd,
                                timeout=timeout)
            rc = cp.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        except OSError as e:
            fh.write(f"cannot start: {e}\n")
            rc = "error"
    return rc, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", action="append", required=True, help="version=folder")
    ap.add_argument("--thr", default="0.2")
    ap.add_argument("--sparsity", default="1")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--extra", action="store_true", help="also time group, high-resolution and Kendall construction")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "_cwd"
    scratch.mkdir(exist_ok=True)
    log = out / "run_log.txt"
    rows = []
    pkgs = [p.split("=", 1) for p in a.pkg]
    for version, pkg in pkgs:
        for weighted in (False, True):
            kind = "weighted" if weighted else "unweighted"
            dst = out / version / kind
            if dst.exists():
                print(version, kind, "already done", flush=True)
                continue
            # PAGANI 2.0 does not create the output folders (and then silently
            # writes nothing, or writes into the data folder)
            (Path(a.data) / kind).mkdir(exist_ok=True)
            for name, tool, args in steps(a.data, a.mask, a.thr, a.sparsity, weighted, version):
                rc, dt = run([exe(pkg, weighted, tool), *args], log, scratch, a.timeout)
                if name == "construct":
                    for f in Path(a.data).glob("*.csr"):
                        shutil.move(str(f), str(Path(a.data) / kind / f.name))
                rows.append({"version": version, "network": kind, "step": name, "tool": tool,
                             "seconds": round(dt, 2), "exit": rc})
                print(f"{version} {kind:10s} {name:14s} {dt:9.1f} s  exit {rc}", flush=True)
            src = Path(a.data) / kind
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.move(str(src), str(dst))
    if a.extra:
        for version, pkg in pkgs:
            extra = [("group bf (weighted)", True, "CUCorMat", [a.data, a.mask, a.thr, "bf", "n", "s", a.sparsity]),
                     ("high-res Pearson", True, "CUCorMatHr", [a.data, a.mask, a.thr, "n", "s", "p", a.sparsity])]
            if version.startswith("2.1"):
                extra.append(("Kendall (all subjects)", True, "CuCorMat-sca-kendall",
                              [a.data, a.mask, a.thr, "n", "s", a.sparsity]))
            for name, weighted, tool, args in extra:
                rc, dt = run([exe(pkg, weighted, tool), *args], log, scratch, a.timeout)
                rows.append({"version": version, "network": "weighted", "step": name, "tool": tool,
                             "seconds": round(dt, 2), "exit": rc})
                print(f"{version} extra      {name:24s} {dt:9.1f} s  exit {rc}", flush=True)
                for sub in ("weighted", "weightedKendall", "unweighted"):
                    src = Path(a.data) / sub
                    if src.exists():
                        shutil.move(str(src), str(out / version / f"extra_{tool}_{sub}"))
    if rows:
        new = not (out / "timing.csv").exists()
        with open(out / "timing.csv", "a", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
    if len(pkgs) == 2:
        (v1, _), (v2, _) = pkgs
        for kind in ("unweighted", "weighted"):
            d1, d2 = out / v1 / kind, out / v2 / kind
            if d1.exists() and d2.exists():
                subprocess.run([sys.executable, str(HERE.parent / "compare_results.py"), str(d1), str(d2),
                                "--csv", str(out / f"compare_{kind}.csv")],
                               stdout=open(out / f"compare_{kind}.txt", "w"), stderr=subprocess.STDOUT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
