"""Run the metric tools of one PAGANI package on networks built by another one,
so that differences in the metrics are not mixed with differences in the
network construction.

    python metrics_same_networks.py --nets <results>/2.1/unweighted --pkg 2.0=<PAGANI 2.0 folder>
                                    --out <results>/2.0_on_2.1/unweighted [--weighted]

The network files (*.csr) of --nets are copied to --out, every metric step of
run_comparison.py except the construction is run there with the given package,
the run times are written to <out>/_log/timing.csv and the results are compared
with those in --nets (<out>/_log/compare.csv, compare.txt).
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_comparison import exe, run, steps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", required=True)
    ap.add_argument("--pkg", required=True, help="version=folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--timeout", type=float, default=7200)
    a = ap.parse_args()
    version, pkg = a.pkg.split("=", 1)
    out = Path(a.out).resolve()
    logs = out / "_log"
    logs.mkdir(parents=True, exist_ok=True)
    nets = sorted(p for p in Path(a.nets).glob("*.csr") if not p.name.endswith("_Maslov.csr"))
    for p in nets:
        shutil.copy(p, out / p.name)
    print(f"{len(nets)} network(s) copied to {out}", flush=True)
    log = logs / "run_log.txt"
    rows = []
    for name, tool, args in steps("unused", "unused", "0", "0", a.weighted, version):
        if name == "construct":
            continue
        args = [str(out)] + args[1:]
        rc, dt = run([exe(pkg, a.weighted, tool), *args], log, out, a.timeout)
        rows.append({"version": version, "step": name, "tool": tool, "seconds": round(dt, 2), "exit": rc})
        print(f"{version} {name:14s} {dt:9.1f} s  exit {rc}", flush=True)
    with open(logs / "timing.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    subprocess.run([sys.executable, str(HERE.parent / "compare_results.py"), str(out), a.nets,
                    "--csv", str(logs / "compare.csv")],
                   stdout=open(logs / "compare.txt", "w"), stderr=subprocess.STDOUT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
