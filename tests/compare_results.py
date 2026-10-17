"""Compare the outputs of two PAGANI runs on the same data (e.g. 2.0 vs 2.1).

    python compare_results.py <old_network_dir> <new_network_dir> [--csv report.csv]

Files are matched by name (the 2.0 name "<net>_pc.nm_Normalized.nm" is mapped to
"<net>_pc_normalized.nm"). For every pair the report gives
  .csr : edge counts and the Jaccard overlap of the edge sets (and the largest
         weight difference on common edges for weighted networks)
  .nm  : Pearson correlation and largest absolute difference
  .txt : first number of each file and their relative difference
Differences are expected where 2.1 fixed a 2.0 problem; see CHANGELOG.md.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pagani_ref as R  # noqa: E402

RENAMES = {"_pc.nm_Normalized.nm": "_pc_normalized.nm"}


def canonical(name):
    for old, new in RENAMES.items():
        if name.endswith(old):
            return name[: -len(old)] + new
    return name


def index(d):
    return {canonical(p.name): p for p in Path(d).iterdir() if p.suffix in (".csr", ".nm", ".txt")}


def compare_csr(a, b):
    A, wa = R.read_csr(a)
    B, wb = R.read_csr(b)
    if A.shape != B.shape:
        return {"note": f"different sizes {A.shape[0]} / {B.shape[0]}"}
    ea = set(zip(*A.nonzero()))
    eb = set(zip(*B.nonzero()))
    common = ea & eb
    res = {"old": len(ea) // 2, "new": len(eb) // 2,
           "similarity": len(common) / max(1, len(ea | eb))}
    if wa and wb and common:
        idx = tuple(np.array(list(common)).T)
        res["max_abs_diff"] = float(np.max(np.abs(np.asarray(A[idx]) - np.asarray(B[idx]))))
    return res


def compare_nm(a, b):
    x, y = R.read_nm(a), R.read_nm(b)
    if x.size != y.size:
        return {"note": f"different lengths {x.size} / {y.size}"}
    r = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else float("nan")
    return {"old": float(x.mean()), "new": float(y.mean()), "similarity": float(r),
            "max_abs_diff": float(np.max(np.abs(x - y)))}


def compare_txt(a, b):
    x, y = R.read_first_number(a), R.read_first_number(b)
    return {"old": x, "new": y, "max_abs_diff": abs(x - y), "similarity": 1 - abs(x - y) / max(abs(x), abs(y), 1e-12)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--csv")
    args = ap.parse_args()
    old, new = index(args.old), index(args.new)
    rows = []
    for name in sorted(set(old) | set(new)):
        row = {"file": name}
        if name not in old or name not in new:
            row["note"] = "only in " + ("new" if name in new else "old")
        else:
            try:
                f = {".csr": compare_csr, ".nm": compare_nm, ".txt": compare_txt}[Path(name).suffix]
                row.update(f(old[name], new[name]))
            except Exception as e:  # unreadable or legacy-format file
                row["note"] = f"error: {e}"
        rows.append(row)
    cols = ["file", "old", "new", "similarity", "max_abs_diff", "note"]
    for r in rows:
        print("  ".join(f"{c}={r[c]:.6g}" if isinstance(r.get(c), float) else f"{c}={r[c]}"
                        for c in cols if c in r))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
