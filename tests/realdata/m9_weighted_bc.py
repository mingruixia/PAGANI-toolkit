"""Paired Windows CPU/CUDA benchmark on the historical 1% weighted CSR cohort."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagani_ref as ref


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class Memory(ctypes.Structure):
    _fields_ = [(k, ctypes.c_ulonglong) for k in ("total", "free", "used")]


class Monitor:
    def __init__(self):
        self.lib = ctypes.WinDLL("nvml.dll")
        self.device = ctypes.c_void_p()
        if self.lib.nvmlInit_v2() or self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.device)):
            raise RuntimeError("Cannot open NVML device 0")

    def used(self):
        memory = Memory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.device, ctypes.byref(memory)):
            raise RuntimeError("NVML memory query failed")
        return memory.used


def prepare(source, case, pct):
    graph, weighted = ref.read_csr(source)
    n = graph.shape[0]
    edges = sparse.triu(graph, k=1, format="coo")
    assert weighted and graph.nnz == 2 * edges.nnz and (graph != graph.T).nnz == 0
    assert np.all(np.isfinite(edges.data)) and np.min(edges.data) > 0
    k = int(np.floor(n*(n-1)/2*pct/100 + 0.5))
    if k > edges.nnz:
        raise ValueError("density exceeds source coverage")
    data = case / "input"
    data.mkdir(parents=True, exist_ok=True)
    path = data / "network.csr"
    if k == edges.nnz:
        shutil.copyfile(source, path)
    else:
        order = np.lexsort((edges.col, edges.row, -edges.data))[:k]
        upper = sparse.csr_matrix((edges.data[order], (edges.row[order], edges.col[order])), shape=(n,n))
        graph = upper + upper.T
        ref.write_csr(path, graph, True)
    groups, labels = connected_components(graph, directed=False)
    return dict(source=source.name, source_sha256=digest(source), csr_sha256=digest(path),
                nodes=n, edges=k, density_pct=200*k/(n*(n-1)), mean_degree=2*k/n,
                isolates=int(np.sum(np.diff(graph.indptr)==0)), components=groups,
                largest_component=int(np.bincount(labels).max()))


def run(exe, case, backend, repeat, monitor, env):
    raw = case / "input" / "network_bc.nm"
    raw.unlink(missing_ok=True)
    stop = threading.Event()
    samples = []
    def sample():
        while not stop.wait(0.025):
            samples.append(monitor.used())
    gpu = backend == "gpu"
    if gpu:
        samples.append(monitor.used())
        sampler = threading.Thread(target=sample)
        sampler.start()
    try:
        start = time.perf_counter()
        proc = subprocess.run([str(exe), str(raw.parent)], capture_output=True, text=True, env=env)
        wall = time.perf_counter() - start
    finally:
        if gpu:
            stop.set()
            sampler.join()
    (case / f"{backend}-{repeat}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    dest = case / f"{backend}-{repeat}.nm"
    raw.replace(dest)
    rec = dict(backend=backend, repeat=repeat, seconds=wall, output_sha256=digest(dest))
    if gpu:
        peak = re.search(r"Weighted BC CUDA buffer peak: (\d+) bytes", proc.stdout)
        assert peak and "GPU 0:" in proc.stdout
        rec.update(buffer_bytes=int(peak[1]), board_start_bytes=samples[0], board_peak_bytes=max(samples),
                   samples=len(samples))
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","package","work"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--indices",nargs="+",type=int,default=[0,3,7])
    parser.add_argument("--densities",nargs="+",type=float,default=[0.05,0.15,0.20,0.50,1.0])
    parser.add_argument("--repeats",type=int,default=2)
    a = parser.parse_args()
    if a.repeats < 1 or any(not 0 < d <= 1 for d in a.densities):
        parser.error("positive repeats and densities in (0, 1] are required")
    if len(set(a.indices)) != len(a.indices) or len(set(a.densities)) != len(a.densities):
        parser.error("indices and densities must be unique")
    a.work.mkdir(parents=True, exist_ok=True)
    if (a.work/"hardware.json").exists() or any(a.work.glob("subject-*")):
        parser.error("use a fresh run directory to preserve existing measurements")
    srcs=sorted(a.source.glob("*_weighted.csr"))
    if not srcs or any(i < 0 or i >= len(srcs) for i in a.indices):
        parser.error("the requested subject indices must occur in the source directory")
    env={**os.environ,"PAGANI_THREADS":"32","PAGANI_GPU":"0"}
    monitor=Monitor()
    exe={b:a.package/(name+".exe") for b,name in (("cpu","CUBC"),("gpu","CUBC_GPU"))}
    profile={b:digest(path) for b,path in exe.items()}
    profile["gpu"]=dict(binary=profile["gpu"], device=subprocess.check_output(
        ["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip())
    profile["source_dir"]=str(a.source)
    profile["cpu_threads"]=32
    profile["cpu_name"]=subprocess.check_output(["powershell","-NoProfile","-Command",
        "(Get-CimInstance Win32_Processor).Name"],text=True).strip()
    save(a.work/"hardware.json",profile)
    for di,pct in enumerate(a.densities):
        for i in a.indices:
            source=srcs[i]
            case=a.work/f"subject-{i+1:02d}"/f"density-{pct:g}"
            case.mkdir(parents=True,exist_ok=True)
            meta=prepare(source,case,pct)
            save(case/"graph.json",meta)
            for repeat in range(1,a.repeats+1):
                order=["cpu","gpu"] if (repeat+di+i)%2 else ["gpu","cpu"]
                for backend in order:
                    rec=run(exe[backend],case,backend,repeat,monitor,env)
                    save(case/f"{backend}-{repeat}.json",rec)
                cpu,gpu=[ref.read_nm(case/f"{b}-{repeat}.nm") for b in ("cpu","gpu")]
                assert len(cpu)==len(gpu)==meta["nodes"] and np.isfinite(cpu).all() and np.isfinite(gpu).all()
                diff=gpu-cpu
                relative=np.divide(np.abs(diff),np.abs(cpu),out=np.full_like(cpu,np.nan),where=cpu!=0)
                np.savez_compressed(case/f"nodes-{repeat}.npz",node_index=np.arange(len(cpu)),cpu=cpu,gpu=gpu,
                                    signed_error=diff,absolute_error=np.abs(diff),relative_error=relative)
                np.testing.assert_allclose(gpu,cpu,rtol=1e-6,atol=1e-8)
                print(case.relative_to(a.work),repeat,"max_abs",np.max(np.abs(diff)),flush=True)


if __name__=="__main__":
    main()
