"""Reference implementations and file helpers used by the PAGANI test-suite.

Everything here is written independently of the C++/CUDA code, using
numpy / scipy / networkx / bctpy, so that the toolkit can be validated
against well-established implementations.
"""
from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

# --------------------------------------------------------------------- files

def write_csr(path, A, weighted: bool):
    """Write a symmetric (dense or sparse) matrix as a PAGANI .csr file."""
    A = sp.csr_matrix(A, dtype=np.float32)
    A.setdiag(0)
    A.eliminate_zeros()
    A.sort_indices()
    n = A.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("<i", n + 1))
        f.write(A.indptr.astype("<u4").tobytes())
        f.write(struct.pack("<I", A.nnz))
        f.write(A.indices.astype("<u4").tobytes())
        if weighted:
            f.write(struct.pack("<I", A.nnz))
            f.write(A.data.astype("<f4").tobytes())


def read_csr(path):
    """Return (scipy csr matrix, weighted flag)."""
    b = Path(path).read_bytes()
    rl = struct.unpack_from("<i", b, 0)[0]
    R = np.frombuffer(b, "<u4", rl, 4)
    off = 4 + 4 * rl
    cl = struct.unpack_from("<I", b, off)[0]
    C = np.frombuffer(b, "<u4", cl, off + 4)
    off += 4 + 4 * cl
    weighted = len(b) > off
    if weighted:
        vl = struct.unpack_from("<I", b, off)[0]
        V = np.frombuffer(b, "<f4", vl, off + 4).astype(np.float64)
    else:
        V = np.ones(cl)
    n = rl - 1
    return sp.csr_matrix((V, C, R), shape=(n, n)), weighted


def read_nm(path):
    b = Path(path).read_bytes()
    n = struct.unpack_from("<i", b, 0)[0]
    assert len(b) == 4 + 4 * n
    return np.frombuffer(b, "<f4", n, 4).astype(np.float64)


def write_nm(path, v):
    v = np.asarray(v, dtype="<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<i", v.size))
        f.write(v.tobytes())


def read_first_number(path):
    return float(Path(path).read_text().split()[0])


# ----------------------------------------------------------------- executables

def bin_dir() -> Path:
    return Path(os.environ.get("PAGANI_BIN", Path(__file__).resolve().parents[1] / "build" / "package"))


def exe(name: str, weighted: bool) -> str:
    d = bin_dir() / ("exefiles_weighted" if weighted else "exefiles")
    p = d / name
    if os.name == "nt":
        p = p.with_suffix(".exe")
    return str(p)


def run(name, weighted, *args, check=True, env=None):
    e = dict(os.environ)
    e.setdefault("PAGANI_SEED", "12345")
    if env:
        e.update(env)
    cp = subprocess.run([exe(name, weighted), *map(str, args)], capture_output=True, text=True, env=e)
    if check and cp.returncode != 0:
        raise AssertionError(f"{name} failed ({cp.returncode}):\n{cp.stdout}\n{cp.stderr}")
    return cp


# ------------------------------------------------------------- synthetic data

def modular_graph(n=240, modules=4, p_in=0.25, p_out=0.02, weighted=False, seed=0,
                  isolated=3, extra_component=True):
    """Symmetric adjacency matrix with planted modules, some isolated nodes and
    (optionally) a small disconnected component."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(modules), n // modules)
    labels = np.concatenate([labels, np.full(n - labels.size, modules - 1)])
    P = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    U = rng.random((n, n))
    A = np.triu(U < P, 1)
    A = (A | A.T).astype(float)
    if extra_component:
        comp = np.arange(n - isolated - 4, n - isolated)
        A[comp, :] = 0
        A[:, comp] = 0
        for i in comp:
            for j in comp:
                if i != j:
                    A[i, j] = 1
    if isolated:
        iso = np.arange(n - isolated, n)
        A[iso, :] = 0
        A[:, iso] = 0
    if weighted:
        W = np.triu(rng.uniform(0.2, 0.95, (n, n)), 1)
        W = W + W.T
        A = A * W
    return A


# ------------------------------------------------------------------ metrics

def degree(A):
    return np.asarray(A.sum(1)).ravel()


def clustering_binary(A):
    B = (np.asarray(A) > 0).astype(float)
    k = B.sum(1)
    t = np.diag(B @ B @ B) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(k >= 2, 2 * t / (k * (k - 1)), 0.0)
    return c


def clustering_onnela(W):
    W = np.asarray(W, float)
    Wn = W / W.max()
    B = (W > 0).astype(float)
    k = B.sum(1)
    C3 = np.cbrt(Wn)
    cyc = np.diag(C3 @ C3 @ C3)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k >= 2, cyc / (k * (k - 1)), 0.0)


def clustering_barrat(W):
    W = np.asarray(W, float)
    B = (W > 0).astype(float)
    n = W.shape[0]
    k = B.sum(1)
    s = W.sum(1)
    c = np.zeros(n)
    for i in range(n):
        nb = np.nonzero(B[i])[0]
        if len(nb) < 2:
            continue
        acc = 0.0
        for a in range(len(nb)):
            for b in range(len(nb)):
                j, h = nb[a], nb[b]
                if j != h and B[j, h]:
                    acc += (W[i, j] + W[i, h]) / 2.0
        c[i] = acc / (s[i] * (k[i] - 1))
    return c


def efficiency(A, weighted):
    A = np.asarray(A, float)
    n = A.shape[0]
    if weighted:
        D = np.zeros_like(A)
        D[A > 0] = 1.0 / A[A > 0]
        dist = csg.shortest_path(sp.csr_matrix(D), directed=False)
    else:
        dist = csg.shortest_path(sp.csr_matrix((A > 0).astype(float)), directed=False, unweighted=True)
    with np.errstate(divide="ignore"):
        inv = 1.0 / dist
    np.fill_diagonal(inv, 0.0)
    inv[~np.isfinite(inv)] = 0.0
    nodal = inv.sum(1) / (n - 1)
    eglob = inv.sum() / (n * (n - 1))
    return 1.0 / eglob, nodal


def betweenness(A):
    import networkx as nx
    G = nx.from_numpy_array((np.asarray(A) > 0).astype(int))
    bc = nx.betweenness_centrality(G, normalized=False)
    return 2.0 * np.array([bc[i] for i in range(A.shape[0])])  # ordered-pair convention


def eigenvector(A):
    A = np.asarray(A, float)
    vals, vecs = np.linalg.eigh(A)
    v = vecs[:, -1]
    v = v * np.sign(v.sum())
    return vals[-1], v / np.linalg.norm(v)


def modularity(A, labels):
    A = np.asarray(A, float)
    k = A.sum(1)
    m2 = k.sum()
    same = labels[:, None] == labels[None, :]
    return float(((A - np.outer(k, k) / m2) * same).sum() / m2)


def participation(A, labels):
    A = np.asarray(A, float)
    k = A.sum(1)
    pc = np.zeros(A.shape[0])
    mods = np.unique(labels[k > 0])
    for i in range(A.shape[0]):
        if k[i] == 0:
            continue
        s = 0.0
        for m in mods:
            s += (A[i, labels == m].sum() / k[i]) ** 2
        pc[i] = 1 - s
    return pc, mods.size


# ---------------------------------------------------------------- NIfTI helpers

_DT = {np.uint8: (2, 8), np.int16: (4, 16), np.int32: (8, 32), np.float32: (16, 32),
       np.float64: (64, 64), np.int8: (256, 8), np.uint16: (512, 16)}


def save_nifti(path, raw, pixdim=4.0, slope=0.0, inter=0.0, extension_bytes=0, big_endian=False):
    """Write a single-file NIfTI-1 image with full control over the header.
    `raw` is stored as-is (its dtype selects datatype); voxel values seen by a
    reader are raw*slope+inter when slope != 0."""
    raw = np.asarray(raw)
    dt, bp = _DT[raw.dtype.type]
    e = ">" if big_endian else "<"
    shape = raw.shape
    hdr = bytearray(348)
    struct.pack_into(e + "i", hdr, 0, 348)
    dims = [len(shape)] + list(shape) + [1] * (7 - len(shape))
    struct.pack_into(e + "8h", hdr, 40, *dims)
    struct.pack_into(e + "hh", hdr, 70, dt, bp)
    struct.pack_into(e + "8f", hdr, 76, 1.0, pixdim, pixdim, pixdim, 2.0, 1, 1, 1)
    vox_offset = 352 + extension_bytes
    struct.pack_into(e + "f", hdr, 108, float(vox_offset))
    struct.pack_into(e + "ff", hdr, 112, slope, inter)
    struct.pack_into(e + "hh", hdr, 252, 1, 1)
    struct.pack_into(e + "4f", hdr, 280, pixdim, 0, 0, 0)
    struct.pack_into(e + "4f", hdr, 296, 0, pixdim, 0, 0)
    struct.pack_into(e + "4f", hdr, 312, 0, 0, pixdim, 0)
    hdr[344:348] = b"n+1\0"
    with open(path, "wb") as f:
        f.write(hdr)
        if extension_bytes:
            f.write(bytes([1, 0, 0, 0]))
            esize = extension_bytes
            f.write(struct.pack(e + "ii", esize, 6))
            f.write(b"x" * (esize - 8))
        else:
            f.write(bytes(4))
        f.write(raw.astype(raw.dtype.newbyteorder(e)).tobytes(order="F"))


def load_nifti(path):
    import nibabel as nib
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float64), img.header
