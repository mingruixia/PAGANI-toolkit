"""Validation of network construction (CUCorMat, CUCorMatHr, Kendall).

The tools are exercised through their CPU-backend builds (build/test-bin);
the CUDA backend is validated with the same checks by tests/gpu_validate.py
on a machine with an NVIDIA GPU.
"""
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import pagani_ref as R

TESTBIN = Path(os.environ.get("PAGANI_TESTBIN", Path(__file__).resolve().parents[1] / "build" / "test-bin"))
GPU = os.environ.get("PAGANI_TEST_GPU") == "1"   # run against the real (CUDA) package instead


def tool(name, weighted):
    root = R.bin_dir() if GPU else TESTBIN
    p = root / ("exefiles_weighted" if weighted else "exefiles") / name
    return str(p.with_suffix(".exe")) if os.name == "nt" else str(p)


def run(name, weighted, *args, check=True):
    import subprocess
    cp = subprocess.run([tool(name, weighted), *map(str, args)], capture_output=True, text=True)
    if check and cp.returncode != 0:
        raise AssertionError(f"{name} failed ({cp.returncode}):\n{cp.stdout}\n{cp.stderr}")
    return cp


SHAPE = (7, 6, 5)


def make_dataset(tmp_path, nsub=3, L=40, seed=0, variants=("float32", "int16_scaled", "float64_ext")):
    """Write subjects + mask; return (dir, mask path, mask threshold, list of masked series (n x L))."""
    rng = np.random.default_rng(seed)
    d = tmp_path / "data"
    d.mkdir()
    prob = rng.random(SHAPE)
    prob[0, 0, 0] = 0.95                       # will hold a constant series
    maskfile = tmp_path / "grey.nii"           # mask outside the data directory
    R.save_nifti(maskfile, np.round(prob * 255).astype(np.uint8), slope=1 / 255.0)
    thr = 0.3
    maskval = np.round(prob * 255) / 255.0
    inmask = maskval.ravel(order="F") >= thr
    nvox = int(np.prod(SHAPE))
    series = []
    for s in range(nsub):
        k = 4
        base = rng.standard_normal((k, L))
        member = rng.integers(0, k, nvox)
        X = base[member] * rng.uniform(0.5, 2.0, (nvox, 1)) + 0.8 * rng.standard_normal((nvox, L)) + 100
        X[0] = 7.0                                # constant voxel (index 0 = (0,0,0))
        variant = variants[s % len(variants)]
        vol = X.reshape(SHAPE + (L,), order="F")
        if variant == "float32":
            R.save_nifti(d / f"sub{s:02d}.nii", vol.astype(np.float32))
            Xs = vol.astype(np.float32).astype(float)
        elif variant == "int16_scaled":
            raw = np.round(vol * 100).astype(np.int16)
            R.save_nifti(d / f"sub{s:02d}.nii", raw, slope=0.01, inter=0.0)
            Xs = raw.astype(float) * np.float32(0.01)
        else:
            R.save_nifti(d / f"sub{s:02d}.nii", vol.astype(np.float64), extension_bytes=48)
            Xs = vol
        series.append(Xs.reshape(nvox, L, order="F")[inmask])
    return d, maskfile, thr, series


def corr(X, kind="pearson"):
    if kind == "spearman":
        X = np.apply_along_axis(stats.rankdata, 1, X)
    if kind == "kendall":
        n = X.shape[0]
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                t = stats.kendalltau(X[i], X[j]).statistic
                C[i, j] = C[j, i] = 0.0 if np.isnan(t) else t
        return C
    Xc = X - X.mean(1, keepdims=True)
    nrm = np.linalg.norm(Xc, axis=1)
    Z = np.divide(Xc, nrm[:, None], out=np.zeros_like(Xc), where=nrm[:, None] > 1e-9)
    C = Z @ Z.T
    np.fill_diagonal(C, 0)
    return C


def upper(C):
    iu = np.triu_indices(C.shape[0], 1)
    return iu, C[iu]


def check_r_network(path, C, t, weighted, tol=2e-5):
    A, w = R.read_csr(path)
    assert w == weighted
    A = A.toarray()
    assert np.allclose(A, A.T)
    iu, vals = upper(C)
    got = A[iu] > 0
    ref = vals > t
    ambiguous = np.abs(vals - t) < tol
    assert np.all(got[~ambiguous] == ref[~ambiguous])
    if weighted:
        np.testing.assert_allclose(A[iu][got], vals[got], atol=tol)


def check_s_network(path, C, s, weighted, tol=2e-5):
    A, w = R.read_csr(path)
    A = A.toarray()
    iu, vals = upper(C)
    m = vals.size
    c = int(np.floor(2 * m * s / 100 + 1e-9))
    c += c % 2
    k = c // 2
    got = A[iu] > 0
    assert got.sum() == k
    rk = np.sort(vals)[::-1][k - 1]
    sure = np.abs(vals - rk) >= tol
    assert np.all(got[sure] == (vals[sure] > rk))
    return rk


def find(dirp, pattern):
    files = sorted(dirp.glob(pattern))
    assert len(files) == 1, f"{pattern}: {files}"
    return files[0]


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_cucormat_individual_r_and_s(tmp_path, weighted):
    d, mask, thr, series = make_dataset(tmp_path)
    n = series[0].shape[0]
    run("CUCorMat", weighted, d, mask, thr, "n", "n", "r", 0.35, 0.5)
    run("CUCorMat", weighted, d, mask, thr, "n", "n", "s", 5, 12.5)
    out = d / ("weighted" if weighted else "unweighted")
    suf = "_weighted.csr" if weighted else ".csr"
    for s, X in enumerate(series):
        C = corr(X)
        assert np.all(C[0] == 0), "constant voxel must have zero correlation"
        for t in (0.35, 0.5):
            check_r_network(find(out, f"sub{s:02d}_{n}_spa*_cor{t:.3f}{suf}"), C, t, weighted)
        for sp in (5, 12.5):
            check_s_network(find(out, f"sub{s:02d}_{n}_spa{sp:.3f}%_cor*{suf}"), C, sp, weighted)


@pytest.mark.parametrize("avg", ["yn", "yf", "bn", "bf"])
def test_cucormat_group(tmp_path, avg):
    weighted = avg.endswith("f")   # exercise both variants
    d, mask, thr, series = make_dataset(tmp_path)
    n = series[0].shape[0]
    run("CUCorMat", weighted, d, mask, thr, avg, "y", "r", 0.3)
    Cs = [corr(X) for X in series]
    if avg[1] == "f":
        G = np.tanh(np.mean([np.arctanh(np.clip(C, -0.9999999, 0.9999999)) for C in Cs], axis=0))
    else:
        G = np.mean(Cs, axis=0)
    np.fill_diagonal(G, 0)
    out = d / ("weighted" if weighted else "unweighted")
    suf = "_weighted.csr" if weighted else ".csr"
    check_r_network(find(out, f"ave_3_{n}_spa*{suf}"), G, 0.3, weighted, tol=1e-4)
    # .cormat of the group matrix
    b = (d / f"ave_3_{n}.cormat").read_bytes()
    cnt = struct.unpack_from("<i", b, 0)[0]
    assert cnt == n * (n - 1) // 2
    packed = np.frombuffer(b, "<f4", cnt, 4)
    np.testing.assert_allclose(packed, upper(G)[1], atol=1e-4)
    individual = list(out.glob(f"sub*{suf}"))
    assert (len(individual) == 3) == avg.startswith("b")


def test_cucormat_save_individual_cormat(tmp_path):
    d, mask, thr, series = make_dataset(tmp_path, nsub=1)
    n = series[0].shape[0]
    run("CUCorMat", False, d, mask, thr, "n", "y", "s", 10)
    b = (d / f"sub00_{n}.cormat").read_bytes()
    packed = np.frombuffer(b, "<f4", struct.unpack_from("<i", b, 0)[0], 4)
    np.testing.assert_allclose(packed, upper(corr(series[0]))[1], atol=2e-5)
    check_s_network(find(d / "unweighted", f"sub00_{n}_spa10.000%*.csr"), corr(series[0]), 10, False)


@pytest.mark.parametrize("coef", ["p", "s", "ps"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_cucormat_hr(tmp_path, coef, weighted):
    d, mask, thr, series = make_dataset(tmp_path, nsub=2)
    n = series[0].shape[0]
    run("CUCorMatHr", weighted, d, mask, thr, "n", "s", coef, 8)
    run("CUCorMatHr", weighted, d, mask, thr, "n", "r", coef, 0.4)
    out = d / ("weighted" if weighted else "unweighted")
    suf = "_weighted.csr" if weighted else "_unweighted.csr"
    kinds = {"p": ["pearson"], "s": ["spearman"], "ps": ["pearson", "spearman"]}[coef]
    for s, X in enumerate(series):
        for kind in kinds:
            C = corr(X, kind)
            check_s_network(find(out, f"sub{s:02d}_{n}_{kind}_spa8.000%*{suf}"), C, 8, weighted)
            check_r_network(find(out, f"sub{s:02d}_{n}_{kind}_spa*_cor0.400{suf}"), C, 0.4, weighted)


def test_kendall(tmp_path):
    d, mask, thr, series = make_dataset(tmp_path, nsub=1, L=25)
    # make ties likely: quantise one subject's data
    n = series[0].shape[0]
    run("CuCorMat-sca-kendall", True, d, mask, thr, "n", "s", 10)
    C = corr(series[0], "kendall")
    check_s_network(find(d / "weightedKendall", f"sub00_{n}_kendall_spa10.000%*_weighted.csr"), C, 10, True, tol=1e-5)


def test_mask_in_data_dir_is_not_a_subject(tmp_path):
    d, mask, thr, series = make_dataset(tmp_path, nsub=1)
    import shutil
    shutil.copy(mask, d / "mymask.nii")
    cp = run("CUCorMat", False, d, d / "mymask.nii", thr, "n", "n", "r", 0.5)
    assert "1 image file(s)" in cp.stdout


def test_usage_errors(tmp_path):
    d, mask, thr, _ = make_dataset(tmp_path, nsub=1)
    assert run("CUCorMat", False, d, mask, thr, "y", "n", "r", 0.5, check=False).returncode == 2
    assert run("CUCorMat", False, d, mask, thr, "n", "n", "x", 0.5, check=False).returncode == 2
    assert run("CUCorMat", False, d, mask, thr, "n", "n", "r", 1.5, check=False).returncode == 2
    assert run("CUCorMatHr", False, d, mask, thr, "bn", "r", "p", 0.5, check=False).returncode == 2
    assert run("CUCorMatHr", False, d, mask, thr, "n", "r", "q", 0.5, check=False).returncode == 2


def test_dimension_mismatch_is_error(tmp_path):
    d, mask, thr, _ = make_dataset(tmp_path, nsub=1)
    R.save_nifti(tmp_path / "small.nii", np.ones((3, 3, 3), np.uint8))
    cp = run("CUCorMat", False, d, tmp_path / "small.nii", 0.5, "n", "n", "r", 0.5, check=False)
    assert cp.returncode == 1 and "different dimensions" in cp.stderr
