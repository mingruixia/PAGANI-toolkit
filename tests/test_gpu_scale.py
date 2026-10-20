"""Larger inputs that exercise the blocked / streamed GPU code paths.

The ordinary tests use ~150 voxels, which fit in a single GPU block. Here the
data are split into many blocks (PAGANI_BLOCK) and the "does not fit in GPU
memory" paths are forced with PAGANI_GPU_MEM_LIMIT_MB, so that every code path
used for whole-brain high-resolution data is compared with numpy/scipy.

Runs with PAGANI_TEST_GPU=1 (real GPU tools) or PAGANI_TEST_SLOW=1 (CPU
builds, to check the tests themselves). A few minutes in total.
"""
import os
import subprocess

import numpy as np
import pytest

import pagani_ref as R
from test_construct import GPU, check_r_network, check_s_network, corr, find, tool

SLOW = GPU or os.environ.get("PAGANI_TEST_SLOW") == "1"
pytestmark = pytest.mark.skipif(not SLOW, reason="set PAGANI_TEST_GPU=1 (or PAGANI_TEST_SLOW=1)")

# block size (multiple of 96) and memory limit that force many blocks / streaming
MODES = {
    "blocks": {"PAGANI_BLOCK": "960"},
    "streamed": {"PAGANI_BLOCK": "960", "PAGANI_GPU_MEM_LIMIT_MB": "300"},
}


def run(name, weighted, *args, env=None):
    exe = tool(name, weighted)
    if not os.path.exists(exe):           # CPU-only tools live in the package
        exe = R.exe(name, weighted)
    cp = subprocess.run([exe, *map(str, args)], capture_output=True, text=True,
                        env={**os.environ, "PAGANI_SEED": "11", **(env or {})})
    if cp.returncode != 0:
        raise AssertionError(f"{name} failed ({cp.returncode}):\n{cp.stdout}\n{cp.stderr}")
    return cp


def big_dataset(tmp_path, nsub=1, shape=(20, 18, 12), L=48, seed=1, continuous=False):
    """4D NIfTI files with modular structure; the mask keeps ~4000 voxels."""
    rng = np.random.default_rng(seed)
    d = tmp_path / "data"
    d.mkdir()
    nvox = int(np.prod(shape))
    prob = rng.random(shape)
    mask = tmp_path / "mask.nii"
    R.save_nifti(mask, prob.astype(np.float32))
    thr = 0.05
    inmask = prob.astype(np.float32).astype(float).ravel(order="F") >= thr
    series = []
    for s in range(nsub):
        base = rng.standard_normal((8, L))
        member = rng.integers(0, 8, nvox)
        X = base[member] * rng.uniform(0.3, 1.5, (nvox, 1)) + rng.standard_normal((nvox, L)) + 50
        if not continuous:
            X = np.round(X, 1)                     # ties for Spearman
        X = X.astype(np.float32)
        R.save_nifti(d / f"s{s}.nii", X.reshape(shape + (L,), order="F"))
        series.append(X.astype(float)[inmask])
    return d, mask, thr, series


def kendall_tau_b(X):
    """Vectorised Kendall tau-b (same as scipy.stats.kendalltau)."""
    L = X.shape[1]
    a, b = np.triu_indices(L, 1)
    S = np.sign(X[:, a] - X[:, b])
    nz = np.count_nonzero(S, axis=1).astype(float)
    den = np.sqrt(np.outer(nz, nz))
    C = np.divide(S @ S.T, den, out=np.zeros((X.shape[0],) * 2), where=den > 0)
    np.fill_diagonal(C, 0)
    return C


@pytest.mark.parametrize("mode", MODES)
def test_cormat_blocks(tmp_path, mode):
    d, mask, thr, series = big_dataset(tmp_path, nsub=2)
    n = series[0].shape[0]
    assert n > 3 * 960
    env = MODES[mode]
    run("CUCorMat", True, d, mask, thr, "bf", "n", "r", 0.45, env=env)
    run("CUCorMat", False, d, mask, thr, "n", "n", "s", 1.5, env=env)
    Cs = [corr(X) for X in series]
    for s, C in enumerate(Cs):
        check_r_network(find(d / "weighted", f"s{s}_{n}_spa*_cor0.450_weighted.csr"), C, 0.45, True)
        check_s_network(find(d / "unweighted", f"s{s}_{n}_spa1.500%*.csr"), C, 1.5, False)
    G = np.tanh(np.mean([np.arctanh(np.clip(C, -0.9999999, 0.9999999)) for C in Cs], axis=0))
    np.fill_diagonal(G, 0)
    check_r_network(find(d / "weighted", f"ave_2_{n}_spa*_weighted.csr"), G, 0.45, True, tol=1e-4)


@pytest.mark.parametrize("mode", MODES)
def test_cormat_hr_blocks(tmp_path, mode):
    d, mask, thr, series = big_dataset(tmp_path)
    n = series[0].shape[0]
    run("CUCorMatHr", True, d, mask, thr, "n", "s", "ps", 2, env=MODES[mode])
    for kind in ("pearson", "spearman"):
        C = corr(series[0], kind)
        check_s_network(find(d / "weighted", f"s0_{n}_{kind}_spa2.000%*_weighted.csr"), C, 2, True)


@pytest.mark.parametrize("mode", MODES)
def test_kendall_blocks(tmp_path, mode):
    # two subjects: the second must not reuse device data of the first (streaming cache)
    d, mask, thr, series = big_dataset(tmp_path, nsub=2, shape=(16, 14, 12), L=30, continuous=True)
    n = series[0].shape[0]
    run("CuCorMat-sca-kendall", True, d, mask, thr, "n", "r", 0.3, env=MODES[mode])
    for s, X in enumerate(series):
        C = kendall_tau_b(X)
        check_r_network(find(d / "weightedKendall", f"s{s}_{n}_kendall_spa*_cor0.300_weighted.csr"), C, 0.3, True,
                        tol=1e-5)


def lp_tool():
    return "CUBFW_Lp" if GPU else "BFS_MulCPU"


def write_net(tmp_path, A, weighted):
    d = tmp_path / "nets"
    d.mkdir(exist_ok=True)
    f = d / ("net_weighted.csr" if weighted else "net.csr")
    R.write_csr(f, A, weighted)
    return d, str(f)[:-4]


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("streamed", [False, True], ids=["resident", "streamed"])
def test_path_length_large(tmp_path, weighted, streamed):
    # n is not a multiple of the tile size (weighted: Floyd-Warshall) nor of
    # the 512 sources of a GPU BFS batch (unweighted)
    A = R.modular_graph(n=3999, modules=5, p_in=0.04, p_out=0.004, seed=21, weighted=weighted)
    d, stem = write_net(tmp_path, A, weighted)
    env = {"PAGANI_GPU_MEM_LIMIT_MB": "300"} if streamed else {}
    cp = run(lp_tool(), weighted, d, 0, "gn", env=env)
    if GPU and streamed and weighted:
        assert "streaming" in cp.stdout
    lp, nodal = R.efficiency(A, weighted)
    np.testing.assert_allclose(R.read_nm(stem + "_eff.nm"), nodal, rtol=1e-4, atol=1e-6)
    assert R.read_first_number(stem + "_Lp.txt") == pytest.approx(lp, rel=1e-4)


def test_path_length_sparse_unweighted(tmp_path):
    # density < 1 % (PAGANI 2.0 and 2.1.0 used the CPU here), several BFS
    # batches, isolated nodes and a separate component
    A = R.modular_graph(n=2100, modules=6, p_in=0.012, p_out=0.0004, seed=24, isolated=5, extra_component=True)
    d, stem = write_net(tmp_path, A, False)
    run(lp_tool(), False, d, 0, "gn")
    lp, nodal = R.efficiency(A, False)
    np.testing.assert_allclose(R.read_nm(stem + "_eff.nm"), nodal, rtol=1e-5, atol=1e-7)
    assert R.read_first_number(stem + "_Lp.txt") == pytest.approx(lp, rel=1e-5)


def test_betweenness_large(tmp_path):
    A = R.modular_graph(n=1200, modules=4, p_in=0.03, p_out=0.004, seed=22)
    d, stem = write_net(tmp_path, A, False)
    run("CUBC", False, d)
    np.testing.assert_allclose(R.read_nm(stem + "_bc.nm"), R.betweenness(A), rtol=1e-4, atol=1e-2)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_eigenvector_large(tmp_path, weighted):
    A = R.modular_graph(n=5000, modules=5, p_in=0.01, p_out=0.002, seed=23, weighted=weighted,
                        isolated=0, extra_component=False)
    d, stem = write_net(tmp_path, A, weighted)
    run("CUEC", weighted, d)
    lam, v = R.eigenvector(A)
    np.testing.assert_allclose(R.read_nm(stem + "_ec.nm"), v, atol=2e-4)
    assert R.read_first_number(stem + "_evalue.txt") == pytest.approx(lam, rel=1e-4)
