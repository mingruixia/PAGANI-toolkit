"""Betweenness, eigenvector centrality and the GPU-only tools.

Without PAGANI_TEST_GPU=1 the CPU-backend builds in build/test-bin are tested
(CUBC, CUEC); with PAGANI_TEST_GPU=1 the real GPU executables of the package
(PAGANI_BIN) are tested, including CUBFW_Lp and CUmodularity.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import pagani_ref as R
from test_construct import GPU, tool

gpu_only = pytest.mark.skipif(not GPU, reason="requires an NVIDIA GPU (set PAGANI_TEST_GPU=1)")


def run(name, weighted, *args):
    cp = subprocess.run([tool(name, weighted), *map(str, args)], capture_output=True, text=True,
                        env={**os.environ, "PAGANI_SEED": "7"})
    if cp.returncode != 0:
        raise AssertionError(f"{name} failed ({cp.returncode}):\n{cp.stdout}\n{cp.stderr}")
    return cp


def write(tmp_path, A, weighted, name="net"):
    d = tmp_path / "nets"
    d.mkdir(exist_ok=True)
    f = d / (name + ("_weighted.csr" if weighted else ".csr"))
    R.write_csr(f, A, weighted)
    return d, str(f)[:-4]


def test_betweenness(tmp_path):
    A = R.modular_graph(n=150, seed=5)
    d, stem = write(tmp_path, A, False)
    run("CUBC", False, d)
    np.testing.assert_allclose(R.read_nm(stem + "_bc.nm"), R.betweenness(A), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_eigenvector(tmp_path, weighted):
    A = R.modular_graph(n=200, seed=6, weighted=weighted, isolated=0, extra_component=False)
    d, stem = write(tmp_path, A, weighted)
    run("CUEC", weighted, d)
    lam, v = R.eigenvector(A)
    got = R.read_nm(stem + "_ec.nm")
    np.testing.assert_allclose(got, v, atol=2e-4)
    assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-4)
    assert R.read_first_number(stem + "_evalue.txt") == pytest.approx(lam, rel=1e-4)


@gpu_only
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("n", [150, 700])
def test_gpu_path_length(tmp_path, weighted, n):
    A = R.modular_graph(n=n, seed=8, weighted=weighted, p_in=0.2, p_out=0.02)
    d, stem = write(tmp_path, A, weighted)
    run("CUBFW_Lp", weighted, d, 2, "gnl")
    lp, nodal = R.efficiency(A, weighted)
    np.testing.assert_allclose(R.read_nm(stem + "_eff.nm"), nodal, rtol=1e-4, atol=1e-6)
    assert R.read_first_number(stem + "_Lp.txt") == pytest.approx(lp, rel=1e-4)


@gpu_only
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_gpu_newman(tmp_path, weighted):
    A = R.modular_graph(n=6000, modules=6, p_in=0.02, p_out=0.001, seed=9, weighted=weighted)
    d, stem = write(tmp_path, A, weighted)
    run("CUmodularity" if not weighted else "CUModularity", weighted, d)
    lab = R.read_nm(stem + "_modu.nm").astype(int)
    q = R.modularity(A, lab)
    # the GPU tool must reproduce the CPU implementation of the same algorithm
    import shutil
    d2 = tmp_path / "cpu"
    d2.mkdir()
    shutil.copy(stem + ".csr", d2)
    subprocess.run([R.exe("Newman_Modularity_CPU", weighted), str(d2)], check=True, capture_output=True)
    q_cpu = R.modularity(A, R.read_nm(str(d2 / Path(stem).name) + "_modu.nm").astype(int))
    assert q == pytest.approx(q_cpu, abs=0.01)
    # leading-eigenvector splitting is greedy; on this symmetric planted graph
    # (6 equal modules, planted Q = 0.633) it reaches about 0.6
    assert q > 0.57
