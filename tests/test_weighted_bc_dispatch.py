"""Weighted BC dispatch integration on a CUDA-equipped machine."""
import os

import numpy as np
import pytest

import pagani_ref as ref

pytestmark = pytest.mark.skipif(os.getenv("PAGANI_TEST_GPU") != "1", reason="needs a CUDA package")


def test_calibrate_and_reuse(tmp_path):
    matrix = np.ones((4, 4)) - np.eye(4)
    ref.write_csr(tmp_path / "a.csr", matrix, True)
    args = ["--threads", "2"]
    env = {"PAGANI_BC_CACHE_DIR": str(tmp_path / "cache")}
    log = ref.run("CUBC_GPU", True, tmp_path, "--backend", "calibrate", *args, env=env).stdout
    assert "backend=calibrate" in log
    before = (tmp_path / "a_bc.nm").read_bytes()
    log = ref.run("CUBC_GPU", True, tmp_path, "--backend", "auto", *args, env=env).stdout
    assert "reason=local calibration" in log
    assert log.count("calculating betweenness for") == 1
    assert (tmp_path / "a_bc.nm").read_bytes() == before
    assert "reason=local calibration" not in ref.run(
        "CUBC_GPU", True, tmp_path, "--backend", "auto", "--threads", "1", env=env).stdout


def test_auto_fallback_and_strict_gpu(tmp_path):
    ref.write_csr(tmp_path / "a.csr", [[0, 1], [1, 0]], True)
    env = {"PAGANI_GPU": "999"}
    log = ref.run("CUBC_GPU", True, tmp_path, "--backend", "auto", env=env).stdout
    assert "backend=cpu reason=PAGANI_GPU=999" in log
    output = tmp_path / "a_bc.nm"
    before = output.read_bytes()
    forced = ref.run("CUBC_GPU", True, tmp_path, "--backend", "gpu", check=False, env=env)
    assert forced.returncode == 1 and "PAGANI_GPU=999" in forced.stderr
    ref.write_csr(tmp_path / "a.csr", [[0, -1], [-1, 0]], True)
    bad = ref.run("CUBC_GPU", True, tmp_path, "--backend", "auto", check=False)
    assert bad.returncode == 1 and "positive finite" in bad.stderr
    assert output.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp.*"))
