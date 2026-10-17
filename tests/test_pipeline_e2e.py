"""End-to-end test: the GUI generates a script, every command is executed and
all expected outputs are produced.

On a machine without GPU the GPU-only tools are substituted by CPU builds
(CUBFW_Lp -> BFS_MulCPU, CUmodularity -> Newman_Modularity_CPU, CUCorMat /
CUBC / CUEC -> CPU-backend builds from build/test-bin).
"""
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

import pagani_ref as R
from test_construct import GPU, TESTBIN, make_dataset

ROOT = Path(__file__).resolve().parents[1]
PIPE = Path(os.environ.get("PAGANI_GUI_TEST", ROOT / "build" / "src" / "gui" / "test_gui_pipeline"))
EXE = ".exe" if os.name == "nt" else ""


def fake_app(tmp_path):
    if GPU:
        return R.bin_dir()
    app = tmp_path / "app"
    for sub in ("exefiles", "exefiles_weighted"):
        d = app / sub
        d.mkdir(parents=True)
        for f in (R.bin_dir() / sub).iterdir():
            shutil.copy(f, d / f.name)
        for f in (TESTBIN / sub).iterdir():
            shutil.copy(f, d / f.name)
        shutil.copy(d / f"BFS_MulCPU{EXE}", d / f"CUBFW_Lp{EXE}")
        shutil.copy(d / f"Newman_Modularity_CPU{EXE}", d / (("CUModularity" if "weighted" in sub else "CUmodularity") + EXE))
    return app


def script_commands(script):
    cmds = []
    for line in script.splitlines():
        s = line.strip()
        low = s.lower()
        if not s or low.startswith(("@", "rem", "echo", "if errorlevel", "goto", ":", "exit")):
            continue
        cmds.append(shlex.split(s.replace("\\", "/") if os.name != "nt" else s, posix=True))
    return cmds


@pytest.mark.skipif(not PIPE.exists() and not PIPE.with_suffix(".exe").exists(), reason="GUI not built")
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_gui_script_runs(tmp_path, weighted):
    app = fake_app(tmp_path)
    d, mask, thr, series = make_dataset(tmp_path, nsub=2)
    settings = {
        "workingDir": str(d), "maskFile": str(mask), "maskThreshold": str(thr), "weighted": weighted,
        "construct": True, "highResolution": False, "sparsityThreshold": True, "thresholds": "10 20",
        "groupNetwork": True, "fisherAverage": True, "saveCorMatrix": False,
        "globalLp": True, "globalCp": True, "gamma": True, "lambda": True, "sigma": True,
        "modularity": True, "modularityAlgorithm": 0, "cpType": 2, "randomNetworks": "3",
        "nodalEfficiency": True, "betweenness": not weighted, "nodalCp": True, "degree": True,
        "eigenvector": True, "participation": True,
    }
    sf = tmp_path / "settings.json"
    sf.write_text(json.dumps(settings))
    cp = subprocess.run([str(PIPE), "--script", str(sf), str(app)], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    env = {**os.environ, "PAGANI_SEED": "3"}
    for cmd in script_commands(cp.stdout):
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert r.returncode == 0, f"{cmd}\n{r.stdout}\n{r.stderr}"
    net = d / ("weighted" if weighted else "unweighted")
    csrs = sorted(net.glob("*.csr"))
    assert len(csrs) == 2 * 2 + 2  # two subjects + group network, two sparsities
    for c in csrs:
        stem = str(c)[:-4]
        for suffix in ("_Lp.txt", "_lambda.txt", "_cp.txt", "_gamma.txt", "_smallWorld.txt", "_modu.txt",
                       "_eff.nii", "_cp.nii", "_deg.nii", "_ec.nii", "_pc.nii", "_pc_normalized.nii", "_modu.nii"):
            assert Path(stem + suffix).exists(), stem + suffix
        if not weighted:
            assert Path(stem + "_bc.nii").exists()
