"""Optional pytest plugin for paired weighted-BC validation on a CUDA machine.

Run from tests with PAGANI_BIN pointing to a CUDA build's package directory:
python -m pytest -p weighted_bc_parity test_weighted_betweenness.py --bc-report report.json
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pytest

import pagani_ref as ref


def pytest_addoption(parser):
    parser.addoption("--bc-report", required=True, help="output JSON with per-command measurements")


def pytest_configure(config):
    config.bc_results = {}


@pytest.fixture(autouse=True)
def compare_backends(request, monkeypatch, tmp_path):
    run = ref.run
    measurements = []
    request.config.bc_results[request.node.nodeid] = {"runs": measurements}

    def paired(tool, weighted, directory, check=True, env=None):
        assert tool == "CUBC" and weighted and directory == tmp_path
        source, = directory.glob("*.csr")
        output = source.with_name(source.stem + "_bc.nm")
        cpu_values = None
        record = {}
        measurements.append(record)
        for backend, executable in (("cpu", "CUBC"), ("gpu", "CUBC_GPU")):
            output.unlink(missing_ok=True)
            start = time.perf_counter()
            process = run(executable, True, directory, check=False, env=env)
            record[backend + "_wall_s"] = time.perf_counter() - start
            record[backend + "_returncode"] = process.returncode
            if backend == "gpu":
                peak = re.search(r"Weighted BC CUDA buffer peak: (\d+) bytes", process.stdout)
                record["gpu_buffer_peak_bytes"] = int(peak[1]) if peak else 0
                record["device"] = next((s for s in process.stdout.splitlines() if s.startswith("GPU ")), None)
            if process.returncode:
                assert not output.exists()
                record[backend + "_error"] = process.stderr.strip()
            else:
                values = ref.read_nm(output)
                if backend == "cpu":
                    cpu_values = values
                else:
                    assert cpu_values is not None
                    if ref.read_csr(source)[0].nnz:
                        assert record["device"] and record["gpu_buffer_peak_bytes"] > 0
                    np.testing.assert_allclose(values, cpu_values, rtol=1e-6, atol=1e-8)
                    record["max_abs_difference"] = float(np.max(np.abs(values - cpu_values), initial=0))
                    if len(values) <= 8:
                        record["nodal_values"] = values.tolist()
        assert record["cpu_returncode"] == record["gpu_returncode"]
        if "cpu_error" in record:
            assert record["cpu_error"].partition("ERROR: ")[2] == record["gpu_error"].partition("ERROR: ")[2]
        if check:
            assert process.returncode == 0, process.stderr
        return process

    monkeypatch.setattr(ref, "run", paired)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    if report.when == "call":
        item.config.bc_results[item.nodeid]["outcome"] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    path = Path(session.config.getoption("--bc-report"))
    path.write_text(json.dumps(session.config.bc_results, indent=2) + "\n", encoding="utf-8")
