"""Weighted Brandes CPU and BCT validation using PAGANI CSR and .nm files."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import pagani_ref as ref


REFERENCE = Path(__file__).parent / "fixtures" / "bct_weighted_betweenness.json"
CASES = json.loads(REFERENCE.read_text(encoding="utf-8"))["cases"]


def run_weighted(directory, matrix, threads=4):
    ref.write_csr(directory / "network_weighted.csr", matrix, True)
    ref.run("CUBC", True, directory, env={"PAGANI_THREADS": str(threads)})
    return ref.read_nm(directory / "network_weighted_bc.nm")


def reference_matrix(case):
    matrix = np.zeros((case["nodes"], case["nodes"]), dtype=np.float32)
    for start, end, weight in case["edges"]:
        start, end = int(start), int(end)
        matrix[start, end] = matrix[end, start] = weight
    return matrix


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_matlab_bct_reference(tmp_path, case):
    actual = run_weighted(tmp_path, reference_matrix(case))
    np.testing.assert_allclose(actual, case["expected"], rtol=1e-6, atol=1e-8)


def test_hand_calculation_ties_and_disconnected_components(tmp_path):
    case = next(case for case in CASES if case["name"] == "disconnected_diamond_chain_isolate")
    # Each diamond vertex contributes 1/2 for each ordering of the opposite
    # pair. The chain middle contributes 1 for each ordering of its endpoints.
    expected = [1, 1, 1, 1, 0, 2, 0, 0]
    np.testing.assert_array_equal(run_weighted(tmp_path, reference_matrix(case)), expected)
    np.testing.assert_array_equal(run_weighted(tmp_path, reference_matrix(case), threads=1), expected)


def test_equal_direct_path_counts_as_one_of_two_paths(tmp_path):
    case = next(case for case in CASES if case["name"] == "equal_direct_and_multihop")
    # The direct length-2 edge and two length-1 edges split the ordered pair.
    np.testing.assert_array_equal(run_weighted(tmp_path, reference_matrix(case)), [0, 1, 0])


@pytest.mark.parametrize("length,tied", [(65536, True), (32768, False)])
def test_floating_point_tolerance_boundary(tmp_path, length, tied):
    # One edge differs by only one float32 ULP. BCT compares its double
    # distances exactly, while PAGANI ties only within the documented threshold.
    weaker = np.nextafter(np.float32(1), np.float32(0))
    graph = np.zeros((4, 4), dtype=np.float32)
    for start, end, weight in [(0, 1, 1/length), (0, 2, 1/length), (1, 3, 1), (2, 3, weaker)]:
        graph[start, end] = graph[end, start] = weight
    shortest, competitor = length + 1.0, length + 1.0 / float(weaker)
    assert ((competitor - shortest) <= 1e-12 * competitor) == tied
    expected = [0, 1, 1, 2] if tied else [0, 2, 0, 2]
    np.testing.assert_array_equal(run_weighted(tmp_path, graph), expected)


def test_against_networkx_weighted_graph(tmp_path):
    matrix = ref.modular_graph(n=64, seed=29, weighted=True).astype(np.float32)
    graph = nx.from_numpy_array(matrix)
    for _, _, attrs in graph.edges(data=True):
        attrs["length"] = 1 / float(attrs["weight"])
    reference = nx.betweenness_centrality(graph, normalized=False, weight="length")
    expected = [2 * reference[node] for node in graph]
    one_thread = run_weighted(tmp_path, matrix, threads=1)
    np.testing.assert_allclose(one_thread, expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(run_weighted(tmp_path, matrix), one_thread, rtol=1e-7, atol=1e-8)


def test_large_path_multiplicity_remains_finite(tmp_path):
    # Two nodes per layer with all four edges between adjacent layers. The
    # number of shortest paths between endpoints exceeds the range of float64.
    layers = 1027
    start = np.repeat(np.arange(2 * (layers - 1)), 2)
    end = np.repeat(2 * (np.arange(layers - 1) + 1), 4) + np.tile([0, 1, 0, 1], layers - 1)
    row, col = np.r_[start, end], np.r_[end, start]
    matrix = csr_matrix((np.ones(row.size), (row, col)), shape=(2 * layers, 2 * layers))
    actual = run_weighted(tmp_path, matrix)
    assert np.isfinite(actual).all()
    middle = layers // 2
    # Endpoints on opposite sides contribute 4*j*(L-1-j). The two pairs of
    # vertices in the immediately adjacent layers add one in total.
    peak = 4 * middle * (layers - 1 - middle) + 1
    np.testing.assert_allclose(actual[2 * middle:2 * middle + 2], [peak, peak], rtol=1e-7)


@pytest.mark.parametrize(
    "matrix,weighted,message",
    [
        ([[0, 1], [1, 0]], False, "a weighted network is required"),
        ([[0, -1], [-1, 0]], True, "positive finite edge weights"),
        ([[0, np.inf], [np.inf, 0]], True, "positive finite edge weights"),
        ([[0, np.nan], [np.nan, 0]], True, "positive finite edge weights"),
        ([[0, 1], [0, 0]], True, "not symmetric"),
        ([[0, 1], [0.5, 0]], True, "weights are not symmetric"),
        ([[0, 1e-30, 0], [1e-30, 0, 1], [0, 1, 0]], True, "exceed the precision"),
    ],
)
def test_weighted_input_failures(tmp_path, matrix, weighted, message):
    ref.write_csr(tmp_path / "network.csr", matrix, weighted)
    result = ref.run("CUBC", True, tmp_path, check=False, env={"PAGANI_THREADS": "2"})
    assert result.returncode == 1
    assert message in result.stderr
    assert not (tmp_path / "network_bc.nm").exists()
