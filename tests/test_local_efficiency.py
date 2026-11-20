"""Validate the LocalEfficiency CLI against NetworkX and MATLAB BCT."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy.sparse.csgraph import floyd_warshall

import pagani_ref as R


def write_network(tmp_path, matrix, weighted=False):
    path = tmp_path / "network.csr"
    R.write_csr(path, matrix, weighted)
    return path.with_suffix("")


@pytest.mark.parametrize("weighted", [False, True])
def test_induced_neighbourhoods(tmp_path, weighted):
    # Node 0 sees an induced three-node path and an isolated neighbour.
    # The endpoints are two hops apart within the neighbourhood.
    graph = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3)])
    graph.add_node(5)
    matrix = nx.to_numpy_array(graph, nodelist=range(6))
    stem = write_network(tmp_path, matrix * (0.4 if weighted else 1), weighted)
    R.run("LocalEfficiency", False, tmp_path, 0, "gn")
    expected = [5/12, 1, 5/6, 1, 0, 0]
    np.testing.assert_allclose(R.read_nm(str(stem) + "_leff.nm"), expected, atol=1e-7)
    assert R.read_first_number(str(stem) + "_leff.txt") == pytest.approx(
        nx.local_efficiency(graph), rel=1e-5
    )


@pytest.mark.parametrize("matrix", [np.zeros((1, 1)), np.zeros((3, 3)), np.ones((4, 4)) - np.eye(4)])
def test_empty_neighbourhoods_and_cliques(tmp_path, matrix):
    graph = nx.from_numpy_array(matrix)
    stem = write_network(tmp_path, matrix)
    R.run("LocalEfficiency", False, tmp_path, 0, "gn")
    assert R.read_first_number(str(stem) + "_leff.txt") == nx.local_efficiency(graph)
    expected = [nx.global_efficiency(graph.subgraph(graph[node])) for node in graph]
    np.testing.assert_array_equal(R.read_nm(str(stem) + "_leff.nm"), expected)


def test_all_nodes_against_networkx(tmp_path):
    matrix = R.modular_graph(n=100, seed=12)
    graph = nx.from_numpy_array(matrix)
    stem = write_network(tmp_path, matrix)
    R.run("LocalEfficiency", False, tmp_path, 0, "gn")
    expected = [nx.global_efficiency(graph.subgraph(graph[node])) for node in graph]
    np.testing.assert_allclose(R.read_nm(str(stem) + "_leff.nm"), expected, rtol=1e-6, atol=1e-8)
    assert R.read_first_number(str(stem) + "_leff.txt") == pytest.approx(
        nx.local_efficiency(graph), rel=1e-5
    )


def test_random_normalization_and_thread_count(tmp_path):
    stem = write_network(tmp_path, R.modular_graph(n=100, seed=13))
    table = stem.with_name("network_leff.txt")
    normalized = stem.with_name("network_leff_norm.txt")
    R.run("LocalEfficiency", False, tmp_path, 4, "gr", env={"PAGANI_SEED": "73", "PAGANI_THREADS": "1"})
    original = table.read_bytes(), normalized.read_bytes()
    values = np.loadtxt(table)
    assert values.size == 5 and values[1:].mean() > 0
    assert R.read_first_number(normalized) == pytest.approx(values[0] / values[1:].mean(), rel=1e-5)
    R.run("LocalEfficiency", False, tmp_path, 4, "gr", env={"PAGANI_SEED": "73", "PAGANI_THREADS": "4"})
    assert (table.read_bytes(), normalized.read_bytes()) == original
    assert not stem.with_name("network_leff.nm").exists()


def test_random_zero_mean_and_flags(tmp_path):
    stem = write_network(tmp_path, np.zeros((3, 3)))
    failure = R.run("LocalEfficiency", False, tmp_path, 0, "r", check=False)
    assert failure.returncode == 2 and "requires at least one" in failure.stderr
    assert not list(tmp_path.glob("*_leff*"))
    invalid = R.run("LocalEfficiency", False, tmp_path, 1, "z", check=False)
    assert invalid.returncode == 2
    result = R.run("LocalEfficiency", False, tmp_path, 1, "r")
    assert "undefined" in result.stdout
    assert np.isnan(R.read_first_number(stem.with_name("network_leff_norm.txt")))
    assert not stem.with_name("network_leff.txt").exists()


def test_asymmetric_graph_rejected(tmp_path):
    write_network(tmp_path, np.array([[0, 1], [0, 0]]))
    failure = R.run("LocalEfficiency", False, tmp_path, 0, "gn", check=False)
    assert failure.returncode == 1 and "not symmetric" in failure.stderr
    assert not list(tmp_path.glob("*_leff*"))


BCT_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "bct_weighted_local_efficiency.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", BCT_CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("scale", ["raw", "normalized"])
def test_weighted_against_matlab_bct(tmp_path, case, scale):
    matrix = np.zeros((case["nodes"], case["nodes"]))
    for start, end, weight in case["edges"]:
        start, end = int(start), int(end)
        matrix[start, end] = matrix[end, start] = weight
    stem = write_network(tmp_path, matrix, True)
    options = ["--normalize-weights"] if scale == "normalized" else []
    process = R.run("LocalEfficiency", True, tmp_path, 0, "gn", *options)
    assert "BCT local=2" in process.stdout
    nodal = R.read_nm(str(stem) + "_leff.nm")
    np.testing.assert_allclose(nodal, case[scale], rtol=1e-6, atol=1e-8)
    assert R.read_first_number(str(stem) + "_leff.txt") == pytest.approx(case[scale + "_mean"], rel=1e-5)


def weighted_reference(matrix):
    """Independent dense all-pairs reference for small symmetric graphs."""
    roots = np.cbrt(matrix)
    length = np.divide(1.0, roots, out=np.zeros_like(roots), where=roots > 0)
    values = np.zeros(len(matrix))
    for center, weights in enumerate(roots):
        neighbours = np.flatnonzero(weights)
        count = len(neighbours)
        if count < 2:
            continue
        distances = floyd_warshall(length[np.ix_(neighbours, neighbours)], directed=False)
        inverse = np.divide(1.0, distances, out=np.zeros_like(distances), where=distances > 0)
        values[center] = (np.outer(weights[neighbours], weights[neighbours]) * inverse).sum() / (
            count * (count - 1)
        )
    return values


def test_weighted_floyd_warshall_and_random_networks(tmp_path):
    matrix = R.modular_graph(n=80, seed=31, weighted=True).astype(np.float32)
    stem = write_network(tmp_path, matrix, True)
    env = {"PAGANI_SEED": "73", "PAGANI_THREADS": "1"}
    R.run("LocalEfficiency", True, tmp_path, 4, "gnr", env=env)
    nodal_path = stem.with_name("network_leff.nm")
    table_path = stem.with_name("network_leff.txt")
    norm_path = stem.with_name("network_leff_norm.txt")
    expected = weighted_reference(matrix.astype(np.float64))
    raw_nodal = R.read_nm(nodal_path)
    np.testing.assert_allclose(raw_nodal, expected, rtol=1e-6, atol=1e-8)
    raw_values = np.loadtxt(table_path)
    ratio = R.read_first_number(norm_path)
    assert raw_values[0] == pytest.approx(expected.mean(), rel=1e-5)
    assert raw_values.size == 5 and raw_values[1:].mean() > 0
    assert ratio == pytest.approx(raw_values[0] / raw_values[1:].mean(), rel=1e-5)
    before = tuple(path.read_bytes() for path in (nodal_path, table_path, norm_path))
    env["PAGANI_THREADS"] = "4"
    R.run("LocalEfficiency", True, tmp_path, 4, "gnr", env=env)
    assert tuple(path.read_bytes() for path in (nodal_path, table_path, norm_path)) == before
    R.run("LocalEfficiency", True, tmp_path, 4, "gnr", "--normalize-weights", env=env)
    scale = float(matrix.max())
    np.testing.assert_allclose(R.read_nm(nodal_path), raw_nodal / scale, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.loadtxt(table_path), raw_values / scale, rtol=1e-5)
    assert R.read_first_number(norm_path) == pytest.approx(ratio, rel=1e-5)


def test_weighted_unit_weights_match_binary(tmp_path):
    matrix = R.modular_graph(n=70, seed=50)
    stem = write_network(tmp_path, matrix, True)
    R.run("LocalEfficiency", False, tmp_path, 0, "gn")
    nodal = R.read_nm(str(stem) + "_leff.nm")
    mean = R.read_first_number(str(stem) + "_leff.txt")
    R.run("LocalEfficiency", True, tmp_path, 0, "gn")
    np.testing.assert_allclose(R.read_nm(str(stem) + "_leff.nm"), nodal, atol=1e-7)
    assert R.read_first_number(str(stem) + "_leff.txt") == mean


def test_weighted_empty_graph_normalization(tmp_path):
    stem = write_network(tmp_path, np.zeros((3, 3)), True)
    process = R.run("LocalEfficiency", True, tmp_path, 1, "gnr", "--normalize-weights")
    np.testing.assert_array_equal(R.read_nm(str(stem) + "_leff.nm"), [0.0, 0.0, 0.0])
    assert "undefined" in process.stdout
    assert np.isnan(R.read_first_number(str(stem) + "_leff_norm.txt"))


@pytest.mark.parametrize("weight", [-1.0, np.inf, np.nan])
def test_weighted_rejects_invalid_weights(tmp_path, weight):
    write_network(tmp_path, [[0, weight], [weight, 0]], True)
    process = R.run("LocalEfficiency", True, tmp_path, 0, "gn", check=False)
    assert process.returncode == 1
    assert "require positive weights" in process.stderr
    assert not list(tmp_path.glob("*_leff*"))


def test_weighted_rejects_asymmetric_weights(tmp_path):
    write_network(tmp_path, [[0, 0.25], [0.5, 0]], True)
    process = R.run("LocalEfficiency", True, tmp_path, 0, "gn", check=False)
    assert process.returncode == 1
    assert "edge weights are not symmetric" in process.stderr
    assert not list(tmp_path.glob("*_leff*"))


def test_weighted_requires_weights_and_rejects_unknown_option(tmp_path):
    write_network(tmp_path, [[0, 1], [1, 0]])
    for weighted, extra, code, error in [
        (True, [], 1, "a weighted network is required"),
        (True, ["--wrong-option"], 2, "--normalize-weights"),
        (False, ["--normalize-weights"], 2, "usage"),
    ]:
        process = R.run("LocalEfficiency", weighted, tmp_path, 0, "gn", *extra, check=False)
        assert process.returncode == code
        if error == "usage":
            assert "LocalEfficiency dir_for_csr" in process.stderr
        else:
            assert error in process.stderr
    assert not list(tmp_path.glob("*_leff*"))
