"""Validation of the CPU command-line tools against reference implementations."""
import shutil

import numpy as np
import pytest

import pagani_ref as R


def make_dir(tmp_path, A, weighted, name="sub01_spa5.000%_cor0.300"):
    d = tmp_path / ("weighted" if weighted else "unweighted")
    d.mkdir(parents=True, exist_ok=True)
    suffix = "_weighted.csr" if weighted else ".csr"
    R.write_csr(d / (name + suffix), A, weighted)
    return d, d / (name + suffix)


@pytest.fixture(params=[False, True], ids=["unweighted", "weighted"])
def weighted(request):
    return request.param


@pytest.fixture
def graph(weighted):
    return R.modular_graph(weighted=weighted, seed=1)


def stem(p):
    return str(p)[: -len(".csr")]


# --------------------------------------------------------------------------
def test_degree(tmp_path, graph, weighted):
    d, f = make_dir(tmp_path, graph, weighted)
    R.run("Degree", weighted, d)
    got = R.read_nm(stem(f) + "_deg.nm")
    ref = R.degree(graph if weighted else (graph > 0))
    np.testing.assert_allclose(got, ref, rtol=1e-5)


def test_unweighted_tools_ignore_weights(tmp_path):
    A = R.modular_graph(weighted=True, seed=2)
    d, f = make_dir(tmp_path, A, True)
    R.run("Degree", False, d)
    np.testing.assert_allclose(R.read_nm(stem(f) + "_deg.nm"), R.degree(A > 0))


def test_weighted_tool_rejects_unweighted_network(tmp_path):
    A = R.modular_graph(weighted=False, seed=2)
    d, _ = make_dir(tmp_path, A, False)
    cp = R.run("Degree", True, d, check=False)
    assert cp.returncode == 1
    assert "unweighted network" in cp.stderr


@pytest.mark.parametrize("tool,args", [("Degree", []), ("Cp", [0, "g"]), ("BFS_MulCPU", [0, "g"]),
                                       ("Louvain_Modularity", []), ("Newman_Modularity_CPU", [])])
def test_weighted_tools_reject_non_positive_weights(tmp_path, tool, args):
    A = R.modular_graph(weighted=True, seed=4)
    A[A > 0] -= 0.5           # some edges get r <= 0
    d, _ = make_dir(tmp_path, A, True)
    cp = R.run(tool, True, d, *args, check=False)
    assert cp.returncode == 1
    assert "weight <= 0" in cp.stderr
    # the same network analysed as unweighted is fine
    assert R.run(tool, False, d, *args, check=False).returncode == 0


@pytest.mark.parametrize("ctype", ["binary", "barrat", "onnela"])
def test_clustering(tmp_path, ctype):
    weighted = ctype != "binary"
    A = R.modular_graph(weighted=weighted, seed=3)
    d, f = make_dir(tmp_path, A, weighted)
    if ctype == "binary":
        R.run("Cp", False, d, 0, "gn")
        ref = R.clustering_binary(A)
    else:
        R.run("Cp", True, d, 0, "1" if ctype == "barrat" else "2", "gn")
        ref = R.clustering_barrat(A) if ctype == "barrat" else R.clustering_onnela(A)
    got = R.read_nm(stem(f) + "_cp.nm")
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-6)
    assert R.read_first_number(stem(f) + "_cp.txt") == pytest.approx(ref.mean(), rel=1e-4)


def test_clustering_weighted_default_type_is_onnela(tmp_path):
    A = R.modular_graph(weighted=True, seed=4)
    d, f = make_dir(tmp_path, A, True)
    R.run("Cp", True, d, 0, "n")  # type omitted (crashed in PAGANI 2.0, audit CP-01)
    np.testing.assert_allclose(R.read_nm(stem(f) + "_cp.nm"), R.clustering_onnela(A), rtol=1e-4, atol=1e-6)


def test_clustering_random_and_gamma(tmp_path, graph, weighted):
    d, f = make_dir(tmp_path, graph, weighted)
    args = (d, 5, "gk") if not weighted else (d, 5, "2", "gk")
    R.run("Cp", weighted, *args)
    vals = [float(x) for x in open(stem(f) + "_cp.txt").read().split()]
    assert len(vals) == 6
    assert len(set(vals[1:])) > 1, "random networks must differ from each other (audit M-02)"
    gamma = R.read_first_number(stem(f) + "_gamma.txt")
    assert gamma == pytest.approx(vals[0] / np.mean(vals[1:]), rel=1e-4)
    assert gamma > 1.5  # modular graph is far more clustered than random


def test_random_networks_reproducible(tmp_path, graph, weighted):
    """Random networks are generated in parallel; with PAGANI_SEED the results
    must not depend on the number of threads, and differ between seeds."""
    d, f = make_dir(tmp_path, graph, weighted)
    args = (d, 6, "g") if not weighted else (d, 6, "2", "g")

    def cp_values(**env):
        R.run("Cp", weighted, *args, env=env)
        return open(stem(f) + "_cp.txt").read()

    one = cp_values(PAGANI_SEED="5", PAGANI_THREADS="1")
    assert cp_values(PAGANI_SEED="5", PAGANI_THREADS="4") == one
    assert cp_values(PAGANI_SEED="5") == one
    assert cp_values(PAGANI_SEED="6") != one


def test_random_networks_shared(tmp_path):
    """With one PAGANI_SEED every tool generates the same random networks for a
    network, whatever the file name, the other files in the folder, or whether
    the weighted or unweighted tool is used."""
    A = (R.modular_graph(weighted=False, seed=7) > 0) * 0.5   # equal weights:
    env = {"PAGANI_SEED": "77"}                               # Barrat Cp == binary Cp

    def rand_cp(d, f, weighted):
        R.run("Cp", weighted, d, 4, *(["1"] if weighted else []), "g", env=env)
        return [float(x) for x in open(stem(f) + "_cp.txt").read().split()][1:]

    d1, f1 = make_dir(tmp_path / "a", A, False, name="net")
    one = rand_cp(d1, f1, False)
    # same network under another name, after another network in the folder
    d2, _ = make_dir(tmp_path / "b", R.modular_graph(weighted=False, seed=8), False, name="aaa")
    _, f2 = make_dir(tmp_path / "b", A, False, name="zzz")
    assert rand_cp(d2, f2, False) == one
    d3, f3 = make_dir(tmp_path / "c", A, True, name="net")
    np.testing.assert_allclose(rand_cp(d3, f3, True), one, rtol=1e-5)
    assert len(set(one)) > 1


@pytest.mark.parametrize("tool", ["BFS_MulCPU"])
def test_efficiency(tmp_path, graph, weighted, tool):
    d, f = make_dir(tmp_path, graph, weighted)
    R.run(tool, weighted, d, 3, "gnl")
    lp_ref, nodal_ref = R.efficiency(graph, weighted)
    np.testing.assert_allclose(R.read_nm(stem(f) + "_eff.nm"), nodal_ref, rtol=1e-5, atol=1e-7)
    vals = [float(x) for x in open(stem(f) + "_Lp.txt").read().split()]
    assert vals[0] == pytest.approx(lp_ref, rel=1e-5)
    assert len(vals) == 4
    lam = R.read_first_number(stem(f) + "_lambda.txt")
    assert lam == pytest.approx(vals[0] / np.mean(vals[1:]), rel=1e-4)


def test_efficiency_small_reachable_set_not_zeroed(tmp_path):
    """PAGANI 2.0 set nodal efficiency < 1/N to zero (audit L-02)."""
    n = 50
    A = np.zeros((n, n))
    A[0, 1] = A[1, 0] = 1  # 0 - 1
    A[1, 2] = A[2, 1] = 1  # 1 - 2 : node 0 reaches 2 nodes
    for i in range(10, n - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    d, f = make_dir(tmp_path, A, False)
    R.run("BFS_MulCPU", False, d, 0, "n")
    _, ref = R.efficiency(A, False)
    np.testing.assert_allclose(R.read_nm(stem(f) + "_eff.nm"), ref, rtol=1e-6)


def test_modularity_louvain(tmp_path, graph, weighted):
    d, f = make_dir(tmp_path, graph, weighted)
    R.run("Louvain_Modularity", weighted, d, 2)
    lab = R.read_nm(stem(f) + "_modu.nm").astype(int)
    k = R.degree(graph)
    assert np.all(lab[k == 0] == 0) and np.all(lab[k > 0] > 0)
    q = R.modularity(graph, lab)
    txt = open(stem(f) + "_modu.txt").read()
    q_file = float(txt.split("Final Modularity Q =")[1].split()[0])
    assert q_file == pytest.approx(q, abs=1e-5)
    import networkx as nx
    G = nx.from_numpy_array(graph)
    q_nx = max(nx.community.modularity(G, nx.community.louvain_communities(G, weight="weight", seed=s))
               for s in range(5))
    assert q >= q_nx - 0.01
    import re
    assert len(re.findall(r"Random Network \d+:", txt)) == 2


def test_modularity_newman(tmp_path, graph, weighted):
    d, f = make_dir(tmp_path, graph, weighted)
    R.run("Newman_Modularity_CPU", weighted, d)
    lab = R.read_nm(stem(f) + "_modu.nm").astype(int)
    q = R.modularity(graph, lab)
    import networkx as nx
    G = nx.from_numpy_array(graph)
    q_ref = nx.community.modularity(G, nx.community.greedy_modularity_communities(G, weight="weight"))
    assert q >= q_ref - 0.02
    assert len(np.unique(lab[lab > 0])) >= 4


def test_participation(tmp_path, graph, weighted):
    d, f = make_dir(tmp_path, graph, weighted)
    lab = np.zeros(graph.shape[0])
    k = R.degree(graph)
    lab[k > 0] = (np.arange(graph.shape[0])[k > 0] // 60) + 1
    R.write_nm(stem(f) + "_modu.nm", lab)
    R.run("PC_CPU", weighted, d)
    pc_ref, nm = R.participation(graph, lab)
    np.testing.assert_allclose(R.read_nm(stem(f) + "_pc.nm"), pc_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(R.read_nm(stem(f) + "_pc_normalized.nm"), pc_ref / ((nm - 1) / nm),
                               rtol=1e-5, atol=1e-6)


def test_participation_requires_modules(tmp_path, graph, weighted):
    d, _ = make_dir(tmp_path, graph, weighted)
    assert R.run("PC_CPU", weighted, d, check=False).returncode == 1


def test_maslov_preserves_degrees_and_weights(tmp_path, graph, weighted):
    d, f = make_dir(tmp_path, graph, weighted)
    R.run("Maslov", weighted, f)
    M, w = R.read_csr(stem(f) + "_Maslov.csr")
    assert w == weighted
    A = M.toarray()
    assert np.allclose(A, A.T)
    assert np.all(np.diag(A) == 0)
    np.testing.assert_array_equal((A > 0).sum(1), (graph > 0).sum(1))
    assert ((A > 0) != (graph > 0)).sum() > 0.5 * (graph > 0).sum()
    if weighted:
        np.testing.assert_allclose(np.sort(A[A > 0]), np.sort(graph[graph > 0].astype(np.float32)), rtol=1e-6)


def test_smallworld_pairing(tmp_path):
    d = tmp_path / "net"
    d.mkdir()
    # names sharing the first characters broke PAGANI 2.0 (audit S-01)
    vals = {"sub001_spa10.000%": (2.0, 1.25), "sub0010_spa10.000%": (3.0, 1.5), "sub002_spa10.000%": (1.8, 1.2)}
    for s, (g, l) in vals.items():
        (d / f"{s}_gamma.txt").write_text(f"{g}\n")
        (d / f"{s}_lambda.txt").write_text(f"{l}\n")
    (d / "lonely_gamma.txt").write_text("2\n")
    R.run("SmallWorldProperty", False, d)
    for s, (g, l) in vals.items():
        assert R.read_first_number(d / f"{s}_smallWorld.txt") == pytest.approx(g / l, rel=1e-6)
    assert not (d / "lonely_smallWorld.txt").exists()


@pytest.mark.parametrize("compressed", [False, True], ids=["nii", "niigz"])
@pytest.mark.parametrize("variant", ["uint8_scaled", "float_ext", "int16_bigendian"])
def test_convertnii(tmp_path, variant, compressed):
    shape = (6, 7, 5)
    rng = np.random.default_rng(0)
    prob = rng.random(shape)
    if variant == "uint8_scaled":   # SPM c1 maps: uint8 with scl_slope = 1/255
        raw = np.round(prob * 255).astype(np.uint8)
        maskfile = tmp_path / "mask.nii"
        R.save_nifti(maskfile, raw, slope=1 / 255.0)
        maskval = raw * (1 / 255.0)
    elif variant == "float_ext":
        maskfile = tmp_path / "mask.nii"
        R.save_nifti(maskfile, prob.astype(np.float32), extension_bytes=32)
        maskval = prob.astype(np.float32)
    else:
        raw = np.round(prob * 1000).astype(np.int16)
        maskfile = tmp_path / "mask.nii"
        R.save_nifti(maskfile, raw, big_endian=True, slope=0.001)
        maskval = raw * 0.001
    if compressed:
        import gzip
        gz = tmp_path / "mask.nii.gz"
        gz.write_bytes(gzip.compress(maskfile.read_bytes()))
        maskfile.unlink()
        maskfile = gz
    thr = 0.4
    inmask = (maskval.ravel(order="F") >= thr)
    d = tmp_path / "res"
    d.mkdir()
    vals = rng.random(inmask.sum()).astype(np.float32)
    R.write_nm(d / "x_deg.nm", vals)
    R.run("ConvertNII", False, d, maskfile, thr)
    img, hdr = R.load_nifti(d / "x_deg.nii")
    expect = np.zeros(inmask.size)
    expect[inmask] = vals
    np.testing.assert_allclose(img.ravel(order="F"), expect, rtol=1e-6)
    assert hdr.get_data_dtype() == np.float32
    import struct
    assert struct.unpack_from("<f", open(d / "x_deg.nii", "rb").read(), 108)[0] == 352
    sl, it = hdr.get_slope_inter()
    assert sl in (None, 1.0) and it in (None, 0.0)


def test_convertnii_mismatch_is_error(tmp_path):
    maskfile = tmp_path / "mask.nii"
    R.save_nifti(maskfile, np.ones((3, 3, 3), np.float32))
    d = tmp_path / "res"
    d.mkdir()
    R.write_nm(d / "a.nm", np.ones(5))
    assert R.run("ConvertNII", False, d, maskfile, 0.5, check=False).returncode == 1


def test_usage_errors_return_2(tmp_path):
    assert R.run("Cp", False, check=False).returncode == 2
    assert R.run("Cp", False, tmp_path, "abc", "g", check=False).returncode == 2
    assert R.run("Cp", False, tmp_path, 0, "k", check=False).returncode == 2
