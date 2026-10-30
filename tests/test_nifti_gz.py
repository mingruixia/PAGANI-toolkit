"""Compressed (.nii.gz) images: same networks and file names as with .nii."""
import gzip
import shutil

import test_construct as C


def gzip_file(src, dst):
    with open(src, "rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g)


def outputs(d):
    """Relative path -> bytes of every file the tools wrote under d (not the images)."""
    return {str(p.relative_to(d)): p.read_bytes()
            for p in sorted(d.rglob("*")) if p.is_file() and not p.name.lower().endswith((".nii", ".nii.gz"))}


def compressed_copy(tmp_path, d, mask):
    """Copy of the dataset with every image and the mask gzip-compressed."""
    dz = tmp_path / "data_gz"
    dz.mkdir()
    for f in d.glob("*.nii"):
        gzip_file(f, dz / (f.name + ".gz"))
    maskz = tmp_path / "grey.nii.gz"
    gzip_file(mask, maskz)
    return dz, maskz


def test_same_networks_as_uncompressed(tmp_path):
    # three subjects: float32, int16 with scaling, float64 with a header extension
    d, mask, thr, _ = C.make_dataset(tmp_path)
    dz, maskz = compressed_copy(tmp_path, d, mask)
    for src, m in ((d, mask), (dz, maskz)):
        C.run("CUCorMat", True, src, m, thr, "bf", "y", "s", 10)
        C.run("CUCorMat", False, src, m, thr, "n", "n", "r", 0.4)
        C.run("CUCorMatHr", False, src, m, thr, "n", "s", "ps", 8)
        C.run("CuCorMat-sca-kendall", True, src, m, thr, "n", "s", 10)
    plain, packed = outputs(d), outputs(dz)
    assert plain.keys() == packed.keys()
    assert any(k.startswith("sub00_") for k in plain)        # .cormat named after the subject
    assert any(k.endswith("_weighted.csr") for k in plain)
    for k in plain:
        assert plain[k] == packed[k], k


def test_mixed_directory(tmp_path):
    d, mask, thr, _ = C.make_dataset(tmp_path)
    f = d / "sub01.nii"
    gzip_file(f, d / "sub01.nii.gz")
    f.unlink()
    cp = C.run("CUCorMat", False, d, mask, thr, "n", "n", "r", 0.4)
    assert "3 image file(s)" in cp.stdout
    names = sorted(p.name.split("_")[0] for p in (d / "unweighted").glob("*.csr"))
    assert names == ["sub00", "sub01", "sub02"]


def test_same_subject_twice_is_error(tmp_path):
    d, mask, thr, _ = C.make_dataset(tmp_path, nsub=1)
    gzip_file(d / "sub00.nii", d / "sub00.nii.gz")
    cp = C.run("CUCorMat", False, d, mask, thr, "n", "n", "r", 0.4, check=False)
    assert cp.returncode == 1 and "same subject name" in cp.stderr


def test_compressed_mask_in_data_dir_is_not_a_subject(tmp_path):
    d, mask, thr, _ = C.make_dataset(tmp_path, nsub=1)
    gzip_file(mask, d / "mymask.nii.gz")
    cp = C.run("CUCorMat", False, d, d / "mymask.nii.gz", thr, "n", "n", "r", 0.5)
    assert "1 image file(s)" in cp.stdout


def test_truncated_and_corrupt_files(tmp_path):
    d, mask, thr, _ = C.make_dataset(tmp_path, nsub=1)
    raw = (d / "sub00.nii").read_bytes()
    (d / "sub00.nii").unlink()
    gz = gzip.compress(raw)
    (d / "sub00.nii.gz").write_bytes(gzip.compress(raw[: len(raw) // 2]))    # valid gzip, image cut short
    cp = C.run("CUCorMat", False, d, mask, thr, "n", "n", "r", 0.4, check=False)
    assert cp.returncode == 1 and "truncated" in cp.stderr
    (d / "sub00.nii.gz").write_bytes(gz[: len(gz) // 2])                     # gzip stream cut short
    cp = C.run("CUCorMat", False, d, mask, thr, "n", "n", "r", 0.4, check=False)
    assert cp.returncode == 1 and ("truncated" in cp.stderr or "corrupt" in cp.stderr)
    bad = bytearray(gz)
    for i in range(40, 60):
        bad[i] ^= 0xFF
    (d / "sub00.nii.gz").write_bytes(bytes(bad))                             # damaged deflate data
    cp = C.run("CUCorMat", False, d, mask, thr, "n", "n", "r", 0.4, check=False)
    assert cp.returncode == 1 and ("corrupt" in cp.stderr or "truncated" in cp.stderr)
