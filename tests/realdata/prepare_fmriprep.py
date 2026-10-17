"""Prepare PAGANI test data from fMRIPrep outputs (validation only).

For each subject: stream the MNI152NLin2009cAsym res-2 preprocessed BOLD
(.nii.gz) volume by volume, block-average to 4 mm, regress out 6 motion
parameters and their derivatives, WM, CSF and a quadratic trend, band-pass
0.01-0.08 Hz, and write an uncompressed float32 4D .nii. Also writes a group
grey-matter probability mask (mean GM probseg x all brain masks) at 4 mm.

    python prepare_fmriprep.py <fmriprep_dir> <out_dir> sub-01 sub-02 ...
"""
import csv
import glob
import os
import sys

import nibabel as nb
import numpy as np

SPACE = "space-MNI152NLin2009cAsym_res-2"
F = 2  # 2 mm -> 4 mm


def block_mean(vol):
    pad = [(0, (-s) % F) for s in vol.shape]
    v = np.pad(vol, pad)
    nx, ny, nz = (s // F for s in v.shape)
    return v.reshape(nx, F, ny, F, nz, F).mean(axis=(1, 3, 5))


def affine4(aff):
    a = aff.copy()
    a[:3, :3] *= F
    a[:3, 3] += aff[:3, :3] @ np.full(3, (F - 1) / 2.0)
    return a


def one(path):
    return sorted(glob.glob(path))[0]


def confounds(tsv):
    cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    cols += [c + "_derivative1" for c in cols] + ["white_matter", "csf"]
    with open(tsv) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    X = np.array([[0.0 if r[c] in ("n/a", "") else float(r[c]) for c in cols] for r in rows])
    t = np.linspace(-1, 1, len(rows))
    return np.column_stack([np.ones_like(t), t, t * t, X])


def bandpass(Y, tr, lo=0.01, hi=0.08):
    n = Y.shape[0]
    f = np.fft.rfftfreq(n, tr)
    S = np.fft.rfft(Y, axis=0)
    S[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(S, n, axis=0)


def main():
    src, out = sys.argv[1], sys.argv[2]
    subs = sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    gm_sum, brain_all, aff = None, None, None
    for s in subs:
        func = os.path.join(src, s, "ses-*", "func")
        anat = os.path.join(src, s, "ses-*", "anat")
        gm = block_mean(nb.load(one(os.path.join(anat, f"*_{SPACE}_label-GM_probseg.nii.gz"))).get_fdata())
        bm = block_mean(nb.load(one(os.path.join(func, f"*run-01_{SPACE}_desc-brain_mask.nii.gz"))).get_fdata()) > 0.5
        gm_sum = gm if gm_sum is None else gm_sum + gm
        brain_all = bm if brain_all is None else brain_all & bm
        dst = os.path.join(out, f"{s}.nii")
        if os.path.exists(dst):
            print(s, "exists", flush=True)
            continue
        img = nb.load(one(os.path.join(func, f"*run-01_{SPACE}_desc-preproc_bold.nii.gz")), keep_file_open=True)
        aff = affine4(img.affine)
        tr = float(img.header.get_zooms()[3]) or 0.75
        T = img.shape[3]
        vols = np.empty(gm.shape + (T,), np.float32)
        for t in range(T):
            vols[..., t] = block_mean(np.asarray(img.dataobj[..., t], np.float32))
            if t % 100 == 0:
                print(s, "volume", t, flush=True)
        work = bm & (gm > 0.05)
        Y = vols[work].T.astype(np.float64)                 # T x V
        X = confounds(one(os.path.join(func, "*run-01_desc-confounds_timeseries.tsv")))
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        Y = bandpass(Y - X @ beta, tr) + 100.0
        vols[:] = 0
        vols[work] = Y.T.astype(np.float32)
        oi = nb.Nifti1Image(vols, aff)
        oi.header.set_xyzt_units("mm", "sec")
        oi.header.set_zooms((4.0, 4.0, 4.0, tr))
        nb.save(oi, dst + ".tmp.nii")
        os.replace(dst + ".tmp.nii", dst)
        print(s, "done", vols.shape, int(work.sum()), "voxels processed", flush=True)
    if aff is None:
        aff = affine4(nb.load(one(os.path.join(src, subs[0], "ses-*", "func",
                                               f"*run-01_{SPACE}_desc-brain_mask.nii.gz"))).affine)
    mask = (gm_sum / len(subs)) * brain_all
    nb.save(nb.Nifti1Image(mask.astype(np.float32), aff), os.path.join(os.path.dirname(out.rstrip("/")), "GM_mask_4mm.nii"))
    print("mask voxels >= 0.2:", int((mask >= 0.2).sum()), " >= 0.3:", int((mask >= 0.3).sum()))


if __name__ == "__main__":
    main()
