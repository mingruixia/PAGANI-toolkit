#!/bin/bash
# Example PAGANI analysis on Linux (unweighted networks at 5 % and 10 %
# sparsity, 10 random networks).  Adjust the paths and options; see
# docs/USER_GUIDE.md, section 4, for every tool and its arguments.
set -e
P="$(cd "$(dirname "$0")" && pwd)/exefiles"   # exefiles_weighted for weighted networks
D=/data/study/bold                            # one 4D image (.nii or .nii.gz) per subject
MASK=/data/study/grey.nii.gz
export PAGANI_SEED=12345                      # same random networks for lambda, gamma, sigma

"$P/CUCorMat" "$D" "$MASK" 0.2 n n s 5 10
"$P/CUBFW_Lp" "$D/unweighted" 10 gnl
"$P/Cp" "$D/unweighted" 10 gnk
"$P/SmallWorldProperty" "$D/unweighted"
"$P/Degree" "$D/unweighted"
"$P/ConvertNII" "$D/unweighted" "$MASK" 0.2
