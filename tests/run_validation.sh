#!/bin/bash
# GPU validation of a PAGANI package on Linux (counterpart of run_validation.bat).
#   tests/run_validation.sh <path to the unpacked PAGANI package>
# Requires Python 3.9+ with: pip install -r tests/requirements.txt
# Writes tests/validation_report.txt and tests/validation_report.xml.
if [ -z "$1" ]; then
  echo "usage: $0 <PAGANI package folder containing exefiles and exefiles_weighted>"
  exit 2
fi
if [ ! -x "$1/exefiles/CUCorMat" ]; then
  echo "'$1' does not look like a PAGANI package (exefiles/CUCorMat not found)"
  exit 2
fi
export PAGANI_BIN="$(cd "$1" && pwd)"
export PAGANI_TEST_GPU=1
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -c "import numpy, scipy, networkx, pytest" 2>/dev/null || {
  echo "Installing the Python packages used by the tests ..."
  $PY -m pip install -r requirements.txt || exit 1
}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > validation_report.txt 2>&1
$PY -m pytest -v -rs --junitxml=validation_report.xml . >> validation_report.txt 2>&1
result=$?
tail -n 20 validation_report.txt
if [ $result -eq 0 ]; then echo; echo "All validation tests passed."; else echo; echo "Some tests FAILED - see tests/validation_report.txt"; fi
exit $result
