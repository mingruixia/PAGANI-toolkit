@echo off
rem GPU validation of a PAGANI package on Windows.
rem   tests\run_validation.bat <path to the unpacked PAGANI package>
rem Requires Python 3.9+ (python.org or conda) with: pip install -r tests\requirements.txt
rem Writes tests\validation_report.txt and tests\validation_report.xml.
setlocal
if "%~1"=="" (
  echo usage: %~nx0 ^<PAGANI package folder containing exefiles and exefiles_weighted^>
  exit /b 2
)
if not exist "%~1\exefiles\CUCorMat.exe" (
  echo "%~1" does not look like a PAGANI package ^(exefiles\CUCorMat.exe not found^)
  exit /b 2
)
set "PAGANI_BIN=%~f1"
set PAGANI_TEST_GPU=1
cd /d "%~dp0"
python -c "import numpy, scipy, networkx, pytest" 2>nul || (
  echo Installing the Python packages used by the tests ...
  python -m pip install -r requirements.txt || exit /b 1
)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > validation_report.txt 2>&1
python -m pytest -v -rs --junitxml=validation_report.xml . >> validation_report.txt 2>&1
set RESULT=%ERRORLEVEL%
powershell -NoProfile -Command "Get-Content validation_report.txt -Tail 20"
if %RESULT%==0 (echo. & echo All validation tests passed.) else (echo. & echo Some tests FAILED - see tests\validation_report.txt)
exit /b %RESULT%
