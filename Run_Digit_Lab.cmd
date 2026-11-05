@echo off
cd /d "%~dp0"
if not exist ".venv312\Scripts\python.exe" (
  py -3.12 -m venv .venv312
  if errorlevel 1 goto python_missing
)
".venv312\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] == (3,12)"
if errorlevel 1 goto python_missing
".venv312\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
if not exist "artifacts\feature_bank.npz" goto data_missing
echo Open http://127.0.0.1:7860 after the server starts.
".venv312\Scripts\python.exe" app.py
pause
exit /b
:python_missing
echo Install Python 3.12 from python.org. Python 3.14 is not supported by these pinned dependencies.
pause
exit /b 1
:data_missing
echo This Git copy excludes dataset-dependent artifacts. Follow README.md to reproduce training with your A1 data, or use the complete local bundle.
pause
exit /b 1
:fail
echo Installation failed. Read the error above.
pause
exit /b 1
