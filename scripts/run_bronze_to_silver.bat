@echo off
setlocal

set PROJECT_ROOT=%~dp0..
cd /d %PROJECT_ROOT%

if exist .venv\Scripts\python.exe (
  set PYTHON_EXE=.venv\Scripts\python.exe
) else (
  set PYTHON_EXE=python
)

set PYTHONPATH=%PROJECT_ROOT%\src
%PYTHON_EXE% -m etl.bronze_to_silver --config config/bronze_to_silver_config.json

endlocal
