@echo off
setlocal

set PROJECT_ROOT=%~dp0..
cd /d %PROJECT_ROOT%

if not exist .env (
  echo .env file not found. Copy .env.example to .env and update values.
  exit /b 1
)

if exist .venv\Scripts\python.exe (
  set PYTHON_EXE=.venv\Scripts\python.exe
) else (
  set PYTHON_EXE=python
)

set PYTHONPATH=%PROJECT_ROOT%\src
%PYTHON_EXE% -m simulator.run_simulator --mode once

endlocal
