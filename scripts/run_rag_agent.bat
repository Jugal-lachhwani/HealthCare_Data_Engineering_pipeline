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

if "%~1"=="" (
  %PYTHON_EXE% -m genai.rag_readmission_agent --model qwen2:7b
) else (
  %PYTHON_EXE% -m genai.rag_readmission_agent --model qwen2:7b --question "%~1"
)

endlocal
