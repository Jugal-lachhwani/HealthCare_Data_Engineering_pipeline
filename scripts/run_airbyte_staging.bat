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
  %PYTHON_EXE% -m ingestion.airbyte_style_staging --config config/airbyte_staging_config.json
) else (
  %PYTHON_EXE% -m ingestion.airbyte_style_staging --config config/airbyte_staging_config.json --source-csv %1
)

endlocal
