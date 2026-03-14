@echo off
setlocal

set PROJECT_ROOT=%~dp0..
cd /d %PROJECT_ROOT%

if exist .venv\Scripts\streamlit.exe (
  set STREAMLIT_EXE=.venv\Scripts\streamlit.exe
) else (
  set STREAMLIT_EXE=streamlit
)

set PYTHONPATH=%PROJECT_ROOT%\src
%STREAMLIT_EXE% run src\dashboard\data_quality_eda_app.py --server.port 8501

endlocal
