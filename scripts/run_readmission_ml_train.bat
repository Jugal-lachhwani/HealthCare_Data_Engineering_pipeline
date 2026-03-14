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
%PYTHON_EXE% -m ml.train_readmission_30d_model --input-csv Data/final/readmission_model_dataset.csv --model-path models/readmission_30d_model.joblib --predictions-csv Data/final/readmission_predictions.csv --metrics-json Data/final/readmission_model_metrics.json --feature-importance-csv Data/final/readmission_feature_importance.csv

endlocal
