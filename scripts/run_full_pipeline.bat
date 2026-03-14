@echo off
setlocal

set PROJECT_ROOT=%~dp0..
cd /d %PROJECT_ROOT%

if not exist .env (
  echo [ERROR] .env file not found. Copy .env.example to .env and update values.
  exit /b 1
)

set SIMULATOR_BATCH_SIZE=500

echo [STEP 1/5] Generating synthetic EHR rows...
call scripts\run_ehr_simulator.bat
if errorlevel 1 goto :fail

echo [STEP 2/5] Staging synthetic CSV to Bronze...
call scripts\run_airbyte_staging.bat Data/synthetic_ehr.csv
if errorlevel 1 goto :fail

echo [STEP 3/5] Transforming Bronze to Silver...
call scripts\run_bronze_to_silver.bat
if errorlevel 1 goto :fail

echo [STEP 4/5] Transforming Silver to Gold...
call scripts\run_silver_to_gold.bat
if errorlevel 1 goto :fail

echo [STEP 5/5] Exporting final CSV outputs...
call scripts\run_final_output_export.bat
if errorlevel 1 goto :fail

echo.
echo [SUCCESS] Full pipeline completed.
echo Outputs are under Data/final and Data/gold/marts.
exit /b 0

:fail
echo.
echo [FAILED] Pipeline stopped due to an error.
exit /b 1
