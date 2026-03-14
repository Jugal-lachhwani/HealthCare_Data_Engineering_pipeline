# Patient Readmission Platform - Parts 1 and 2

This repository now includes the implementation foundation and synthetic source generation for the healthcare pipeline.

## Scope implemented

- Part 1: Foundation setup
  - Project structure for `src`, `scripts`, `config`, `Data/bronze`, `Data/silver`, `Data/gold`, `airflow/dags`, `dbt`.
  - Environment template and dependency file.
  - JSON schema draft for synthetic EHR events.

- Part 2: Data source simulation and generation
  - Python EHR event simulator.
  - MongoDB writer.
  - Two run modes:
    - `once` mode for a single batch insert.
    - `continuous` mode using a built-in timer loop.
  - Windows Task Scheduler registration script.

## Files added

- `src/simulator/config.py`
- `src/simulator/generate_ehr.py`
- `src/simulator/mongodb_writer.py`
- `src/simulator/run_simulator.py`
- `config/ehr_event_schema.json`
- `scripts/run_ehr_simulator.bat`
- `scripts/register_ehr_task.ps1`
- `.env.example`
- `requirements.txt`

## Quick start (Windows)

1. Create environment file:

```powershell
Copy-Item .env.example .env
```

2. Install dependencies in your virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Make sure MongoDB is running locally or update `MONGO_URI` in `.env`.

4. Run one batch:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m simulator.run_simulator --mode once
```

5. Run continuously in-process:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m simulator.run_simulator --mode continuous --interval-seconds 60 --batch-size 5
```

6. Register scheduled task (every 5 minutes):

```powershell
.\scripts\register_ehr_task.ps1 -EveryMinutes 5
```

## Event payload coverage

Synthetic events include:

- Patient demographics
- Encounter timestamps and diagnoses
- Lab results and medication lists
- Proxy readmission risk label
- Source and ingestion timestamps
