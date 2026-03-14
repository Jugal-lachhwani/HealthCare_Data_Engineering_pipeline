# Patient Readmission Platform - Parts 1 and 2

This repository now includes the implementation foundation and synthetic source generation for the healthcare pipeline.

It also includes Part 3 ingestion as an Airbyte-style staging process from CSV into Bronze raw Parquet.

Airflow orchestration is included to run simulator then staging on a recurring schedule.

## Scope implemented

- Part 1: Foundation setup
  - Project structure for `src`, `scripts`, `config`, `Data/bronze`, `Data/silver`, `Data/gold`, `airflow/dags`, `dbt`.
  - Environment template and dependency file.
  - JSON schema draft for synthetic EHR events.

- Part 2: Data source simulation and generation
  - Python EHR.csv-like synthetic row generator.
  - MongoDB and CSV writers.
  - Two run modes:
    - `once` mode for a single batch insert.
    - `continuous` mode using a built-in timer loop.
  - Windows Task Scheduler registration script.

- Part 3: Airbyte-style staging to Bronze
  - Incremental sync from CSV source using persisted state.
  - Writes partitioned raw Parquet files under Bronze.
  - Adds Airbyte-like metadata columns (`_airbyte_ab_id`, `_airbyte_emitted_at`, `_airbyte_stream`, `_airbyte_data`).

- Part 4: Bronze to Silver ETL
  - Cleans and normalizes Bronze records into Silver analytical format.
  - Casts numeric types, standardizes categorical values, and derives helper features.
  - Produces a data quality report and incremental processing state.

- Part 5: Data Quality and EDA Dashboard
  - Streamlit dashboard on top of Silver parquet and quality reports.
  - Displays KPI cards, null checks, discharge status mix, LOS distribution, and top diagnoses.

- Part 6: Persona Identification and Analytical Dashboards
  - Silver-to-Gold marts for doctor and administrator personas.
  - Doctor mart highlights patient-level risk and triage prioritization.
  - Admin marts expose hospital/unit-level operational KPIs for management reporting.

- Gold to SQL publishing
  - Loads Gold parquet marts into PostgreSQL tables under `analytics` schema.
  - Makes data directly queryable for ML feature pulls, BI, and downstream services.

- Orchestration: Airflow DAG
  - DAG file chains `generate_synthetic_csv_batch` then `stage_csv_to_bronze_raw`.
  - Schedule is every 5 minutes (`*/5 * * * *`).

## Files added

- `src/simulator/config.py`
- `src/simulator/csv_like_generator.py`
- `src/simulator/csv_writer.py`
- `src/simulator/mongodb_writer.py`
- `src/simulator/run_simulator.py`
- `src/ingestion/airbyte_style_staging.py`
- `config/ehr_event_schema.json`
- `config/airbyte_staging_config.json`
- `scripts/run_airbyte_staging.bat`
- `src/etl/bronze_to_silver.py`
- `config/bronze_to_silver_config.json`
- `scripts/run_bronze_to_silver.bat`
- `src/dashboard/data_quality_eda_app.py`
- `config/dashboard_config.json`
- `scripts/run_quality_dashboard.bat`
- `src/etl/silver_to_gold.py`
- `config/silver_to_gold_config.json`
- `scripts/run_silver_to_gold.bat`
- `src/etl/gold_to_sql.py`
- `config/gold_to_sql_config.json`
- `scripts/run_gold_to_sql.bat`
- `src/orchestration/pipeline_tasks.py`
- `airflow/dags/ehr_simulator_to_bronze_dag.py`
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

3. Make sure MongoDB is running locally if sink is `mongo` or `both`.

4. Run one batch and write synthetic EHR-like rows to CSV only:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m simulator.run_simulator --mode once --sink csv --output-csv Data/synthetic_ehr.csv
```

5. Run one batch to both CSV and MongoDB:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m simulator.run_simulator --mode once --sink both
```

6. Run continuously in-process:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m simulator.run_simulator --mode continuous --interval-seconds 60 --batch-size 5
```

7. Register scheduled task (every 5 minutes):

```powershell
.\scripts\register_ehr_task.ps1 -EveryMinutes 5
```

8. Run Airbyte-style staging sync from CSV to Bronze:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m ingestion.airbyte_style_staging --config config/airbyte_staging_config.json --source-csv Data/synthetic_ehr.csv
```

9. Run with Windows batch wrapper:

```powershell
.\scripts\run_airbyte_staging.bat Data\synthetic_ehr.csv
```

10. Run Bronze-to-Silver ETL:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m etl.bronze_to_silver --config config/bronze_to_silver_config.json
```

11. Run with Windows batch wrapper:

```powershell
.\scripts\run_bronze_to_silver.bat
```

12. Run Step 5 quality dashboard:

```powershell
.\scripts\run_quality_dashboard.bat
```

13. Open dashboard:

- URL: `http://localhost:8501`

14. Run Step 6 Silver-to-Gold marts:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m etl.silver_to_gold --config config/silver_to_gold_config.json
```

15. Run with Windows batch wrapper:

```powershell
.\scripts\run_silver_to_gold.bat
```

16. Load Gold marts into SQL tables:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m etl.gold_to_sql --config config/gold_to_sql_config.json
```

17. Run with Windows batch wrapper:

```powershell
.\scripts\run_gold_to_sql.bat
```

## Synthetic dataset coverage

Generated rows follow the same columns as `Data/EHR.csv`, including:

- ICU/hospital stay identifiers
- Demographics and ethnicity
- APACHE diagnosis text
- Admission/discharge times and offsets
- Unit type, stay type, source, and location fields
- Weight/height fields with realistic missingness

## Bronze layout

- Raw staged files: `Data/bronze/airbyte_raw/ehr_visits/load_date=YYYY-MM-DD/load_hour=HH/*.parquet`
- Sync state file: `Data/bronze/_airbyte_state/ehr_visits_state.json`

## Silver layout

- Silver files: `Data/silver/ehr_visits/process_date=YYYY-MM-DD/*.parquet`
- ETL state: `Data/silver/_state/bronze_to_silver_state.json`
- Quality report: `Data/silver/_quality/ehr_visits_quality_report.json`

## Step 5 Dashboard Panels

- Data quality KPI summary (rows, duplicates, expired counts, source file count)
- Required-column null count chart
- Gender distribution and discharge status charts
- Unit LOS histogram
- Top admission diagnosis chart
- Silver data preview table

## Step 6 Gold Marts

- Doctor mart: `Data/gold/marts/doctor_patient_risk_mart.parquet`
- Admin hospital KPI mart: `Data/gold/marts/admin_hospital_kpi_mart.parquet`
- Admin department KPI mart: `Data/gold/marts/admin_department_kpi_mart.parquet`

## SQL Tables (Published from Gold)

- Database: `airflow`
- Schema: `analytics`
- Tables:
  - `analytics.doctor_patient_risk_mart`
  - `analytics.admin_hospital_kpi_mart`
  - `analytics.admin_department_kpi_mart`

## Airflow Task Chain

- Task dependency: Task 1 >> Task 2 >> Task 3 >> Task 4 >> Task 5
- Task 5: `load_gold_to_sql`

## Step 6 Persona Tabs (Streamlit)

- Doctor Persona: high-risk queue, risk-band mix, risk vs LOS, unit-level high-risk burden
- Admin Persona: hospital volumes, mortality and high-risk rates, daily KPI trend, hospital-unit heatmap

## Airflow DAG

- DAG ID: `ehr_simulator_to_bronze_staging`
- Task 1: Generate synthetic EHR-like rows to CSV
- Task 2: Incrementally stage new rows to Bronze raw Parquet
- Task 3: Transform Bronze records into Silver normalized dataset
- Task 4: Publish Silver-derived Gold persona marts
- Task dependency: Task 1 >> Task 2 >> Task 3 >> Task 4

## Airflow Local Runtime (Docker)

Prerequisite: Docker Desktop must be running with Linux containers enabled.

1. Start Airflow stack:

```powershell
.\scripts\start_airflow_local.ps1
```

2. Open Airflow UI:

- URL: `http://localhost:8080`
- Username: `admin`
- Password: `admin`

3. Enable DAG `ehr_simulator_to_bronze_staging` and trigger manually, or wait for the 5-minute schedule.

4. Stop Airflow stack:

```powershell
.\scripts\stop_airflow_local.ps1
```

Compose and image files:

- `docker-compose.airflow.yml`
- `airflow/Dockerfile`
