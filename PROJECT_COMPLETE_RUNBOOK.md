# Healthcare Data Engineering Pipeline - Complete Runbook

This document explains the full project end-to-end:
1. What each step does
2. Which tables are created
3. Which tables are used for KPIs
4. Where data is stored
5. How to run the project

## 1. Project Objective

Build a medallion data pipeline for healthcare readmission analytics:
1. Generate source EHR-like data
2. Stage raw data into Bronze
3. Clean and model data into Silver
4. Build KPI-ready marts in Gold
5. Publish Gold marts to PostgreSQL
6. Visualize and monitor in Streamlit

## 2. High-Level Architecture

Source -> Bronze -> Silver -> Gold -> SQL/Streamlit

1. Source: synthetic EHR records (CSV and optional MongoDB)
2. Bronze: raw Airbyte-style parquet with metadata
3. Silver: fully cleaned, modeled tables (4 tables only)
4. Gold: KPI/dashboard marts (3 tables only)
5. Serving: PostgreSQL analytics schema + Streamlit dashboard

## 3. Step-by-Step Pipeline

## Step 1: Source Simulation

Code:
1. src/simulator/run_simulator.py
2. src/simulator/csv_like_generator.py
3. src/simulator/mongodb_writer.py

What it does:
1. Generates synthetic patient-visit records
2. Writes to CSV, MongoDB, or both (based on env)

Main outputs:
1. Data/synthetic_ehr.csv
2. MongoDB: healthcare_ehr.raw_patient_visits

## Step 2: Bronze Ingestion

Code:
1. src/ingestion/airbyte_style_staging.py

What it does:
1. Reads source CSV incrementally
2. Writes raw Bronze parquet with metadata columns
3. Tracks ingestion state

Main outputs:
1. Data/bronze/airbyte_raw/ehr_visits/load_date=YYYY-MM-DD/load_hour=HH/*.parquet
2. Data/bronze/_airbyte_state/ehr_visits_state.json

## Step 3: Bronze -> Silver Transformation

Code:
1. src/etl/bronze_to_silver.py

What it does:
1. Cleans types and values
2. Normalizes fields and creates derived features
3. Splits Silver into modeled domain tables
4. Produces data quality report + state

Silver tables created (only these 4):
1. visits_core
2. diagnosis_care
3. utilization_outcomes
4. clinical_features

Main outputs:
1. Data/silver/ehr_visits/visits_core/process_date=YYYY-MM-DD/*.parquet
2. Data/silver/ehr_visits/diagnosis_care/process_date=YYYY-MM-DD/*.parquet
3. Data/silver/ehr_visits/utilization_outcomes/process_date=YYYY-MM-DD/*.parquet
4. Data/silver/ehr_visits/clinical_features/process_date=YYYY-MM-DD/*.parquet
5. Data/silver/_quality/ehr_visits_quality_report.json
6. Data/silver/_state/bronze_to_silver_state.json

## Step 4: Silver -> Gold Transformation

Code:
1. src/etl/silver_to_gold.py

What it does:
1. Reads only the 4 Silver tables above
2. Joins them by record_hash
3. Builds Gold marts for patient and admin KPIs

Gold tables created (only these 3):
1. doctor_patient_risk_mart
2. admin_hospital_kpi_mart
3. admin_department_kpi_mart

Main outputs:
1. Data/gold/marts/doctor_patient_risk_mart.parquet
2. Data/gold/marts/admin_hospital_kpi_mart.parquet
3. Data/gold/marts/admin_department_kpi_mart.parquet

## Step 5: Gold -> SQL Publishing

Code:
1. src/etl/gold_to_sql.py

What it does:
1. Loads Gold parquet marts into PostgreSQL schema analytics
2. Enforces primary keys

SQL tables created:
1. analytics.doctor_patient_risk_mart (PK: record_hash)
2. analytics.admin_hospital_kpi_mart (PK: as_of_date, hospitalid)
3. analytics.admin_department_kpi_mart (PK: as_of_date, hospitalid, unittype)

## Step 6: Dashboard and Monitoring

Code:
1. src/dashboard/data_quality_eda_app.py

What it does:
1. Shows KPI cards and analytical plots
2. Supports cohort, risk, and readmission analysis

## 4. Silver Table Model

## visits_core

Grain:
1. One row per visit (record_hash)

Role:
1. Identity, encounter context, demographics, timestamps, discharge status

## diagnosis_care

Grain:
1. One row per visit (record_hash)

Role:
1. Diagnosis and care plan context

## utilization_outcomes

Grain:
1. One row per visit (record_hash)

Role:
1. LOS, readmission labels, utilization, chronic burden, cost signals

## clinical_features

Grain:
1. One row per visit (record_hash)

Role:
1. Vitals and lab measurements used for risk and analysis

## 5. Gold Table Model

## doctor_patient_risk_mart

Grain:
1. One row per patient visit

Role:
1. Patient-level risk scoring and triage

## admin_hospital_kpi_mart

Grain:
1. One row per hospital per as_of_date

Role:
1. Hospital-level KPI reporting

## admin_department_kpi_mart

Grain:
1. One row per hospital + unit per as_of_date

Role:
1. Unit/department-level KPI reporting

## 6. KPI -> Table Used

Dashboard and analytics KPIs use these Gold tables:

1. Total Admissions -> doctor_patient_risk_mart
2. Total Readmissions (Any) -> doctor_patient_risk_mart
3. 30-Day Readmission Count -> doctor_patient_risk_mart
4. 30-Day Readmission Rate -> doctor_patient_risk_mart (dashboard), admin_hospital_kpi_mart, admin_department_kpi_mart
5. Avg Days to Readmission -> doctor_patient_risk_mart
6. High-Risk Patients Flagged -> doctor_patient_risk_mart
7. Any Readmission Rate -> doctor_patient_risk_mart
8. High-Risk Rate -> doctor_patient_risk_mart
9. Avg LOS (Days) -> doctor_patient_risk_mart, admin_hospital_kpi_mart, admin_department_kpi_mart
10. Median LOS (Days) -> doctor_patient_risk_mart
11. Avg LACE Score -> doctor_patient_risk_mart
12. Avg Chronic Conditions -> doctor_patient_risk_mart
13. Patients with >=2 Chronic -> doctor_patient_risk_mart
14. Avg Initial Stay Cost -> doctor_patient_risk_mart
15. Avg Care Plan Cost -> doctor_patient_risk_mart
16. Total Visits -> admin_hospital_kpi_mart, admin_department_kpi_mart
17. Unique Patients -> admin_hospital_kpi_mart
18. Avg Age Years -> admin_hospital_kpi_mart
19. Expired Count -> admin_hospital_kpi_mart, admin_department_kpi_mart
20. Mortality Rate % -> admin_hospital_kpi_mart, admin_department_kpi_mart
21. High Risk Count -> admin_hospital_kpi_mart, admin_department_kpi_mart
22. High Risk Rate % -> admin_hospital_kpi_mart, admin_department_kpi_mart
23. Readmissions Any -> admin_hospital_kpi_mart, admin_department_kpi_mart
24. Readmission Any Rate % -> admin_hospital_kpi_mart, admin_department_kpi_mart
25. Readmissions Under 30 Days -> admin_hospital_kpi_mart, admin_department_kpi_mart
26. Readmission 30D Rate % -> admin_hospital_kpi_mart, admin_department_kpi_mart

## 7. Where Data Is Stored

## Source
1. Data/synthetic_ehr.csv
2. MongoDB: healthcare_ehr.raw_patient_visits

## Bronze
1. Data/bronze/airbyte_raw/ehr_visits/load_date=.../load_hour=.../*.parquet
2. Data/bronze/_airbyte_state/ehr_visits_state.json

## Silver
1. Data/silver/ehr_visits/visits_core/process_date=.../*.parquet
2. Data/silver/ehr_visits/diagnosis_care/process_date=.../*.parquet
3. Data/silver/ehr_visits/utilization_outcomes/process_date=.../*.parquet
4. Data/silver/ehr_visits/clinical_features/process_date=.../*.parquet
5. Data/silver/_quality/ehr_visits_quality_report.json
6. Data/silver/_state/bronze_to_silver_state.json

## Gold
1. Data/gold/marts/doctor_patient_risk_mart.parquet
2. Data/gold/marts/admin_hospital_kpi_mart.parquet
3. Data/gold/marts/admin_department_kpi_mart.parquet

## SQL Serving
1. analytics.doctor_patient_risk_mart
2. analytics.admin_hospital_kpi_mart
3. analytics.admin_department_kpi_mart

## Final CSV Exports
1. Data/final/readmission_model_dataset.csv
2. Data/final/readmission_reason_summary.csv
3. Data/final/readmission_driver_summary.csv

## 8. How To Run the Project

## Prerequisites
1. Windows machine
2. Python virtual environment with requirements installed
3. Docker Desktop running (for Airflow/Postgres path)
4. Optional MongoDB local instance (if using mongo sink)

## A) Manual Local Run (step by step)

1. Generate source data
- scripts/run_ehr_simulator.bat

2. Stage source -> Bronze
- scripts/run_airbyte_staging.bat

3. Transform Bronze -> Silver
- scripts/run_bronze_to_silver.bat

4. Transform Silver -> Gold
- scripts/run_silver_to_gold.bat

5. Load Gold -> SQL
- scripts/run_gold_to_sql.bat

6. Export final CSV outputs
- scripts/run_final_output_export.bat

7. Launch dashboard
- scripts/run_quality_dashboard.bat

Dashboard URL:
1. http://localhost:8501

## B) Airflow Orchestrated Run

1. Start local Airflow stack
- scripts/start_airflow_local.ps1

2. Open Airflow UI
- http://localhost:8080

3. Trigger DAG
- ehr_simulator_to_bronze_staging

4. Stop Airflow stack when done
- scripts/stop_airflow_local.ps1

## 9. Quick Verification Checklist

After run, verify:

1. Bronze parquet files exist under Data/bronze/airbyte_raw/ehr_visits
2. Silver 4 modeled table folders contain parquet files
3. Gold 3 marts exist under Data/gold/marts
4. SQL analytics schema has 3 tables
5. Streamlit dashboard loads KPIs and charts

## 10. Notes

1. Gold must use only these Silver tables: visits_core, diagnosis_care, utilization_outcomes, clinical_features.
2. No extra Silver tables are required for Gold processing.
3. Primary keys are enforced on SQL Gold tables during load.
