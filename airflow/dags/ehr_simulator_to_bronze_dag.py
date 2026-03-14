from __future__ import annotations
# pyright: reportMissingImports=false

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator


REPO_ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from orchestration.pipeline_tasks import generate_synthetic_ehr_for_pipeline, stage_ehr_to_bronze_for_pipeline
from orchestration.pipeline_tasks import export_final_outputs_for_pipeline
from orchestration.pipeline_tasks import load_gold_to_sql_for_pipeline
from orchestration.pipeline_tasks import transform_bronze_to_silver_for_pipeline
from orchestration.pipeline_tasks import transform_silver_to_gold_for_pipeline


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


default_args = {
    "owner": "data-eng",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="ehr_simulator_to_bronze_staging",
    description="Scheduled healthcare pipeline: optional simulation, Bronze, Silver, Gold, final outputs, and SQL serving.",
    default_args=default_args,
    start_date=datetime(2026, 3, 14),
    schedule=os.getenv("PIPELINE_SCHEDULE_CRON", "*/15 * * * *"),
    catchup=False,
    max_active_runs=int(os.getenv("PIPELINE_MAX_ACTIVE_RUNS", "1")),
    tags=["healthcare", "ingestion", "bronze", "silver", "gold", "sql", "airbyte-style"],
) as dag:
    simulator_enabled = _env_bool("PIPELINE_SIMULATOR_ENABLED", True)
    simulator_batch_size = int(os.getenv("PIPELINE_SIMULATOR_BATCH_SIZE", "200"))
    simulator_sink = os.getenv("PIPELINE_SIMULATOR_SINK", "csv")

    if simulator_enabled:
        source_task = PythonOperator(
            task_id="generate_synthetic_csv_batch",
            python_callable=generate_synthetic_ehr_for_pipeline,
            op_kwargs={"batch_size": simulator_batch_size, "sink": simulator_sink},
        )
    else:
        source_task = EmptyOperator(task_id="skip_simulator_use_existing_source")

    stage_csv_to_bronze_raw = PythonOperator(
        task_id="stage_csv_to_bronze_raw",
        python_callable=stage_ehr_to_bronze_for_pipeline,
    )

    transform_bronze_to_silver = PythonOperator(
        task_id="transform_bronze_to_silver",
        python_callable=transform_bronze_to_silver_for_pipeline,
    )

    transform_silver_to_gold = PythonOperator(
        task_id="transform_silver_to_gold",
        python_callable=transform_silver_to_gold_for_pipeline,
    )

    export_final_outputs = PythonOperator(
        task_id="export_final_outputs",
        python_callable=export_final_outputs_for_pipeline,
    )

    load_gold_to_sql = PythonOperator(
        task_id="load_gold_to_sql",
        python_callable=load_gold_to_sql_for_pipeline,
    )

    source_task >> stage_csv_to_bronze_raw >> transform_bronze_to_silver >> transform_silver_to_gold >> export_final_outputs >> load_gold_to_sql
