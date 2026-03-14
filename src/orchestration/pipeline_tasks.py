from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from etl.bronze_to_silver import load_config as load_bronze_to_silver_config
from etl.bronze_to_silver import run_job as run_bronze_to_silver_job
from etl.final_output_export import load_config as load_final_output_config
from etl.final_output_export import run_job as run_final_output_job
from etl.gold_to_sql import load_config as load_gold_to_sql_config
from etl.gold_to_sql import run_job as run_gold_to_sql_job
from etl.silver_to_gold import load_config as load_silver_to_gold_config
from etl.silver_to_gold import run_job as run_silver_to_gold_job
from ingestion.airbyte_style_staging import load_staging_config, run_sync
from simulator.config import load_config
from simulator.run_simulator import run_once


REPO_ROOT = Path(__file__).resolve().parents[2]



def _resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)



def _load_env() -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)



def generate_synthetic_ehr_for_pipeline(batch_size: int = 200, sink: str = "csv") -> int:
    _load_env()
    cfg = load_config()

    cfg.batch_size = batch_size
    cfg.sink = sink
    cfg.template_csv_path = _resolve_path(cfg.template_csv_path)
    cfg.output_csv_path = _resolve_path(cfg.output_csv_path)

    return run_once(cfg)



def stage_ehr_to_bronze_for_pipeline(config_path: str | None = None) -> int:
    _load_env()

    config_env = os.getenv("AIRBYTE_STAGING_CONFIG", "config/airbyte_staging_config.json")
    raw_config_path = config_path if config_path else config_env
    resolved_config_path = _resolve_path(raw_config_path)

    cfg = load_staging_config(resolved_config_path)
    source_override = os.getenv("AIRBYTE_SOURCE_CSV")
    if source_override:
        cfg.source_csv = source_override

    cfg.source_csv = _resolve_path(cfg.source_csv)
    cfg.destination_root = _resolve_path(cfg.destination_root)
    cfg.state_path = _resolve_path(cfg.state_path)

    return run_sync(cfg)


def transform_bronze_to_silver_for_pipeline(config_path: str | None = None) -> int:
    _load_env()

    config_env = os.getenv("BRONZE_TO_SILVER_CONFIG", "config/bronze_to_silver_config.json")
    raw_config_path = config_path if config_path else config_env
    resolved_config_path = _resolve_path(raw_config_path)

    cfg = load_bronze_to_silver_config(resolved_config_path)
    cfg.source_root = _resolve_path(cfg.source_root)
    cfg.destination_root = _resolve_path(cfg.destination_root)
    cfg.quality_report_path = _resolve_path(cfg.quality_report_path)
    cfg.state_path = _resolve_path(cfg.state_path)

    return run_bronze_to_silver_job(cfg)


def transform_silver_to_gold_for_pipeline(config_path: str | None = None) -> int:
    _load_env()

    config_env = os.getenv("SILVER_TO_GOLD_CONFIG", "config/silver_to_gold_config.json")
    raw_config_path = config_path if config_path else config_env
    resolved_config_path = _resolve_path(raw_config_path)

    cfg = load_silver_to_gold_config(resolved_config_path)
    cfg.silver_root = _resolve_path(cfg.silver_root)
    cfg.gold_root = _resolve_path(cfg.gold_root)
    cfg.doctor_mart_path = _resolve_path(cfg.doctor_mart_path)
    cfg.admin_mart_path = _resolve_path(cfg.admin_mart_path)
    cfg.department_mart_path = _resolve_path(cfg.department_mart_path)
    cfg.state_path = _resolve_path(cfg.state_path)

    return run_silver_to_gold_job(cfg)


def export_final_outputs_for_pipeline(config_path: str | None = None) -> int:
    _load_env()

    config_env = os.getenv("FINAL_OUTPUT_CONFIG", "config/final_output_config.json")
    raw_config_path = config_path if config_path else config_env
    resolved_config_path = _resolve_path(raw_config_path)

    cfg = load_final_output_config(resolved_config_path)
    cfg.silver_root = _resolve_path(cfg.silver_root)
    cfg.model_dataset_csv_path = _resolve_path(cfg.model_dataset_csv_path)
    cfg.reason_summary_csv_path = _resolve_path(cfg.reason_summary_csv_path)
    cfg.driver_summary_csv_path = _resolve_path(cfg.driver_summary_csv_path)

    return run_final_output_job(cfg)


def load_gold_to_sql_for_pipeline(config_path: str | None = None) -> int:
    _load_env()

    config_env = os.getenv("GOLD_TO_SQL_CONFIG", "config/gold_to_sql_config.json")
    raw_config_path = config_path if config_path else config_env
    resolved_config_path = _resolve_path(raw_config_path)

    cfg = load_gold_to_sql_config(resolved_config_path)
    cfg.doctor_mart_path = _resolve_path(cfg.doctor_mart_path)
    cfg.admin_mart_path = _resolve_path(cfg.admin_mart_path)
    cfg.department_mart_path = _resolve_path(cfg.department_mart_path)

    return run_gold_to_sql_job(cfg)
