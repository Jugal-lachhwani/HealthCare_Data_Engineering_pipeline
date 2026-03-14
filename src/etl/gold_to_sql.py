from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("gold-to-sql")


IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class GoldToSqlConfig:
    doctor_mart_path: str
    admin_mart_path: str
    department_mart_path: str
    schema_name: str
    doctor_table: str
    admin_table: str
    department_table: str



def load_config(config_path: str) -> GoldToSqlConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return GoldToSqlConfig(
        doctor_mart_path=raw["doctor_mart_path"],
        admin_mart_path=raw["admin_mart_path"],
        department_mart_path=raw["department_mart_path"],
        schema_name=raw.get("schema_name", "analytics"),
        doctor_table=raw.get("doctor_table", "doctor_patient_risk_mart"),
        admin_table=raw.get("admin_table", "admin_hospital_kpi_mart"),
        department_table=raw.get("department_table", "admin_department_kpi_mart"),
    )



def build_sqlalchemy_url() -> str:
    explicit_url = os.getenv("GOLD_SQLALCHEMY_URL")
    if explicit_url:
        return explicit_url

    user = os.getenv("GOLD_DB_USER", "airflow")
    password = os.getenv("GOLD_DB_PASSWORD", "airflow")
    host = os.getenv("GOLD_DB_HOST", "localhost")
    port = os.getenv("GOLD_DB_PORT", "5432")
    db_name = os.getenv("GOLD_DB_NAME", "airflow")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"



def _read_mart(path: str) -> pd.DataFrame:
    mart_path = Path(path)
    if not mart_path.exists():
        raise FileNotFoundError(f"Gold mart not found: {mart_path}")
    return pd.read_parquet(mart_path)



def _normalize_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype) == "bool":
            out[col] = out[col].astype(bool)
        if str(out[col].dtype) == "object":
            out[col] = out[col].where(out[col].notna(), None)
    return out



def _write_table(df: pd.DataFrame, schema_name: str, table_name: str, engine) -> None:
    ready = _normalize_for_sql(df)
    ready.to_sql(table_name, engine, schema=schema_name, if_exists="replace", index=False, method="multi", chunksize=1000)


def _validate_ident(value: str) -> str:
    if not IDENT_RE.match(value):
        raise ValueError(f"Unsafe SQL identifier: {value}")
    return value


def _quote_ident(value: str) -> str:
    safe = _validate_ident(value)
    return f'"{safe}"'


def _validate_primary_key_columns(df: pd.DataFrame, table_name: str, pk_cols: list[str]) -> None:
    missing = [col for col in pk_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot add primary key for {table_name}. Missing columns: {missing}")

    null_mask = df[pk_cols].isna().any(axis=1)
    if bool(null_mask.any()):
        null_count = int(null_mask.sum())
        raise ValueError(
            f"Cannot add primary key for {table_name}. Found {null_count} rows with NULL values in PK columns {pk_cols}."
        )

    duplicate_mask = df.duplicated(subset=pk_cols, keep=False)
    if bool(duplicate_mask.any()):
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            f"Cannot add primary key for {table_name}. Found {duplicate_count} rows with duplicate PK values for columns {pk_cols}."
        )


def _add_primary_key(conn, schema_name: str, table_name: str, pk_cols: list[str]) -> None:
    quoted_schema = _quote_ident(schema_name)
    quoted_table = _quote_ident(table_name)
    quoted_cols = ", ".join(_quote_ident(col) for col in pk_cols)

    constraint_name = f"{table_name}_pk"
    if len(constraint_name) > 63:
        constraint_name = f"{table_name[:60]}_pk"
    quoted_constraint = _quote_ident(constraint_name)

    conn.execute(text(f"ALTER TABLE {quoted_schema}.{quoted_table} ADD CONSTRAINT {quoted_constraint} PRIMARY KEY ({quoted_cols})"))


def _write_table_with_primary_key(df: pd.DataFrame, schema_name: str, table_name: str, pk_cols: list[str], engine) -> None:
    _validate_primary_key_columns(df, table_name, pk_cols)
    _write_table(df, schema_name, table_name, engine)
    with engine.begin() as conn:
        _add_primary_key(conn, schema_name, table_name, pk_cols)



def run_job(cfg: GoldToSqlConfig, sqlalchemy_url: str | None = None) -> int:
    db_url = sqlalchemy_url if sqlalchemy_url else build_sqlalchemy_url()
    engine = create_engine(db_url)

    doctor_df = _read_mart(cfg.doctor_mart_path)
    admin_df = _read_mart(cfg.admin_mart_path)
    department_df = _read_mart(cfg.department_mart_path)

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {cfg.schema_name}"))

    _write_table_with_primary_key(
        doctor_df,
        cfg.schema_name,
        cfg.doctor_table,
        ["record_hash"],
        engine,
    )
    _write_table_with_primary_key(
        admin_df,
        cfg.schema_name,
        cfg.admin_table,
        ["as_of_date", "hospitalid"],
        engine,
    )
    _write_table_with_primary_key(
        department_df,
        cfg.schema_name,
        cfg.department_table,
        ["as_of_date", "hospitalid", "unittype"],
        engine,
    )

    logger.info(
        "Loaded Gold marts to SQL schema=%s doctor_rows=%s admin_rows=%s department_rows=%s",
        cfg.schema_name,
        len(doctor_df),
        len(admin_df),
        len(department_df),
    )
    return len(doctor_df)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Gold marts into SQL tables")
    parser.add_argument("--config", default="config/gold_to_sql_config.json", help="Path to Gold SQL config")
    parser.add_argument("--sqlalchemy-url", default=None, help="Optional override SQLAlchemy connection URL")
    return parser.parse_args()



def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config)
    run_job(cfg, sqlalchemy_url=args.sqlalchemy_url)


if __name__ == "__main__":
    main()
