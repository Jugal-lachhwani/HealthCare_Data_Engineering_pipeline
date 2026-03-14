from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("silver-to-gold")


SILVER_TABLES = ["visits_core", "diagnosis_care", "utilization_outcomes", "clinical_features"]


@dataclass
class SilverToGoldConfig:
    silver_root: str
    gold_root: str
    doctor_mart_path: str
    admin_mart_path: str
    department_mart_path: str
    state_path: str



def load_config(config_path: str) -> SilverToGoldConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return SilverToGoldConfig(
        silver_root=raw["silver_root"],
        gold_root=raw["gold_root"],
        doctor_mart_path=raw["doctor_mart_path"],
        admin_mart_path=raw["admin_mart_path"],
        department_mart_path=raw["department_mart_path"],
        state_path=raw.get("state_path", "Data/gold/_state/silver_to_gold_state.json"),
    )



def _read_state(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()

    with state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return set(payload.get("processed_silver_files", []))



def _write_state(state_path: Path, processed_files: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_silver_files": sorted(processed_files),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)



def _list_pending_silver_files(silver_root: Path, processed_files: set[str]) -> list[Path]:
    all_files = sorted(silver_root.rglob("*.parquet"))
    return [path for path in all_files if str(path.as_posix()) not in processed_files]



def _read_silver_files(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def _table_from_path(silver_root: Path, file_path: Path) -> str | None:
    try:
        rel_parts = file_path.relative_to(silver_root).parts
    except ValueError:
        return None
    if not rel_parts:
        return None
    table = rel_parts[0]
    return table if table in SILVER_TABLES else None


def _assemble_modeled_silver(table_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    core = table_frames.get("visits_core", pd.DataFrame())
    if core.empty:
        return pd.DataFrame()

    assembled = core.copy()
    for table_name in ["diagnosis_care", "utilization_outcomes", "clinical_features"]:
        part = table_frames.get(table_name, pd.DataFrame())
        if part.empty:
            continue

        cols = [col for col in part.columns if col == "record_hash" or col not in assembled.columns]
        if "record_hash" not in cols:
            continue
        assembled = assembled.merge(part[cols], on="record_hash", how="left")

    return assembled


def _read_modeled_silver_from_files(silver_root: Path, files: list[Path]) -> pd.DataFrame:
    grouped: dict[str, list[Path]] = {name: [] for name in SILVER_TABLES}
    for path in files:
        table_name = _table_from_path(silver_root, path)
        if table_name is not None:
            grouped[table_name].append(path)

    table_frames: dict[str, pd.DataFrame] = {}
    for table_name, table_files in grouped.items():
        table_frames[table_name] = _read_silver_files(table_files)

    return _assemble_modeled_silver(table_frames)


def _read_all_modeled_silver(silver_root: Path) -> pd.DataFrame:
    table_frames: dict[str, pd.DataFrame] = {}
    for table_name in SILVER_TABLES:
        table_root = silver_root / table_name
        table_frames[table_name] = _read_silver_files(sorted(table_root.rglob("*.parquet"))) if table_root.exists() else pd.DataFrame()
    return _assemble_modeled_silver(table_frames)



def _read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()



def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")



def _derive_risk_score(df: pd.DataFrame) -> pd.Series:
    age = _safe_numeric(df.get("age_years", pd.Series(index=df.index, dtype=float))).fillna(0)
    los = _safe_numeric(df.get("los", pd.Series(index=df.index, dtype=float))).fillna(0)
    visit_number = _safe_numeric(df.get("unitvisitnumber", pd.Series(index=df.index, dtype=float))).fillna(1)
    lace = _safe_numeric(df.get("LACE_Score", pd.Series(index=df.index, dtype=float))).fillna(0)
    chronic = _safe_numeric(df.get("chronic_conditions", pd.Series(index=df.index, dtype=float))).fillna(0)

    age_norm = np.clip(age / 100.0, 0, 1)
    los_norm = np.clip(los / 30.0, 0, 1)
    lace_norm = np.clip(lace / 19.0, 0, 1)
    chronic_norm = np.clip(chronic / 15.0, 0, 1)
    revisit_flag = (visit_number > 1).astype(float)

    risk = 0.2 * age_norm + 0.2 * los_norm + 0.3 * lace_norm + 0.2 * chronic_norm + 0.1 * revisit_flag
    return np.clip(risk, 0.0, 1.0)



def build_doctor_mart(silver_df: pd.DataFrame) -> pd.DataFrame:
    df = silver_df.copy()
    df["risk_score"] = _derive_risk_score(df)
    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[-0.001, 0.35, 0.7, 1.0],
        labels=["low", "medium", "high"],
    )
    df["is_high_risk"] = df["risk_score"] >= 0.7

    if "processed_at_utc" in df.columns:
        processed_ts = pd.to_datetime(df["processed_at_utc"], errors="coerce", utc=True)
        df["as_of_date"] = processed_ts.dt.strftime("%Y-%m-%d")
    else:
        df["as_of_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    selected_cols = [
        "as_of_date",
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "hospitalid",
        "wardid",
        "unittype",
        "apacheadmissiondx",
        "Condition",
        "insurance_provider",
        "ethnic_group_c",
        "gender_normalized",
        "age_years",
        "los",
        "LACE_Score",
        "chronic_conditions",
        "glucose",
        "creatinine",
        "wbc",
        "hemoglobin",
        "unitvisitnumber",
        "is_expired",
        "readmitted",
        "readmitted_under_30_days",
        "days_until_readmission",
        "risk_score",
        "risk_band",
        "is_high_risk",
        "record_hash",
        "processed_at_utc",
    ]

    return df[[col for col in selected_cols if col in df.columns]].copy()



def build_admin_marts(doctor_mart: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = doctor_mart.copy()

    if "as_of_date" not in base.columns:
        base["as_of_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    hospital_group = ["as_of_date", "hospitalid"]
    hospital_kpi = (
        base.groupby(hospital_group, dropna=False)
        .agg(
            total_visits=("patientunitstayid", "count"),
            unique_patients=("uniquepid", "nunique"),
            avg_age_years=("age_years", "mean"),
            avg_los_days=("los", "mean"),
            expired_count=("is_expired", "sum"),
            high_risk_count=("is_high_risk", "sum"),
            readmissions_any=("readmitted", "sum"),
            readmissions_under_30_days=("readmitted_under_30_days", "sum"),
        )
        .reset_index()
    )
    hospital_kpi["mortality_rate_pct"] = (hospital_kpi["expired_count"] / hospital_kpi["total_visits"] * 100).round(2)
    hospital_kpi["high_risk_rate_pct"] = (hospital_kpi["high_risk_count"] / hospital_kpi["total_visits"] * 100).round(2)
    hospital_kpi["readmission_any_rate_pct"] = (hospital_kpi["readmissions_any"] / hospital_kpi["total_visits"] * 100).round(2)
    hospital_kpi["readmission_30d_rate_pct"] = (hospital_kpi["readmissions_under_30_days"] / hospital_kpi["total_visits"] * 100).round(2)

    dept_group = ["as_of_date", "hospitalid", "unittype"]
    department_kpi = (
        base.groupby(dept_group, dropna=False)
        .agg(
            total_visits=("patientunitstayid", "count"),
            avg_los_days=("los", "mean"),
            expired_count=("is_expired", "sum"),
            high_risk_count=("is_high_risk", "sum"),
            readmissions_any=("readmitted", "sum"),
            readmissions_under_30_days=("readmitted_under_30_days", "sum"),
        )
        .reset_index()
    )
    department_kpi["mortality_rate_pct"] = (department_kpi["expired_count"] / department_kpi["total_visits"] * 100).round(2)
    department_kpi["high_risk_rate_pct"] = (department_kpi["high_risk_count"] / department_kpi["total_visits"] * 100).round(2)
    department_kpi["readmission_any_rate_pct"] = (department_kpi["readmissions_any"] / department_kpi["total_visits"] * 100).round(2)
    department_kpi["readmission_30d_rate_pct"] = (department_kpi["readmissions_under_30_days"] / department_kpi["total_visits"] * 100).round(2)

    return hospital_kpi, department_kpi



def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)



def _merge_doctor_mart(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = incoming.copy()
    elif incoming.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, incoming], ignore_index=True)

    if merged.empty:
        return merged

    if "record_hash" in merged.columns and merged["record_hash"].notna().all():
        return merged.drop_duplicates(subset=["record_hash"], keep="last").reset_index(drop=True)

    dedupe_keys = [col for col in ["patientunitstayid", "patienthealthsystemstayid", "processed_at_utc"] if col in merged.columns]
    if dedupe_keys:
        return merged.drop_duplicates(subset=dedupe_keys, keep="last").reset_index(drop=True)

    return merged.drop_duplicates(keep="last").reset_index(drop=True)



def run_job(cfg: SilverToGoldConfig) -> int:
    silver_root = Path(cfg.silver_root)
    state_path = Path(cfg.state_path)
    doctor_path = Path(cfg.doctor_mart_path)
    admin_path = Path(cfg.admin_mart_path)
    department_path = Path(cfg.department_mart_path)

    processed_files = _read_state(state_path)
    pending_files = _list_pending_silver_files(silver_root, processed_files)

    existing_doctor = _read_parquet_if_exists(doctor_path)
    incoming_doctor = pd.DataFrame()

    if pending_files:
        silver_df = _read_modeled_silver_from_files(silver_root, pending_files)
        if not silver_df.empty:
            incoming_doctor = build_doctor_mart(silver_df)
        processed_files.update(str(path.as_posix()) for path in pending_files)
        _write_state(state_path, processed_files)

    # Rebuild from all modeled silver tables when doctor mart does not yet exist.
    if existing_doctor.empty and incoming_doctor.empty:
        silver_all = _read_all_modeled_silver(silver_root)
        if not silver_all.empty:
            incoming_doctor = build_doctor_mart(silver_all)

    doctor_mart = _merge_doctor_mart(existing_doctor, incoming_doctor)
    if doctor_mart.empty:
        logger.info("No Silver files found. Skipping Gold build.")
        return 0

    admin_mart, department_mart = build_admin_marts(doctor_mart)

    _write_parquet(doctor_mart, doctor_path)
    _write_parquet(admin_mart, admin_path)
    _write_parquet(department_mart, department_path)

    logger.info(
        "Gold marts built: doctor_rows=%s admin_rows=%s department_rows=%s pending_silver_files=%s",
        len(doctor_mart),
        len(admin_mart),
        len(department_mart),
        len(pending_files),
    )
    return len(doctor_mart)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver to Gold marts for persona analytics")
    parser.add_argument("--config", default="config/silver_to_gold_config.json", help="Path to Gold config")
    return parser.parse_args()



def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config)
    run_job(cfg)


if __name__ == "__main__":
    main()
