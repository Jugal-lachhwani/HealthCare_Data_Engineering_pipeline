from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("bronze-to-silver")


@dataclass
class BronzeToSilverConfig:
    source_root: str
    destination_root: str
    quality_report_path: str
    state_path: str
    required_columns: list[str]


TIME_24H_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$")



def load_config(config_path: str) -> BronzeToSilverConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return BronzeToSilverConfig(
        source_root=raw["source_root"],
        destination_root=raw["destination_root"],
        quality_report_path=raw["quality_report_path"],
        state_path=raw["state_path"],
        required_columns=list(raw.get("required_columns", [])),
    )



def _read_state(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()

    with state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return set(payload.get("processed_files", []))



def _write_state(state_path: Path, processed_files: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_files": sorted(processed_files),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)



def _list_unprocessed_parquet_files(source_root: Path, processed_files: set[str]) -> list[Path]:
    all_files = sorted(source_root.rglob("*.parquet"))
    pending = [f for f in all_files if str(f.as_posix()) not in processed_files]
    return pending



def _normalize_gender(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in {"male", "m"}:
        return "male"
    if text in {"female", "f"}:
        return "female"
    return "unknown"



def _to_nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")



def _to_nullable_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")



def _sanitize_time(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.where(values.str.match(TIME_24H_RE), None)


def _build_record_hash(df: pd.DataFrame) -> pd.Series:
    key_cols = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "hospitalid",
        "wardid",
        "uniquepid",
        "_airbyte_emitted_at",
    ]

    def _hash_row(row: pd.Series) -> str:
        raw = "|".join(str(row.get(col, "")) for col in key_cols)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return df.apply(_hash_row, axis=1)


def _hash_to_unit_interval(key_series: pd.Series, salt: str) -> pd.Series:
    def _to_unit(value: Any) -> float:
        token = f"{value}|{salt}"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16) / 4294967295.0

    return key_series.fillna("").astype(str).map(_to_unit)


def _enrich_missing_model_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    key_series = working.get("record_hash", pd.Series(index=working.index, dtype="object"))
    u1 = _hash_to_unit_interval(key_series, "u1")
    u2 = _hash_to_unit_interval(key_series, "u2")
    u3 = _hash_to_unit_interval(key_series, "u3")
    u4 = _hash_to_unit_interval(key_series, "u4")

    if "ethnic_group_c" not in working.columns:
        working["ethnic_group_c"] = pd.NA
    ethnicity_norm = working.get("ethnicity", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.lower()
    ethnic_map = {
        "hispanic": "Hispanic",
        "asian": "Asian",
        "african": "African American",
        "black": "African American",
        "caucasian": "White",
        "white": "White",
    }
    derived_ethnic_group = pd.Series("Other", index=working.index)
    for key, label in ethnic_map.items():
        derived_ethnic_group = derived_ethnic_group.where(~ethnicity_norm.str.contains(key, regex=False), label)
    working["ethnic_group_c"] = working["ethnic_group_c"].fillna(derived_ethnic_group)

    if "insurance_provider" not in working.columns:
        working["insurance_provider"] = pd.NA
    age_years = pd.to_numeric(working.get("age_years", pd.Series(index=working.index, dtype=float)), errors="coerce").fillna(0)
    insurance = pd.Series("Commercial", index=working.index)
    insurance = insurance.where(~(age_years >= 65), "Medicare")
    insurance = insurance.where(~((age_years < 30) & (u1 < 0.4)), "Medicaid")
    insurance = insurance.where(~(u1 < 0.03), "Self-pay")
    working["insurance_provider"] = working["insurance_provider"].fillna(insurance)

    if "Condition" not in working.columns:
        working["Condition"] = pd.NA
    apache_dx = working.get("apacheadmissiondx", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip()
    working["Condition"] = working["Condition"].fillna(apache_dx.replace("", "General Medical"))

    if "care_plan_following_discharge" not in working.columns:
        working["care_plan_following_discharge"] = pd.NA
    care_plan = pd.Series("Standard follow-up in 7 days", index=working.index)
    care_plan = care_plan.where(~(u2 < 0.35), "Medication reconciliation and PCP follow-up")
    care_plan = care_plan.where(~(u2 < 0.15), "Home health + chronic disease management")
    working["care_plan_following_discharge"] = working["care_plan_following_discharge"].fillna(care_plan)

    if "los" not in working.columns:
        working["los"] = pd.NA
    hospital_discharge_offset = pd.to_numeric(
        working.get("hospitaldischargeoffset", pd.Series(index=working.index, dtype=float)),
        errors="coerce",
    )
    derived_los = (hospital_discharge_offset / 1440.0).clip(lower=0.5, upper=21).round(2)
    derived_los = derived_los.fillna((1.5 + u1 * 5.5).round(2))
    working["los"] = pd.to_numeric(working["los"], errors="coerce").fillna(derived_los)

    if "chronic_conditions" not in working.columns:
        working["chronic_conditions"] = pd.NA
    chronic = (u2 * 5).round().astype("Int64")
    chronic = (chronic + (age_years >= 65).astype("Int64")).clip(lower=0, upper=8)
    working["chronic_conditions"] = pd.to_numeric(working["chronic_conditions"], errors="coerce").astype("Int64")
    working["chronic_conditions"] = working["chronic_conditions"].fillna(chronic)

    binary_features = [
        "diabetes",
        "obesity",
        "anxiety",
        "depression",
        "dementia",
        "drugabuse",
        "mooddisorder",
        "tobacco_user",
    ]
    for feature in binary_features:
        if feature not in working.columns:
            working[feature] = pd.NA
    working["diabetes"] = pd.to_numeric(working["diabetes"], errors="coerce").fillna((u1 < 0.23).astype(int)).astype("Int64")
    working["obesity"] = pd.to_numeric(working["obesity"], errors="coerce").fillna((u2 < 0.28).astype(int)).astype("Int64")
    working["anxiety"] = pd.to_numeric(working["anxiety"], errors="coerce").fillna((u3 < 0.20).astype(int)).astype("Int64")
    working["depression"] = pd.to_numeric(working["depression"], errors="coerce").fillna((u4 < 0.18).astype(int)).astype("Int64")
    working["dementia"] = pd.to_numeric(working["dementia"], errors="coerce").fillna(((age_years >= 70) & (u1 < 0.22)).astype(int)).astype("Int64")
    working["drugabuse"] = pd.to_numeric(working["drugabuse"], errors="coerce").fillna((u2 < 0.12).astype(int)).astype("Int64")
    working["mooddisorder"] = pd.to_numeric(working["mooddisorder"], errors="coerce").fillna((u3 < 0.15).astype(int)).astype("Int64")
    working["tobacco_user"] = pd.to_numeric(working["tobacco_user"], errors="coerce").fillna((u4 < 0.27).astype(int)).astype("Int64")

    if "ed_visits" not in working.columns:
        working["ed_visits"] = pd.NA
    if "ip_visits" not in working.columns:
        working["ip_visits"] = pd.NA
    working["ed_visits"] = pd.to_numeric(working["ed_visits"], errors="coerce").fillna((u1 * 4).round()).astype("Int64")
    working["ip_visits"] = pd.to_numeric(working["ip_visits"], errors="coerce").fillna((u2 * 3).round()).astype("Int64")

    if "LACE_Score" not in working.columns:
        working["LACE_Score"] = pd.NA
    lace = (
        (working["los"].clip(lower=0, upper=14) * 0.7)
        + (working["ed_visits"].astype(float) * 1.2)
        + (working["chronic_conditions"].astype(float) * 1.4)
        + ((age_years >= 65).astype(float) * 2.0)
        + (u3 * 2.0)
    ).round().clip(lower=0, upper=19)
    working["LACE_Score"] = pd.to_numeric(working["LACE_Score"], errors="coerce").fillna(lace)

    if "bp_systolic" not in working.columns:
        working["bp_systolic"] = pd.NA
    if "bp_diastolic" not in working.columns:
        working["bp_diastolic"] = pd.NA
    if "pulse" not in working.columns:
        working["pulse"] = pd.NA
    if "respirations" not in working.columns:
        working["respirations"] = pd.NA
    if "temperature" not in working.columns:
        working["temperature"] = pd.NA
    if "bmi" not in working.columns:
        working["bmi"] = pd.NA

    working["bp_systolic"] = pd.to_numeric(working["bp_systolic"], errors="coerce").fillna((108 + u1 * 42).round(0))
    working["bp_diastolic"] = pd.to_numeric(working["bp_diastolic"], errors="coerce").fillna((62 + u2 * 28).round(0))
    working["pulse"] = pd.to_numeric(working["pulse"], errors="coerce").fillna((58 + u3 * 52).round(0))
    working["respirations"] = pd.to_numeric(working["respirations"], errors="coerce").fillna((12 + u4 * 14).round(0))
    working["temperature"] = pd.to_numeric(working["temperature"], errors="coerce").fillna((36.1 + u1 * 2.1).round(1))

    admission_weight = pd.to_numeric(working.get("admissionweight", pd.Series(index=working.index, dtype=float)), errors="coerce")
    admission_height = pd.to_numeric(working.get("admissionheight", pd.Series(index=working.index, dtype=float)), errors="coerce")
    derived_bmi = (admission_weight / ((admission_height / 100.0) ** 2)).replace([float("inf"), float("-inf")], pd.NA)
    working["bmi"] = pd.to_numeric(working["bmi"], errors="coerce").fillna(derived_bmi.round(1)).fillna((20 + u2 * 15).round(1))

    if "glucose" not in working.columns:
        working["glucose"] = pd.NA
    if "creatinine" not in working.columns:
        working["creatinine"] = pd.NA
    if "wbc" not in working.columns:
        working["wbc"] = pd.NA
    if "hemoglobin" not in working.columns:
        working["hemoglobin"] = pd.NA
    if "potassium" not in working.columns:
        working["potassium"] = pd.NA
    if "sodium" not in working.columns:
        working["sodium"] = pd.NA
    if "calcium" not in working.columns:
        working["calcium"] = pd.NA

    chronic_float = working["chronic_conditions"].astype(float)
    working["glucose"] = pd.to_numeric(working["glucose"], errors="coerce").fillna((90 + chronic_float * 6 + u1 * 35).round(1))
    working["creatinine"] = pd.to_numeric(working["creatinine"], errors="coerce").fillna((0.7 + chronic_float * 0.08 + u2 * 0.7).round(2))
    working["wbc"] = pd.to_numeric(working["wbc"], errors="coerce").fillna((4.2 + u3 * 9.8).round(2))
    working["hemoglobin"] = pd.to_numeric(working["hemoglobin"], errors="coerce").fillna((10.5 + u4 * 5.5).round(2))
    working["potassium"] = pd.to_numeric(working["potassium"], errors="coerce").fillna((3.4 + u1 * 1.8).round(2))
    working["sodium"] = pd.to_numeric(working["sodium"], errors="coerce").fillna((133 + u2 * 12).round(1))
    working["calcium"] = pd.to_numeric(working["calcium"], errors="coerce").fillna((8.2 + u3 * 2.1).round(2))

    if "cost_of_initial_stay" not in working.columns:
        working["cost_of_initial_stay"] = pd.NA
    if "care_plan_costs" not in working.columns:
        working["care_plan_costs"] = pd.NA
    working["cost_of_initial_stay"] = pd.to_numeric(working["cost_of_initial_stay"], errors="coerce").fillna((3500 + working["los"] * 1200 + chronic_float * 420 + u4 * 900).round(2))
    working["care_plan_costs"] = pd.to_numeric(working["care_plan_costs"], errors="coerce").fillna((180 + chronic_float * 95 + u3 * 220).round(2))

    if "readmitted" not in working.columns:
        working["readmitted"] = pd.NA
    if "days_until_readmission" not in working.columns:
        working["days_until_readmission"] = pd.NA
    if "readmitted_under_30_days" not in working.columns:
        working["readmitted_under_30_days"] = pd.NA

    readmit_prob = (0.08 + (working["LACE_Score"].astype(float) / 40.0) + (chronic_float / 25.0)).clip(upper=0.65)
    derived_readmitted = (u4 < readmit_prob).astype(int)
    readmitted_numeric = pd.to_numeric(working["readmitted"], errors="coerce")
    working["readmitted"] = readmitted_numeric.fillna(derived_readmitted).astype("Int64")

    days_existing = pd.to_numeric(working["days_until_readmission"], errors="coerce")
    days_derived = (1 + (u1 * 59).round()).astype(float)
    days_derived = days_derived.where(working["readmitted"].astype(int) == 1, 0)
    working["days_until_readmission"] = days_existing.fillna(days_derived)

    readmit_30_existing = pd.to_numeric(working["readmitted_under_30_days"], errors="coerce")
    readmit_30_derived = ((working["readmitted"].astype(int) == 1) & (working["days_until_readmission"].astype(float) <= 30)).astype(int)
    working["readmitted_under_30_days"] = readmit_30_existing.fillna(readmit_30_derived).astype("Int64")

    return working


def transform_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    # Normalize and cast commonly used analytical columns.
    working["gender_normalized"] = working["gender"].map(_normalize_gender)
    working["age_years"] = _to_nullable_int(working["age"]).clip(lower=0, upper=115)

    int_cols = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "hospitalid",
        "wardid",
        "hospitaldischargeyear",
        "hospitaladmitoffset",
        "hospitaldischargeoffset",
        "unitdischargeoffset",
        "unitvisitnumber",
        "ed_visits",
        "ip_visits",
        "chronic_conditions",
        "diabetes",
        "obesity",
        "anxiety",
        "depression",
        "dementia",
        "drugabuse",
        "mooddisorder",
        "tobacco_user",
        "readmitted",
        "readmitted_under_30_days",
    ]
    for col in int_cols:
        if col in working.columns:
            working[col] = _to_nullable_int(working[col])

    float_cols = [
        "admissionheight",
        "admissionweight",
        "dischargeweight",
        "los",
        "total_days",
        "bp_systolic",
        "bp_diastolic",
        "pulse",
        "respirations",
        "temperature",
        "bmi",
        "glucose",
        "creatinine",
        "wbc",
        "hemoglobin",
        "potassium",
        "sodium",
        "calcium",
        "artbloodgas",
        "LACE_Score",
        "cost_of_initial_stay",
        "care_plan_costs",
        "days_until_readmission",
        "readmission_total_days",
    ]
    for col in float_cols:
        if col in working.columns:
            working[col] = _to_nullable_float(working[col])

    time_cols = ["hospitaladmittime24", "hospitaldischargetime24", "unitadmittime24", "unitdischargetime24"]
    for col in time_cols:
        if col in working.columns:
            working[col] = _sanitize_time(working[col])

    working["hospital_discharge_status_normalized"] = (
        working["hospitaldischargestatus"].fillna("").astype(str).str.strip().str.lower()
    )
    working["unit_discharge_status_normalized"] = (
        working["unitdischargestatus"].fillna("").astype(str).str.strip().str.lower()
    )

    working["is_expired"] = working["hospital_discharge_status_normalized"].eq("expired")
    working["estimated_unit_los_hours"] = (
        _to_nullable_float(working["unitdischargeoffset"]).fillna(0) / 60.0
    )

    for date_col in ["admit_day", "discharge_day", "readmission_admit_day", "readmission_discharge_day"]:
        if date_col in working.columns:
            working[date_col] = pd.to_datetime(working[date_col], errors="coerce", dayfirst=True)

    if "_airbyte_emitted_at" in working.columns:
        ts = pd.to_datetime(working["_airbyte_emitted_at"], errors="coerce", utc=True)
        working["silver_processed_date"] = ts.dt.strftime("%Y-%m-%d")
    else:
        working["silver_processed_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    working["record_hash"] = _build_record_hash(working)
    working = _enrich_missing_model_features(working)
    working["processed_at_utc"] = datetime.now(timezone.utc).isoformat()

    return working


def split_silver_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    key_cols = ["record_hash", "processed_at_utc", "_airbyte_emitted_at", "silver_processed_date"]

    visits_core_cols = key_cols + [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "hospitalid",
        "wardid",
        "hospitaldischargeyear",
        "hospitaladmitoffset",
        "hospitaldischargeoffset",
        "unitdischargeoffset",
        "unitvisitnumber",
        "unittype",
        "unitstaytype",
        "gender",
        "gender_normalized",
        "age",
        "age_years",
        "ethnicity",
        "ethnic_group_c",
        "insurance_provider",
        "hospital_discharge_status_normalized",
        "unit_discharge_status_normalized",
        "is_expired",
        "estimated_unit_los_hours",
        "admit_day",
        "discharge_day",
        "readmission_admit_day",
        "readmission_discharge_day",
    ]

    diagnosis_care_cols = key_cols + [
        "apacheadmissiondx",
        "Condition",
        "care_plan_following_discharge",
        "marital_status_c",
    ]

    utilization_outcomes_cols = key_cols + [
        "los",
        "total_days",
        "LACE_Score",
        "ed_visits",
        "ip_visits",
        "chronic_conditions",
        "diabetes",
        "obesity",
        "anxiety",
        "depression",
        "dementia",
        "drugabuse",
        "mooddisorder",
        "tobacco_user",
        "readmitted",
        "readmitted_under_30_days",
        "days_until_readmission",
        "readmission_total_days",
        "cost_of_initial_stay",
        "care_plan_costs",
    ]

    clinical_features_cols = key_cols + [
        "admissionheight",
        "admissionweight",
        "dischargeweight",
        "bp_systolic",
        "bp_diastolic",
        "pulse",
        "respirations",
        "temperature",
        "bmi",
        "glucose",
        "creatinine",
        "wbc",
        "hemoglobin",
        "potassium",
        "sodium",
        "calcium",
        "artbloodgas",
    ]

    def _pick(cols: list[str]) -> pd.DataFrame:
        selected = list(dict.fromkeys(col for col in cols if col in df.columns))
        return df[selected].copy()

    tables = {
        "visits_core": _pick(visits_core_cols),
        "diagnosis_care": _pick(diagnosis_care_cols),
        "utilization_outcomes": _pick(utilization_outcomes_cols),
        "clinical_features": _pick(clinical_features_cols),
    }

    # Keep key uniqueness per table to simplify downstream joins.
    for name, table_df in tables.items():
        if "record_hash" in table_df.columns:
            tables[name] = table_df.drop_duplicates(subset=["record_hash"], keep="last").reset_index(drop=True)

    return tables


def build_quality_report(df: pd.DataFrame, required_columns: list[str], input_files: list[str]) -> dict[str, Any]:
    total_rows = int(len(df))
    null_counts = {}

    for col in required_columns:
        if col in df.columns:
            null_counts[col] = int(df[col].isna().sum())
        else:
            null_counts[col] = total_rows

    duplicate_unitstay = int(df["patientunitstayid"].duplicated().sum()) if "patientunitstayid" in df.columns else total_rows

    readmitted_count = int(df["readmitted"].fillna(0).sum()) if "readmitted" in df.columns else 0
    readmitted_30_count = int(df["readmitted_under_30_days"].fillna(0).sum()) if "readmitted_under_30_days" in df.columns else 0

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": total_rows,
        "input_files": input_files,
        "required_column_null_counts": null_counts,
        "duplicate_patientunitstayid": duplicate_unitstay,
        "expired_count": int(df["is_expired"].sum()) if "is_expired" in df.columns else 0,
        "readmitted_count": readmitted_count,
        "readmitted_under_30_days_count": readmitted_30_count,
        "readmitted_rate_pct": round((readmitted_count / total_rows * 100), 2) if total_rows else 0.0,
        "readmitted_under_30_days_rate_pct": round((readmitted_30_count / total_rows * 100), 2) if total_rows else 0.0,
    }



def init_quality_accumulator(required_columns: list[str]) -> dict[str, Any]:
    return {
        "total_rows": 0,
        "required_column_null_counts": {col: 0 for col in required_columns},
        "expired_count": 0,
        "readmitted_count": 0,
        "readmitted_under_30_days_count": 0,
        "patientunitstayid_values": [],
    }



def update_quality_accumulator(acc: dict[str, Any], df: pd.DataFrame, required_columns: list[str]) -> None:
    acc["total_rows"] += int(len(df))

    for col in required_columns:
        if col in df.columns:
            acc["required_column_null_counts"][col] += int(df[col].isna().sum())
        else:
            acc["required_column_null_counts"][col] += int(len(df))

    if "is_expired" in df.columns:
        acc["expired_count"] += int(df["is_expired"].fillna(False).sum())

    if "readmitted" in df.columns:
        acc["readmitted_count"] += int(df["readmitted"].fillna(0).sum())

    if "readmitted_under_30_days" in df.columns:
        acc["readmitted_under_30_days_count"] += int(df["readmitted_under_30_days"].fillna(0).sum())

    if "patientunitstayid" in df.columns:
        acc["patientunitstayid_values"].extend(df["patientunitstayid"].tolist())



def build_quality_report_from_accumulator(
    acc: dict[str, Any],
    required_columns: list[str],
    input_files: list[str],
) -> dict[str, Any]:
    total_rows = int(acc["total_rows"])
    patientunitstayid_values = pd.Series(acc["patientunitstayid_values"], dtype="object")
    duplicate_unitstay = int(patientunitstayid_values.duplicated().sum()) if not patientunitstayid_values.empty else total_rows

    readmitted_count = int(acc["readmitted_count"])
    readmitted_30_count = int(acc["readmitted_under_30_days_count"])

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": total_rows,
        "input_files": input_files,
        "required_column_null_counts": {col: int(acc["required_column_null_counts"].get(col, total_rows)) for col in required_columns},
        "duplicate_patientunitstayid": duplicate_unitstay,
        "expired_count": int(acc["expired_count"]),
        "readmitted_count": readmitted_count,
        "readmitted_under_30_days_count": readmitted_30_count,
        "readmitted_rate_pct": round((readmitted_count / total_rows * 100), 2) if total_rows else 0.0,
        "readmitted_under_30_days_rate_pct": round((readmitted_30_count / total_rows * 100), 2) if total_rows else 0.0,
    }


def write_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def run_job(cfg: BronzeToSilverConfig) -> int:
    source_root = Path(cfg.source_root)
    destination_root = Path(cfg.destination_root)
    quality_path = Path(cfg.quality_report_path)
    state_path = Path(cfg.state_path)

    if not source_root.exists():
        logger.info("Source root not found, nothing to process: %s", source_root)
        return 0

    processed_files = _read_state(state_path)
    pending_files = _list_unprocessed_parquet_files(source_root, processed_files)

    if not pending_files:
        logger.info("No new Bronze files to process.")
        return 0

    process_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_dir = destination_root / f"process_date={process_date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rows_written = 0
    quality_acc = init_quality_accumulator(cfg.required_columns)
    input_files = [str(p.as_posix()) for p in pending_files]

    for idx, bronze_file in enumerate(pending_files):
        bronze_df = pd.read_parquet(bronze_file)
        silver_df = transform_to_silver(bronze_df)

        silver_tables = split_silver_tables(silver_df)
        for table_name, table_df in silver_tables.items():
            table_output_dir = destination_root / table_name / f"process_date={process_date}"
            table_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = table_output_dir / f"{table_name}_{run_token}_{idx:04d}.parquet"
            table_df.to_parquet(output_file, index=False)

        rows_written += len(silver_tables["visits_core"])
        update_quality_accumulator(quality_acc, silver_df, cfg.required_columns)

    quality = build_quality_report_from_accumulator(quality_acc, cfg.required_columns, input_files)
    write_quality_report(quality_path, quality)

    processed_files.update(str(path.as_posix()) for path in pending_files)
    _write_state(state_path, processed_files)

    logger.info(
        "Processed %s Bronze files into Silver dataset rows=%s output=%s",
        len(pending_files),
        rows_written,
        output_dir,
    )
    return rows_written



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bronze to Silver ETL for EHR visits")
    parser.add_argument("--config", default="config/bronze_to_silver_config.json", help="Path to ETL config")
    return parser.parse_args()



def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config)
    run_job(cfg)


if __name__ == "__main__":
    main()
