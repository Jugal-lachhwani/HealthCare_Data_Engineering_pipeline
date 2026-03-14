from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("final-output-export")


SILVER_TABLES = ["visits_core", "diagnosis_care", "utilization_outcomes", "clinical_features"]


@dataclass
class FinalOutputConfig:
    silver_root: str
    model_dataset_csv_path: str
    reason_summary_csv_path: str
    driver_summary_csv_path: str


MODEL_COLUMNS = [
    "patientunitstayid",
    "uniquepid",
    "hospitalid",
    "wardid",
    "unittype",
    "gender",
    "ethnicity",
    "ethnic_group_c",
    "insurance_provider",
    "age",
    "los",
    "LACE_Score",
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
    "cost_of_initial_stay",
    "care_plan_costs",
    "Condition",
    "care_plan_following_discharge",
    "readmitted",
    "readmitted_under_30_days",
    "days_until_readmission",
    "processed_at_utc",
]


NUMERIC_COLUMNS = [
    "age",
    "hospitalid",
    "wardid",
    "los",
    "LACE_Score",
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
    "cost_of_initial_stay",
    "care_plan_costs",
    "readmitted",
    "readmitted_under_30_days",
    "days_until_readmission",
]

TEXT_DEFAULTS = {
    "gender": "unknown",
    "ethnicity": "unknown",
    "ethnic_group_c": "unknown",
    "insurance_provider": "unknown",
    "unittype": "unknown",
    "Condition": "unspecified",
    "care_plan_following_discharge": "No documented plan",
}



def load_config(config_path: str) -> FinalOutputConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return FinalOutputConfig(
        silver_root=raw["silver_root"],
        model_dataset_csv_path=raw["model_dataset_csv_path"],
        reason_summary_csv_path=raw["reason_summary_csv_path"],
        driver_summary_csv_path=raw["driver_summary_csv_path"],
    )



def _read_silver(silver_root: Path) -> pd.DataFrame:
    table_frames: dict[str, pd.DataFrame] = {}

    for table_name in SILVER_TABLES:
        table_root = silver_root / table_name
        files = sorted(table_root.rglob("*.parquet")) if table_root.exists() else []
        if not files:
            table_frames[table_name] = pd.DataFrame()
            continue
        table_frames[table_name] = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)

    core = table_frames.get("visits_core", pd.DataFrame())
    if not core.empty:
        assembled = core.copy()
        for table_name in ["diagnosis_care", "utilization_outcomes", "clinical_features"]:
            part = table_frames.get(table_name, pd.DataFrame())
            if part.empty:
                continue
            cols = [col for col in part.columns if col == "record_hash" or col not in assembled.columns]
            if "record_hash" not in cols:
                continue
            assembled = assembled.merge(part[cols], on="record_hash", how="left")

        if not assembled.empty:
            return assembled

    # Legacy fallback: older silver layout used a single wide table.
    legacy_files = sorted(silver_root.rglob("*.parquet"))
    if not legacy_files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in legacy_files], ignore_index=True)



def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out



def build_model_dataset(silver_df: pd.DataFrame) -> pd.DataFrame:
    working = _to_numeric(silver_df)

    if "readmitted" in working.columns:
        working["readmitted"] = working["readmitted"].fillna(0).astype(int)
    if "readmitted_under_30_days" in working.columns:
        working["readmitted_under_30_days"] = working["readmitted_under_30_days"].fillna(0).astype(int)

    selected = [col for col in MODEL_COLUMNS if col in working.columns]
    model_df = working[selected].copy()

    for col in NUMERIC_COLUMNS:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0)

    for col, default_value in TEXT_DEFAULTS.items():
        if col in model_df.columns:
            model_df[col] = model_df[col].fillna(default_value)
            model_df[col] = model_df[col].replace("", default_value)

    return model_df



def build_reason_summary(model_df: pd.DataFrame) -> pd.DataFrame:
    if "Condition" not in model_df.columns:
        return pd.DataFrame()

    grp = (
        model_df.groupby("Condition", dropna=False)
        .agg(
            total_admissions=("patientunitstayid", "count"),
            total_readmissions=("readmitted", "sum"),
            readmissions_under_30_days=("readmitted_under_30_days", "sum"),
            avg_days_to_readmission=("days_until_readmission", "mean"),
            avg_lace_score=("LACE_Score", "mean"),
            avg_los=("los", "mean"),
        )
        .reset_index()
    )

    grp["readmission_rate_pct"] = (grp["total_readmissions"] / grp["total_admissions"] * 100).round(2)
    grp["readmission_30d_rate_pct"] = (grp["readmissions_under_30_days"] / grp["total_admissions"] * 100).round(2)
    return grp.sort_values("readmissions_under_30_days", ascending=False)



def build_driver_summary(model_df: pd.DataFrame) -> pd.DataFrame:
    needed = ["LACE_Score", "los", "chronic_conditions", "readmitted_under_30_days"]
    if any(col not in model_df.columns for col in needed):
        return pd.DataFrame()

    bins = pd.cut(model_df["LACE_Score"], bins=[-1, 5, 10, 15, 25], labels=["0-5", "6-10", "11-15", "16+"])
    temp = model_df.copy()
    temp["lace_band"] = bins

    summary = (
        temp.groupby("lace_band", observed=False)
        .agg(
            patients=("patientunitstayid", "count"),
            readmission_30d_count=("readmitted_under_30_days", "sum"),
            avg_los=("los", "mean"),
            avg_chronic_conditions=("chronic_conditions", "mean"),
        )
        .reset_index()
    )
    summary["readmission_30d_rate_pct"] = (
        summary["readmission_30d_count"] / summary["patients"] * 100
    ).round(2)

    return summary



def _write_csv(df: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def run_job(cfg: FinalOutputConfig) -> int:
    silver_df = _read_silver(Path(cfg.silver_root))
    if silver_df.empty:
        logger.info("No Silver data available to export.")
        return 0

    model_df = build_model_dataset(silver_df)
    reason_df = build_reason_summary(model_df)
    driver_df = build_driver_summary(model_df)

    _write_csv(model_df, cfg.model_dataset_csv_path)
    _write_csv(reason_df, cfg.reason_summary_csv_path)
    _write_csv(driver_df, cfg.driver_summary_csv_path)

    logger.info(
        "Final CSV exports created: model_rows=%s reason_rows=%s driver_rows=%s",
        len(model_df),
        len(reason_df),
        len(driver_df),
    )
    return len(model_df)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final model-ready CSV outputs")
    parser.add_argument("--config", default="config/final_output_config.json", help="Path to final output config")
    return parser.parse_args()



def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config)
    run_job(cfg)


if __name__ == "__main__":
    main()
