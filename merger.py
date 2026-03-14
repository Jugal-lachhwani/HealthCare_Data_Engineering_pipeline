"""
╔══════════════════════════════════════════════════════════════════════╗
║  EHR Master Merge Script                                             ║
║  Merges all 5 CSVs into one master flat table                        ║
║                                                                      ║
║  INPUT  (folder: synthetic_ehr_output/):                             ║
║    patients.csv · diagnoses.csv · lab_results.csv                    ║
║    medications.csv · prior_admissions.csv                            ║
║                                                                      ║
║  OUTPUT:                                                             ║
║    master_ehr.csv         ← ML training (wide, one row per patient)  ║ 
║    master_ehr_summary.csv ← Dashboard / EDA (aggregated stats)       ║
║                                                                      ║
║  INSTALL:  pip install pandas numpy tqdm                             ║
║  RUN:      python merge_ehr.py                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

INPUT_DIR  = "./synthetic_ehr_output"
OUTPUT_DIR = "./synthetic_ehr_output"

print("\n🔗  EHR Master Merge — Loading tables...\n")

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD ALL 5 TABLES
# ══════════════════════════════════════════════════════════════════════════════

patients   = pd.read_csv(f"{INPUT_DIR}/patients.csv")
diagnoses  = pd.read_csv(f"{INPUT_DIR}/diagnoses.csv")
labs       = pd.read_csv(f"{INPUT_DIR}/lab_results.csv")
meds       = pd.read_csv(f"{INPUT_DIR}/medications.csv")
prior      = pd.read_csv(f"{INPUT_DIR}/prior_admissions.csv")

print(f"  patients.csv         {len(patients):>9,} rows")
print(f"  diagnoses.csv        {len(diagnoses):>9,} rows")
print(f"  lab_results.csv      {len(labs):>9,} rows")
print(f"  medications.csv      {len(meds):>9,} rows")
print(f"  prior_admissions.csv {len(prior):>9,} rows")
print()

KEY = "patientunitstayid"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PIVOT LAB RESULTS WIDE
#  Each lab test becomes its own column: lab_WBC_mean, lab_WBC_max, etc.
# ══════════════════════════════════════════════════════════════════════════════

print("🔬  Pivoting lab results wide (mean + max per lab per patient)...")

lab_pivot = (
    labs
    .groupby([KEY, "lab_name"])["lab_value"]
    .agg(["mean", "max", "min"])
    .reset_index()
)
lab_pivot.columns = [KEY, "lab_name", "mean", "max", "min"]

# Pivot to wide: one column per lab × stat
lab_wide = lab_pivot.pivot_table(
    index=KEY,
    columns="lab_name",
    values=["mean", "max", "min"],
    aggfunc="first"
)
# Flatten multi-level columns → lab_Glucose_mean, lab_Glucose_max, etc.
lab_wide.columns = [f"lab_{stat}_{name.replace(' ', '_')}" for stat, name in lab_wide.columns]
lab_wide = lab_wide.reset_index()

# Add critical flag counts (how many CRITICAL lab readings per patient)
critical_counts = (
    labs[labs["lab_flag"].str.startswith("CRITICAL", na=False)]
    .groupby(KEY)
    .size()
    .reset_index(name="lab_critical_flag_count")
)

# Add number of lab rounds (proxy for ICU severity / stay length)
lab_rounds = (
    labs.groupby(KEY)["round_number"]
    .max()
    .reset_index(name="lab_total_rounds")
)

print(f"  ✅ Lab pivot: {lab_wide.shape[1]} columns for {len(lab_wide):,} patients\n")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — AGGREGATE MEDICATIONS
#  Count of unique drugs, drug classes, total duration, key drug flags
# ══════════════════════════════════════════════════════════════════════════════

print("💊  Aggregating medications...")

med_agg = meds.groupby(KEY).agg(
    med_unique_drugs      = ("drug_name",       "nunique"),
    med_unique_classes    = ("drug_class",       "nunique"),
    med_total_duration_min= ("duration_minutes", "sum"),
    med_has_vasopressor   = ("drug_class",       lambda x: int("Vasopressor" in x.values)),
    med_has_antibiotic    = ("drug_class",       lambda x: int("Antibiotic" in x.values)),
    med_has_anticoagulant = ("drug_class",       lambda x: int("Anticoagulant" in x.values)),
    med_has_opioid        = ("drug_class",       lambda x: int("Opioid Analgesic" in x.values)),
    med_has_insulin       = ("drug_class",       lambda x: int("Antidiabetic" in x.values)),
    med_has_diuretic      = ("drug_class",       lambda x: int("Diuretic" in x.values)),
    med_prn_count         = ("prn",              lambda x: (x == "Yes").sum()),
    med_iv_count          = ("route_of_admin",   lambda x: (x == "IV").sum()),
).reset_index()

print(f"  ✅ Medication aggregation: {len(med_agg):,} patients, {med_agg.shape[1]} columns\n")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — AGGREGATE DIAGNOSES
#  Primary ICD code, comorbidity count, high-risk flags
# ══════════════════════════════════════════════════════════════════════════════

print("🩺  Aggregating diagnoses...")

# Primary diagnosis per patient
primary_dx = (
    diagnoses[diagnoses["diagnosis_type"] == "Primary"]
    [["patientunitstayid", "icd10_code", "icd10_description"]]
    .rename(columns={"icd10_code": "primary_icd10", "icd10_description": "primary_icd10_desc"})
    .drop_duplicates(subset=KEY)
)

# Comorbidity count
comorbidity_count = (
    diagnoses[diagnoses["diagnosis_type"] == "Comorbidity"]
    .groupby(KEY)
    .size()
    .reset_index(name="dx_comorbidity_count")
)

# High-risk comorbidity flags
HIGH_RISK_CODES = {"I50.9", "I50.22", "N17.9", "N18.6", "A41.9", "A41.51",
                   "J44.1", "J44.0", "I63.9", "I61.9"}
diag_flags = (
    diagnoses.groupby(KEY)["icd10_code"]
    .apply(lambda codes: int(bool(set(codes) & HIGH_RISK_CODES)))
    .reset_index(name="dx_has_high_risk_comorbidity")
)

# Total distinct ICD codes
icd_count = (
    diagnoses.groupby(KEY)["icd10_code"]
    .nunique()
    .reset_index(name="dx_total_icd_codes")
)

print(f"  ✅ Diagnosis aggregation done\n")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — AGGREGATE PRIOR ADMISSIONS
# ══════════════════════════════════════════════════════════════════════════════

print("📁  Aggregating prior admissions...")

prior_agg = prior.groupby(KEY).agg(
    prior_total_admissions     = ("sequence_number",           "count"),
    prior_avg_los_days         = ("prior_los_days",            "mean"),
    prior_max_los_days         = ("prior_los_days",            "max"),
    prior_icu_admissions       = ("prior_icu_admission",       lambda x: (x == "Yes").sum()),
    prior_deaths               = ("prior_discharge_status",    lambda x: (x == "Expired").sum()),
    prior_min_days_since       = ("days_since_prior_admission","min"),   # most recent
    prior_any_snf_discharge    = ("prior_discharge_location",  lambda x: int("Skilled Nursing Facility" in x.values)),
).reset_index()
prior_agg["prior_avg_los_days"] = prior_agg["prior_avg_los_days"].round(1)

print(f"  ✅ Prior admission aggregation: {len(prior_agg):,} patients\n")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — MERGE EVERYTHING INTO MASTER TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("🔗  Merging all tables into master_ehr.csv...")

master = patients.copy()

merges = [
    (primary_dx,        KEY, "primary diagnosis"),
    (comorbidity_count, KEY, "comorbidity count"),
    (diag_flags,        KEY, "diagnosis flags"),
    (icd_count,         KEY, "ICD code count"),
    (lab_wide,          KEY, "lab pivot"),
    (critical_counts,   KEY, "critical lab flags"),
    (lab_rounds,        KEY, "lab rounds"),
    (med_agg,           KEY, "medication aggregates"),
    (prior_agg,         KEY, "prior admissions"),
]

for df, key, label in tqdm(merges, desc="  Merging"):
    master = master.merge(df, on=key, how="left")

# Fill NaN for patients with no prior admissions / no critical flags
master["prior_total_admissions"]     = master["prior_total_admissions"].fillna(0).astype(int)
master["prior_icu_admissions"]       = master["prior_icu_admissions"].fillna(0).astype(int)
master["prior_deaths"]               = master["prior_deaths"].fillna(0).astype(int)
master["prior_any_snf_discharge"]    = master["prior_any_snf_discharge"].fillna(0).astype(int)
master["lab_critical_flag_count"]    = master["lab_critical_flag_count"].fillna(0).astype(int)
master["dx_comorbidity_count"]       = master["dx_comorbidity_count"].fillna(0).astype(int)
master["dx_has_high_risk_comorbidity"]= master["dx_has_high_risk_comorbidity"].fillna(0).astype(int)
master["dx_total_icd_codes"]         = master["dx_total_icd_codes"].fillna(0).astype(int)

# Encode ML target as binary
master["readmission_binary"] = (master["readmission_within_30days"] == "Yes").astype(int)

master.to_csv(f"{OUTPUT_DIR}/master_ehr.csv", index=False)
print(f"\n  ✅ master_ehr.csv — {len(master):,} rows × {master.shape[1]} columns")
print(f"     Readmission rate: {master['readmission_binary'].mean()*100:.1f}%\n")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — BUILD DASHBOARD SUMMARY CSV
#  Aggregated stats by archetype — for EDA dashboards
# ══════════════════════════════════════════════════════════════════════════════

print("📊  Building dashboard summary (master_ehr_summary.csv)...")

summary = master.groupby("archetype").agg(
    total_patients            = (KEY,                          "count"),
    readmission_rate_pct      = ("readmission_binary",         lambda x: round(x.mean()*100, 1)),
    avg_age                   = ("age",                        "mean"),
    pct_male                  = ("gender",                     lambda x: round((x=="Male").mean()*100,1)),
    avg_severity_score        = ("severity_score",             "mean"),
    avg_los_hours             = ("unitdischargeoffset",        "mean"),
    avg_prior_admissions      = ("prior_total_admissions",     "mean"),
    avg_comorbidities         = ("dx_comorbidity_count",       "mean"),
    avg_unique_drugs          = ("med_unique_drugs",           "mean"),
    pct_vasopressor           = ("med_has_vasopressor",        lambda x: round(x.mean()*100,1)),
    pct_antibiotic            = ("med_has_antibiotic",         lambda x: round(x.mean()*100,1)),
    pct_discharged_home       = ("hospitaldischargelocation",  lambda x: round((x=="Home").mean()*100,1)),
    pct_discharged_snf        = ("hospitaldischargelocation",  lambda x: round((x=="Skilled Nursing Facility").mean()*100,1)),
    pct_expired               = ("hospitaldischargestatus",    lambda x: round((x=="Expired").mean()*100,1)),
    avg_critical_lab_flags    = ("lab_critical_flag_count",    "mean"),
).reset_index()

summary = summary.round(2)
summary.to_csv(f"{OUTPUT_DIR}/master_ehr_summary.csv", index=False)
print(f"  ✅ master_ehr_summary.csv — {len(summary)} rows (one per archetype)\n")

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 64)
print("  ✅  MERGE COMPLETE")
print("=" * 64)
print(f"""
  Output files:
  ─────────────────────────────────────────────────────────
  master_ehr.csv         {len(master):>8,} rows × {master.shape[1]:>3} columns
  master_ehr_summary.csv {len(summary):>8} rows × {summary.shape[1]:>3} columns
  ─────────────────────────────────────────────────────────

  master_ehr.csv column groups:
    Patient core    : {len(patients.columns)} cols  (demographics, ICU stay, target)
    Diagnosis       : 4 cols  (primary ICD, comorbidity count, flags)
    Labs (pivoted)  : ~{len([c for c in master.columns if c.startswith('lab_')])} cols  (mean/max/min per lab + flags)
    Medications     : {med_agg.shape[1]-1} cols  (drug counts, class flags)
    Prior admissions: {prior_agg.shape[1]-1} cols  (history, recency, ICU count)

  ML target columns:
    readmission_within_30days  →  "Yes" / "No"  (string)
    readmission_binary         →  1 / 0          (integer, use this for ML)

  Quick ML start:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv('synthetic_ehr_output/master_ehr.csv')

    # Select numeric feature columns only
    feature_cols = [c for c in df.columns if c.startswith('lab_')
                    or c.startswith('med_') or c.startswith('prior_')
                    or c.startswith('dx_')
                    or c in ['age','severity_score','num_prior_admissions']]

    X = df[feature_cols].fillna(0)
    y = df['readmission_binary']

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
""")