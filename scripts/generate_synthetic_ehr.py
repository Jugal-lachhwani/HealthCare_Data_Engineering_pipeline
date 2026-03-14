"""
Synthetic EHR Feature Generator
================================
Adds medically realistic, correlated features to the existing EHR.csv.
Every value is derived from the patient's age, gender, diagnosis, unit type,
and length of stay — NOT random.

Phase 1: Enrich existing 1,447 rows with ~35 new feature columns.
"""

import pandas as pd
import numpy as np
import os
import re
import sys

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# 1. LOAD & PREP
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "Data", "EHR.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "Data", "EHR_enriched.csv")

df = pd.read_csv(INPUT_PATH)
n = len(df)
print(f"Loaded {n} rows, {df.shape[1]} columns")

# Parse age to numeric ("> 89" → 92)
def parse_age(val):
    try:
        if isinstance(val, str) and '>' in val:
            return 92
        return int(float(val))
    except (ValueError, TypeError):
        return 65  # median fallback

df['_age_num'] = df['age'].apply(parse_age)

# Parse LOS from unitdischargeoffset (minutes → days)
df['los'] = np.maximum(df['unitdischargeoffset'] / 1440.0, 0.1).round(2)

# Standardize sex from gender
df['sex'] = df['gender'].map({'Male': 'M', 'Female': 'F'}).fillna('Unknown')

# ─────────────────────────────────────────────────────────────
# 2. CLASSIFY PATIENT ARCHETYPE FROM DIAGNOSIS
# ─────────────────────────────────────────────────────────────
def classify_archetype(row):
    """Assign a clinical archetype based on diagnosis + age + unit type."""
    dx = str(row.get('apacheadmissiondx', '')).lower()
    age = row['_age_num']
    unit = str(row.get('unittype', '')).lower()

    # Cardiac
    if any(k in dx for k in ['myocardial', 'cabg', 'cardiac', 'coronary', 'heart',
                               'arrhythm', 'chf', 'angina', 'aortic', 'valve',
                               'endarterect', 'ablation', 'pacemaker', 'stent']):
        return 'cardiac'
    # Respiratory
    if any(k in dx for k in ['pneumonia', 'copd', 'emphysema', 'bronchitis',
                               'asthma', 'respiratory', 'pulmonary', 'ventilat',
                               'pleural', 'tracheostomy']):
        return 'respiratory'
    # Sepsis / Infectious
    if any(k in dx for k in ['sepsis', 'infection', 'cellulitis', 'abscess',
                               'meningitis', 'endocarditis']):
        return 'sepsis'
    # Neurological
    if any(k in dx for k in ['stroke', 'cerebrovascular', 'seizure', 'neurolog',
                               'intracranial', 'subarachnoid', 'subdural', 'coma']):
        return 'neuro'
    # Diabetic / Endocrine
    if any(k in dx for k in ['diabetic', 'ketoacidosis', 'dka', 'hyperglycemia',
                               'thyroid', 'adrenal']):
        return 'diabetic'
    # GI / Surgical
    if any(k in dx for k in ['gi bleed', 'pancreatitis', 'cholecystectomy', 'bowel',
                               'hepatic', 'liver', 'esophageal', 'varices',
                               'appendectomy', 'hernia', 'colectomy', 'laparotomy']):
        return 'gi_surgical'
    # Renal
    if any(k in dx for k in ['renal', 'kidney', 'dialysis', 'nephr']):
        return 'renal'
    # Overdose / Substance
    if any(k in dx for k in ['overdose', 'alcohol', 'withdrawal', 'drug',
                               'toxicity', 'poison']):
        return 'substance'
    # Trauma / Surgical
    if any(k in dx for k in ['trauma', 'fracture', 'surgical', 'postoperative',
                               'transplant', 'amputation']):
        return 'trauma_surgical'
    # Cancer
    if any(k in dx for k in ['cancer', 'neoplasm', 'tumor', 'lymphoma', 'leukemia',
                               'malignant', 'metasta']):
        return 'cancer'
    # Elderly multi-morbid (age-based fallback)
    if age >= 75:
        return 'elderly_multimorbid'
    # Default healthy adult
    return 'general'

df['_archetype'] = df.apply(classify_archetype, axis=1)
print(f"\nArchetype distribution:\n{df['_archetype'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────
# 3. HELPER: BOUNDED NORMAL SAMPLING
# ─────────────────────────────────────────────────────────────
def bounded_normal(mean, std, low, high, size=1):
    """Sample from truncated normal distribution."""
    vals = np.random.normal(mean, std, size)
    return np.clip(vals, low, high)

def bounded_normal_int(mean, std, low, high, size=1):
    return bounded_normal(mean, std, low, high, size).astype(int)

# ─────────────────────────────────────────────────────────────
# 4. VITALS — Age/Archetype-correlated
# ─────────────────────────────────────────────────────────────
print("Generating vitals...")

# Blood Pressure
bp_sys_base = np.where(df['_age_num'] < 40, 118,
              np.where(df['_age_num'] < 60, 125,
              np.where(df['_age_num'] < 75, 135, 142)))
bp_sys_adjust = np.where(df['_archetype'] == 'cardiac', 15,
                np.where(df['_archetype'] == 'renal', 12,
                np.where(df['_archetype'] == 'sepsis', -20,
                np.where(df['_archetype'] == 'substance', -10, 0))))
df['bp_systolic'] = np.clip(
    np.round(bp_sys_base + bp_sys_adjust + np.random.normal(0, 8, n)),
    70, 220).astype(int)

df['bp_diastolic'] = np.clip(
    np.round(df['bp_systolic'] * np.random.uniform(0.55, 0.68, n)),
    40, 130).astype(int)

# Pulse — higher in sepsis, cardiac, lower in healthy
pulse_base = np.where(df['_archetype'] == 'sepsis', 105,
             np.where(df['_archetype'] == 'cardiac', 88,
             np.where(df['_archetype'] == 'respiratory', 92,
             np.where(df['_age_num'] >= 75, 78, 75))))
df['pulse'] = np.clip(
    np.round(pulse_base + np.random.normal(0, 10, n)),
    45, 160).astype(int)

# Respirations
resp_base = np.where(df['_archetype'] == 'respiratory', 24,
            np.where(df['_archetype'] == 'sepsis', 24,
            np.where(df['_age_num'] >= 75, 20, 17)))
df['respirations'] = np.clip(
    np.round(resp_base + np.random.normal(0, 3, n)),
    10, 42).astype(int)

# Temperature (Fahrenheit)
temp_base = np.where(df['_archetype'] == 'sepsis', 101.2,
            np.where(df['_archetype'].isin(['elderly_multimorbid']), 97.4,
            np.full(n, 98.2)))
df['temperature'] = np.round(temp_base + np.random.normal(0, 0.7, n), 1)
df['temperature'] = np.clip(df['temperature'], 95.0, 105.0)

# ─────────────────────────────────────────────────────────────
# 5. BMI — from height/weight
# ─────────────────────────────────────────────────────────────
print("Generating BMI...")
height_m = df['admissionheight'].fillna(170) / 100.0
weight_kg = df['admissionweight'].fillna(80)
df['bmi'] = np.round(weight_kg / (height_m ** 2), 1)
df['bmi'] = np.clip(df['bmi'], 14.0, 65.0)

# ─────────────────────────────────────────────────────────────
# 6. LAB VALUES — Archetype-correlated
# ─────────────────────────────────────────────────────────────
print("Generating lab values...")

# Glucose (mg/dL) — diabetics have higher
gluc_base = np.where(df['_archetype'] == 'diabetic', 220,
            np.where(df['_archetype'] == 'sepsis', 160,
            np.where(df['_age_num'] >= 75, 115,
            np.full(n, 100.0))))
df['glucose'] = np.clip(np.round(gluc_base + np.random.normal(0, 25, n), 1), 50, 500)

# Creatinine (mg/dL) — renal patients have higher
creat_base = np.where(df['_archetype'] == 'renal', 3.5,
             np.where(df['_archetype'] == 'sepsis', 1.8,
             np.where(df['_archetype'] == 'cardiac', 1.3,
             np.where(df['_age_num'] >= 75, 1.2,
             np.full(n, 0.95)))))
df['creatinine'] = np.clip(np.round(creat_base + np.random.normal(0, 0.3, n), 2), 0.3, 12.0)

# WBC (K/uL) — sepsis/infection → elevated
wbc_base = np.where(df['_archetype'] == 'sepsis', 18.0,
           np.where(df['_archetype'] == 'cancer', 4.5,
           np.where(df['_archetype'] == 'trauma_surgical', 12.0,
           np.full(n, 8.0))))
df['wbc'] = np.clip(np.round(wbc_base + np.random.normal(0, 2.5, n), 1), 1.0, 45.0)

# Hemoglobin (g/dL) — cancer/renal → lower
hb_base = np.where(df['sex'] == 'M', 14.5, 13.0)
hb_adjust = np.where(df['_archetype'] == 'cancer', -3.0,
            np.where(df['_archetype'] == 'renal', -2.5,
            np.where(df['_archetype'] == 'gi_surgical', -2.0,
            np.where(df['_age_num'] >= 75, -1.5, 0.0))))
df['hemoglobin'] = np.clip(np.round(hb_base + hb_adjust + np.random.normal(0, 1.0, n), 1), 5.0, 19.0)

# Potassium (mEq/L) — renal → hyperkalemia
k_base = np.where(df['_archetype'] == 'renal', 5.2,
         np.where(df['_archetype'] == 'diabetic', 4.8, np.full(n, 4.1)))
df['potassium'] = np.clip(np.round(k_base + np.random.normal(0, 0.4, n), 1), 2.5, 7.5)

# Sodium (mEq/L)
na_base = np.where(df['_archetype'] == 'elderly_multimorbid', 136,
          np.where(df['_archetype'] == 'substance', 133, np.full(n, 140.0)))
df['sodium'] = np.clip(np.round(na_base + np.random.normal(0, 3, n), 0), 120, 160).astype(int)

# Calcium (mg/dL)
ca_base = np.where(df['_archetype'] == 'cancer', 10.8,
          np.where(df['_archetype'] == 'renal', 8.2, np.full(n, 9.3)))
df['calcium'] = np.clip(np.round(ca_base + np.random.normal(0, 0.5, n), 1), 6.0, 14.0)

# Arterial Blood Gas (pH-like score, 0-1 scale for simplicity → 7.25-7.55)
abg_base = np.where(df['_archetype'] == 'respiratory', 7.32,
           np.where(df['_archetype'] == 'sepsis', 7.28,
           np.where(df['_archetype'] == 'renal', 7.30, np.full(n, 7.40))))
df['artbloodgas'] = np.clip(np.round(abg_base + np.random.normal(0, 0.04, n), 2), 7.10, 7.60)

# ─────────────────────────────────────────────────────────────
# 7. MEDICATIONS — count per category, correlated to archetype
# ─────────────────────────────────────────────────────────────
print("Generating medication counts...")

def med_count(archetype_col, primary_archetypes, primary_range, secondary_archetypes, secondary_range, default_range):
    """Generate medication counts based on archetype relevance."""
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        arch = archetype_col.iloc[i]
        if arch in primary_archetypes:
            counts[i] = np.random.randint(primary_range[0], primary_range[1] + 1)
        elif arch in secondary_archetypes:
            counts[i] = np.random.randint(secondary_range[0], secondary_range[1] + 1)
        else:
            counts[i] = np.random.randint(default_range[0], default_range[1] + 1)
    return counts

arch = df['_archetype']

df['meds_gastro'] = med_count(arch,
    ['gi_surgical'], (3, 8),
    ['elderly_multimorbid', 'sepsis'], (1, 4),
    (0, 2))

df['meds_respiratory'] = med_count(arch,
    ['respiratory'], (4, 10),
    ['sepsis', 'cardiac', 'elderly_multimorbid'], (1, 5),
    (0, 2))

df['meds_anesthetics'] = med_count(arch,
    ['trauma_surgical', 'gi_surgical'], (3, 8),
    ['cardiac', 'neuro'], (1, 4),
    (0, 2))

df['meds_infective'] = med_count(arch,
    ['sepsis'], (5, 12),
    ['respiratory', 'trauma_surgical', 'renal'], (2, 6),
    (0, 2))

df['meds_central_nervous_system'] = med_count(arch,
    ['neuro', 'substance'], (4, 10),
    ['elderly_multimorbid'], (2, 5),
    (0, 3))

df['meds_hematological'] = med_count(arch,
    ['cardiac', 'cancer'], (3, 8),
    ['trauma_surgical', 'renal'], (1, 4),
    (0, 2))

df['meds_cardio_agents'] = med_count(arch,
    ['cardiac'], (5, 12),
    ['elderly_multimorbid', 'renal', 'diabetic'], (2, 5),
    (0, 2))

df['meds_endocrine'] = med_count(arch,
    ['diabetic'], (4, 9),
    ['elderly_multimorbid', 'renal'], (1, 4),
    (0, 1))

df['meds_topical'] = med_count(arch,
    ['trauma_surgical'], (2, 5),
    ['elderly_multimorbid', 'cancer'], (1, 3),
    (0, 2))

df['meds_nutrition'] = med_count(arch,
    ['cancer', 'gi_surgical', 'renal'], (3, 7),
    ['elderly_multimorbid', 'sepsis'], (1, 4),
    (0, 2))

df['meds_neuromuscular'] = med_count(arch,
    ['neuro', 'trauma_surgical'], (2, 6),
    ['elderly_multimorbid'], (1, 3),
    (0, 1))

df['meds_genitourinary'] = med_count(arch,
    ['renal'], (3, 7),
    ['elderly_multimorbid', 'diabetic'], (1, 3),
    (0, 1))

df['meds_biological'] = med_count(arch,
    ['cancer'], (3, 8),
    ['sepsis', 'renal'], (1, 3),
    (0, 1))

df['meds_neoplastic'] = med_count(arch,
    ['cancer'], (4, 10),
    [],  (0, 0),
    (0, 1))

# ─────────────────────────────────────────────────────────────
# 8. CLINICAL FLAGS & CONDITIONS — archetype-probability based
# ─────────────────────────────────────────────────────────────
print("Generating conditions and clinical flags...")

def bernoulli_by_archetype(archetype_col, prob_map, default_prob):
    """Generate binary flags based on archetype-specific probabilities."""
    probs = archetype_col.map(prob_map).fillna(default_prob).values
    return (np.random.random(n) < probs).astype(int)

# ICU flag
df['icu_yn'] = bernoulli_by_archetype(arch,
    {'cardiac': 0.75, 'sepsis': 0.85, 'respiratory': 0.70, 'neuro': 0.65,
     'trauma_surgical': 0.70, 'cancer': 0.40, 'elderly_multimorbid': 0.55,
     'renal': 0.50, 'substance': 0.40, 'general': 0.25, 'diabetic': 0.45,
     'gi_surgical': 0.50}, 0.30)

# Ventilator
df['vent'] = bernoulli_by_archetype(arch,
    {'respiratory': 0.60, 'sepsis': 0.50, 'neuro': 0.35, 'cardiac': 0.25,
     'trauma_surgical': 0.30, 'substance': 0.20, 'cancer': 0.15,
     'elderly_multimorbid': 0.20}, 0.05)

# Chest tube
df['chest_tube'] = bernoulli_by_archetype(arch,
    {'cardiac': 0.35, 'trauma_surgical': 0.30, 'respiratory': 0.20,
     'gi_surgical': 0.10, 'cancer': 0.10}, 0.02)

# Chronic conditions count (0-6)
chronic_mean = np.where(arch == 'elderly_multimorbid', 4.0,
               np.where(arch == 'cardiac', 3.0,
               np.where(arch == 'diabetic', 3.0,
               np.where(arch == 'renal', 3.5,
               np.where(arch == 'cancer', 2.5,
               np.where(df['_age_num'] >= 65, 2.5, 1.0))))))
df['chronic_conditions'] = np.clip(
    np.round(chronic_mean + np.random.normal(0, 0.8, n)).astype(int), 0, 8)

# Individual condition flags
df['diabetes'] = bernoulli_by_archetype(arch,
    {'diabetic': 0.95, 'cardiac': 0.40, 'renal': 0.45,
     'elderly_multimorbid': 0.35, 'general': 0.08}, 0.12)

df['obesity'] = (df['bmi'] >= 30).astype(int)

df['anxiety'] = bernoulli_by_archetype(arch,
    {'substance': 0.65, 'neuro': 0.35, 'elderly_multimorbid': 0.30,
     'cardiac': 0.25}, 0.12)

df['depression'] = bernoulli_by_archetype(arch,
    {'substance': 0.55, 'cancer': 0.45, 'elderly_multimorbid': 0.35,
     'neuro': 0.30}, 0.10)

df['dementia'] = bernoulli_by_archetype(arch,
    {'elderly_multimorbid': 0.40, 'neuro': 0.25}, 0.03)

df['drugabuse'] = bernoulli_by_archetype(arch,
    {'substance': 0.85}, 0.04)

df['mooddisorder'] = bernoulli_by_archetype(arch,
    {'substance': 0.50, 'neuro': 0.20, 'elderly_multimorbid': 0.15}, 0.06)

# Tobacco user
df['tobacco_user'] = bernoulli_by_archetype(arch,
    {'respiratory': 0.70, 'cardiac': 0.45, 'cancer': 0.40,
     'substance': 0.55}, 0.18)

# ─────────────────────────────────────────────────────────────
# 9. CLINICAL SCORES & VISITS
# ─────────────────────────────────────────────────────────────
print("Generating visits, costs, and scores...")

# ED visits (past 6 months) — higher for chronic patients
ed_base = np.where(arch.isin(['elderly_multimorbid', 'substance']), 3,
          np.where(arch.isin(['cardiac', 'respiratory', 'diabetic', 'renal']), 2,
          np.full(n, 1)))
df['ed_visits'] = np.clip(
    np.round(ed_base + np.random.exponential(1.0, n)).astype(int), 0, 15)

# Inpatient visits (past year)
ip_base = np.where(arch.isin(['elderly_multimorbid', 'cancer', 'renal']), 2,
          np.where(arch.isin(['cardiac', 'respiratory', 'diabetic']), 1,
          np.full(n, 0)))
df['ip_visits'] = np.clip(
    np.round(ip_base + np.random.exponential(0.8, n)).astype(int), 0, 10)

# Pain score (0-10)
pain_base = np.where(arch.isin(['trauma_surgical', 'cancer']), 6,
            np.where(arch.isin(['gi_surgical', 'cardiac']), 4,
            np.where(arch.isin(['respiratory', 'sepsis']), 3, np.full(n, 2))))
df['pet_pain_score_c'] = np.clip(
    np.round(pain_base + np.random.normal(0, 1.5, n)).astype(int), 0, 10)

# LACE Score (Length of stay + Acuity + Comorbidity + ED visits)
# L: LOS days → score 0-7
los_days = df['los'].values
l_score = np.where(los_days < 1, 0,
          np.where(los_days < 2, 1,
          np.where(los_days < 3, 2,
          np.where(los_days < 4, 3,
          np.where(los_days < 7, 4,
          np.where(los_days < 14, 5,
          np.where(los_days < 21, 6, 7)))))))
# A: Acuity (admitted via ED = 3, else 0)
a_score = np.where(df['hospitaladmitsource'] == 'Emergency Department', 3, 0)
# C: Comorbidity (Charlson index proxy)
c_score = np.clip(df['chronic_conditions'], 0, 6)
# E: ED visits in past 6 months, capped at 4
e_score = np.clip(df['ed_visits'], 0, 4)
df['LACE_Score'] = (l_score + a_score + c_score + e_score).astype(int)

# ─────────────────────────────────────────────────────────────
# 10. COSTS — LOS + ICU + procedures driven
# ─────────────────────────────────────────────────────────────
# Base daily cost varies by unit type
daily_cost = np.where(df['icu_yn'] == 1, np.random.uniform(3500, 7000, n),
                                         np.random.uniform(1800, 3500, n))
df['cost_of_initial_stay'] = np.round(daily_cost * df['los'] + 
    np.random.uniform(500, 2500, n), 2)  # add procedure/supply costs
df['cost_of_initial_stay'] = np.maximum(df['cost_of_initial_stay'], 1500)

# Care plan costs (follow-up, rehab, meds)
care_multi = np.where(arch.isin(['cardiac', 'cancer', 'renal']), 0.35,
             np.where(arch.isin(['elderly_multimorbid', 'neuro']), 0.30,
             np.full(n, 0.20)))
df['care_plan_costs'] = np.round(
    df['cost_of_initial_stay'] * care_multi + np.random.uniform(200, 1500, n), 2)

# ─────────────────────────────────────────────────────────────
# 11. DEMOGRAPHICS & MISC
# ─────────────────────────────────────────────────────────────
print("Generating demographics and misc...")

# Marital status (correlated with age)
def assign_marital(age):
    if age < 30:
        return np.random.choice(['Single', 'Married', 'Domestic Partner'],
                                p=[0.65, 0.25, 0.10])
    elif age < 55:
        return np.random.choice(['Married', 'Single', 'Divorced', 'Domestic Partner'],
                                p=[0.55, 0.20, 0.20, 0.05])
    elif age < 75:
        return np.random.choice(['Married', 'Widowed', 'Divorced', 'Single'],
                                p=[0.50, 0.20, 0.20, 0.10])
    else:
        return np.random.choice(['Widowed', 'Married', 'Single', 'Divorced'],
                                p=[0.40, 0.35, 0.10, 0.15])

df['marital_status_c'] = df['_age_num'].apply(assign_marital)

# Ethnic group (map from existing ethnicity column for consistency)
ethnic_map = {
    'Caucasian': 'White', 'African American': 'Black or African American',
    'Hispanic': 'Hispanic or Latino', 'Asian': 'Asian',
    'Native American': 'American Indian or Alaska Native',
    'Other/Unknown': 'Other'
}
df['ethnic_group_c'] = df['ethnicity'].map(ethnic_map).fillna('Other')

# Insurance provider (age-correlated)
def assign_insurance(age):
    if age >= 65:
        return np.random.choice(['Medicare', 'Medicare Advantage', 'Private'],
                                p=[0.55, 0.30, 0.15])
    elif age < 30:
        return np.random.choice(['Private', 'Medicaid', 'Self-Pay', 'Other'],
                                p=[0.40, 0.35, 0.15, 0.10])
    else:
        return np.random.choice(['Private', 'Medicare', 'Medicaid', 'Self-Pay', 'Other'],
                                p=[0.50, 0.15, 0.20, 0.10, 0.05])

df['insurance_provider'] = df['_age_num'].apply(assign_insurance)

# Condition (primary diagnosis category — cleaner version)
condition_map = {
    'cardiac': 'Cardiovascular Disease',
    'respiratory': 'Respiratory Disease',
    'sepsis': 'Sepsis/Infection',
    'neuro': 'Neurological Disorder',
    'diabetic': 'Diabetes/Endocrine',
    'gi_surgical': 'Gastrointestinal/Surgical',
    'renal': 'Renal Disease',
    'substance': 'Substance Abuse/Overdose',
    'trauma_surgical': 'Trauma/Surgical',
    'cancer': 'Cancer/Neoplasm',
    'elderly_multimorbid': 'Multi-morbid/Elderly',
    'general': 'General Medical'
}
df['Condition'] = arch.map(condition_map)

# Care plan following discharge
def assign_care_plan(row):
    arch = row['_archetype']
    plans = []
    if arch in ['cardiac', 'elderly_multimorbid']:
        plans = ['Cardiac Rehab', 'Follow-up Cardiology', 'Home Health', 'Medication Management']
    elif arch == 'respiratory':
        plans = ['Pulmonary Rehab', 'Follow-up Pulmonology', 'Home O2 Therapy', 'Smoking Cessation']
    elif arch == 'sepsis':
        plans = ['IV Antibiotics Completion', 'Follow-up Infectious Disease', 'Home Health', 'Wound Care']
    elif arch == 'neuro':
        plans = ['Physical Therapy', 'Occupational Therapy', 'Follow-up Neurology', 'Speech Therapy']
    elif arch == 'diabetic':
        plans = ['Diabetes Education', 'Endocrinology Follow-up', 'Nutrition Counseling', 'Home Glucose Monitoring']
    elif arch == 'gi_surgical':
        plans = ['Surgical Follow-up', 'Wound Care', 'Dietary Modification', 'Pain Management']
    elif arch == 'renal':
        plans = ['Dialysis Schedule', 'Nephrology Follow-up', 'Dietary Restriction', 'Fluid Management']
    elif arch == 'substance':
        plans = ['Addiction Counseling', 'Psychiatric Follow-up', 'Support Group', 'Medication-Assisted Treatment']
    elif arch == 'trauma_surgical':
        plans = ['Physical Therapy', 'Surgical Follow-up', 'Wound Care', 'Pain Management']
    elif arch == 'cancer':
        plans = ['Oncology Follow-up', 'Chemotherapy Schedule', 'Palliative Care', 'Nutrition Support']
    else:
        plans = ['PCP Follow-up', 'Medication Review', 'Lifestyle Counseling']
    
    # Pick 1-3 plans
    k = min(np.random.randint(1, 4), len(plans))
    return '; '.join(np.random.choice(plans, k, replace=False))

df['care_plan_following_discharge'] = df.apply(assign_care_plan, axis=1)

# ─────────────────────────────────────────────────────────────
# 12. CLEAN UP & SAVE
# ─────────────────────────────────────────────────────────────
# Drop internal helper columns
df.drop(columns=['_age_num', '_archetype'], inplace=True)

# Reorder: original columns first, then new columns
original_cols = ['patientunitstayid', 'patienthealthsystemstayid', 'gender', 'age',
                 'ethnicity', 'hospitalid', 'wardid', 'apacheadmissiondx',
                 'admissionheight', 'hospitaladmittime24', 'hospitaladmitoffset',
                 'hospitaladmitsource', 'hospitaldischargeyear', 'hospitaldischargetime24',
                 'hospitaldischargeoffset', 'hospitaldischargelocation',
                 'hospitaldischargestatus', 'unittype', 'unitadmittime24',
                 'unitadmitsource', 'unitvisitnumber', 'unitstaytype',
                 'admissionweight', 'dischargeweight', 'unitdischargetime24',
                 'unitdischargeoffset', 'unitdischargelocation', 'unitdischargestatus',
                 'uniquepid']
new_cols = [c for c in df.columns if c not in original_cols]
df = df[original_cols + new_cols]

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved enriched dataset to: {OUTPUT_PATH}")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   New features added: {len(new_cols)}")
print(f"\n   New columns: {new_cols}")

# Quick sanity check
print("\n=== Sample Statistics ===")
vitals = ['bp_systolic', 'bp_diastolic', 'pulse', 'respirations', 'temperature']
labs = ['glucose', 'creatinine', 'wbc', 'hemoglobin', 'potassium', 'sodium', 'calcium']
print("\nVitals summary:")
print(df[vitals].describe().round(1).to_string())
print("\nLabs summary:")
print(df[labs].describe().round(2).to_string())
print("\nCost summary:")
print(df[['cost_of_initial_stay', 'care_plan_costs']].describe().round(2).to_string())
print("\nCondition flags:")
cond_cols = ['diabetes', 'obesity', 'anxiety', 'depression', 'dementia', 
             'drugabuse', 'mooddisorder', 'tobacco_user']
print(df[cond_cols].mean().round(3).to_string())
