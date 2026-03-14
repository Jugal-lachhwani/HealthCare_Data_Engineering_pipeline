"""
Add readmitted and readmitted_under_30_days features to EHR_enriched.csv

Logic for REALISM:
- Scores patients based on the feature importance screenshot provided by the user.
- Heaviest weights: cost_of_initial_stay, care_plan_costs, meds_gastro, ed_visits, meds_respiratory, meds_infective, glucose, age, bmi, los.
- The risk score is converted into probabilities.
- Probabilities are scaled to hit the EXACT target rates expected (14% and 7%).
"""

import pandas as pd
import numpy as np
import os, sys

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "Data", "EHR_enriched.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "Data", "EHR_enriched.csv")

df = pd.read_csv(INPUT_PATH)
n = len(df)
print(f"Loaded {n} rows, {df.shape[1]} columns")

# Drop the columns if they already exist from a previous run so we can cleanly recreate them
df = df.drop(columns=['readmitted', 'readmitted_under_30_days'], errors='ignore')

def norm(col):
    q99 = col.quantile(0.99)
    if q99 == 0: q99 = col.max()
    if q99 == 0: q99 = 1.0
    return np.clip(col / q99, 0, 1.2)

age_num = df['age'].apply(lambda x: 92 if isinstance(x, str) and '>' in str(x) else pd.to_numeric(x, errors='coerce')).fillna(65)

# 1. Feature Importance Weights
weights = {
    'cost_of_initial_stay': 0.080, 'care_plan_costs': 0.060, 'meds_gastro': 0.040,
    'ed_visits': 0.035, 'meds_respiratory': 0.033, 'meds_anesthetics': 0.033,
    'meds_infective': 0.033, 'glucose': 0.033, 'age_num': 0.032, 'bmi': 0.032,
    'meds_central_nervous_system': 0.031, 'meds_hematological': 0.030,
    'los': 0.028, 'meds_cardio_agents': 0.026, 'meds_endocrine': 0.026,
    'creatinine': 0.025, 'meds_topical': 0.022, 'chronic_conditions': 0.015,
}

df['_age_num'] = age_num
raw_score = np.zeros(n)

for feature, weight in weights.items():
    if feature == 'age_num': col = df['_age_num']
    elif feature in df.columns: col = df[feature]
    else: continue
    raw_score += norm(col) * weight

# Base probabilities
base_p = raw_score.copy()

# Target: 14% readmitted overall (user asked for 13-15%)
target_readmit = 0.14
# Scale base_p so its mean is exactly the target
scale_factor = target_readmit / np.mean(base_p)
prob_readmit = np.clip(base_p * scale_factor, 0, 0.95)

# Keep sampling until we are strictly inside the 13-15% range
for _ in range(100):
    readmitted = (np.random.rand(n) < prob_readmit).astype(int)
    r_rate = readmitted.mean()
    if 0.13 <= r_rate <= 0.15:
        break

df['readmitted'] = readmitted

# Now early readmissions. Target: 7% overall (half of 14, in the 5-10% range)
# We make acute factors (ed_visits, meds_infective, los) stronger predictors for *early* readmission
acute_score = norm(df['ed_visits']) + norm(df['meds_infective']) + norm(df['los'])
prob_early_given_re = base_p + 0.3 * acute_score

# We only consider rows where readmitted == 1
re_idx = df['readmitted'] == 1
n_re = re_idx.sum()

target_early_overall = 0.075 # 7.5% overall
target_early_given_re = target_early_overall / r_rate # ~53% of the readmitted ones

# Scale the probabilities for just the readmitted subset
scale_early = target_early_given_re / np.mean(prob_early_given_re[re_idx])
prob_early = np.clip(prob_early_given_re * scale_early, 0, 0.98)

readmitted_30 = np.zeros(n, dtype=int)

# Sample until we hit 5-10% range
for _ in range(100):
    # Only sample for the indices where readmitted == 1
    sampled = (np.random.rand(n_re) < prob_early[re_idx]).astype(int)
    readmitted_30[re_idx] = sampled
    r30_rate = readmitted_30.mean()
    if 0.05 <= r30_rate <= 0.10:
        break

df['readmitted_under_30_days'] = readmitted_30

# Cleanup
df.drop(columns=['_age_num'], inplace=True)

# Reorder
cols = list(df.columns)
for c in ['readmitted', 'readmitted_under_30_days']: cols.remove(c)
cond_idx = cols.index('Condition')
cols = cols[:cond_idx] + ['readmitted', 'readmitted_under_30_days'] + cols[cond_idx:]
df = df[cols]

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved to: {OUTPUT_PATH}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

print("\n=== READMISSION TARGET RATES ===")
print(f"Readmitted overall:                {df['readmitted'].mean()*100:.1f}%  (Target: 13-15%)")
print(f"Readmitted under 30 days overall:  {df['readmitted_under_30_days'].mean()*100:.1f}%  (Target: 5-10%)")

print("\n=== COMPARISON (R=Readmitted, NR=Not) ===")
r = df['readmitted'] == 1
nr = df['readmitted'] == 0

for col in ['cost_of_initial_stay', 'care_plan_costs', 'meds_gastro', 'ed_visits', 'los', 'glucose', 'age', 'bmi', 'creatinine']:
    if col == 'age':
        print(f"  age: R={age_num[r].mean():.1f}  NR={age_num[nr].mean():.1f}")
    else:
        print(f"  {col}: R={df[col][r].mean():.1f}  NR={df[col][nr].mean():.1f}")
