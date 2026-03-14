"""
Add explicit readmission dates to EHR_enriched.csv based on the readmission flags.

Logic:
- For patients where `readmitted` == 1:
  - Generate a `days_until_readmission` gap.
  - If `readmitted_under_30_days` == 1, the gap is between 1 and 29 days.
  - If `readmitted` == 1 but not under 30 days, the gap is between 30 and 365 days.
- Calculate `readmission_admit_day` = original `discharge_day` + gap.
- Calculate `readmission_discharge_day` = `readmission_admit_day` + a new length of stay calculation (similar to their original LOS but slightly adjusted for severity).
- For non-readmitted patients, leave these fields blank.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, sys

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "Data", "EHR_enriched.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "Data", "EHR_enriched.csv")

df = pd.read_csv(INPUT_PATH)
n = len(df)
print(f"Loaded {n} rows, {df.shape[1]} columns")

# Drop if they already exist from prior runs
df = df.drop(columns=['readmission_admit_day', 'readmission_discharge_day', 'readmission_total_days', 'days_until_readmission'], errors='ignore')

# We need to parse dates back to objects
# Current format: DD-MM-YY (e.g. 05-02-15) - handle standard formatting
# Need datetime objects for addition
try:
    # Try parsing with specific format first
    orig_discharge = pd.to_datetime(df['discharge_day'], format='%d-%m-%y')
except (ValueError, Exception):
    # Fallback to general parsing with dayfirst
    orig_discharge = pd.to_datetime(df['discharge_day'], dayfirst=True)

days_until = []
readmit_admits = []
readmit_discharges = []
readmit_los = []

for i in range(n):
    is_readmit = df['readmitted'].iloc[i] == 1
    is_under_30 = df['readmitted_under_30_days'].iloc[i] == 1
    orig_los = int(df['total_days'].iloc[i])
    cond = df['Condition'].iloc[i]
    
    if not is_readmit:
        # Not readmitted
        days_until.append(np.nan)
        readmit_admits.append(np.nan)
        readmit_discharges.append(np.nan)
        readmit_los.append(np.nan)
        continue
        
    # Generate gap before readmission
    if is_under_30:
        # Heavily front-load early readmissions (days 2-15 are most common)
        # Using a gamma distribution or bounded pareto is realistic
        gap = int(np.clip(np.random.gamma(shape=2.0, scale=4.0), 1, 29))
    else:
        # Readmission after 30 days but within the year
        # Gap between 30 and 365, skewed toward earlier
        gap = int(np.clip(np.random.gamma(shape=2.5, scale=40.0) + 30, 30, 365))
        
    # Generate length of readmission stay
    # Usually, readmissions are slightly shorter or equal to index admission, unless it's a severe complication
    if orig_los > 10:
        new_los = int(np.clip(np.random.normal(orig_los * 0.7, 3), 3, 30))
    else:
        new_los = int(np.clip(np.random.normal(orig_los, 2), 1, 15))
        
    # Calculate exact dates
    base_date = orig_discharge.iloc[i]
    if pd.isna(base_date):
        base_date = datetime(2015, 6, 1) # Fallback
        
    r_admit = base_date + timedelta(days=gap)
    r_discharge = r_admit + timedelta(days=new_los)
    
    days_until.append(gap)
    readmit_admits.append(r_admit.strftime('%d-%m-%y'))
    readmit_discharges.append(r_discharge.strftime('%d-%m-%y'))
    readmit_los.append(new_los)

df['days_until_readmission'] = days_until
df['readmission_admit_day'] = readmit_admits
df['readmission_discharge_day'] = readmit_discharges
df['readmission_total_days'] = readmit_los

# Move columns exactly after the original readmitted flags
cols = list(df.columns)
# Pop off the new ones
for c in ['days_until_readmission', 'readmission_admit_day', 'readmission_discharge_day', 'readmission_total_days']:
    cols.remove(c)

# Insert after readmitted_under_30_days
idx = cols.index('readmitted_under_30_days') + 1
cols = cols[:idx] + ['days_until_readmission', 'readmission_admit_day', 'readmission_discharge_day', 'readmission_total_days'] + cols[idx:]
df = df[cols]

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved to: {OUTPUT_PATH}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# VERIFICATION
print("\n=== SAMPLE READMISSION DATES ===")
# Show a few under 30 days
print("\nUnder 30 Days sample:")
sample_cols = ['Condition', 'discharge_day', 'readmission_admit_day', 'days_until_readmission', 'readmission_discharge_day', 'readmission_total_days']
u30 = df[df['readmitted_under_30_days'] == 1].head(5)
print(u30[sample_cols].to_string())

# Show a few over 30 days
print("\nOver 30 Days sample:")
o30 = df[(df['readmitted'] == 1) & (df['readmitted_under_30_days'] == 0)].head(5)
print(o30[sample_cols].to_string())

print("\n=== DAYS UNTIL READMISSION STATS ===")
print("Under 30 days:")
print(df[df['readmitted_under_30_days'] == 1]['days_until_readmission'].describe().round(1).to_string())
print("\nOver 30 days:")
print(df[(df['readmitted'] == 1) & (df['readmitted_under_30_days'] == 0)]['days_until_readmission'].describe().round(1).to_string())
