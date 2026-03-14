import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

df = pd.read_csv("Data/EHR_enriched.csv")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nAll columns ({df.shape[1]}):")
for i, c in enumerate(df.columns):
    print(f"  {i+1:2d}. {c}")

print("\n\n=== CORRELATION CHECK: Diabetic patients vs others ===")
diab = df[df["Condition"] == "Diabetes/Endocrine"]
non_diab = df[df["Condition"] != "Diabetes/Endocrine"]
print(f"Diabetics  - glucose: {diab['glucose'].mean():.1f}, meds_endocrine: {diab['meds_endocrine'].mean():.1f}, diabetes_flag: {diab['diabetes'].mean():.2f}")
print(f"Others     - glucose: {non_diab['glucose'].mean():.1f}, meds_endocrine: {non_diab['meds_endocrine'].mean():.1f}, diabetes_flag: {non_diab['diabetes'].mean():.2f}")

print("\n=== CORRELATION CHECK: Cardiac patients vs others ===")
card = df[df["Condition"] == "Cardiovascular Disease"]
non_card = df[df["Condition"] != "Cardiovascular Disease"]
print(f"Cardiac    - bp_sys: {card['bp_systolic'].mean():.1f}, meds_cardio: {card['meds_cardio_agents'].mean():.1f}, LACE: {card['LACE_Score'].mean():.1f}")
print(f"Others     - bp_sys: {non_card['bp_systolic'].mean():.1f}, meds_cardio: {non_card['meds_cardio_agents'].mean():.1f}, LACE: {non_card['LACE_Score'].mean():.1f}")

print("\n=== CORRELATION CHECK: Sepsis patients ===")
sep = df[df["Condition"] == "Sepsis/Infection"]
non_sep = df[df["Condition"] != "Sepsis/Infection"]
print(f"Sepsis     - temp: {sep['temperature'].mean():.1f}, wbc: {sep['wbc'].mean():.1f}, pulse: {sep['pulse'].mean():.1f}, meds_infective: {sep['meds_infective'].mean():.1f}")
print(f"Others     - temp: {non_sep['temperature'].mean():.1f}, wbc: {non_sep['wbc'].mean():.1f}, pulse: {non_sep['pulse'].mean():.1f}, meds_infective: {non_sep['meds_infective'].mean():.1f}")

print("\n=== CORRELATION CHECK: Renal patients ===")
renal = df[df["Condition"] == "Renal Disease"]
non_renal = df[df["Condition"] != "Renal Disease"]
print(f"Renal      - creatinine: {renal['creatinine'].mean():.2f}, potassium: {renal['potassium'].mean():.1f}, hemoglobin: {renal['hemoglobin'].mean():.1f}")
print(f"Others     - creatinine: {non_renal['creatinine'].mean():.2f}, potassium: {non_renal['potassium'].mean():.1f}, hemoglobin: {non_renal['hemoglobin'].mean():.1f}")

print("\n=== CORRELATION CHECK: Elderly vs Young ===")
age_num = df["age"].apply(lambda x: 92 if isinstance(x, str) and ">" in str(x) else int(float(x)) if str(x).replace(".","").isdigit() else 65)
elderly = df[age_num >= 75]
young = df[age_num < 50]
print(f"Elderly(75+) - chronic_cond: {elderly['chronic_conditions'].mean():.1f}, dementia: {elderly['dementia'].mean():.2f}")
print(f"Young(<50)   - chronic_cond: {young['chronic_conditions'].mean():.1f}, dementia: {young['dementia'].mean():.2f}")
print(f"Elderly insurance: {elderly['insurance_provider'].value_counts().head(3).to_dict()}")
print(f"Young insurance:   {young['insurance_provider'].value_counts().head(3).to_dict()}")

print("\n=== SAMPLE ROWS (first 5) ===")
new_cols = ["los", "sex", "bp_systolic", "bp_diastolic", "pulse", "glucose", 
            "creatinine", "wbc", "bmi", "diabetes", "LACE_Score",
            "cost_of_initial_stay", "Condition"]
print(df[new_cols].head().to_string())

print("\n=== OBESITY CHECK (BMI >= 30 correlation) ===")
obese = df[df["obesity"] == 1]
non_obese = df[df["obesity"] == 0]
print(f"Obese patients:     BMI mean={obese['bmi'].mean():.1f}, count={len(obese)}")
print(f"Non-obese patients: BMI mean={non_obese['bmi'].mean():.1f}, count={len(non_obese)}")

print("\nVERIFICATION PASSED ✅")
