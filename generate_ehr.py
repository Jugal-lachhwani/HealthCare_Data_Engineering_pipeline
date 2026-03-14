"""
╔══════════════════════════════════════════════════════════════════════╗
║  Synthetic EHR Generator — KD&A-1: Patient Readmission Prediction    ║ 
║  Profile-based generation: 10 clinical archetypes                    ║
║  Every patient's labs, meds, and outcome are medically consistent    ║
║                                                                      ║
║  OUTPUT (folder: synthetic_ehr_output/):                             ║
║    patients.csv · diagnoses.csv · lab_results.csv                    ║
║    medications.csv · prior_admissions.csv                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm

np.random.seed(42)
random.seed(42)

N_PATIENTS = 500_000
OUTPUT_DIR = "synthetic_ehr_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n🏥  Synthetic EHR Generator — {N_PATIENTS:,} patients across 10 clinical archetypes\n")

# ══════════════════════════════════════════════════════════════════════════════
#  CLINICAL ARCHETYPE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

ARCHETYPES = {

    "Sepsis": {
        "weight": 0.13,
        "age_range": (45, 85),
        "gender_bias": 0.52,
        "severity": 3,
        "los_hours_mean": 8000,
        "readmit_base": 0.32,
        "apache_dx": "Sepsis, pulmonary",
        "unit_types": ["MICU", "Med-Surg ICU", "SICU"],
        "admit_sources": ["Emergency Department", "Floor"],
        "icd_primary": ["A41.9", "A41.51"],
        "icd_comorbidities": ["I10", "E11.9", "N17.9", "J44.1", "K74.60"],
        "discharge_locs": {
            "Home": 0.25, "Skilled Nursing Facility": 0.22, "Death": 0.20,
            "Other Hospital": 0.12, "Rehabilitation": 0.08,
            "Other": 0.08, "Nursing Home": 0.03, "Other External": 0.02,
        },
        "lab_deltas": {
            "WBC": 2.2, "Lactate": 3.5, "Procalcitonin": 15.0, "CRP": 12.0,
            "Heart Rate": 1.3, "Resp Rate": 1.4, "Systolic BP": 0.75,
            "Diastolic BP": 0.72, "Platelets": 0.55, "Albumin": 0.65,
            "Bicarbonate": 0.78, "Creatinine": 2.0, "BUN": 2.5,
            "Glucose": 1.3, "Temperature": 1.025, "SpO2": 0.94,
        },
        "mandatory_meds": ["Vancomycin", "Piperacillin/Tazo", "Norepinephrine", "IV Fluid (NS)", "Pantoprazole"],
        "optional_meds": ["Vasopressin", "Hydrocortisone", "Heparin", "Insulin Regular", "Furosemide", "Meropenem"],
    },

    "Acute_MI": {
        "weight": 0.12,
        "age_range": (50, 82),
        "gender_bias": 0.62,
        "severity": 3,
        "los_hours_mean": 3500,
        "readmit_base": 0.28,
        "apache_dx": "Infarction, acute myocardial (MI)",
        "unit_types": ["Cardiac ICU", "CCU-CTICU", "Med-Surg ICU"],
        "admit_sources": ["Emergency Department", "Direct Admit"],
        "icd_primary": ["I21.9", "I21.0"],
        "icd_comorbidities": ["I10", "E78.5", "I25.10", "E11.9", "I48.91", "Z87.891"],
        "discharge_locs": {
            "Home": 0.50, "Skilled Nursing Facility": 0.18, "Death": 0.10,
            "Rehabilitation": 0.12, "Other Hospital": 0.06, "Other": 0.04,
        },
        "lab_deltas": {
            "Troponin I": 25.0, "BNP": 8.0, "Heart Rate": 1.25,
            "Systolic BP": 0.85, "WBC": 1.3, "Glucose": 1.2,
            "Creatinine": 1.3, "Platelets": 0.9, "CRP": 3.0, "Lactate": 1.8,
        },
        "mandatory_meds": ["Aspirin", "Heparin", "Atorvastatin", "Metoprolol", "Nitroglycerin"],
        "optional_meds": ["Clopidogrel", "Ticagrelor", "Lisinopril", "Furosemide", "Morphine", "Warfarin"],
    },

    "CHF": {
        "weight": 0.11,
        "age_range": (60, 90),
        "gender_bias": 0.50,
        "severity": 2,
        "los_hours_mean": 4500,
        "readmit_base": 0.35,
        "apache_dx": "CHF, congestive heart failure",
        "unit_types": ["Cardiac ICU", "Med-Surg ICU", "CCU-CTICU"],
        "admit_sources": ["Emergency Department", "Direct Admit", "Floor"],
        "icd_primary": ["I50.9", "I50.22"],
        "icd_comorbidities": ["I10", "I48.91", "E78.5", "N18.6", "E11.9", "I25.10"],
        "discharge_locs": {
            "Home": 0.42, "Skilled Nursing Facility": 0.22, "Death": 0.08,
            "Rehabilitation": 0.10, "Other Hospital": 0.08, "Nursing Home": 0.06, "Other": 0.04,
        },
        "lab_deltas": {
            "BNP": 12.0, "Troponin I": 2.5, "Creatinine": 1.6, "BUN": 1.8,
            "Sodium": 0.97, "Potassium": 1.05, "Albumin": 0.80,
            "Heart Rate": 1.2, "Systolic BP": 0.88, "SpO2": 0.95,
            "Resp Rate": 1.3, "CRP": 2.0,
        },
        "mandatory_meds": ["Furosemide", "Spironolactone", "Lisinopril", "Metoprolol", "Digoxin"],
        "optional_meds": ["Carvedilol", "Hydralazine", "Heparin", "Warfarin", "Pantoprazole", "Atorvastatin", "Insulin Glargine"],
    },

    "COPD": {
        "weight": 0.09,
        "age_range": (55, 85),
        "gender_bias": 0.53,
        "severity": 2,
        "los_hours_mean": 3200,
        "readmit_base": 0.30,
        "apache_dx": "Emphysema/bronchitis",
        "unit_types": ["MICU", "Med-Surg ICU"],
        "admit_sources": ["Emergency Department", "Direct Admit"],
        "icd_primary": ["J44.1", "J44.0"],
        "icd_comorbidities": ["J18.9", "I10", "I25.10", "Z87.891", "E11.9", "I50.9"],
        "discharge_locs": {
            "Home": 0.48, "Skilled Nursing Facility": 0.20, "Death": 0.07,
            "Rehabilitation": 0.10, "Other Hospital": 0.08, "Other": 0.07,
        },
        "lab_deltas": {
            "pCO2": 1.45, "pO2": 0.72, "SpO2": 0.92, "pH": 0.985,
            "Bicarbonate": 1.15, "Resp Rate": 1.5, "WBC": 1.4,
            "CRP": 4.0, "Procalcitonin": 3.0, "Hemoglobin": 1.05, "Heart Rate": 1.2,
        },
        "mandatory_meds": ["Albuterol", "Ipratropium", "Methylprednisolone"],
        "optional_meds": ["Ceftriaxone", "Azithromycin", "Theophylline", "Furosemide", "Pantoprazole", "Heparin"],
    },

    "DKA": {
        "weight": 0.08,
        "age_range": (18, 60),
        "gender_bias": 0.48,
        "severity": 2,
        "los_hours_mean": 2200,
        "readmit_base": 0.22,
        "apache_dx": "Diabetic ketoacidosis",
        "unit_types": ["MICU", "Med-Surg ICU"],
        "admit_sources": ["Emergency Department"],
        "icd_primary": ["E13.10", "E11.65"],
        "icd_comorbidities": ["I10", "N17.9", "E78.5", "I25.10"],
        "discharge_locs": {
            "Home": 0.68, "Skilled Nursing Facility": 0.08, "Death": 0.02,
            "Rehabilitation": 0.06, "Other Hospital": 0.08, "Other": 0.08,
        },
        "lab_deltas": {
            "Glucose": 4.0, "HbA1c": 1.8, "Bicarbonate": 0.55, "pH": 0.974,
            "Potassium": 1.25, "Sodium": 0.97, "Creatinine": 1.4, "BUN": 1.6,
            "Lactate": 1.4, "WBC": 1.3, "Heart Rate": 1.2, "Resp Rate": 1.3,
        },
        "mandatory_meds": ["Insulin Regular", "IV Fluid (NS)", "Potassium Chloride"],
        "optional_meds": ["Insulin Glargine", "Pantoprazole", "Metformin", "Ondansetron", "Heparin"],
    },

    "Stroke": {
        "weight": 0.09,
        "age_range": (55, 88),
        "gender_bias": 0.52,
        "severity": 3,
        "los_hours_mean": 5500,
        "readmit_base": 0.26,
        "apache_dx": "CVA, cerebrovascular accident/stroke",
        "unit_types": ["Neuro ICU", "Med-Surg ICU"],
        "admit_sources": ["Emergency Department"],
        "icd_primary": ["I63.9", "I61.9"],
        "icd_comorbidities": ["I10", "I48.91", "E11.9", "E78.5", "I25.10", "Z87.891"],
        "discharge_locs": {
            "Home": 0.28, "Skilled Nursing Facility": 0.30, "Death": 0.12,
            "Rehabilitation": 0.20, "Other Hospital": 0.06, "Other": 0.04,
        },
        "lab_deltas": {
            "Glucose": 1.25, "WBC": 1.2, "CRP": 3.0, "Heart Rate": 1.1,
            "Systolic BP": 1.18, "Diastolic BP": 1.15, "Troponin I": 1.5,
            "Sodium": 0.98, "Creatinine": 1.2,
        },
        "mandatory_meds": ["Aspirin", "Atorvastatin", "Heparin"],
        "optional_meds": ["Clopidogrel", "Warfarin", "Lisinopril", "Metoprolol", "Insulin Regular", "Pantoprazole", "Labetalol"],
    },

    "Renal_Failure": {
        "weight": 0.09,
        "age_range": (50, 85),
        "gender_bias": 0.55,
        "severity": 3,
        "los_hours_mean": 6000,
        "readmit_base": 0.33,
        "apache_dx": "Renal failure, acute",
        "unit_types": ["MICU", "Med-Surg ICU", "SICU"],
        "admit_sources": ["Emergency Department", "Floor"],
        "icd_primary": ["N17.9", "N18.6"],
        "icd_comorbidities": ["I10", "E11.9", "I50.9", "E78.5", "A41.9"],
        "discharge_locs": {
            "Home": 0.28, "Skilled Nursing Facility": 0.25, "Death": 0.18,
            "Rehabilitation": 0.10, "Other Hospital": 0.12, "Other": 0.07,
        },
        "lab_deltas": {
            "Creatinine": 4.5, "BUN": 3.8, "Potassium": 1.35, "Phosphorus": 1.6,
            "eGFR": 0.20, "Bicarbonate": 0.80, "Hemoglobin": 0.75,
            "Albumin": 0.75, "Sodium": 0.97, "pH": 0.985, "WBC": 1.2,
        },
        "mandatory_meds": ["Furosemide", "Sodium Bicarbonate", "Pantoprazole"],
        "optional_meds": ["Calcium Gluconate", "Insulin Regular", "Heparin", "Erythropoietin", "Vancomycin", "Albuterol"],
    },

    "Post_Surgical": {
        "weight": 0.10,
        "age_range": (40, 80),
        "gender_bias": 0.50,
        "severity": 2,
        "los_hours_mean": 2500,
        "readmit_base": 0.15,
        "apache_dx": "CABG alone, coronary artery bypass grafting",
        "unit_types": ["CTICU", "CSICU", "Med-Surg ICU"],
        "admit_sources": ["Operating Room", "Recovery Room"],
        "icd_primary": ["I35.0", "I25.10"],
        "icd_comorbidities": ["I10", "E11.9", "E78.5", "I48.91", "Z87.891"],
        "discharge_locs": {
            "Home": 0.55, "Skilled Nursing Facility": 0.15, "Death": 0.04,
            "Rehabilitation": 0.16, "Other Hospital": 0.06, "Other": 0.04,
        },
        "lab_deltas": {
            "Hemoglobin": 0.80, "Hematocrit": 0.80, "Platelets": 0.85,
            "Creatinine": 1.2, "WBC": 1.3, "CRP": 2.5,
            "Troponin I": 2.0, "Glucose": 1.15, "PT/INR": 1.2, "Lactate": 1.3,
        },
        "mandatory_meds": ["Aspirin", "Heparin", "Atorvastatin", "Metoprolol", "Pantoprazole"],
        "optional_meds": ["Morphine", "Fentanyl", "Amiodarone", "Furosemide", "Insulin Regular", "Ceftriaxone"],
    },

    "Trauma": {
        "weight": 0.10,
        "age_range": (18, 65),
        "gender_bias": 0.65,
        "severity": 3,
        "los_hours_mean": 5000,
        "readmit_base": 0.18,
        "apache_dx": "Trauma - multiple",
        "unit_types": ["SICU", "Med-Surg ICU"],
        "admit_sources": ["Emergency Department"],
        "icd_primary": ["S72.001", "S06.300"],
        "icd_comorbidities": ["D62", "R57.0", "J96.00", "N17.9"],
        "discharge_locs": {
            "Home": 0.40, "Rehabilitation": 0.25, "Skilled Nursing Facility": 0.15,
            "Death": 0.10, "Other Hospital": 0.06, "Other": 0.04,
        },
        "lab_deltas": {
            "Hemoglobin": 0.65, "Hematocrit": 0.65, "Platelets": 0.70,
            "Lactate": 2.8, "PT/INR": 1.5, "PTT": 1.4,
            "WBC": 1.5, "CRP": 5.0, "Heart Rate": 1.3,
            "Systolic BP": 0.82, "Creatinine": 1.3, "Glucose": 1.2,
        },
        "mandatory_meds": ["Morphine", "IV Fluid (NS)", "Heparin", "Pantoprazole", "Ceftriaxone"],
        "optional_meds": ["Fentanyl", "Propofol", "Lorazepam", "Norepinephrine", "Tranexamic Acid", "Vitamin K"],
    },

    "Overdose": {
        "weight": 0.09,
        "age_range": (18, 55),
        "gender_bias": 0.45,
        "severity": 2,
        "los_hours_mean": 1800,
        "readmit_base": 0.20,
        "apache_dx": "Overdose, sedatives, hypnotics, antipsychotics, benzodiazepines",
        "unit_types": ["MICU", "Med-Surg ICU"],
        "admit_sources": ["Emergency Department"],
        "icd_primary": ["T39.1X1", "T40.601"],
        "icd_comorbidities": ["F10.20", "G93.1", "J96.00", "R57.0"],
        "discharge_locs": {
            "Home": 0.55, "Other Hospital": 0.15, "Death": 0.06,
            "Other": 0.12, "Rehabilitation": 0.08, "Skilled Nursing Facility": 0.04,
        },
        "lab_deltas": {
            "pH": 0.978, "pO2": 0.78, "SpO2": 0.93, "Bicarbonate": 0.85,
            "Glucose": 0.85, "Heart Rate": 0.72, "Systolic BP": 0.85,
            "Resp Rate": 0.68, "Creatinine": 1.2, "ALT": 1.5, "AST": 1.6, "Lactate": 1.6,
        },
        "mandatory_meds": ["Naloxone", "IV Fluid (NS)", "Thiamine"],
        "optional_meds": ["Lorazepam", "Propofol", "Pantoprazole", "Norepinephrine", "Activated Charcoal"],
    },
}

ARCHETYPE_NAMES   = list(ARCHETYPES.keys())
ARCHETYPE_WEIGHTS = [ARCHETYPES[a]["weight"] for a in ARCHETYPE_NAMES]

# ── ICD-10 descriptions ───────────────────────────────────────────────────────

ICD10_DESC = {
    "A41.9":"Sepsis, unspecified organism", "A41.51":"Sepsis due to E. coli",
    "I21.9":"Acute myocardial infarction, unspecified", "I21.0":"STEMI of anterior wall",
    "I50.9":"Heart failure, unspecified", "I50.22":"Chronic systolic heart failure",
    "I63.9":"Cerebral infarction, unspecified", "I61.9":"Nontraumatic intracerebral hemorrhage",
    "J44.1":"COPD with acute exacerbation", "J44.0":"COPD with acute lower respiratory infection",
    "N17.9":"Acute kidney failure, unspecified", "N18.6":"End stage renal disease",
    "E13.10":"Diabetic ketoacidosis without coma", "E11.65":"T2DM with hyperglycemia",
    "I35.0":"Nonrheumatic aortic valve stenosis", "I25.10":"Atherosclerotic heart disease",
    "S72.001":"Fracture of right femoral head", "S06.300":"Unspecified focal traumatic brain injury",
    "T39.1X1":"Poisoning by analgesics, accidental", "T40.601":"Poisoning by unspecified opioids, accidental",
    "I10":"Essential hypertension", "E11.9":"Type 2 diabetes mellitus",
    "E78.5":"Hyperlipidemia, unspecified", "I48.91":"Unspecified atrial fibrillation",
    "J18.9":"Pneumonia, unspecified", "K74.60":"Unspecified cirrhosis of liver",
    "Z87.891":"Personal history of nicotine dependence",
    "D62":"Acute posthemorrhagic anemia", "R57.0":"Cardiogenic shock",
    "J96.00":"Acute respiratory failure", "F10.20":"Alcohol use disorder, moderate",
    "G93.1":"Anoxic brain damage",
}

# ── Normal lab values: (mean, std, critical_low, critical_high) ───────────────

LAB_NORMALS = {
    "WBC":           (7.5,   2.5,    2.0,   20.0),
    "Hemoglobin":    (13.0,  2.0,    6.0,   18.0),
    "Hematocrit":    (39.0,  6.0,   18.0,   55.0),
    "Platelets":     (220.0, 70.0,  50.0,  600.0),
    "Sodium":        (140.0,  4.0, 120.0,  160.0),
    "Potassium":     (4.0,    0.6,   2.5,    6.5),
    "Creatinine":    (1.0,    0.4,   0.3,   12.0),
    "BUN":           (15.0,   7.0,   3.0,  100.0),
    "Glucose":       (100.0, 30.0,  40.0,  600.0),
    "Bicarbonate":   (24.0,   3.0,  10.0,   40.0),
    "Chloride":      (102.0,  5.0,  80.0,  120.0),
    "Calcium":       (9.2,    0.8,   6.0,   14.0),
    "Albumin":       (4.0,    0.6,   1.5,    5.5),
    "ALT":           (25.0,  15.0,   3.0,  500.0),
    "AST":           (28.0,  18.0,   5.0,  500.0),
    "Total Bilirubin":(0.8,   0.4,   0.1,   20.0),
    "Troponin I":    (0.02,  0.04,  0.00,    5.0),
    "BNP":           (100.0,150.0,  10.0, 5000.0),
    "Lactate":       (1.2,   0.8,   0.5,   15.0),
    "PT/INR":        (1.1,   0.3,   0.8,    8.0),
    "PTT":           (30.0,  8.0,  15.0,  150.0),
    "SpO2":          (97.0,  2.5,  80.0,  100.0),
    "Heart Rate":    (80.0, 18.0,  30.0,  200.0),
    "Systolic BP":   (120.0,22.0,  60.0,  220.0),
    "Diastolic BP":  (75.0, 14.0,  30.0,  140.0),
    "Temperature":   (37.0,  0.8,  34.0,   42.0),
    "Resp Rate":     (16.0,  5.0,   6.0,   40.0),
    "pH":            (7.40,  0.06,  6.9,    7.8),
    "pO2":           (90.0, 20.0,  40.0,  200.0),
    "pCO2":          (40.0,  8.0,  20.0,   80.0),
    "Magnesium":     (2.0,   0.4,   1.0,    4.0),
    "Phosphorus":    (3.5,   0.8,   1.0,    8.0),
    "eGFR":          (75.0, 30.0,   5.0,  130.0),
    "HbA1c":         (6.0,   1.5,   4.0,   14.0),
    "CRP":           (5.0,  10.0,   0.1,  300.0),
    "Procalcitonin": (0.1,   0.5,   0.05,  50.0),
}

LAB_UNITS = {
    "WBC":"10^3/uL", "Hemoglobin":"g/dL", "Hematocrit":"%", "Platelets":"10^3/uL",
    "Sodium":"mEq/L", "Potassium":"mEq/L", "Creatinine":"mg/dL", "BUN":"mg/dL",
    "Glucose":"mg/dL", "Bicarbonate":"mEq/L", "Chloride":"mEq/L", "Calcium":"mg/dL",
    "Albumin":"g/dL", "ALT":"U/L", "AST":"U/L", "Total Bilirubin":"mg/dL",
    "Troponin I":"ng/mL", "BNP":"pg/mL", "Lactate":"mmol/L", "PT/INR":"ratio",
    "PTT":"seconds", "SpO2":"%", "Heart Rate":"bpm", "Systolic BP":"mmHg",
    "Diastolic BP":"mmHg", "Temperature":"°C", "Resp Rate":"breaths/min",
    "pH":"", "pO2":"mmHg", "pCO2":"mmHg", "Magnesium":"mg/dL",
    "Phosphorus":"mg/dL", "eGFR":"mL/min/1.73m2", "HbA1c":"%",
    "CRP":"mg/L", "Procalcitonin":"ng/mL",
}

DRUG_META = {
    "Vancomycin":        ("Antibiotic",      ["1g","1.5g","2g"],                ["IV"]),
    "Piperacillin/Tazo": ("Antibiotic",      ["3.375g","4.5g"],                 ["IV"]),
    "Meropenem":         ("Antibiotic",      ["500mg","1g"],                    ["IV"]),
    "Ceftriaxone":       ("Antibiotic",      ["1g","2g"],                       ["IV"]),
    "Azithromycin":      ("Antibiotic",      ["250mg","500mg"],                 ["Oral","IV"]),
    "Norepinephrine":    ("Vasopressor",     ["0.05mcg/kg/min","0.1mcg/kg/min"],["IV"]),
    "Vasopressin":       ("Vasopressor",     ["0.03U/min","0.04U/min"],         ["IV"]),
    "Dopamine":          ("Vasopressor",     ["5mcg/kg/min","10mcg/kg/min"],    ["IV"]),
    "IV Fluid (NS)":     ("IV Fluid",        ["500mL","1000mL"],                ["IV"]),
    "Pantoprazole":      ("PPI",             ["40mg"],                          ["Oral","IV"]),
    "Hydrocortisone":    ("Corticosteroid",  ["50mg","100mg"],                  ["IV"]),
    "Heparin":           ("Anticoagulant",   ["5000U","10000U"],                ["IV","Subcutaneous"]),
    "Insulin Regular":   ("Antidiabetic",    ["5U","10U","20U"],                ["IV","Subcutaneous"]),
    "Insulin Glargine":  ("Antidiabetic",    ["10U","20U","30U"],               ["Subcutaneous"]),
    "Furosemide":        ("Diuretic",        ["20mg","40mg","80mg"],            ["Oral","IV"]),
    "Aspirin":           ("Antiplatelet",    ["81mg","325mg"],                  ["Oral"]),
    "Atorvastatin":      ("Statin",          ["20mg","40mg","80mg"],            ["Oral"]),
    "Metoprolol":        ("Beta-blocker",    ["25mg","50mg","100mg"],           ["Oral","IV"]),
    "Nitroglycerin":     ("Nitrate",         ["0.4mg","5mcg/min"],              ["Sublingual","IV"]),
    "Clopidogrel":       ("Antiplatelet",    ["75mg"],                          ["Oral"]),
    "Ticagrelor":        ("Antiplatelet",    ["90mg"],                          ["Oral"]),
    "Lisinopril":        ("ACE Inhibitor",   ["5mg","10mg","20mg"],             ["Oral"]),
    "Warfarin":          ("Anticoagulant",   ["2mg","5mg","10mg"],              ["Oral"]),
    "Spironolactone":    ("Diuretic",        ["25mg","50mg"],                   ["Oral"]),
    "Digoxin":           ("Antiarrhythmic",  ["0.125mg","0.25mg"],              ["Oral","IV"]),
    "Carvedilol":        ("Beta-blocker",    ["6.25mg","12.5mg"],               ["Oral"]),
    "Hydralazine":       ("Vasodilator",     ["25mg","50mg"],                   ["Oral","IV"]),
    "Albuterol":         ("Bronchodilator",  ["2.5mg"],                         ["Nebulizer"]),
    "Ipratropium":       ("Bronchodilator",  ["0.5mg"],                         ["Nebulizer"]),
    "Methylprednisolone":("Corticosteroid",  ["40mg","125mg"],                  ["IV"]),
    "Theophylline":      ("Bronchodilator",  ["200mg","400mg"],                 ["Oral"]),
    "Potassium Chloride":("Electrolyte",     ["20mEq","40mEq"],                 ["IV","Oral"]),
    "Sodium Bicarbonate":("Electrolyte",     ["50mEq","100mEq"],                ["IV"]),
    "Calcium Gluconate": ("Electrolyte",     ["1g","2g"],                       ["IV"]),
    "Erythropoietin":    ("Hematopoietic",   ["3000U","6000U"],                 ["Subcutaneous"]),
    "Morphine":          ("Opioid Analgesic",["2mg","4mg","8mg"],               ["IV","Oral"]),
    "Fentanyl":          ("Opioid Analgesic",["25mcg","50mcg","100mcg"],        ["IV"]),
    "Propofol":          ("Sedative",        ["10mg/mL"],                       ["IV"]),
    "Lorazepam":         ("Benzodiazepine",  ["0.5mg","1mg","2mg"],             ["IV","Oral"]),
    "Tranexamic Acid":   ("Antifibrinolytic",["1g"],                            ["IV"]),
    "Vitamin K":         ("Coagulant",       ["1mg","5mg","10mg"],              ["IV","Oral"]),
    "Naloxone":          ("Opioid Antagonist",["0.4mg","2mg"],                  ["IV","IM"]),
    "Thiamine":          ("Vitamin",         ["100mg"],                         ["IV","Oral"]),
    "Activated Charcoal":("Antidote",        ["25g","50g"],                     ["Oral"]),
    "Labetalol":         ("Beta-blocker",    ["20mg","40mg"],                   ["IV"]),
    "Amiodarone":        ("Antiarrhythmic",  ["200mg","400mg"],                 ["Oral","IV"]),
    "Ondansetron":       ("Antiemetic",      ["4mg","8mg"],                     ["IV","Oral"]),
    "Metformin":         ("Antidiabetic",    ["500mg","1000mg"],                ["Oral"]),
}

ETHNICITY_CHOICES = ["Caucasian","African American","Hispanic","Asian","Native American","Other/Unknown"]
ETHNICITY_WEIGHTS = [0.60,0.18,0.11,0.06,0.01,0.04]
HOSPITAL_IDS      = list(range(60, 220))

def clamp(v, lo, hi):    return max(lo, min(hi, v))
def maybe_null(v, p=0.05): return None if random.random() < p else v
def w_choice(opts, wts): return random.choices(opts, weights=wts, k=1)[0]
def rtime():             return f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"

def gender_age_offset(lab_name, gender, age):
    mo = 0.0
    if gender == "Male":
        if lab_name == "Hemoglobin":  mo += 1.5
        if lab_name == "Hematocrit":  mo += 4.0
        if lab_name == "Creatinine":  mo += 0.2
        if lab_name == "eGFR":        mo -= 5.0
    if gender == "Female":
        if lab_name == "BNP":         mo += 20.0
        if lab_name == "Calcium":     mo -= 0.1
    if age >= 70:
        extra_decades = (age - 70) / 10
        if lab_name in ("Creatinine","BUN"): mo += 0.3 * extra_decades
        if lab_name == "eGFR":               mo -= 8.0 * extra_decades
        if lab_name == "Albumin":            mo -= 0.2
        if lab_name == "Sodium":             mo -= 1.0
    return mo

# ══════════════════════════════════════════════════════════════════════════════
#  TABLE 1 — PATIENTS
# ══════════════════════════════════════════════════════════════════════════════

print("📋  Generating patients table...")
patient_rows   = []
archetype_list = []
patient_ids    = list(range(200_001, 200_001 + N_PATIENTS))

for i in tqdm(range(N_PATIENTS), desc="  Patients"):
    pid       = patient_ids[i]
    arch_name = w_choice(ARCHETYPE_NAMES, ARCHETYPE_WEIGHTS)
    arch      = ARCHETYPES[arch_name]
    archetype_list.append(arch_name)

    age_lo, age_hi = arch["age_range"]
    age    = int(clamp(np.random.normal((age_lo+age_hi)/2, (age_hi-age_lo)/4), age_lo, age_hi))
    gender = "Male" if random.random() < arch["gender_bias"] else "Female"
    eth    = w_choice(ETHNICITY_CHOICES, ETHNICITY_WEIGHTS)

    if gender == "Male":
        height = clamp(round(np.random.normal(177, 7), 1), 150, 215)
        weight = clamp(round(np.random.normal(88, 20), 1), 40, 200)
    else:
        height = clamp(round(np.random.normal(163, 7), 1), 140, 200)
        weight = clamp(round(np.random.normal(72, 17), 1), 35, 170)

    hospital_id  = random.choice(HOSPITAL_IDS)
    ward_id      = random.randint(1, 120)
    unit_type    = random.choice(arch["unit_types"])
    admit_source = random.choice(arch["admit_sources"])
    sev_score    = clamp(int(np.random.normal(arch["severity"]*3, 1.5)), 1, 10)
    los_hours    = clamp(int(np.random.exponential(arch["los_hours_mean"])), 60, 130_000)

    disc_loc_d   = arch["discharge_locs"]
    disc_loc     = w_choice(list(disc_loc_d.keys()), list(disc_loc_d.values()))
    disc_stat    = "Expired" if disc_loc == "Death" else "Alive"
    unit_disc_loc= disc_loc if random.random() > 0.15 else w_choice(list(disc_loc_d.keys()), list(disc_loc_d.values()))
    unit_stay    = "readmit" if random.random() < 0.10 else w_choice(["admit","stepdown/other","transfer"],[0.78,0.14,0.08])

    chronic      = arch_name in ("CHF","Renal_Failure","COPD","DKA")
    prior_w      = [0.20,0.25,0.22,0.16,0.10,0.07] if chronic else [0.42,0.28,0.16,0.08,0.04,0.02]
    n_prior      = random.choices([0,1,2,3,4,5], weights=prior_w)[0]

    risk = arch["readmit_base"]
    if sev_score >= 8:   risk += 0.12
    elif sev_score >= 6: risk += 0.06
    if n_prior >= 3:     risk += 0.10
    elif n_prior >= 1:   risk += 0.05
    if disc_loc in ("Skilled Nursing Facility","Nursing Home"): risk += 0.08
    if disc_loc == "Home":   risk -= 0.05
    if unit_stay == "readmit": risk += 0.12
    if age >= 80:  risk += 0.10
    elif age >= 70: risk += 0.05
    if disc_stat == "Expired": risk = 0.0
    risk = clamp(risk, 0.0, 0.92)

    readmit = "Yes" if (disc_stat == "Alive" and random.random() < risk) else "No"

    patient_rows.append({
        "patientunitstayid":         pid,
        "patienthealthsystemstayid": pid + 700_000,
        "uniquepid":                 f"pid-{pid:08d}",
        "archetype":                 arch_name,
        "gender":                    maybe_null(gender, 0.004),
        "age":                       maybe_null(age, 0.002),
        "ethnicity":                 maybe_null(eth, 0.028),
        "hospitalid":                hospital_id,
        "wardid":                    ward_id,
        "apacheadmissiondx":         maybe_null(arch["apache_dx"], 0.10),
        "admissionheight":           maybe_null(height, 0.03),
        "admissionweight":           maybe_null(round(weight, 1), 0.08),
        "dischargeweight":           maybe_null(round(weight + np.random.normal(-1, 2), 1), 0.38),
        "hospitaladmittime24":       rtime(),
        "hospitaladmitoffset":       random.randint(-1440, 0),
        "hospitaladmitsource":       maybe_null(admit_source, 0.13),
        "hospitaldischargeyear":     random.randint(2010, 2024),
        "hospitaldischargetime24":   rtime(),
        "hospitaldischargeoffset":   los_hours + random.randint(0, 2880),
        "hospitaldischargelocation": maybe_null(disc_loc, 0.005),
        "hospitaldischargestatus":   maybe_null(disc_stat, 0.005),
        "unittype":                  unit_type,
        "unitadmittime24":           rtime(),
        "unitadmitsource":           maybe_null(admit_source, 0.012),
        "unitvisitnumber":           random.randint(1, 5),
        "unitstaytype":              unit_stay,
        "unitdischargetime24":       rtime(),
        "unitdischargeoffset":       los_hours,
        "unitdischargelocation":     maybe_null(unit_disc_loc, 0.003),
        "unitdischargestatus":       maybe_null(disc_stat, 0.001),
        "severity_score":            sev_score,
        "num_prior_admissions":      n_prior,
        "readmission_risk_score":    round(risk, 4),
        "readmission_within_30days": readmit,            # ← ML TARGET
    })

patients_df = pd.DataFrame(patient_rows)
patients_df.to_csv(f"{OUTPUT_DIR}/patients.csv", index=False)
rr = (patients_df["readmission_within_30days"] == "Yes").mean()
print(f"  ✅ patients.csv — {len(patients_df):,} rows, {len(patients_df.columns)} cols")
print(f"     Readmission rate: {rr*100:.1f}%")
for name in ARCHETYPE_NAMES:
    n = (patients_df["archetype"] == name).sum()
    print(f"     {name:<20} {n:>7,} ({n/N_PATIENTS*100:.1f}%)")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  TABLE 2 — DIAGNOSES
# ══════════════════════════════════════════════════════════════════════════════

print("🩺  Generating diagnoses table...")
diag_rows = []
for i, row in enumerate(tqdm(patient_rows, desc="  Diagnoses")):
    arch = ARCHETYPES[archetype_list[i]]
    pid  = row["patientunitstayid"]
    yr   = row["hospitaldischargeyear"]
    primary = random.choice(arch["icd_primary"])
    diag_rows.append({"diagnosisid": len(diag_rows)+1, "patientunitstayid": pid,
        "icd10_code": primary, "icd10_description": ICD10_DESC.get(primary, primary),
        "diagnosis_priority": 1, "diagnosis_type": "Primary",
        "active_upondischarge": "Yes", "diagnosis_year": yr, "diagnosis_offset_min": 0})
    pool  = arch["icd_comorbidities"]
    n_com = random.choices([1,2,3,4,5], weights=[0.10,0.25,0.30,0.22,0.13])[0]
    for rank, code in enumerate(random.sample(pool, min(n_com, len(pool))), 2):
        diag_rows.append({"diagnosisid": len(diag_rows)+1, "patientunitstayid": pid,
            "icd10_code": code, "icd10_description": ICD10_DESC.get(code, code),
            "diagnosis_priority": rank, "diagnosis_type": "Comorbidity",
            "active_upondischarge": random.choice(["Yes","No"]),
            "diagnosis_year": yr, "diagnosis_offset_min": random.randint(0,1440)})

diagnoses_df = pd.DataFrame(diag_rows)
diagnoses_df.to_csv(f"{OUTPUT_DIR}/diagnoses.csv", index=False)
print(f"  ✅ diagnoses.csv — {len(diagnoses_df):,} rows\n")

# ══════════════════════════════════════════════════════════════════════════════
#  TABLE 3 — LAB RESULTS & VITALS
# ══════════════════════════════════════════════════════════════════════════════

print("🔬  Generating lab results & vitals table...")
lab_rows = []
for i, row in enumerate(tqdm(patient_rows, desc="  Labs")):
    arch   = ARCHETYPES[archetype_list[i]]
    pid    = row["patientunitstayid"]
    gender = row.get("gender") or "Male"
    age    = row.get("age") or 60
    sev    = row["severity_score"]
    deltas = arch["lab_deltas"]
    n_rds  = random.choices([1,2,3,4,5], weights=[0.15,0.30,0.28,0.18,0.09])[0]

    for rnd in range(n_rds):
        offset = rnd * random.randint(300, 800)
        for lab_name, (norm_mean, norm_std, lo_c, hi_c) in LAB_NORMALS.items():
            delta    = deltas.get(lab_name, 1.0)
            adj_mean = norm_mean * delta + gender_age_offset(lab_name, gender, age)
            # Severity amplifies abnormal values
            if sev >= 8 and delta != 1.0:
                adj_mean = norm_mean + (adj_mean - norm_mean) * 1.3
            elif sev <= 3 and delta != 1.0:
                adj_mean = norm_mean + (adj_mean - norm_mean) * 0.6
            # Treatment recovery on later rounds
            if rnd > 0 and delta != 1.0:
                adj_mean = adj_mean*(1 - 0.08*rnd) + norm_mean*(0.08*rnd)
            val  = round(float(np.clip(np.random.normal(adj_mean, norm_std), lo_c, hi_c)), 2)
            norm_lo = norm_mean - 2*norm_std
            norm_hi = norm_mean + 2*norm_std
            if   val < lo_c*1.05:  flag = "CRITICAL_LOW"
            elif val > hi_c*0.95:  flag = "CRITICAL_HIGH"
            elif val < norm_lo:    flag = "LOW"
            elif val > norm_hi:    flag = "HIGH"
            else:                  flag = "NORMAL"
            lab_rows.append({
                "labresultid":       len(lab_rows)+1,
                "patientunitstayid": pid,
                "lab_name":          lab_name,
                "lab_value":         val,
                "lab_unit":          LAB_UNITS.get(lab_name, ""),
                "lab_flag":          flag,
                "round_number":      rnd+1,
                "result_offset_min": offset + random.randint(-20,20),
            })

labs_df = pd.DataFrame(lab_rows)
labs_df.to_csv(f"{OUTPUT_DIR}/lab_results.csv", index=False)
print(f"  ✅ lab_results.csv — {len(labs_df):,} rows\n")

# ══════════════════════════════════════════════════════════════════════════════
#  TABLE 4 — MEDICATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("💊  Generating medications table...")
med_rows = []
for i, row in enumerate(tqdm(patient_rows, desc="  Medications")):
    arch    = ARCHETYPES[archetype_list[i]]
    pid     = row["patientunitstayid"]
    sev     = row["severity_score"]
    los_min = row["unitdischargeoffset"]
    mand    = list(arch["mandatory_meds"])
    optl    = arch["optional_meds"]
    n_opt   = random.choices([0,1,2,3,4], weights=[0.05,0.20,0.35,0.28,0.12])[0]
    if sev >= 8: n_opt = min(n_opt+2, len(optl))
    chosen  = list(dict.fromkeys(mand + random.sample(optl, min(n_opt, len(optl)))))
    for drug in chosen:
        meta = DRUG_META.get(drug)
        if not meta: continue
        drug_class, doses, routes = meta
        route = random.choice(routes)
        freq  = "Continuous" if route == "IV" and drug_class in ("Vasopressor","IV Fluid","Sedative") \
                else random.choice(["Once","BID","TID","QID","PRN"])
        start = random.randint(0, max(los_min//3, 60))
        stop  = min(start + random.randint(60, max(los_min-start, 120)), los_min)
        med_rows.append({
            "medicationid":      len(med_rows)+1,
            "patientunitstayid": pid,
            "drug_name":         drug,
            "drug_class":        drug_class,
            "dosage":            random.choice(doses),
            "route_of_admin":    route,
            "frequency":         freq,
            "start_offset_min":  start,
            "stop_offset_min":   stop,
            "duration_minutes":  stop-start,
            "prn":               "Yes" if freq=="PRN" else "No",
        })

meds_df = pd.DataFrame(med_rows)
meds_df.to_csv(f"{OUTPUT_DIR}/medications.csv", index=False)
print(f"  ✅ medications.csv — {len(meds_df):,} rows\n")

# ══════════════════════════════════════════════════════════════════════════════
#  TABLE 5 — PRIOR ADMISSIONS
# ══════════════════════════════════════════════════════════════════════════════

print("📁  Generating prior admissions table...")
prior_rows = []
for i, row in enumerate(tqdm(patient_rows, desc="  Prior admissions")):
    n_prior = row["num_prior_admissions"]
    if n_prior == 0: continue
    arch     = ARCHETYPES[archetype_list[i]]
    pid      = row["patientunitstayid"]
    cur_year = row["hospitaldischargeyear"]
    disc_d   = arch["discharge_locs"]
    for seq in range(n_prior):
        yback = random.choices([1,2,3,4,5,6,7,8,9,10],[0.22,0.18,0.14,0.12,0.10,0.08,0.06,0.04,0.03,0.03])[0]
        if random.random() < 0.65:
            pri_code = random.choice(arch["icd_primary"])
        else:
            all_primary = [c for a in ARCHETYPES.values() for c in a["icd_primary"]]
            pri_code = random.choice(all_primary)
        pdl  = w_choice(list(disc_d.keys()), list(disc_d.values()))
        pds  = "Expired" if pdl == "Death" else "Alive"
        prior_rows.append({
            "prior_admission_id":        len(prior_rows)+1,
            "patientunitstayid":         pid,
            "sequence_number":           seq+1,
            "prior_hospital_id":         random.choice(HOSPITAL_IDS),
            "prior_admission_year":      max(2005, cur_year-yback),
            "prior_apache_dx":           arch["apache_dx"],
            "prior_icd10_primary":       pri_code,
            "prior_icd10_description":   ICD10_DESC.get(pri_code, pri_code),
            "prior_los_days":            max(1, int(np.random.exponential(5))),
            "prior_discharge_location":  pdl,
            "prior_discharge_status":    pds,
            "prior_icu_admission":       random.choice(["Yes","No"]),
            "days_since_prior_admission":yback*365 + random.randint(0,364),
        })

prior_df = pd.DataFrame(prior_rows)
prior_df.to_csv(f"{OUTPUT_DIR}/prior_admissions.csv", index=False)
print(f"  ✅ prior_admissions.csv — {len(prior_df):,} rows\n")

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

total = len(patients_df)+len(diagnoses_df)+len(labs_df)+len(meds_df)+len(prior_df)
print("=" * 64)
print("  ✅  GENERATION COMPLETE")
print("=" * 64)
print(f"""
  Output : ./{OUTPUT_DIR}/
  ──────────────────────────────────────────────────────────
  patients.csv          {len(patients_df):>9,} rows  {len(patients_df.columns)} cols
  diagnoses.csv         {len(diagnoses_df):>9,} rows  {len(diagnoses_df.columns)} cols
  lab_results.csv       {len(labs_df):>9,} rows  {len(labs_df.columns)} cols
  medications.csv       {len(meds_df):>9,} rows  {len(meds_df.columns)} cols
  prior_admissions.csv  {len(prior_df):>9,} rows  {len(prior_df.columns)} cols
  ──────────────────────────────────────────────────────────
  TOTAL ROWS            {total:>9,}
  ──────────────────────────────────────────────────────────

  ML target   : readmission_within_30days (Yes / No)
  Readmission : {rr*100:.1f}%
  Join key    : patientunitstayid

  Clinical guarantees:
    ✔  Labs match archetype   (MI → Troponin 25x, Sepsis → WBC 2.2x)
    ✔  Meds match diagnosis   (Sepsis → antibiotics + vasopressors)
    ✔  Gender/age adjusts lab norms
    ✔  Severity amplifies abnormal values
    ✔  Treatment effect: later lab rounds trend toward normal
    ✔  Readmission derived from archetype + severity + context

  Quick load:
    import pandas as pd
    pts  = pd.read_csv('{OUTPUT_DIR}/patients.csv')
    labs = pd.read_csv('{OUTPUT_DIR}/lab_results.csv')
    df   = pts.merge(labs, on='patientunitstayid')
""")