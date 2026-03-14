from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any


DIAGNOSIS_CODES = [
    "I10",
    "E11.9",
    "J18.9",
    "N18.9",
    "I50.9",
    "A41.9",
    "J44.1",
    "K52.9",
]

LAB_TEST_NAMES = [
    "hemoglobin",
    "wbc",
    "platelets",
    "creatinine",
    "glucose",
    "sodium",
    "potassium",
]

MEDICATIONS = [
    "metformin",
    "lisinopril",
    "atorvastatin",
    "furosemide",
    "insulin_glargine",
    "amoxicillin",
]

DEPARTMENTS = ["Cardiology", "Internal Medicine", "Pulmonology", "Nephrology", "Emergency"]



def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()



def _sample_labs(rng: random.Random) -> list[dict[str, Any]]:
    labs = []
    for lab in rng.sample(LAB_TEST_NAMES, k=4):
        if lab == "hemoglobin":
            value = round(rng.uniform(8.0, 16.0), 2)
        elif lab == "wbc":
            value = round(rng.uniform(3.0, 18.0), 2)
        elif lab == "platelets":
            value = round(rng.uniform(120.0, 450.0), 1)
        elif lab == "creatinine":
            value = round(rng.uniform(0.4, 3.5), 2)
        elif lab == "glucose":
            value = round(rng.uniform(65.0, 320.0), 1)
        elif lab == "sodium":
            value = round(rng.uniform(125.0, 150.0), 1)
        else:
            value = round(rng.uniform(2.8, 6.3), 2)

        labs.append(
            {
                "name": lab,
                "value": value,
                "unit": "standard",
                "abnormal_flag": rng.random() < 0.22,
            }
        )
    return labs



def generate_visit_event(rng: random.Random, hospital_id: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    age = rng.randint(18, 95)
    previous_admissions = rng.randint(0, 7)
    length_of_stay_days = rng.randint(1, 14)

    admit_time = now - timedelta(days=length_of_stay_days)
    discharge_time = now

    visit_id = str(uuid.uuid4())
    patient_id = f"PT-{rng.randint(100000, 999999)}"

    readmission_risk_proxy = min(0.95, max(0.02, 0.08 + previous_admissions * 0.08 + rng.random() * 0.2))

    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "patient_visit_discharge",
        "event_time_utc": _iso_utc(now),
        "source": "synthetic_ehr_simulator",
        "hospital_id": hospital_id,
        "patient": {
            "patient_id": patient_id,
            "age": age,
            "sex": rng.choice(["F", "M"]),
        },
        "encounter": {
            "visit_id": visit_id,
            "department": rng.choice(DEPARTMENTS),
            "admit_time_utc": _iso_utc(admit_time),
            "discharge_time_utc": _iso_utc(discharge_time),
            "length_of_stay_days": length_of_stay_days,
            "diagnosis_codes": rng.sample(DIAGNOSIS_CODES, k=2),
            "previous_admissions_12m": previous_admissions,
        },
        "clinical": {
            "lab_results": _sample_labs(rng),
            "medications": rng.sample(MEDICATIONS, k=3),
        },
        "labels": {
            "simulated_readmission_risk": round(readmission_risk_proxy, 4),
        },
    }



def generate_visit_batch(batch_size: int, random_seed: int, hospital_id: str) -> list[dict[str, Any]]:
    rng = random.Random(random_seed + int(datetime.now(timezone.utc).timestamp()))
    return [generate_visit_event(rng, hospital_id) for _ in range(batch_size)]
