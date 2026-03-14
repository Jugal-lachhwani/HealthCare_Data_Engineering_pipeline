from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


EHR_COLUMNS = [
    "patientunitstayid",
    "patienthealthsystemstayid",
    "gender",
    "age",
    "ethnicity",
    "hospitalid",
    "wardid",
    "apacheadmissiondx",
    "admissionheight",
    "hospitaladmittime24",
    "hospitaladmitoffset",
    "hospitaladmitsource",
    "hospitaldischargeyear",
    "hospitaldischargetime24",
    "hospitaldischargeoffset",
    "hospitaldischargelocation",
    "hospitaldischargestatus",
    "unittype",
    "unitadmittime24",
    "unitadmitsource",
    "unitvisitnumber",
    "unitstaytype",
    "admissionweight",
    "dischargeweight",
    "unitdischargetime24",
    "unitdischargeoffset",
    "unitdischargelocation",
    "unitdischargestatus",
    "uniquepid",
]


@dataclass
class TemplateStats:
    genders: list[str]
    ethnicities: list[str]
    apache_dx: list[str]
    hospital_sources: list[str]
    discharge_locations: list[str]
    unit_types: list[str]
    unit_admit_sources: list[str]
    unit_stay_types: list[str]
    unit_discharge_locations: list[str]
    min_hospital_id: int
    max_hospital_id: int
    min_ward_id: int
    max_ward_id: int


DEFAULT_GENDERS = ["Male", "Female"]
DEFAULT_ETHNICITIES = ["Caucasian", "African American", "Hispanic", "Asian"]
DEFAULT_APACHE_DX = [
    "Hypertension, uncontrolled",
    "Renal failure, acute",
    "Diabetic ketoacidosis",
    "Infarction, acute myocardial (MI)",
    "Drug withdrawal",
]
DEFAULT_HOSPITAL_SOURCES = ["Emergency Department", "Direct Admit", "Operating Room", "Floor"]
DEFAULT_DISCHARGE_LOCATIONS = ["Home", "Skilled Nursing Facility", "Nursing Home", "Other"]
DEFAULT_UNIT_TYPES = ["Med-Surg ICU", "MICU", "CTICU", "Neuro ICU"]
DEFAULT_UNIT_ADMIT_SOURCES = ["Emergency Department", "Operating Room", "Direct Admit", "Floor"]
DEFAULT_UNIT_STAY_TYPES = ["admit", "stepdown/other"]



def _format_time_24(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")



def _safe_int(value: str, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default



def _clean_nonempty(values: set[str], defaults: list[str]) -> list[str]:
    cleaned = sorted({v.strip() for v in values if v is not None and v.strip() != ""})
    return cleaned if cleaned else defaults



def load_template_stats(template_csv_path: str) -> TemplateStats:
    path = Path(template_csv_path)
    if not path.exists():
        return TemplateStats(
            genders=DEFAULT_GENDERS,
            ethnicities=DEFAULT_ETHNICITIES,
            apache_dx=DEFAULT_APACHE_DX,
            hospital_sources=DEFAULT_HOSPITAL_SOURCES,
            discharge_locations=DEFAULT_DISCHARGE_LOCATIONS,
            unit_types=DEFAULT_UNIT_TYPES,
            unit_admit_sources=DEFAULT_UNIT_ADMIT_SOURCES,
            unit_stay_types=DEFAULT_UNIT_STAY_TYPES,
            unit_discharge_locations=DEFAULT_DISCHARGE_LOCATIONS,
            min_hospital_id=60,
            max_hospital_id=90,
            min_ward_id=80,
            max_ward_id=140,
        )

    genders: set[str] = set()
    ethnicities: set[str] = set()
    apache_dx: set[str] = set()
    hospital_sources: set[str] = set()
    discharge_locations: set[str] = set()
    unit_types: set[str] = set()
    unit_admit_sources: set[str] = set()
    unit_stay_types: set[str] = set()
    unit_discharge_locations: set[str] = set()

    hospital_ids: list[int] = []
    ward_ids: list[int] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            genders.add(row.get("gender", ""))
            ethnicities.add(row.get("ethnicity", ""))
            apache_dx.add(row.get("apacheadmissiondx", ""))
            hospital_sources.add(row.get("hospitaladmitsource", ""))
            discharge_locations.add(row.get("hospitaldischargelocation", ""))
            unit_types.add(row.get("unittype", ""))
            unit_admit_sources.add(row.get("unitadmitsource", ""))
            unit_stay_types.add(row.get("unitstaytype", ""))
            unit_discharge_locations.add(row.get("unitdischargelocation", ""))

            hospital_ids.append(_safe_int(row.get("hospitalid", ""), 70))
            ward_ids.append(_safe_int(row.get("wardid", ""), 95))

    return TemplateStats(
        genders=_clean_nonempty(genders, DEFAULT_GENDERS),
        ethnicities=_clean_nonempty(ethnicities, DEFAULT_ETHNICITIES),
        apache_dx=_clean_nonempty(apache_dx, DEFAULT_APACHE_DX),
        hospital_sources=_clean_nonempty(hospital_sources, DEFAULT_HOSPITAL_SOURCES),
        discharge_locations=_clean_nonempty(discharge_locations, DEFAULT_DISCHARGE_LOCATIONS),
        unit_types=_clean_nonempty(unit_types, DEFAULT_UNIT_TYPES),
        unit_admit_sources=_clean_nonempty(unit_admit_sources, DEFAULT_UNIT_ADMIT_SOURCES),
        unit_stay_types=_clean_nonempty(unit_stay_types, DEFAULT_UNIT_STAY_TYPES),
        unit_discharge_locations=_clean_nonempty(unit_discharge_locations, DEFAULT_DISCHARGE_LOCATIONS),
        min_hospital_id=min(hospital_ids) if hospital_ids else 60,
        max_hospital_id=max(hospital_ids) if hospital_ids else 90,
        min_ward_id=min(ward_ids) if ward_ids else 80,
        max_ward_id=max(ward_ids) if ward_ids else 140,
    )



def generate_ehr_like_row(rng: random.Random, stats: TemplateStats) -> dict[str, str]:
    patient_unit_stay_id = rng.randint(200000, 399999)
    patient_health_system_stay_id = rng.randint(130000, 299999)

    age = str(rng.randint(18, 90))
    hospital_id = str(rng.randint(stats.min_hospital_id, stats.max_hospital_id))
    ward_id = str(rng.randint(stats.min_ward_id, stats.max_ward_id))

    hospital_discharge_year = str(rng.choice([2014, 2015, 2016]))

    now = datetime.utcnow().replace(microsecond=0)
    admit_time = now - timedelta(minutes=rng.randint(30, 3000))
    discharge_time = now
    unit_admit_time = admit_time + timedelta(minutes=rng.randint(0, 240))
    unit_discharge_time = discharge_time - timedelta(minutes=rng.randint(0, 180))

    hospital_admit_offset = -rng.randint(0, 3000)
    hospital_discharge_offset = rng.randint(90, 12000)
    unit_discharge_offset = rng.randint(1, 5000)

    admission_weight = round(rng.uniform(45.0, 140.0), 1)
    discharge_weight = round(max(35.0, admission_weight + rng.uniform(-4.5, 2.5)), 1)
    admission_height = round(rng.uniform(145.0, 195.0), 1)

    unique_pid = f"002-{rng.randint(1000, 19999)}"

    row = {
        "patientunitstayid": str(patient_unit_stay_id),
        "patienthealthsystemstayid": str(patient_health_system_stay_id),
        "gender": rng.choice(stats.genders),
        "age": age,
        "ethnicity": rng.choice(stats.ethnicities),
        "hospitalid": hospital_id,
        "wardid": ward_id,
        "apacheadmissiondx": rng.choice(stats.apache_dx),
        "admissionheight": f"{admission_height:.1f}",
        "hospitaladmittime24": _format_time_24(admit_time),
        "hospitaladmitoffset": str(hospital_admit_offset),
        "hospitaladmitsource": rng.choice(stats.hospital_sources),
        "hospitaldischargeyear": hospital_discharge_year,
        "hospitaldischargetime24": _format_time_24(discharge_time),
        "hospitaldischargeoffset": str(hospital_discharge_offset),
        "hospitaldischargelocation": rng.choice(stats.discharge_locations),
        "hospitaldischargestatus": rng.choice(["Alive", "Expired"]),
        "unittype": rng.choice(stats.unit_types),
        "unitadmittime24": _format_time_24(unit_admit_time),
        "unitadmitsource": rng.choice(stats.unit_admit_sources),
        "unitvisitnumber": str(rng.choice([1, 1, 1, 2, 3])),
        "unitstaytype": rng.choice(stats.unit_stay_types),
        "admissionweight": f"{admission_weight:.1f}",
        "dischargeweight": f"{discharge_weight:.1f}",
        "unitdischargetime24": _format_time_24(unit_discharge_time),
        "unitdischargeoffset": str(unit_discharge_offset),
        "unitdischargelocation": rng.choice(stats.unit_discharge_locations),
        "unitdischargestatus": rng.choice(["Alive", "Expired"]),
        "uniquepid": unique_pid,
    }

    # Keep sparsity in selected fields to resemble real ICU exports.
    if rng.random() < 0.15:
        row["apacheadmissiondx"] = ""
    if rng.random() < 0.1:
        row["hospitaladmitsource"] = ""
    if rng.random() < 0.15:
        row["admissionweight"] = ""
    if rng.random() < 0.08:
        row["dischargeweight"] = ""
    if rng.random() < 0.05:
        row["admissionheight"] = ""

    return row



def generate_ehr_like_batch(batch_size: int, random_seed: int, stats: TemplateStats) -> list[dict[str, str]]:
    rng = random.Random(random_seed + int(datetime.utcnow().timestamp()))
    return [generate_ehr_like_row(rng, stats) for _ in range(batch_size)]
