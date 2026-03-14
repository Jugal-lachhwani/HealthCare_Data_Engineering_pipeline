from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SimulatorConfig:
    mongo_uri: str
    mongo_db: str
    mongo_collection: str
    interval_seconds: int
    batch_size: int
    random_seed: int
    hospital_id: str



def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)



def load_config() -> SimulatorConfig:
    return SimulatorConfig(
        mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        mongo_db=os.getenv("MONGO_DB", "healthcare_ehr"),
        mongo_collection=os.getenv("MONGO_COLLECTION", "raw_patient_visits"),
        interval_seconds=_get_int_env("SIMULATOR_INTERVAL_SECONDS", 60),
        batch_size=_get_int_env("SIMULATOR_BATCH_SIZE", 5),
        random_seed=_get_int_env("SIMULATOR_RANDOM_SEED", 42),
        hospital_id=os.getenv("HOSPITAL_ID", "HOSP_001"),
    )
