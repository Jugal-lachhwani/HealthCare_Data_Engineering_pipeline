from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("airbyte-style-staging")


@dataclass
class StagingConfig:
    source_csv: str
    stream_name: str
    destination_root: str
    state_path: str
    file_prefix: str
    max_rows_per_file: int



def load_staging_config(config_path: str) -> StagingConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    return StagingConfig(
        source_csv=cfg["source_csv"],
        stream_name=cfg.get("stream_name", "ehr_visits"),
        destination_root=cfg.get("destination_root", "Data/bronze/airbyte_raw"),
        state_path=cfg.get("state_path", "Data/bronze/_airbyte_state/ehr_visits_state.json"),
        file_prefix=cfg.get("file_prefix", "ehr_visits_raw"),
        max_rows_per_file=int(cfg.get("max_rows_per_file", 5000)),
    )



def read_state(state_path: Path) -> int:
    if not state_path.exists():
        return 0

    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    return int(state.get("last_processed_row", 0))



def write_state(state_path: Path, last_processed_row: int, synced_rows: int) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_processed_row": last_processed_row,
        "last_synced_rows": synced_rows,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def to_airbyte_raw_record(row: dict[str, Any], stream_name: str, emitted_at: str) -> dict[str, Any]:
    return {
        "_airbyte_ab_id": str(uuid.uuid4()),
        "_airbyte_emitted_at": emitted_at,
        "_airbyte_stream": stream_name,
        "_airbyte_data": json.dumps(row, ensure_ascii=True),
        **row,
    }



def build_output_dir(cfg: StagingConfig, now: datetime) -> Path:
    load_date = now.strftime("%Y-%m-%d")
    load_hour = now.strftime("%H")
    base_dir = Path(cfg.destination_root) / cfg.stream_name / f"load_date={load_date}" / f"load_hour={load_hour}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir



def write_chunk(base_dir: Path, cfg: StagingConfig, timestamp_token: str, file_index: int, records: list[dict[str, Any]]) -> int:
    if not records:
        return 0

    file_name = f"{cfg.file_prefix}_{timestamp_token}_{file_index:04d}.parquet"
    output_path = base_dir / file_name
    pd.DataFrame(records).to_parquet(output_path, index=False)
    return len(records)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Airbyte-style CSV to Bronze raw Parquet staging")
    parser.add_argument("--config", default="config/airbyte_staging_config.json", help="Path to staging config JSON")
    parser.add_argument("--source-csv", default=None, help="Override source CSV path")
    return parser.parse_args()



def run_sync(cfg: StagingConfig) -> int:
    source_csv = Path(cfg.source_csv)
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    state_path = Path(cfg.state_path)
    last_processed = read_state(state_path)

    now = datetime.now(timezone.utc)
    emitted_at = now.isoformat()
    timestamp_token = now.strftime("%Y%m%dT%H%M%S")
    base_dir = build_output_dir(cfg, now)

    current_row = 0
    written = 0
    file_index = 0
    buffer: list[dict[str, Any]] = []
    chunk_size = max(1, cfg.max_rows_per_file)

    for chunk in pd.read_csv(source_csv, chunksize=chunk_size):
        for row in chunk.to_dict(orient="records"):
            current_row += 1
            if current_row <= last_processed:
                continue
            buffer.append(to_airbyte_raw_record(row, cfg.stream_name, emitted_at))

        if buffer:
            written += write_chunk(base_dir, cfg, timestamp_token, file_index, buffer)
            buffer = []
            file_index += 1

    if not written:
        logger.info("No new rows to sync. source=%s last_processed_row=%s", source_csv, last_processed)
        return 0

    write_state(state_path, current_row, written)

    logger.info(
        "Synced %s new rows to Bronze raw. stream=%s source=%s destination=%s",
        written,
        cfg.stream_name,
        source_csv,
        cfg.destination_root,
    )
    return written



def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_staging_config(args.config)

    if args.source_csv is not None:
        cfg.source_csv = args.source_csv

    run_sync(cfg)


if __name__ == "__main__":
    main()
