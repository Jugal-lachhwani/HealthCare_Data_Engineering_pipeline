from __future__ import annotations

import argparse
import logging
import time
from dotenv import load_dotenv

from simulator.csv_like_generator import EHR_COLUMNS, generate_ehr_like_batch, load_template_stats
from simulator.csv_writer import CsvBatchWriter
from simulator.config import SimulatorConfig, load_config
from simulator.mongodb_writer import MongoVisitWriter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("ehr-simulator")



def run_once(config: SimulatorConfig) -> int:
    stats = load_template_stats(config.template_csv_path)
    rows = generate_ehr_like_batch(
        batch_size=config.batch_size,
        random_seed=config.random_seed,
        stats=stats,
    )

    inserted_mongo = 0
    inserted_csv = 0

    if config.sink in {"mongo", "both"}:
        mongo_writer = MongoVisitWriter(config.mongo_uri, config.mongo_db, config.mongo_collection)
        inserted_mongo = mongo_writer.write_events(rows)

    if config.sink in {"csv", "both"}:
        csv_writer = CsvBatchWriter(config.output_csv_path, EHR_COLUMNS)
        inserted_csv = csv_writer.append_rows(rows)

    logger.info(
        "Generated %s rows | mongo_inserted=%s csv_appended=%s output_csv=%s",
        len(rows),
        inserted_mongo,
        inserted_csv,
        config.output_csv_path,
    )
    return len(rows)



def run_forever(config: SimulatorConfig) -> None:
    logger.info(
        "Starting continuous simulator. interval_seconds=%s batch_size=%s sink=%s template_csv=%s",
        config.interval_seconds,
        config.batch_size,
        config.sink,
        config.template_csv_path,
    )

    # Trigger immediately on startup so first batch is not delayed.
    run_once(config)

    while True:
        time.sleep(config.interval_seconds)
        run_once(config)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic EHR event simulator")
    parser.add_argument(
        "--mode",
        default="once",
        choices=["once", "continuous"],
        help="Run once or continuously using in-process scheduler.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=None,
        help="Override SIMULATOR_INTERVAL_SECONDS.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override SIMULATOR_BATCH_SIZE.",
    )
    parser.add_argument(
        "--sink",
        choices=["mongo", "csv", "both"],
        default=None,
        help="Override SIMULATOR_SINK.",
    )
    parser.add_argument(
        "--template-csv",
        default=None,
        help="Override EHR_TEMPLATE_CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Override SYNTHETIC_OUTPUT_CSV.",
    )
    return parser.parse_args()



def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config()

    if args.interval_seconds is not None:
        config.interval_seconds = args.interval_seconds
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.sink is not None:
        config.sink = args.sink
    if args.template_csv is not None:
        config.template_csv_path = args.template_csv
    if args.output_csv is not None:
        config.output_csv_path = args.output_csv

    if args.mode == "once":
        run_once(config)
    else:
        run_forever(config)


if __name__ == "__main__":
    main()
