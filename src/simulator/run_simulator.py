from __future__ import annotations

import argparse
import logging
import time
from dotenv import load_dotenv

from simulator.config import SimulatorConfig, load_config
from simulator.generate_ehr import generate_visit_batch
from simulator.mongodb_writer import MongoVisitWriter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("ehr-simulator")



def run_once(config: SimulatorConfig) -> int:
    events = generate_visit_batch(
        batch_size=config.batch_size,
        random_seed=config.random_seed,
        hospital_id=config.hospital_id,
    )
    writer = MongoVisitWriter(config.mongo_uri, config.mongo_db, config.mongo_collection)
    inserted = writer.write_events(events)
    logger.info("Inserted %s synthetic EHR events into MongoDB.", inserted)
    return inserted



def run_forever(config: SimulatorConfig) -> None:
    logger.info(
        "Starting continuous simulator. interval_seconds=%s batch_size=%s db=%s collection=%s",
        config.interval_seconds,
        config.batch_size,
        config.mongo_db,
        config.mongo_collection,
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
    return parser.parse_args()



def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config()

    if args.interval_seconds is not None:
        config.interval_seconds = args.interval_seconds
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.mode == "once":
        run_once(config)
    else:
        run_forever(config)


if __name__ == "__main__":
    main()
