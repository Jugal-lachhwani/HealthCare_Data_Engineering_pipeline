from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection


class MongoVisitWriter:
    def __init__(self, uri: str, db_name: str, collection_name: str) -> None:
        self.client = MongoClient(uri)
        self.collection: Collection = self.client[db_name][collection_name]

    def write_events(self, events: list[dict[str, Any]]) -> int:
        if not events:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        payload = []
        for event in events:
            event = dict(event)
            event["ingested_at_utc"] = now
            payload.append(event)

        result = self.collection.insert_many(payload)
        return len(result.inserted_ids)
