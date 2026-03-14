from __future__ import annotations

import csv
from pathlib import Path


class CsvBatchWriter:
    def __init__(self, file_path: str, fieldnames: list[str]) -> None:
        self.file_path = Path(file_path)
        self.fieldnames = fieldnames

    def append_rows(self, rows: list[dict[str, str]]) -> int:
        if not rows:
            return 0

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.file_path.exists()

        with self.file_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

        return len(rows)
