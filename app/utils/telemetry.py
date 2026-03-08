from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from app.config import Settings


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return value


class StructuredLogger:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.enable_structured_logs
        self.path = settings.telemetry_log_path

    def log_event(self, event_type: str, payload: Any) -> None:
        if not self.enabled:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": _to_jsonable(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    def read_recent(self, limit: int = 20) -> List[dict]:
        if not self.path.exists() or limit <= 0:
            return []

        lines = self.path.read_text(encoding="utf-8").splitlines()
        recent = lines[-limit:]
        events = []
        for line in recent:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
