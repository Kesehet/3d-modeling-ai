from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_history(root: Path) -> list[dict[str, Any]]:
    path = root / "history.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def append_history(root: Path, event: str, **data: Any) -> dict[str, Any]:
    history = load_history(root)
    record = {
        "seq": len(history) + 1,
        "at": datetime.now(UTC).isoformat(),
        "event": event,
        **data,
    }
    history.append(record)
    tmp = root / "history.json.tmp"
    tmp.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(root / "history.json")
    return record
