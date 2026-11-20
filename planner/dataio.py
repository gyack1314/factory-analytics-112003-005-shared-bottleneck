from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load(name: str, root: Path | None = None) -> Any:
    """Load one JSON table from the data pack."""
    base = root or DATA
    return json.loads((base / f"{name}.json").read_text(encoding="utf-8"))


def load_all(root: Path | None = None) -> dict[str, Any]:
    base = root or DATA
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(base.glob("*.json"))}
