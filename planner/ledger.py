"""Append-only ledger of decision runs.

Management asked to be able to go back and see what demand and constraints an
abandoned option was evaluated against.  Every ``compare``/``phase2`` run can
be written here with a content hash: the ledger stores the exact world reading
(rejection set, material slip, release span, demand multiplier, route mode),
the option packages that were feasible then, and the per-package outcome.  A
later reader can reproduce a run byte-for-byte from the stored inputs.

Nothing is overwritten: the file is a JSON array appended in place, so an old
decision can always be re-read even after the pack changes.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .compare import ComparisonReport, Phase2Report


def _digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def record_comparison(report: ComparisonReport, note: str = "",
                      path: Path | None = None) -> dict:
    entry = {
        "kind": "comparison",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "world": report.world.as_dict(),
        "route_mode": report.route_mode,
        "baseline": report.baseline.summary(),
        "packages": [
            {
                "options": sorted(p.options),
                "cost_cents": p.cost_cents,
                "delivered_units": p.result.delivered_units,
                "on_time_units": p.result.on_time_units,
                "late_unit_minutes": p.result.late_unit_minutes,
                "backlog_units": p.result.backlog_units,
                "buffer_block_minutes": p.result.buffer_block_minutes,
                "committed_breach": p.committed_breach,
                "unblocked_orders": sorted(p.unblocked),
                "still_blocked_orders": sorted(p.blocked_by_other),
            }
            for p in report.packages
        ],
        "infeasible": report.infeasible,
        "differentiation": report.differentiation,
        "notes": report.notes,
    }
    entry["digest"] = _digest({k: v for k, v in entry.items() if k != "digest"})
    if path is not None:
        _append(path, entry)
    return entry


def record_phase2(report: Phase2Report, note: str = "", path: Path | None = None) -> dict:
    entry = {
        "kind": "phase2",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "phase1_state": report.start_state.as_dict(),
        "phase2_new_options": sorted(report.phase2_options),
        "phase2_baseline": report.baseline_phase2.summary(),
        "phase2_with_new_options": report.phase2.summary(),
        "marginal_on_time_units": report.marginal_units,
        "marginal_late_unit_minutes": report.marginal_lum,
    }
    entry["digest"] = _digest({k: v for k, v in entry.items() if k != "digest"})
    if path is not None:
        _append(path, entry)
    return entry


def _append(path: Path, entry: dict) -> None:
    path = Path(path)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except json.JSONDecodeError:
            existing = []
    existing.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def find(entries: list[dict], digest: str) -> dict | None:
    return next((e for e in entries if e.get("digest") == digest), None)
