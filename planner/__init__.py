"""Shared-station capacity attribution and retrofit scenario engine.

The package is deliberately standard-library only (Python 3.11) so the
decision tool can run inside the plant's air-gapped planning desktop.

Entry point: ``python3 -m planner.cli --help``.
"""

__all__ = ["dataio", "model", "history", "attribution", "scenario", "cli"]
