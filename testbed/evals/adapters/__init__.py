"""Checker adapters — read eval deliverables back THROUGH a platform.

The grader never trusts the engineer's run venv or files: a deliverable counts
only if the platform's own read path returns it. One adapter per platform
implements the small `Checker` surface the assertion suites need; the eval
families stay platform-neutral by talking to this protocol only.

Trust model: an adapter must run against a TRUSTED client install (for
Hopsworks: the wheel built from the committed pinned ref), with the grader's
credentials, outside the engineer boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class TableInfo:
    """Platform-neutral metadata for a feature table."""
    name: str
    version: int | None = None
    primary_key: list[str] = field(default_factory=list)
    event_time: str | None = None          # None where the platform has no concept
    schema: dict[str, str] = field(default_factory=dict)  # column -> type string


@runtime_checkable
class Checker(Protocol):
    def get_feature_table(self, name: str, version: int | None = None) -> TableInfo | None:
        """Metadata for a feature table, or None if it does not exist."""
        ...

    def read_rows(self, name: str, version: int | None = None) -> pd.DataFrame:
        """Full contents of a feature table, read through the platform."""
        ...

    def read_training_dataset(self, name: str, version: int = 1) -> pd.DataFrame:
        """A versioned training dataset, read through the platform."""
        ...
