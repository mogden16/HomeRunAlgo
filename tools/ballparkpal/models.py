"""Shared data models for Ballpark Pal validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DownloadedWorkbook:
    export_name: str
    requested_date: str
    original_filename: str
    saved_path: Path
    source_url: str


@dataclass(frozen=True)
class ValidationFinding:
    valid: bool
    export_name: str
    requested_date: str
    saved_path: Path
    workbook_date: str | None = None
    sheet_names: tuple[str, ...] = ()
    row_count: int = 0
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    sha256: str | None = None
    file_size_bytes: int | None = None

    def as_manifest_row(self) -> dict[str, Any]:
        return {
            "export_name": self.export_name,
            "requested_date": self.requested_date,
            "saved_filename": self.saved_path.name,
            "saved_path": str(self.saved_path),
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "validation_result": "valid" if self.valid else "invalid",
            "workbook_date_detected": self.workbook_date,
            "sheet_names": list(self.sheet_names),
            "row_count": self.row_count,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RunManifest:
    requested_date: str
    pulled_at: str
    overall_valid: bool
    downloads: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

