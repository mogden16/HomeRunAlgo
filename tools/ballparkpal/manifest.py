"""Manifest writer for Ballpark Pal validation runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import RunManifest, ValidationFinding


def build_manifest(
    *,
    requested_date: str,
    validations: list[ValidationFinding],
    notes: list[str] | None = None,
) -> RunManifest:
    pulled_at = datetime.now(timezone.utc).isoformat()
    return RunManifest(
        requested_date=requested_date,
        pulled_at=pulled_at,
        overall_valid=all(item.valid for item in validations),
        downloads=[item.as_manifest_row() for item in validations],
        notes=list(notes or []),
    )


def write_manifest(manifest: RunManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "requested_date": manifest.requested_date,
        "pulled_at": manifest.pulled_at,
        "overall_valid": manifest.overall_valid,
        "downloads": manifest.downloads,
        "notes": manifest.notes,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path

