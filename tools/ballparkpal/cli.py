"""CLI entrypoint for Ballpark Pal validation runs."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .downloader import download_ballparkpal_exports
from .manifest import build_manifest, write_manifest
from .overlay import build_validation_overlay, write_validation_overlay
from .snapshot import build_ballparkpal_snapshot, write_ballparkpal_snapshot, write_latest_ballparkpal_snapshot
from .selectors import DEFAULT_EXPORT_CENTER_URL, DEFAULT_LOGIN_URL
from .validator import validate_workbook_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requested-date", required=True, help="Requested slate date in YYYY-MM-DD format.")
    parser.add_argument("--output-root", default="data/ballparkpal", help="Root directory for raw files, logs, and manifests.")
    parser.add_argument("--env-file", default=None, help="Optional path to a local .env file.")
    parser.add_argument("--login-url", default=DEFAULT_LOGIN_URL, help="Override the Ballpark Pal login URL.")
    parser.add_argument("--export-center-url", default=DEFAULT_EXPORT_CENTER_URL, help="Override the Ballpark Pal export center URL.")
    parser.add_argument("--headed", action="store_true", help="Run Playwright with a visible browser window.")
    parser.add_argument("--slow-mo-ms", type=int, default=0, help="Optional Playwright slow-motion delay in milliseconds.")
    parser.add_argument("--navigation-timeout-ms", type=int, default=60_000, help="Navigation timeout in milliseconds.")
    parser.add_argument("--overlay-output", default=None, help="Optional path to write a validation overlay JSON file.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Logging level.")
    return parser.parse_args()


def _configure_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def run_ballparkpal_validation(
    *,
    requested_date: str,
    output_root: Path,
    env_file: Path | None = None,
    login_url: str = DEFAULT_LOGIN_URL,
    export_center_url: str = DEFAULT_EXPORT_CENTER_URL,
    headless: bool = True,
    slow_mo_ms: int = 0,
    navigation_timeout_ms: int = 60_000,
    overlay_output: Path | None = None,
    log_level: str = "INFO",
) -> dict[str, Any]:
    run_started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = output_root / "raw" / requested_date
    logs_dir = output_root / "logs"
    manifest_path = raw_dir / "manifest.json"
    log_path = logs_dir / f"{requested_date}_{run_started_at}.log"
    _configure_logging(log_path, log_level)
    logger = logging.getLogger("ballparkpal")

    logger.info("Starting Ballpark Pal validation for %s", requested_date)
    downloads = download_ballparkpal_exports(
        requested_date=requested_date,
        output_dir=raw_dir,
        headless=headless,
        env_file=env_file,
        login_url=login_url,
        export_center_url=export_center_url,
        slow_mo_ms=slow_mo_ms,
        navigation_timeout_ms=navigation_timeout_ms,
    )

    validations = [
        validate_workbook_file(download.saved_path, requested_date=requested_date, export_name=download.export_name)
        for download in downloads
    ]

    manifest = build_manifest(
        requested_date=requested_date,
        validations=validations,
        notes=[f"log_file={log_path}"],
    )
    write_manifest(manifest, manifest_path)
    logger.info("Wrote manifest to %s", manifest_path)

    if overlay_output is not None and manifest.overall_valid:
        overlay_payload = build_validation_overlay({item.export_name: item.saved_path for item in downloads}, requested_date)
        write_validation_overlay(overlay_payload, overlay_output)
        logger.info("Wrote validation overlay to %s", overlay_output)

    snapshot_path = None
    if manifest.overall_valid:
        snapshot_payload = build_ballparkpal_snapshot(
            requested_date=requested_date,
            pulled_at=manifest.pulled_at,
            downloads=downloads,
            validations=validations,
        )
        snapshot_dir = output_root / "validated" / requested_date
        snapshot_path = snapshot_dir / "ballparkpal_snapshot.json"
        write_ballparkpal_snapshot(snapshot_payload, snapshot_path)
        write_latest_ballparkpal_snapshot(snapshot_payload, output_root / "validated" / "latest_snapshot.json")
        logger.info("Wrote normalized snapshot to %s", snapshot_path)

    invalid_exports = [item for item in validations if not item.valid]
    if invalid_exports:
        for item in invalid_exports:
            logger.error("Invalid export %s: %s", item.export_name, "; ".join(item.errors))
        raise RuntimeError("Ballpark Pal validation failed.")

    logger.info("Ballpark Pal validation passed for %s", requested_date)
    return {
        "requested_date": requested_date,
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "downloads": [item.as_manifest_row() for item in validations],
    }


def main() -> None:
    args = parse_args()
    result = run_ballparkpal_validation(
        requested_date=args.requested_date,
        output_root=Path(args.output_root),
        env_file=Path(args.env_file) if args.env_file else None,
        login_url=args.login_url,
        export_center_url=args.export_center_url,
        headless=not args.headed,
        slow_mo_ms=args.slow_mo_ms,
        navigation_timeout_ms=args.navigation_timeout_ms,
        overlay_output=Path(args.overlay_output) if args.overlay_output else None,
        log_level=args.log_level,
    )
    print(result["manifest_path"])


if __name__ == "__main__":
    main()

