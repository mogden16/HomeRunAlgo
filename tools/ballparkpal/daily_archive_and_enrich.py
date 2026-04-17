"""Daily Ballpark Pal archive runner and pick enrichment utility.

This script is intended for forward daily collection of the live slate and
cross-referencing your picks against the four export workbooks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from tools.ballparkpal.download_exports import (
    DEFAULT_BASE_URL,
    DEFAULT_EXPORT_URL_TEMPLATE,
    DEFAULT_OUTPUT_DIR,
    login_if_needed,
    parse_bool_flag,
    run_for_date,
    setup_logger,
)
from tools.ballparkpal.feature_join import (
    BALLPARKPAL_PICK_OVERLAY_MAX_ABS_SCORE,
    BALLPARKPAL_PICK_OVERLAY_RULES,
    PICK_BATTER_FEATURE_COLUMNS,
    PICK_PITCHER_FEATURE_COLUMNS,
    enrich_picks_with_ballparkpal,
)

DEFAULT_PICKS_CANDIDATES = (
    ROOT_DIR / "data" / "live" / "current_picks.json",
    ROOT_DIR / "data" / "live" / "current_picks_smoke.json",
)
DEFAULT_ENRICHED_DIR = Path("data/ballparkpal/enriched")
DEFAULT_RUN_DIR = Path("data/ballparkpal/runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Target slate date in YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--headless", type=parse_bool_flag, default=True, help="Run browser in headless mode (true/false).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory root for raw exports.")
    parser.add_argument("--enriched-output-dir", default=str(DEFAULT_ENRICHED_DIR), help="Directory root for enriched picks outputs.")
    parser.add_argument("--run-output-dir", default=str(DEFAULT_RUN_DIR), help="Directory root for combined run manifests.")
    parser.add_argument("--picks-path", help="Path to your picks file (JSON or CSV).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw export files if present.")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Validate login/navigation and selector detection without downloading or enriching files.",
    )
    return parser.parse_args()


def _resolve_target_date(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    return datetime.now().astimezone().date()


def _resolve_picks_path(explicit_path: str | None) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    for candidate in DEFAULT_PICKS_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _load_picks_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path)
    raise ValueError(f"Unsupported picks file format: {path.suffix}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_enriched_outputs(df: pd.DataFrame, output_dir: Path, requested_date: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{requested_date}_ballparkpal_picks_enriched.csv"
    json_path = output_dir / f"{requested_date}_ballparkpal_picks_enriched.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2, date_format="iso")
    return {"csv": str(csv_path), "json": str(json_path)}


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    args = parse_args()

    target_date = _resolve_target_date(args.date)
    requested_date = target_date.isoformat()
    output_dir = Path(args.output_dir)
    enriched_output_dir = Path(args.enriched_output_dir) / requested_date
    run_output_dir = Path(args.run_output_dir)

    base_url = os.environ.get("BALLPARKPAL_BASE_URL", DEFAULT_BASE_URL).strip()
    export_url_template = os.environ.get("BALLPARKPAL_EXPORT_URL_TEMPLATE", DEFAULT_EXPORT_URL_TEMPLATE).strip()
    email = (os.environ.get("BALLPARKPAL_EMAIL") or "").strip()
    password = (os.environ.get("BALLPARKPAL_PASSWORD") or "").strip()
    if not email or not password:
        raise RuntimeError("BALLPARKPAL_EMAIL and BALLPARKPAL_PASSWORD must be set in environment variables.")

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = output_dir.parent / "logs" / f"ballparkpal_daily_{run_timestamp}.log"
    logger = setup_logger(log_path)
    logger.info("Ballpark Pal daily archive runner start.")
    logger.info("Target date: %s", requested_date)
    logger.info("Headless=%s TestMode=%s OutputDir=%s", args.headless, args.test_mode, output_dir)

    snapshot_manifest: dict[str, Any] | None = None
    date_dir = output_dir / requested_date
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=args.headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        try:
            login_if_needed(page, base_url=base_url, email=email, password=password, logger=logger)
            snapshot_manifest = run_for_date(
                page,
                target_date=target_date,
                base_url=base_url,
                export_url_template=export_url_template,
                output_dir=output_dir,
                overwrite=args.overwrite,
                test_mode=args.test_mode,
                logger=logger,
            )
        finally:
            context.close()
            browser.close()

    enriched_manifest: dict[str, Any] | None = None
    picks_path = _resolve_picks_path(args.picks_path)
    if args.test_mode:
        logger.info("Test mode enabled; skipping picks enrichment.")
    elif picks_path is None:
        logger.warning("No picks file found; skipping enrichment.")
    else:
        logger.info("Loading picks from %s", picks_path)
        picks_df = _load_picks_frame(picks_path)
        if picks_df.empty:
            logger.warning("Picks file %s is empty; skipping enrichment.", picks_path)
        else:
            enriched_df, coverage, overlay_summary = enrich_picks_with_ballparkpal(picks_df, date_dir)
            output_paths = _write_enriched_outputs(enriched_df, enriched_output_dir, requested_date)
            enriched_manifest = {
                "requested_date": requested_date,
                "pulled_at": datetime.now(timezone.utc).isoformat(),
                "picks_path": str(picks_path),
                "snapshot_dir": str(date_dir),
                "output_paths": output_paths,
                "coverage": coverage.to_dict(),
                "overlay_summary": overlay_summary.to_dict(),
                "feature_columns": {
                    "batter": PICK_BATTER_FEATURE_COLUMNS,
                    "pitcher": PICK_PITCHER_FEATURE_COLUMNS,
                },
                "overlay_rules": BALLPARKPAL_PICK_OVERLAY_RULES,
                "overlay_max_abs_score": BALLPARKPAL_PICK_OVERLAY_MAX_ABS_SCORE,
            }
            _write_json(enriched_output_dir / "picks_enrichment_manifest.json", enriched_manifest)
            logger.info(
                "Pick enrichment complete: batter_coverage=%.3f pitcher_coverage=%.3f signed_mean=%.2f",
                coverage.batter_coverage,
                coverage.pitcher_coverage,
                overlay_summary.mean_signed_score,
            )

    run_manifest = {
        "run_timestamp": run_timestamp,
        "requested_date": requested_date,
        "raw_snapshot_dir": str(date_dir),
        "log_path": str(log_path),
        "download_manifest": snapshot_manifest,
        "enrichment_manifest": enriched_manifest,
    }
    run_output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_output_dir / f"ballparkpal_daily_run_{run_timestamp}.json", run_manifest)

    if snapshot_manifest is None or not snapshot_manifest["summary"]["all_success"]:
        logger.error("Daily archive run failed for %s.", requested_date)
        raise SystemExit(1)

    logger.info("Daily archive run completed successfully.")


if __name__ == "__main__":
    main()
