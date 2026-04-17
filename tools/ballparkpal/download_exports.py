"""Download Ballpark Pal Export Center XLSX files for one date or date ranges.

Historical downloads are useful for automation/schema testing, but may not represent
true point-in-time pregame snapshots from those historical dates.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from tools.ballparkpal.selectors import (
    EMAIL_PATTERN,
    EXPORT_CENTER_HINT_PATTERN,
    EXPORT_SPECS,
    LOGIN_BUTTON_PATTERN,
    LOGIN_LINK_PATTERN,
    PASSWORD_PATTERN,
    ExportSelectorSpec,
)
from tools.ballparkpal.validation import sha256sum, validate_export_file


DEFAULT_BASE_URL = "https://www.ballparkpal.com"
DEFAULT_EXPORT_URL_TEMPLATE = "{base_url}/Export-Center.php"
DEFAULT_OUTPUT_DIR = Path("data/ballparkpal/raw")


@dataclass
class ExportDownloadResult:
    export_key: str
    status: str
    original_download_filename: str | None
    saved_filename: str | None
    file_size_bytes: int | None
    sha256: str | None
    validation: dict[str, Any] | None
    notes: list[str]


def parse_bool_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean flag value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Single date in YYYY-MM-DD.")
    parser.add_argument("--start-date", help="Start date (inclusive) in YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date (inclusive) in YYYY-MM-DD.")
    parser.add_argument("--headless", type=parse_bool_flag, default=True, help="Run browser in headless mode (true/false).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory root for raw exports.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files if present.")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Dry run: validate login/navigation and selector detection without saving downloads.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ballparkpal_downloader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resolve_target_dates(args: argparse.Namespace) -> list[date]:
    if args.date and (args.start_date or args.end_date):
        raise ValueError("Use either --date or --start-date/--end-date, not both.")
    if args.date:
        return [date.fromisoformat(args.date)]
    if args.start_date and args.end_date:
        start = date.fromisoformat(args.start_date)
        end = date.fromisoformat(args.end_date)
        if end < start:
            raise ValueError("--end-date must be >= --start-date.")
        values: list[date] = []
        current = start
        while current <= end:
            values.append(current)
            current += timedelta(days=1)
        return values
    raise ValueError("Provide either --date or both --start-date and --end-date.")


def _locator_count_safe(locator: Any) -> int:
    try:
        return int(locator.count())
    except Exception:
        return 0


def _has_export_hints(page: Page) -> bool:
    if _locator_count_safe(page.get_by_role("link", name=EXPORT_CENTER_HINT_PATTERN)) > 0:
        return True
    if _locator_count_safe(page.get_by_role("button", name=EXPORT_CENTER_HINT_PATTERN)) > 0:
        return True
    if _locator_count_safe(page.locator("a", has_text=EXPORT_CENTER_HINT_PATTERN)) > 0:
        return True
    return False


def _is_likely_logged_in(page: Page) -> bool:
    has_password_input = _locator_count_safe(page.locator("input[type='password']")) > 0
    has_login_link = _locator_count_safe(page.get_by_role("link", name=LOGIN_LINK_PATTERN)) > 0
    has_logout_hint = _locator_count_safe(
        page.get_by_role("button", name=re.compile(r"(logout|log out|account|profile)", re.IGNORECASE))
    ) > 0
    return has_logout_hint or (not has_password_input and not has_login_link)


def _find_first_locator(page: Page, locator_builders: list[tuple[str, Any]]) -> tuple[str, Any] | None:
    for description, locator in locator_builders:
        if _locator_count_safe(locator) > 0:
            return description, locator.first
    return None


def _fill_login_form(page: Page, email: str, password: str, logger: logging.Logger) -> None:
    email_candidates = [
        ("email_by_label", page.get_by_label(EMAIL_PATTERN)),
        ("email_by_placeholder", page.get_by_placeholder(EMAIL_PATTERN)),
        ("email_by_type", page.locator("input[type='email']")),
        ("email_generic_name", page.locator("input[name*='email' i]")),
    ]
    password_candidates = [
        ("password_by_label", page.get_by_label(PASSWORD_PATTERN)),
        ("password_by_placeholder", page.get_by_placeholder(PASSWORD_PATTERN)),
        ("password_by_type", page.locator("input[type='password']")),
        ("password_generic_name", page.locator("input[name*='password' i]")),
    ]

    email_locator = _find_first_locator(page, email_candidates)
    password_locator = _find_first_locator(page, password_candidates)
    if email_locator is None or password_locator is None:
        raise RuntimeError("Could not locate email/password inputs on login page.")

    logger.info("Filling login form via selectors: email=%s password=%s", email_locator[0], password_locator[0])
    email_locator[1].fill(email)
    password_locator[1].fill(password)

    submit_candidates = [
        ("submit_by_role", page.get_by_role("button", name=LOGIN_BUTTON_PATTERN)),
        ("submit_input", page.locator("input[type='submit']")),
        ("submit_button_type", page.locator("button[type='submit']")),
    ]
    submit_locator = _find_first_locator(page, submit_candidates)
    if submit_locator is None:
        raise RuntimeError("Could not locate login submit button.")
    logger.info("Submitting login form via selector: %s", submit_locator[0])
    submit_locator[1].click(timeout=10000)


def login_if_needed(page: Page, base_url: str, email: str, password: str, logger: logging.Logger) -> None:
    logger.info("Login start: opening base URL.")
    page.goto(base_url, wait_until="domcontentloaded", timeout=30000)
    if _is_likely_logged_in(page):
        logger.info("Session already appears authenticated.")
        return

    login_link_candidates = [
        ("login_link_role", page.get_by_role("link", name=LOGIN_LINK_PATTERN)),
        ("login_button_role", page.get_by_role("button", name=LOGIN_LINK_PATTERN)),
        ("login_link_text", page.locator("a", has_text=LOGIN_LINK_PATTERN)),
    ]
    login_locator = _find_first_locator(page, login_link_candidates)
    if login_locator is not None:
        logger.info("Clicking login entry point via selector: %s", login_locator[0])
        login_locator[1].click(timeout=10000)
        page.wait_for_timeout(1000)

    _fill_login_form(page, email=email, password=password, logger=logger)
    page.wait_for_timeout(2000)
    if not _is_likely_logged_in(page):
        raise RuntimeError("Login may have failed; authenticated state not detected.")
    logger.info("Login success.")


def build_export_urls(base_url: str, date_str: str, export_url_template: str) -> list[str]:
    candidates: list[str] = []
    candidates.append(export_url_template.format(base_url=base_url.rstrip("/"), date=date_str))
    # Conservative fallbacks in case site paths drift.
    candidates.append(f"{base_url.rstrip('/')}/Export-Center.php")
    candidates.append(f"{base_url.rstrip('/')}/Export-Center.php?date={date_str}")
    candidates.append(f"{base_url.rstrip('/')}/export?date={date_str}")
    candidates.append(f"{base_url.rstrip('/')}/exports?date={date_str}")

    deduped: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def navigate_to_export_center(
    page: Page,
    *,
    target_date: str,
    base_url: str,
    export_url_template: str,
    logger: logging.Logger,
) -> None:
    urls = build_export_urls(base_url=base_url, date_str=target_date, export_url_template=export_url_template)
    last_error: Exception | None = None

    for url in urls:
        for attempt in range(1, 3):
            logger.info("Navigation attempt %s to export center URL: %s", attempt, url)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(1000)
                if _has_export_hints(page):
                    logger.info("Export center hints detected at URL: %s", page.url)
                    return
            except (PlaywrightTimeoutError, PlaywrightError) as exc:
                last_error = exc
                logger.warning("Navigation issue at %s (attempt %s): %s", url, attempt, exc)
            time.sleep(1.5)

    raise RuntimeError(f"Unable to confirm export center for date {target_date}. Last error: {last_error}")


def _find_export_locator(page: Page, spec: ExportSelectorSpec) -> tuple[str, Any] | None:
    label_pattern = spec.text_patterns[0]
    combined_pattern = spec.text_patterns[1] if len(spec.text_patterns) > 1 else label_pattern
    candidates: list[tuple[str, Any]] = [
        (f"{spec.export_key}_link_role", page.get_by_role("link", name=label_pattern)),
        (f"{spec.export_key}_button_role", page.get_by_role("button", name=label_pattern)),
        (f"{spec.export_key}_link_text", page.locator("a", has_text=label_pattern)),
        (f"{spec.export_key}_button_text", page.locator("button", has_text=label_pattern)),
        (
            f"{spec.export_key}_download_combo",
            page.locator("*", has_text=combined_pattern).filter(has_text=label_pattern),
        ),
    ]
    for token in spec.href_tokens:
        candidates.append(
            (
                f"{spec.export_key}_href_{token}",
                page.locator(f"a[href*='{token}']").filter(has_text=label_pattern),
            )
        )
        candidates.append((f"{spec.export_key}_href_any_{token}", page.locator(f"a[href*='{token}']")))

    return _find_first_locator(page, candidates)


def _download_via_authenticated_request(page: Page, href: str, destination_path: Path) -> None:
    response = page.context.request.get(href, timeout=30000)
    if not response.ok:
        raise RuntimeError(f"Authenticated fallback request failed with HTTP {response.status}.")
    destination_path.write_bytes(response.body())


def download_single_export(
    page: Page,
    *,
    spec: ExportSelectorSpec,
    destination_path: Path,
    overwrite: bool,
    test_mode: bool,
    logger: logging.Logger,
) -> ExportDownloadResult:
    if destination_path.exists() and not overwrite:
        note = "File already exists; skipped (use --overwrite to force)."
        logger.info("%s: %s", spec.label, note)
        return ExportDownloadResult(
            export_key=spec.export_key,
            status="skipped_exists",
            original_download_filename=None,
            saved_filename=destination_path.name,
            file_size_bytes=destination_path.stat().st_size,
            sha256=sha256sum(destination_path),
            validation=validate_export_file(destination_path, spec.export_key).to_dict(),
            notes=[note],
        )

    locator_info: tuple[str, Any] | None = None
    for selector_attempt in range(1, 3):
        locator_info = _find_export_locator(page, spec)
        if locator_info is not None:
            break
        logger.warning(
            "Export control missing for %s on selector attempt %s; waiting/reloading once.",
            spec.label,
            selector_attempt,
        )
        page.wait_for_timeout(1200)
        if selector_attempt == 1:
            page.reload(wait_until="domcontentloaded", timeout=30000)
    if locator_info is None:
        note = f"Could not locate export control for {spec.label}."
        logger.error(note)
        return ExportDownloadResult(
            export_key=spec.export_key,
            status="failed_selector_missing",
            original_download_filename=None,
            saved_filename=None,
            file_size_bytes=None,
            sha256=None,
            validation=None,
            notes=[note],
        )

    locator_description, locator = locator_info
    logger.info("Download start for %s via selector %s", spec.label, locator_description)
    if test_mode:
        note = "Test mode enabled; selector detected but download skipped."
        logger.info("%s", note)
        return ExportDownloadResult(
            export_key=spec.export_key,
            status="test_mode_skipped_download",
            original_download_filename=None,
            saved_filename=None,
            file_size_bytes=None,
            sha256=None,
            validation=None,
            notes=[note],
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    suggested_filename: str | None = None

    for attempt in range(1, 3):
        try:
            with page.expect_download(timeout=30000) as download_info:
                locator.click(timeout=10000)
            download = download_info.value
            suggested_filename = download.suggested_filename
            download.save_as(str(destination_path))
            break
        except Exception as exc:
            last_error = exc
            logger.warning("Download click failed for %s on attempt %s: %s", spec.label, attempt, exc)
            href = locator.get_attribute("href")
            if href:
                normalized_href = urljoin(page.url, href)
                logger.info("Attempting authenticated request fallback for %s: %s", spec.label, normalized_href)
                try:
                    _download_via_authenticated_request(page, normalized_href, destination_path)
                    break
                except Exception as fallback_exc:
                    last_error = fallback_exc
                    logger.warning("Fallback download failed for %s: %s", spec.label, fallback_exc)
            page.wait_for_timeout(1200)
    else:
        note = f"Failed to download {spec.label}. Last error: {last_error}"
        logger.error(note)
        return ExportDownloadResult(
            export_key=spec.export_key,
            status="failed_download",
            original_download_filename=suggested_filename,
            saved_filename=None,
            file_size_bytes=None,
            sha256=None,
            validation=None,
            notes=[note],
        )

    validation = validate_export_file(destination_path, spec.export_key)
    checksum = sha256sum(destination_path)
    file_size = destination_path.stat().st_size
    status = "success" if validation.is_valid else "failed_validation"
    if validation.is_valid:
        logger.info("Validation success for %s (%s bytes).", spec.label, file_size)
    else:
        logger.error("Validation failed for %s: %s", spec.label, validation.error)
    return ExportDownloadResult(
        export_key=spec.export_key,
        status=status,
        original_download_filename=suggested_filename,
        saved_filename=destination_path.name,
        file_size_bytes=file_size,
        sha256=checksum,
        validation=validation.to_dict(),
        notes=[validation.error] if validation.error else [],
    )


def write_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_for_date(
    page: Page,
    *,
    target_date: date,
    base_url: str,
    export_url_template: str,
    output_dir: Path,
    overwrite: bool,
    test_mode: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    requested_date = target_date.isoformat()
    date_dir = output_dir / requested_date
    pulled_at_utc = datetime.now(timezone.utc)
    timestamp_label = pulled_at_utc.astimezone().strftime("%H%M")
    logger.info("Processing date %s", requested_date)

    navigate_to_export_center(
        page,
        target_date=requested_date,
        base_url=base_url,
        export_url_template=export_url_template,
        logger=logger,
    )

    export_results: dict[str, dict[str, Any]] = {}
    for spec in EXPORT_SPECS:
        destination = date_dir / f"{requested_date}_{timestamp_label}_{spec.export_key}.xlsx"
        result = download_single_export(
            page,
            spec=spec,
            destination_path=destination,
            overwrite=overwrite,
            test_mode=test_mode,
            logger=logger,
        )
        export_results[spec.export_key] = asdict(result)

    statuses = [item["status"] for item in export_results.values()]
    passing_statuses = {"success", "test_mode_skipped_download", "skipped_exists"}
    successful = sum(1 for status in statuses if status in passing_statuses)
    failed = sum(1 for status in statuses if status not in passing_statuses)
    manifest_payload = {
        "requested_date": requested_date,
        "pulled_at": pulled_at_utc.isoformat(),
        "base_url": base_url,
        "export_url_template": export_url_template,
        "test_mode": test_mode,
        "exports": export_results,
        "summary": {
            "success_count": successful,
            "failure_count": failed,
            "all_success": failed == 0,
        },
        "notes": [
            "Historical export pulls may not be true point-in-time pregame snapshots.",
            "Use historical ranges first for automation and schema validation; trust forward daily archives for backtesting.",
        ],
    }
    write_manifest(date_dir / "download_manifest.json", manifest_payload)
    logger.info(
        "Date %s summary: success=%s failure=%s",
        requested_date,
        manifest_payload["summary"]["success_count"],
        manifest_payload["summary"]["failure_count"],
    )
    return manifest_payload


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    args = parse_args()
    target_dates = resolve_target_dates(args)
    output_dir = Path(args.output_dir)

    base_url = os.environ.get("BALLPARKPAL_BASE_URL", DEFAULT_BASE_URL).strip()
    export_url_template = os.environ.get("BALLPARKPAL_EXPORT_URL_TEMPLATE", DEFAULT_EXPORT_URL_TEMPLATE).strip()
    email = (os.environ.get("BALLPARKPAL_EMAIL") or "").strip()
    password = (os.environ.get("BALLPARKPAL_PASSWORD") or "").strip()
    if not email or not password:
        raise RuntimeError("BALLPARKPAL_EMAIL and BALLPARKPAL_PASSWORD must be set in environment variables.")

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = output_dir.parent / "logs" / f"ballparkpal_download_{run_timestamp}.log"
    logger = setup_logger(log_path)
    logger.info("Ballpark Pal export downloader start.")
    logger.info("Target dates: %s", [d.isoformat() for d in target_dates])
    logger.info("Headless=%s TestMode=%s OutputDir=%s", args.headless, args.test_mode, output_dir)

    date_manifests: list[dict[str, Any]] = []
    failure_dates: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=args.headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        try:
            login_if_needed(page, base_url=base_url, email=email, password=password, logger=logger)
            for target_date in target_dates:
                try:
                    manifest = run_for_date(
                        page,
                        target_date=target_date,
                        base_url=base_url,
                        export_url_template=export_url_template,
                        output_dir=output_dir,
                        overwrite=args.overwrite,
                        test_mode=args.test_mode,
                        logger=logger,
                    )
                    date_manifests.append(manifest)
                    if not manifest["summary"]["all_success"]:
                        failure_dates.append(target_date.isoformat())
                except Exception as exc:
                    logger.exception("Date %s failed: %s", target_date.isoformat(), exc)
                    failure_dates.append(target_date.isoformat())
        finally:
            context.close()
            browser.close()

    run_manifest = {
        "run_timestamp": run_timestamp,
        "target_dates": [d.isoformat() for d in target_dates],
        "test_mode": args.test_mode,
        "date_manifests": date_manifests,
        "failure_dates": failure_dates,
    }
    write_manifest(output_dir.parent / f"run_manifest_{run_timestamp}.json", run_manifest)

    if failure_dates:
        logger.error("Run completed with failures for dates: %s", failure_dates)
        raise SystemExit(1)
    logger.info("Run completed successfully.")


if __name__ == "__main__":
    main()
