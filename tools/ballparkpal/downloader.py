"""Playwright-based Ballpark Pal download workflow."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import DownloadedWorkbook
from .selectors import (
    DATE_FILTER_APPLY_SELECTORS,
    DATE_FILTER_SELECTORS,
    DEFAULT_EXPORT_CENTER_URL,
    DEFAULT_LOGIN_URL,
    EXPORT_CENTER_SELECTORS,
    EXPORT_LINK_TEXTS,
    EXPORT_NAMES,
    LOGIN_EMAIL_SELECTORS,
    LOGIN_PASSWORD_SELECTORS,
    LOGIN_SUBMIT_SELECTORS,
)


@dataclass(frozen=True)
class BallparkPalCredentials:
    email: str
    password: str


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def load_credentials(*, env_file: Path | None = None) -> BallparkPalCredentials:
    candidates: list[Path] = []
    if env_file is not None:
        candidates.append(env_file)
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parents[2] / ".env")

    merged: dict[str, str] = dict(os.environ)
    for candidate in candidates:
        merged.update(_load_env_file(candidate))

    email = merged.get("BALLPARKPAL_EMAIL") or merged.get("BALLPARKPAL_USERNAME")
    password = merged.get("BALLPARKPAL_PASSWORD")
    if not email or not password:
        raise RuntimeError(
            "Missing Ballpark Pal credentials. Set BALLPARKPAL_EMAIL and BALLPARKPAL_PASSWORD in the environment or a local .env file."
        )
    return BallparkPalCredentials(email=email, password=password)


def _normalize_filename(filename: str, export_name: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix != ".xlsx":
        suffix = ".xlsx"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("_")
    if not safe:
        safe = export_name
    return f"{safe}{suffix}"


def _ensure_visible_export_center(page: Any) -> None:
    page.wait_for_load_state("domcontentloaded")
    return


def _fill_first(page: Any, selectors: tuple[str, ...], value: str) -> None:
    for selector in selectors:
        try:
            locator = page.locator(selector)
            if locator.count() > 0:
                locator.first.fill(value)
                return
        except Exception:
            pass
    if selectors is LOGIN_EMAIL_SELECTORS:
        for locator in (
            page.get_by_label("Email Address", exact=False),
            page.get_by_placeholder("Enter your email address", exact=False),
        ):
            try:
                if locator.count() > 0:
                    locator.first.fill(value)
                    return
            except Exception:
                continue
    if selectors is LOGIN_PASSWORD_SELECTORS:
        for locator in (
            page.get_by_label("Password", exact=False),
            page.get_by_placeholder("Enter your password", exact=False),
        ):
            try:
                if locator.count() > 0:
                    locator.first.fill(value)
                    return
            except Exception:
                continue
    raise RuntimeError(f"Could not find any matching input for selectors: {selectors}")


def _click_first(page: Any, selectors: tuple[str, ...]) -> None:
    for selector in selectors:
        try:
            locator = page.locator(selector)
            if locator.count() > 0:
                locator.first.click()
                return
        except Exception:
            pass
    if selectors is LOGIN_SUBMIT_SELECTORS:
        for locator in (
            page.get_by_role("button", name="Sign In"),
            page.get_by_role("button", name="Login"),
            page.get_by_role("button", name="Log in"),
            page.get_by_text("Sign In", exact=False),
        ):
            try:
                if locator.count() > 0:
                    locator.first.click()
                    return
            except Exception:
                continue
    raise RuntimeError(f"Could not find any matching button for selectors: {selectors}")


def _maybe_apply_requested_date(page: Any, requested_date: str) -> None:
    for selector in DATE_FILTER_SELECTORS:
        locator = page.locator(selector)
        if locator.count() > 0:
            try:
                if not locator.first.is_visible():
                    continue
            except Exception:
                continue
            locator.first.fill(requested_date)
            for apply_selector in DATE_FILTER_APPLY_SELECTORS:
                apply_locator = page.locator(apply_selector)
                if apply_locator.count() > 0:
                    apply_locator.first.click()
                    page.wait_for_load_state("networkidle")
                    return
            page.wait_for_timeout(500)
            return


def _download_export(page: Any, export_name: str, requested_date: str, output_dir: Path) -> DownloadedWorkbook:
    form_action = f"Export{export_name.title()}.php"
    action_locators = [
        page.locator(f"form[action='{form_action}'] button[type='submit']"),
        page.locator(f"form[action='{form_action}'] button"),
    ]
    for locator in action_locators:
        try:
            if locator.count() == 0:
                continue
            with page.expect_download() as download_info:
                locator.first.click()
            download = download_info.value
            suggested_filename = download.suggested_filename
            saved_filename = _normalize_filename(suggested_filename, export_name)
            saved_path = output_dir / saved_filename
            download.save_as(saved_path)
            return DownloadedWorkbook(
                export_name=export_name,
                requested_date=requested_date,
                original_filename=suggested_filename,
                saved_path=saved_path,
                source_url=page.url,
            )
        except Exception:
            continue
    for text in EXPORT_LINK_TEXTS[export_name]:
        try:
            locator = page.get_by_role("button", name=text, exact=False)
            if locator.count() == 0:
                continue
            with page.expect_download() as download_info:
                locator.first.click()
            download = download_info.value
            suggested_filename = download.suggested_filename
            saved_filename = _normalize_filename(suggested_filename, export_name)
            saved_path = output_dir / saved_filename
            download.save_as(saved_path)
            return DownloadedWorkbook(
                export_name=export_name,
                requested_date=requested_date,
                original_filename=suggested_filename,
                saved_path=saved_path,
                source_url=page.url,
            )
        except Exception:
            continue
    raise RuntimeError(f"Could not find a download link for export '{export_name}'.")


def download_ballparkpal_exports(
    *,
    requested_date: str,
    output_dir: Path,
    headless: bool,
    env_file: Path | None = None,
    login_url: str = DEFAULT_LOGIN_URL,
    export_center_url: str = DEFAULT_EXPORT_CENTER_URL,
    slow_mo_ms: int = 0,
    navigation_timeout_ms: int = 60_000,
) -> list[DownloadedWorkbook]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Playwright is required. Install the 'playwright' package and browser binaries.") from exc

    credentials = load_credentials(env_file=env_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloads: list[DownloadedWorkbook] = []
    export_url = export_center_url
    if requested_date and "date=" not in export_url:
        separator = "&" if "?" in export_url else "?"
        export_url = f"{export_url}{separator}date={requested_date}"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless, slow_mo=slow_mo_ms)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(navigation_timeout_ms)

        page.goto(export_url, wait_until="domcontentloaded")
        if not any(page.locator(selector).count() > 0 for selector in EXPORT_CENTER_SELECTORS):
            page.goto(login_url, wait_until="domcontentloaded")
            _fill_first(page, LOGIN_EMAIL_SELECTORS, credentials.email)
            _fill_first(page, LOGIN_PASSWORD_SELECTORS, credentials.password)
            _click_first(page, LOGIN_SUBMIT_SELECTORS)
            page.wait_for_load_state("networkidle")
            page.goto(export_url, wait_until="domcontentloaded")

        _ensure_visible_export_center(page)
        _maybe_apply_requested_date(page, requested_date)

        requested_output_dir = output_dir / requested_date
        requested_output_dir.mkdir(parents=True, exist_ok=True)
        for export_name in EXPORT_NAMES:
            downloads.append(_download_export(page, export_name, requested_date, requested_output_dir))

        context.close()
        browser.close()
    return downloads

