"""Selectors and URLs used by the Ballpark Pal browser workflow."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_BASE_URL = "https://ballparkpal.com"
DEFAULT_LOGIN_URL = f"{DEFAULT_BASE_URL}/login"
DEFAULT_EXPORT_CENTER_URL = f"{DEFAULT_BASE_URL}/export-center"

EXPORT_NAMES = ("batters", "pitchers", "teams", "games")

LOGIN_EMAIL_SELECTORS = (
    "input[type='email']",
    "input[name='email']",
    "input[name='username']",
    "input[id*='email' i]",
    "input[id*='username' i]",
)
LOGIN_PASSWORD_SELECTORS = (
    "input[type='password']",
    "input[name='password']",
    "input[id*='password' i]",
)
LOGIN_SUBMIT_SELECTORS = (
    "button[type='submit']",
    "input[type='submit']",
    "button:has-text('Sign in')",
    "button:has-text('Log in')",
    "button:has-text('Login')",
)
EXPORT_CENTER_SELECTORS = (
    "a:has-text('Export Center')",
    "button:has-text('Export Center')",
    "text=Export Center",
)
DATE_FILTER_SELECTORS = (
    "input[type='date']",
    "input[name*='date' i]",
    "input[id*='date' i]",
    "input[aria-label*='date' i]",
)
DATE_FILTER_APPLY_SELECTORS = (
    "button:has-text('Apply')",
    "button:has-text('Load')",
    "button:has-text('Refresh')",
    "button:has-text('Update')",
)
EXPORT_LINK_TEXTS = {
    "batters": (
        "Batters",
        "Batters Export",
        "Download Batters",
        "Export Batters",
    ),
    "pitchers": (
        "Pitchers",
        "Pitchers Export",
        "Download Pitchers",
        "Export Pitchers",
    ),
    "teams": (
        "Teams",
        "Teams Export",
        "Download Teams",
        "Export Teams",
    ),
    "games": (
        "Games",
        "Games Export",
        "Download Games",
        "Export Games",
    ),
}


@dataclass(frozen=True)
class BallparkPalUrls:
    login_url: str = DEFAULT_LOGIN_URL
    export_center_url: str = DEFAULT_EXPORT_CENTER_URL

