"""Centralized selector strategy for Ballpark Pal login and export downloads.

Selectors are intentionally grouped in one module so adjustments after first live run
are localized and low-risk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern


@dataclass(frozen=True)
class ExportSelectorSpec:
    """Text/href patterns used to locate one export type."""

    export_key: str
    label: str
    href_tokens: tuple[str, ...]
    text_patterns: tuple[Pattern[str], ...]


EXPORT_SPECS: tuple[ExportSelectorSpec, ...] = (
    ExportSelectorSpec(
        export_key="batters",
        label="Batters",
        href_tokens=("batter",),
        text_patterns=(
            re.compile(r"\bbatters?\b", re.IGNORECASE),
            re.compile(r"\bdownload\b", re.IGNORECASE),
        ),
    ),
    ExportSelectorSpec(
        export_key="pitchers",
        label="Pitchers",
        href_tokens=("pitcher",),
        text_patterns=(
            re.compile(r"\bpitchers?\b", re.IGNORECASE),
            re.compile(r"\bdownload\b", re.IGNORECASE),
        ),
    ),
    ExportSelectorSpec(
        export_key="teams",
        label="Teams",
        href_tokens=("team",),
        text_patterns=(
            re.compile(r"\bteams?\b", re.IGNORECASE),
            re.compile(r"\bdownload\b", re.IGNORECASE),
        ),
    ),
    ExportSelectorSpec(
        export_key="games",
        label="Games",
        href_tokens=("game",),
        text_patterns=(
            re.compile(r"\bgames?\b", re.IGNORECASE),
            re.compile(r"\bdownload\b", re.IGNORECASE),
        ),
    ),
)


LOGIN_LINK_PATTERN = re.compile(r"(log in|login|sign in|signin)", re.IGNORECASE)
LOGIN_BUTTON_PATTERN = re.compile(r"(log in|login|sign in|signin|continue|submit)", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"(email|e-mail)", re.IGNORECASE)
PASSWORD_PATTERN = re.compile(r"password", re.IGNORECASE)
EXPORT_CENTER_HINT_PATTERN = re.compile(r"(export|download|batters|pitchers|teams|games)", re.IGNORECASE)

