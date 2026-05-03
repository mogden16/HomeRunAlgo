"""Shared live refresh artifact lists used by automation and tests."""

from __future__ import annotations


LIVE_REFRESH_TRACKED_ARTIFACTS: tuple[str, ...] = (
    "data/live/current_picks.json",
    "data/live/pick_history.json",
    "data/live/morning_baseline_picks.json",
    "data/live/daily_board_state.json",
    "cloudflare-app/data/dashboard.json",
)

LIVE_REFRESH_FORCE_ADD_BY_MODE: dict[str, tuple[str, ...]] = {
    "prepare": (
        "data/live/model_bundle.pkl",
        "data/live/model_metadata.json",
        "data/live/draft_picks.json",
    ),
    "auto": (
        "data/live/model_bundle.pkl",
        "data/live/model_metadata.json",
        "data/live/draft_picks.json",
    ),
}

