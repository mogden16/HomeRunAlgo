from __future__ import annotations

from scripts.live_refresh_artifacts import (
    LIVE_REFRESH_FORCE_ADD_BY_MODE,
    LIVE_REFRESH_TRACKED_ARTIFACTS,
)


def test_tracked_artifacts_include_daily_board_state() -> None:
    assert "data/live/daily_board_state.json" in LIVE_REFRESH_TRACKED_ARTIFACTS


def test_force_add_modes_are_declared_for_prepare_and_auto() -> None:
    assert "prepare" in LIVE_REFRESH_FORCE_ADD_BY_MODE
    assert "auto" in LIVE_REFRESH_FORCE_ADD_BY_MODE


def test_force_add_lists_include_model_and_draft_artifacts() -> None:
    required = {
        "data/live/model_bundle.pkl",
        "data/live/model_metadata.json",
        "data/live/draft_picks.json",
    }
    for mode in ("prepare", "auto"):
        assert required.issubset(set(LIVE_REFRESH_FORCE_ADD_BY_MODE[mode]))

