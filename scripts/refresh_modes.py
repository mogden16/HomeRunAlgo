"""Shared Prepare, Publish, Settle, and auto-orchestration refresh mode functions."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from config import (
    LIVE_CURRENT_PICKS_PATH,
    LIVE_DRAFT_PICKS_PATH,
    LIVE_MORNING_BASELINE_PICKS_PATH,
    LIVE_MODEL_BUNDLE_PATH,
    LIVE_MODEL_DATA_PATH,
    LIVE_MODEL_METADATA_PATH,
    LIVE_MODEL_START_DATE,
    LIVE_PICK_HISTORY_PATH,
)
from scripts.build_dashboard_artifacts import DEFAULT_OUTPUT_DIR, build_dashboard_artifacts
from scripts.live_pipeline import (
    build_slate_state,
    default_publish_date,
    fetch_schedule_games,
    load_json_array,
    normalize_game_date,
    parse_game_datetime,
    write_current_picks,
)
from scripts.prepare_live_board import run_prepare_live_board
from scripts.publish_live_picks import (
    DEFAULT_MAX_PICKS,
    DEFAULT_MAX_PICKS_PER_GAME,
    DEFAULT_MAX_PICKS_PER_TEAM,
    DEFAULT_MIN_CONFIDENCE_TIER,
    publish_live_picks,
)
from scripts.refresh_live_results import default_training_end_date, refresh_live_dataset
from scripts.settle_live_results import run_settle_live_results
from scripts.verify_public_live_artifacts import verify_public_live_artifacts

ET_ZONE = ZoneInfo("America/New_York")
PREPARE_START_HOUR_ET = 6
REFRESH_MODES = ("settle", "prepare", "publish", "auto")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_reference_time(reference_time: datetime | None = None) -> datetime:
    if reference_time is None:
        return datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        return reference_time.replace(tzinfo=ET_ZONE).astimezone(timezone.utc)
    return reference_time.astimezone(timezone.utc)


def _prepare_is_current_for_today(
    *,
    metadata_path: Path,
    draft_output_path: Path,
    publish_date: str,
) -> bool:
    yesterday = (datetime.fromisoformat(publish_date) - timedelta(days=1)).date().isoformat()
    metadata = _read_json_object(metadata_path)
    trained_through = normalize_game_date(metadata.get("trained_through"))
    if trained_through and trained_through >= yesterday:
        return True
    draft_rows = load_json_array(draft_output_path)
    draft_dates = {normalize_game_date(row.get("game_date")) for row in draft_rows if normalize_game_date(row.get("game_date"))}
    return publish_date in draft_dates


def _has_pregame_games(slate_state: dict[str, Any]) -> bool:
    return any(str(game.get("game_state") or "") == "pregame" for game in slate_state.get("games", []))


def _write_morning_baseline_if_needed(
    *,
    baseline_path: Path,
    publish_date: str,
    draft_rows: list[dict[str, Any]],
) -> Path:
    existing_rows = load_json_array(baseline_path)
    existing_dates = {
        normalize_game_date(row.get("game_date"))
        for row in existing_rows
        if normalize_game_date(row.get("game_date"))
    }
    if existing_dates == {publish_date} and existing_rows:
        return baseline_path
    baseline_rows = [
        dict(row)
        for row in draft_rows
        if normalize_game_date(row.get("game_date")) == publish_date
    ]
    write_current_picks(baseline_rows or [dict(row) for row in draft_rows], baseline_path)
    return baseline_path


def resolve_auto_refresh_mode(
    *,
    current_picks_path: Path = LIVE_CURRENT_PICKS_PATH,
    metadata_path: Path = LIVE_MODEL_METADATA_PATH,
    draft_output_path: Path = LIVE_DRAFT_PICKS_PATH,
    reference_time: datetime | None = None,
) -> str:
    reference_utc = _resolve_reference_time(reference_time)
    reference_et = reference_utc.astimezone(ET_ZONE)
    publish_date = reference_et.date().isoformat()
    before_prepare_window = reference_et.hour < PREPARE_START_HOUR_ET
    current_rows = load_json_array(current_picks_path)
    active_dates = sorted({normalize_game_date(row.get("game_date")) for row in current_rows if normalize_game_date(row.get("game_date"))})
    active_date = active_dates[-1] if active_dates else None

    if active_date:
        schedule_games = fetch_schedule_games(active_date)
        slate_state = build_slate_state(schedule_games, reference_time=reference_utc)
        if active_date < publish_date:
            return "settle"
        if before_prepare_window:
            return "idle"
        if slate_state["all_final"]:
            return "settle"
        if _has_pregame_games(slate_state):
            return "mixed"
        last_game_datetime = parse_game_datetime(slate_state.get("last_game_datetime"))
        if last_game_datetime and reference_utc >= last_game_datetime:
            return "settle"
        return "settle"

    if before_prepare_window:
        return "idle"

    schedule_games = fetch_schedule_games(publish_date)
    if not schedule_games:
        return "idle"
    if not _prepare_is_current_for_today(
        metadata_path=metadata_path,
        draft_output_path=draft_output_path,
        publish_date=publish_date,
    ):
        return "prepare"

    slate_state = build_slate_state(schedule_games, reference_time=reference_utc)
    if _has_pregame_games(slate_state):
        return "mixed"
    return "settle"


def rebuild_and_verify_public_artifacts(
    *,
    current_picks_path: Path = LIVE_CURRENT_PICKS_PATH,
    history_path: Path = LIVE_PICK_HISTORY_PATH,
    morning_baseline_path: Path = LIVE_MORNING_BASELINE_PICKS_PATH,
    dashboard_output_dir: Path = DEFAULT_OUTPUT_DIR,
    refresh_script_path: Path = Path("scripts/refresh_dashboard.ps1"),
    live_pipeline_path: Path = Path("scripts/live_pipeline.py"),
    verify_public_artifacts: bool = True,
) -> Path:
    dashboard_path = build_dashboard_artifacts(
        current_picks_path=current_picks_path,
        history_path=history_path,
        draft_picks_path=morning_baseline_path,
        output_dir=dashboard_output_dir,
    )
    if verify_public_artifacts:
        verify_public_live_artifacts(
            current_picks=Path(current_picks_path),
            pick_history=Path(history_path),
            dashboard=dashboard_path,
            refresh_script=refresh_script_path,
            live_pipeline=live_pipeline_path,
        )
    return dashboard_path


def run_settle_refresh(
    *,
    dataset_path: Path = LIVE_MODEL_DATA_PATH,
    current_picks_path: Path = LIVE_CURRENT_PICKS_PATH,
    history_path: Path = LIVE_PICK_HISTORY_PATH,
    morning_baseline_path: Path = LIVE_MORNING_BASELINE_PICKS_PATH,
    dashboard_output_dir: Path = DEFAULT_OUTPUT_DIR,
    start_date: str = LIVE_MODEL_START_DATE,
    end_date: str | None = None,
    force_refresh: bool = False,
    rebuild_dashboard: bool = True,
    verify_public_artifacts: bool = True,
    **_: Any,
) -> dict[str, Any]:
    current_rows = load_json_array(current_picks_path)
    active_dates = sorted({normalize_game_date(row.get("game_date")) for row in current_rows if normalize_game_date(row.get("game_date"))})
    resolved_end_date = end_date or (active_dates[-1] if active_dates else default_publish_date())
    refresh_live_dataset(
        output_path=dataset_path,
        start_date=start_date,
        end_date=resolved_end_date,
        force_refresh=force_refresh,
    )
    result = run_settle_live_results(
        dataset_path=dataset_path,
        current_picks_path=current_picks_path,
        history_path=history_path,
    )
    if rebuild_dashboard:
        dashboard_path = rebuild_and_verify_public_artifacts(
            current_picks_path=current_picks_path,
            history_path=history_path,
            morning_baseline_path=morning_baseline_path,
            dashboard_output_dir=dashboard_output_dir,
            verify_public_artifacts=verify_public_artifacts,
        )
        result["dashboard_path"] = dashboard_path
    return result


def run_prepare_refresh(
    *,
    dataset_path: Path = LIVE_MODEL_DATA_PATH,
    bundle_path: Path = LIVE_MODEL_BUNDLE_PATH,
    metadata_path: Path = LIVE_MODEL_METADATA_PATH,
    current_picks_path: Path = LIVE_CURRENT_PICKS_PATH,
    history_path: Path = LIVE_PICK_HISTORY_PATH,
    draft_output_path: Path = LIVE_DRAFT_PICKS_PATH,
    morning_baseline_path: Path = LIVE_MORNING_BASELINE_PICKS_PATH,
    dashboard_output_dir: Path = DEFAULT_OUTPUT_DIR,
    start_date: str = LIVE_MODEL_START_DATE,
    train_end_date: str | None = None,
    publish_date: str | None = None,
    force_refresh: bool = False,
    model: str = "histgb",
    feature_profile: str = "live_usable_candidate_v1",
    calibration: str = "sigmoid",
    selection_metric: str = "pr_auc",
    missingness_threshold: float | None = None,
    training_mode: str = "fast_refit",
    hitters_per_team: int = 9,
    max_picks: int | None = DEFAULT_MAX_PICKS,
    rebuild_dashboard: bool = True,
    verify_public_artifacts: bool = True,
    **_: Any,
) -> list[dict[str, Any]]:
    draft_rows = run_prepare_live_board(
        dataset_path=dataset_path,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        current_picks_path=current_picks_path,
        history_path=history_path,
        draft_output_path=draft_output_path,
        start_date=start_date,
        train_end_date=train_end_date,
        publish_date=publish_date,
        force_refresh=force_refresh,
        model=model,
        feature_profile=feature_profile,
        calibration=calibration,
        selection_metric=selection_metric,
        missingness_threshold=missingness_threshold,
        training_mode=training_mode,
        hitters_per_team=hitters_per_team,
        max_picks=max_picks,
    )
    resolved_publish_date = publish_date or default_publish_date()
    _write_morning_baseline_if_needed(
        baseline_path=morning_baseline_path,
        publish_date=resolved_publish_date,
        draft_rows=draft_rows,
    )
    if rebuild_dashboard:
        rebuild_and_verify_public_artifacts(
            current_picks_path=current_picks_path,
            history_path=history_path,
            morning_baseline_path=morning_baseline_path,
            dashboard_output_dir=dashboard_output_dir,
            verify_public_artifacts=verify_public_artifacts,
        )
    return draft_rows


def run_mixed_refresh(
    *,
    dataset_path: Path = LIVE_MODEL_DATA_PATH,
    bundle_path: Path = LIVE_MODEL_BUNDLE_PATH,
    metadata_path: Path = LIVE_MODEL_METADATA_PATH,
    current_picks_path: Path = LIVE_CURRENT_PICKS_PATH,
    history_path: Path = LIVE_PICK_HISTORY_PATH,
    morning_baseline_path: Path = LIVE_MORNING_BASELINE_PICKS_PATH,
    dashboard_output_dir: Path = DEFAULT_OUTPUT_DIR,
    start_date: str = LIVE_MODEL_START_DATE,
    end_date: str | None = None,
    schedule_date: str | None = None,
    hitters_per_team: int = 9,
    max_picks: int | None = DEFAULT_MAX_PICKS,
    min_confidence_tier: str | None = DEFAULT_MIN_CONFIDENCE_TIER,
    max_picks_per_team: int | None = DEFAULT_MAX_PICKS_PER_TEAM,
    max_picks_per_game: int | None = DEFAULT_MAX_PICKS_PER_GAME,
    force_refresh: bool = False,
    rebuild_dashboard: bool = True,
    verify_public_artifacts: bool = True,
    **_: Any,
) -> dict[str, Any]:
    current_rows = load_json_array(current_picks_path)
    active_dates = sorted({normalize_game_date(row.get("game_date")) for row in current_rows if normalize_game_date(row.get("game_date"))})
    resolved_schedule_date = schedule_date or (active_dates[-1] if active_dates else default_publish_date())
    refresh_live_dataset(
        output_path=dataset_path,
        start_date=start_date,
        end_date=end_date or resolved_schedule_date,
        force_refresh=force_refresh,
    )
    settle_result = run_settle_live_results(
        dataset_path=dataset_path,
        current_picks_path=current_picks_path,
        history_path=history_path,
    )
    published_rows = publish_live_picks(
        dataset_path=dataset_path,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        output_path=current_picks_path,
        history_path=history_path,
        dashboard_output_dir=dashboard_output_dir,
        schedule_date=resolved_schedule_date,
        hitters_per_team=hitters_per_team,
        max_picks=max_picks,
        min_confidence_tier=min_confidence_tier,
        max_picks_per_team=max_picks_per_team,
        max_picks_per_game=max_picks_per_game,
    )
    result: dict[str, Any] = {
        "resolved_schedule_date": resolved_schedule_date,
        "settle_result": settle_result,
        "published_rows": published_rows,
    }
    if rebuild_dashboard:
        dashboard_path = rebuild_and_verify_public_artifacts(
            current_picks_path=current_picks_path,
            history_path=history_path,
            morning_baseline_path=morning_baseline_path,
            dashboard_output_dir=dashboard_output_dir,
            verify_public_artifacts=verify_public_artifacts,
        )
        result["dashboard_path"] = dashboard_path
    return result


def run_publish_refresh(
    *,
    dataset_path: Path = LIVE_MODEL_DATA_PATH,
    bundle_path: Path = LIVE_MODEL_BUNDLE_PATH,
    metadata_path: Path = LIVE_MODEL_METADATA_PATH,
    current_picks_path: Path = LIVE_CURRENT_PICKS_PATH,
    history_path: Path = LIVE_PICK_HISTORY_PATH,
    morning_baseline_path: Path = LIVE_MORNING_BASELINE_PICKS_PATH,
    dashboard_output_dir: Path = DEFAULT_OUTPUT_DIR,
    start_date: str = LIVE_MODEL_START_DATE,
    end_date: str | None = None,
    schedule_date: str | None = None,
    hitters_per_team: int = 9,
    max_picks: int = 20,
    min_confidence_tier: str | None = DEFAULT_MIN_CONFIDENCE_TIER,
    max_picks_per_team: int | None = DEFAULT_MAX_PICKS_PER_TEAM,
    max_picks_per_game: int | None = DEFAULT_MAX_PICKS_PER_GAME,
    refresh_results_before_publish: bool = False,
    rebuild_dashboard: bool = True,
    verify_public_artifacts: bool = True,
    **_: Any,
) -> list[dict[str, Any]]:
    if refresh_results_before_publish:
        resolved_end_date = end_date or default_training_end_date()
        refresh_live_dataset(
            output_path=dataset_path,
            start_date=start_date,
            end_date=resolved_end_date,
            force_refresh=False,
        )
    published_rows = publish_live_picks(
        dataset_path=dataset_path,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        output_path=current_picks_path,
        history_path=history_path,
        dashboard_output_dir=dashboard_output_dir,
        schedule_date=schedule_date,
        hitters_per_team=hitters_per_team,
        max_picks=max_picks,
        min_confidence_tier=min_confidence_tier,
        max_picks_per_team=max_picks_per_team,
        max_picks_per_game=max_picks_per_game,
    )
    if rebuild_dashboard:
        rebuild_and_verify_public_artifacts(
            current_picks_path=current_picks_path,
            history_path=history_path,
            morning_baseline_path=morning_baseline_path,
            dashboard_output_dir=dashboard_output_dir,
            verify_public_artifacts=verify_public_artifacts,
        )
    return published_rows


def run_refresh_mode(
    mode: str,
    **kwargs: Any,
) -> Any:
    if mode == "auto":
        resolved_mode = resolve_auto_refresh_mode(
            current_picks_path=Path(kwargs.get("current_picks_path", LIVE_CURRENT_PICKS_PATH)),
            metadata_path=Path(kwargs.get("metadata_path", LIVE_MODEL_METADATA_PATH)),
            draft_output_path=Path(kwargs.get("draft_output_path", LIVE_DRAFT_PICKS_PATH)),
            reference_time=kwargs.get("reference_time"),
        )
        if resolved_mode == "idle":
            return {"mode": "idle", "result": {"status": "idle"}}
        result = run_refresh_mode(resolved_mode, **kwargs)
        return {"mode": resolved_mode, "result": result}
    if mode == "settle":
        return run_settle_refresh(**kwargs)
    if mode == "prepare":
        return run_prepare_refresh(**kwargs)
    if mode == "publish":
        return run_publish_refresh(**kwargs)
    if mode == "mixed":
        return run_mixed_refresh(**kwargs)
    raise ValueError(f"Unsupported refresh mode: {mode}")
