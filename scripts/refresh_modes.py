"""Shared Prepare, Publish, Settle, and auto-orchestration refresh mode functions."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from config import (
    LIVE_CURRENT_PICKS_PATH,
    LIVE_DAILY_BOARD_STATE_PATH,
    LIVE_DRAFT_PICKS_PATH,
    LIVE_MORNING_BASELINE_PICKS_PATH,
    LIVE_MODEL_BUNDLE_PATH,
    LIVE_MODEL_DATA_PATH,
    LIVE_MODEL_METADATA_PATH,
    LIVE_MODEL_START_DATE,
    LIVE_PICK_HISTORY_PATH,
)
from scripts.board_state import (
    apply_major_alerts,
    board_entries_to_current_rows,
    create_daily_board_snapshot,
    load_daily_board_state,
    update_board_entry_status,
    write_daily_board_state,
)
from scripts.build_dashboard_artifacts import DEFAULT_OUTPUT_DIR, build_dashboard_artifacts
from scripts.live_pipeline import (
    build_slate_state,
    default_publish_date,
    enrich_ballparkpal_rows,
    fetch_schedule_games,
    load_json_array,
    load_live_dataset,
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


def _row_is_terminal_for_refresh(row: dict[str, Any]) -> bool:
    result_token = str(row.get("result_label") or row.get("result") or "").strip().lower()
    if result_token in {"hr", "no hr", "postponed"}:
        return True
    game_state = str(row.get("game_state") or "").strip().lower()
    if game_state == "final":
        return True
    status_token = str(row.get("game_status") or row.get("status") or "").strip().lower()
    return any(token in status_token for token in ("final", "game over", "completed early", "postponed", "cancelled"))


def _stale_terminal_rows_should_not_block_publish(
    *,
    active_date: str | None,
    publish_date: str,
    current_rows: list[dict[str, Any]],
) -> bool:
    if not active_date or active_date >= publish_date:
        return False
    active_rows = [
        dict(row)
        for row in current_rows
        if normalize_game_date(row.get("game_date")) == active_date
    ]
    return bool(active_rows) and all(_row_is_terminal_for_refresh(row) for row in active_rows)


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


def _same_day_rows(rows: list[dict[str, Any]], schedule_date: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if normalize_game_date(row.get("game_date")) == schedule_date
    ]


def publish_prepared_baseline_to_current(
    *,
    baseline_path: Path,
    current_picks_path: Path,
    board_state_path: Path,
    schedule_date: str,
    dataset_df: Any | None = None,
    resolved_through_date: str | None = None,
) -> list[dict[str, Any]]:
    baseline_rows = _same_day_rows(load_json_array(baseline_path), schedule_date)
    if not baseline_rows:
        return []

    schedule_games = fetch_schedule_games(schedule_date)
    created_at = str(baseline_rows[0].get("created_at") or baseline_rows[0].get("published_at") or "")
    board_state = create_daily_board_snapshot(
        baseline_rows,
        schedule_date,
        created_at=created_at,
    )
    if dataset_df is not None and resolved_through_date:
        board_state, _ = update_board_entry_status(
            board_state,
            dataset_df,
            resolved_through_date=resolved_through_date,
            schedule_games=schedule_games,
        )
    board_state, alert_count = apply_major_alerts(
        board_state,
        schedule_games=schedule_games,
    )
    write_daily_board_state(board_state, board_state_path)
    published_rows = enrich_ballparkpal_rows(
        board_entries_to_current_rows(board_state),
        schedule_date=schedule_date,
    )
    write_current_picks(published_rows, current_picks_path)
    print(
        f"Published prepared baseline for {schedule_date}: "
        f"{len(published_rows)} rows, {alert_count} alerts"
    )
    return published_rows


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
    if _stale_terminal_rows_should_not_block_publish(
        active_date=active_date,
        publish_date=publish_date,
        current_rows=current_rows,
    ):
        active_date = None

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
        season_hr_source="live",
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
    board_state_path: Path = LIVE_DAILY_BOARD_STATE_PATH,
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
        board_state_path=board_state_path,
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
    board_state_path: Path = LIVE_DAILY_BOARD_STATE_PATH,
    draft_output_path: Path = LIVE_DRAFT_PICKS_PATH,
    morning_baseline_path: Path = LIVE_MORNING_BASELINE_PICKS_PATH,
    dashboard_output_dir: Path = DEFAULT_OUTPUT_DIR,
    start_date: str = LIVE_MODEL_START_DATE,
    train_end_date: str | None = None,
    publish_date: str | None = None,
    force_refresh: bool = False,
    model: str = "histgb",
    feature_profile: str = "pregame_safe_v1",
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
        board_state_path=board_state_path,
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
    board_state_path: Path = LIVE_DAILY_BOARD_STATE_PATH,
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
        board_state_path=board_state_path,
    )
    refreshed_current_rows = load_json_array(current_picks_path)
    existing_same_day_rows = [
        dict(row)
        for row in refreshed_current_rows
        if normalize_game_date(row.get("game_date")) == resolved_schedule_date
    ]
    if existing_same_day_rows:
        schedule_games = fetch_schedule_games(resolved_schedule_date)
        board_state = load_daily_board_state(board_state_path)
        if normalize_game_date(board_state.get("board_date")) != resolved_schedule_date or not board_state.get("entries"):
            board_state = create_daily_board_snapshot(
                existing_same_day_rows,
                resolved_schedule_date,
                created_at=str(existing_same_day_rows[0].get("created_at") or existing_same_day_rows[0].get("published_at") or ""),
            )
        dataset_df = load_live_dataset(dataset_path)
        resolved_through_date = str(dataset_df["game_date"].max().date())
        board_state, status_updates = update_board_entry_status(
            board_state,
            dataset_df,
            resolved_through_date=resolved_through_date,
            schedule_games=schedule_games,
        )
        board_state, alert_count = apply_major_alerts(
            board_state,
            schedule_games=schedule_games,
        )
        write_daily_board_state(board_state, board_state_path)
        published_rows = enrich_ballparkpal_rows(
            board_entries_to_current_rows(board_state),
            schedule_date=resolved_schedule_date,
        )
        write_current_picks(published_rows, current_picks_path)
        print(
            f"Stable mixed refresh for {resolved_schedule_date}: "
            f"{len(published_rows)} rows kept in original order, {status_updates} status updates, {alert_count} alerts"
        )
    else:
        baseline_rows = _same_day_rows(load_json_array(morning_baseline_path), resolved_schedule_date)
        if baseline_rows:
            dataset_df = load_live_dataset(dataset_path)
            resolved_through_date = str(dataset_df["game_date"].max().date())
            published_rows = publish_prepared_baseline_to_current(
                baseline_path=morning_baseline_path,
                current_picks_path=current_picks_path,
                board_state_path=board_state_path,
                schedule_date=resolved_schedule_date,
                dataset_df=dataset_df,
                resolved_through_date=resolved_through_date,
            )
        else:
            published_rows = publish_live_picks(
                dataset_path=dataset_path,
                bundle_path=bundle_path,
                metadata_path=metadata_path,
                output_path=current_picks_path,
                history_path=history_path,
                board_state_path=board_state_path,
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
    board_state_path: Path = LIVE_DAILY_BOARD_STATE_PATH,
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
        board_state_path=board_state_path,
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
        if resolved_mode == "prepare":
            result = run_prepare_refresh(**kwargs)
            current_picks_path = Path(kwargs.get("current_picks_path", LIVE_CURRENT_PICKS_PATH))
            board_state_path = Path(kwargs.get("board_state_path", LIVE_DAILY_BOARD_STATE_PATH))
            morning_baseline_path = Path(kwargs.get("morning_baseline_path", LIVE_MORNING_BASELINE_PICKS_PATH))
            history_path = Path(kwargs.get("history_path", LIVE_PICK_HISTORY_PATH))
            dashboard_output_dir = Path(kwargs.get("dashboard_output_dir", DEFAULT_OUTPUT_DIR))
            publish_date = normalize_game_date(kwargs.get("publish_date")) or default_publish_date()
            published_rows = publish_prepared_baseline_to_current(
                baseline_path=morning_baseline_path,
                current_picks_path=current_picks_path,
                board_state_path=board_state_path,
                schedule_date=publish_date,
            )
            if published_rows and kwargs.get("rebuild_dashboard", True):
                rebuild_and_verify_public_artifacts(
                    current_picks_path=current_picks_path,
                    history_path=history_path,
                    morning_baseline_path=morning_baseline_path,
                    dashboard_output_dir=dashboard_output_dir,
                    verify_public_artifacts=kwargs.get("verify_public_artifacts", True),
                )
            return {"mode": resolved_mode, "result": result, "published_rows": published_rows}
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
