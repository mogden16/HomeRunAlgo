"""Stable daily board snapshot helpers for Today/History lifecycle management."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import LIVE_CURRENT_PICKS_PATH, LIVE_DAILY_BOARD_STATE_PATH
from scripts.live_pipeline import (
    build_pick_id,
    build_slate_state,
    fetch_forecast_weather,
    normalize_game_date,
    parse_game_datetime,
    settle_pick_records,
    weather_code_label,
)

BOARD_STATUS_PENDING = "pending"
BOARD_STATUS_HOME_RUN = "home_run"
BOARD_STATUS_NO_HOME_RUN = "no_home_run"
BOARD_STATUS_INACTIVE = "inactive"
BOARD_STATUS_POSTPONED = "postponed"
BOARD_STATUS_GAME_IN_PROGRESS = "game_in_progress"
BOARD_STATUS_FINAL = "final"

ALERT_FLAG_WEATHER = "weather_alert"
ALERT_FLAG_LINEUP = "lineup_alert"
ALERT_FLAG_PITCHER_CHANGE = "pitcher_change_alert"

DISPLAY_STYLE_DEFAULT = "default"
DISPLAY_STYLE_MUTED = "muted"
DISPLAY_STYLE_STRIKETHROUGH = "strikethrough"

TERMINAL_BOARD_STATUSES = {
    BOARD_STATUS_HOME_RUN,
    BOARD_STATUS_NO_HOME_RUN,
    BOARD_STATUS_INACTIVE,
    BOARD_STATUS_POSTPONED,
    BOARD_STATUS_FINAL,
}
SEVERE_WEATHER_CODES = {51, 53, 55, 61, 63, 65, 66, 67, 75, 80, 81, 82, 95, 96, 99}
WEATHER_FALLBACK_LABEL = "Weather unavailable"


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, pd.Timestamp):
            serialized[key] = value.isoformat()
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, float):
            if pd.isna(value):
                serialized[key] = None
            else:
                serialized[key] = round(float(value), 6)
        else:
            serialized[key] = value
    return serialized


def _sort_entry_key(entry: dict[str, Any]) -> tuple[int, float, str]:
    original_rank = entry.get("original_rank", entry.get("rank", 999))
    try:
        resolved_rank = int(original_rank)
    except (TypeError, ValueError):
        resolved_rank = 999
    score = entry.get("original_score", entry.get("predicted_hr_score"))
    try:
        resolved_score = float(score) if score not in (None, "") else float("-inf")
    except (TypeError, ValueError):
        resolved_score = float("-inf")
    return (resolved_rank, -resolved_score, str(entry.get("batter_name") or ""))


def load_daily_board_state(path: Path = LIVE_DAILY_BOARD_STATE_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_board_state_path(
    *,
    board_state_path: Path | None,
    current_picks_path: Path,
) -> Path:
    if board_state_path is None:
        return current_picks_path.parent / "daily_board_state.json"
    if Path(board_state_path) == LIVE_DAILY_BOARD_STATE_PATH and Path(current_picks_path) != LIVE_CURRENT_PICKS_PATH:
        return current_picks_path.parent / "daily_board_state.json"
    return Path(board_state_path)


def write_daily_board_state(board_state: dict[str, Any], path: Path = LIVE_DAILY_BOARD_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialize_row(board_state), indent=2), encoding="utf-8")


def _coerce_flag_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    flags: list[str] = []
    for item in value:
        token = str(item or "").strip()
        if token and token not in flags:
            flags.append(token)
    return flags


def _player_day_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_game_date(row.get("board_date") or row.get("game_date")),
        str(row.get("batter_id") or "").strip() or str(row.get("batter_name") or "").strip().lower(),
        str(row.get("team") or "").strip().upper(),
    )


def _normalize_reason_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _reason_list(row: dict[str, Any]) -> list[str]:
    return [str(row.get(column) or "").strip() for column in ("top_reason_1", "top_reason_2", "top_reason_3") if str(row.get(column) or "").strip()]


def _apply_reason_list(row: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    updated = dict(row)
    normalized_reasons = [reason.strip() for reason in reasons if reason.strip()]
    for index, column in enumerate(("top_reason_1", "top_reason_2", "top_reason_3")):
        updated[column] = normalized_reasons[index] if index < len(normalized_reasons) else ""
    return updated


def _merge_reason_lists(existing_reasons: list[str], new_reasons: list[str]) -> list[str]:
    merged = list(existing_reasons)
    seen = {_normalize_reason_text(reason) for reason in merged}
    for reason in new_reasons:
        normalized_reason = _normalize_reason_text(reason)
        if not normalized_reason or normalized_reason in seen:
            continue
        if len(merged) < 3:
            merged.append(reason.strip())
        else:
            merged[-1] = f"{merged[-1].rstrip()} Update: {reason.strip()}"
        seen.add(normalized_reason)
    return merged


def _merge_board_entries(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key in {"pick_id", "created_at", "published_at", "original_rank", "original_score", "original_tier"}:
            continue
        if value in (None, "") and key not in {"alert_flags", "inactive_flag"}:
            continue
        merged[key] = value

    merged["pick_id"] = str(existing.get("pick_id") or incoming.get("pick_id") or "")
    merged["created_at"] = str(existing.get("created_at") or incoming.get("created_at") or "")
    merged["published_at"] = str(existing.get("published_at") or incoming.get("published_at") or "")
    merged["original_rank"] = int(existing.get("original_rank") or existing.get("rank") or incoming.get("original_rank") or incoming.get("rank") or 999)
    merged["original_score"] = existing.get("original_score", existing.get("predicted_hr_score", incoming.get("original_score", incoming.get("predicted_hr_score"))))
    merged["original_tier"] = str(existing.get("original_tier") or existing.get("confidence_tier") or incoming.get("original_tier") or incoming.get("confidence_tier") or "watch")
    merged["alert_flags"] = _coerce_flag_list([*_coerce_flag_list(existing.get("alert_flags")), *_coerce_flag_list(incoming.get("alert_flags"))])
    merged["inactive_flag"] = bool(existing.get("inactive_flag") or incoming.get("inactive_flag") or incoming.get("is_inactive"))

    weather_label = str(merged.get("weather_label") or "").strip()
    if not weather_label or weather_label.lower() == "unknown":
        if merged.get("weather_code") is not None:
            merged["weather_label"] = weather_code_label(merged.get("weather_code"))
        else:
            merged["weather_label"] = WEATHER_FALLBACK_LABEL

    merged_reasons = _merge_reason_lists(_reason_list(existing), _reason_list(incoming))
    return _apply_reason_list(merged, merged_reasons)


def _dedupe_board_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_player_day: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _player_day_key(row)
        previous = by_player_day.get(key)
        by_player_day[key] = _merge_board_entries(previous, row) if previous is not None else dict(row)
    return sorted(by_player_day.values(), key=_sort_entry_key)


def _base_entry(row: dict[str, Any], created_at: str) -> dict[str, Any]:
    game_date = normalize_game_date(row.get("game_date"))
    batter_name = str(row.get("batter_name") or "Unknown hitter")
    pitcher_name = str(row.get("pitcher_name") or "")
    game_pk = row.get("game_pk")
    batter_id = row.get("batter_id")
    pitcher_id = row.get("pitcher_id")
    original_rank = row.get("original_rank", row.get("rank", 999))
    original_score = row.get("original_score", row.get("predicted_hr_score"))
    original_tier = row.get("original_tier", row.get("confidence_tier", "watch"))
    result_label = str(row.get("result_label") or row.get("result") or "Pending")
    actual_hr = row.get("actual_hit_hr")
    inactive_flag = bool(row.get("inactive_flag") or row.get("is_inactive"))
    display_style = str(row.get("display_style") or (DISPLAY_STYLE_STRIKETHROUGH if inactive_flag else DISPLAY_STYLE_DEFAULT))
    alert_flags = _coerce_flag_list(row.get("alert_flags"))
    game_state = str(row.get("game_state") or "pregame")
    current_status = str(row.get("current_status") or "")
    if not current_status:
        current_status = derive_current_status(
            result_label=result_label,
            game_state=game_state,
            game_status=str(row.get("game_status") or ""),
            inactive_flag=inactive_flag,
        )
    weather_label = str(row.get("weather_label") or "").strip()
    if weather_label.lower() in {"", "unknown", "n/a", "na"}:
        weather_label = weather_code_label(row.get("weather_code")) or WEATHER_FALLBACK_LABEL
    return {
        **dict(row),
        "pick_id": str(
            row.get("pick_id")
            or build_pick_id(
                game_date,
                int(game_pk) if game_pk not in (None, "") else None,
                int(batter_id) if batter_id not in (None, "") else None,
                batter_name,
                int(pitcher_id) if pitcher_id not in (None, "") else None,
                pitcher_name,
            )
        ),
        "published_at": str(row.get("published_at") or created_at),
        "board_date": normalize_game_date(row.get("board_date") or row.get("game_date")),
        "created_at": str(row.get("created_at") or created_at),
        "original_rank": int(original_rank) if original_rank not in (None, "") else 999,
        "original_score": original_score,
        "original_tier": str(original_tier or "watch"),
        "rank": int(original_rank) if original_rank not in (None, "") else 999,
        "predicted_hr_score": original_score if row.get("predicted_hr_score") in (None, "") else row.get("predicted_hr_score"),
        "confidence_tier": str(row.get("confidence_tier") or original_tier or "watch"),
        "current_status": current_status,
        "alert_flags": alert_flags,
        "actual_hit_hr": actual_hr,
        "inactive_flag": inactive_flag,
        "display_style": display_style,
        "result": result_label,
        "result_label": result_label,
        "weather_label": weather_label,
    }


def create_daily_board_snapshot(
    rows: list[dict[str, Any]],
    board_date: str,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    created_timestamp = created_at or datetime.now(timezone.utc).isoformat()
    entries = [_base_entry(dict(row), created_timestamp) for row in rows if normalize_game_date(row.get("game_date")) == board_date]
    ordered_entries = _dedupe_board_entries(entries)
    return {
        "board_date": board_date,
        "created_at": created_timestamp,
        "finalized_at": None,
        "is_finalized": False,
        "entries": ordered_entries,
    }


def board_entries_to_current_rows(board_state: dict[str, Any]) -> list[dict[str, Any]]:
    entries = board_state.get("entries") or []
    if not isinstance(entries, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in _dedupe_board_entries([dict(item) for item in entries if isinstance(item, dict)]):
        row = dict(entry)
        row["rank"] = int(row.get("original_rank") or row.get("rank") or 999)
        row["predicted_hr_score"] = row.get("original_score", row.get("predicted_hr_score"))
        row["confidence_tier"] = str(row.get("original_tier") or row.get("confidence_tier") or "watch")
        rows.append(row)
    return rows


def derive_current_status(
    *,
    result_label: str,
    game_state: str,
    game_status: str,
    inactive_flag: bool,
) -> str:
    status_token = str(game_status or "").strip().lower()
    if "postponed" in status_token or "cancelled" in status_token:
        return BOARD_STATUS_POSTPONED
    if inactive_flag:
        return BOARD_STATUS_INACTIVE
    if result_label == "HR":
        return BOARD_STATUS_HOME_RUN
    if result_label == "No HR":
        return BOARD_STATUS_NO_HOME_RUN
    if str(game_state or "").strip().lower() == "live":
        return BOARD_STATUS_GAME_IN_PROGRESS
    if str(game_state or "").strip().lower() == "final":
        return BOARD_STATUS_FINAL
    return BOARD_STATUS_PENDING


def _team_lineup_lookup(schedule_games: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for game in schedule_games:
        game_pk = game.get("game_pk")
        if game_pk in (None, ""):
            continue
        try:
            normalized_game_pk = int(game_pk)
        except (TypeError, ValueError):
            continue
        for side in ("home", "away"):
            team = str(game.get(f"{side}_team") or "")
            if not team:
                continue
            lineup_rows = game.get(f"{side}_projected_lineup") or []
            lookup[(normalized_game_pk, team)] = {
                "lineup_ids": {
                    int(player.get("batter_id"))
                    for player in lineup_rows
                    if isinstance(player, dict) and player.get("batter_id") not in (None, "")
                },
                "lineup_source": str(game.get(f"{side}_lineup_source") or "projected"),
                "opponent_pitcher_id": game.get("away_pitcher_id") if side == "home" else game.get("home_pitcher_id"),
                "opponent_pitcher_name": game.get("away_pitcher_name") if side == "home" else game.get("home_pitcher_name"),
            }
    return lookup


def _weather_alert_lookup(board_entries: list[dict[str, Any]], board_date: str) -> dict[tuple[int, str], list[str]]:
    if not board_entries:
        return {}
    home_teams = sorted({str(row.get("opponent_team") or "") for row in board_entries if str(row.get("team") or "") and str(row.get("opponent_team") or "")})
    if not home_teams:
        return {}
    try:
        weather_df = fetch_forecast_weather(home_teams, board_date)
    except Exception:
        return {}
    if weather_df.empty:
        return {}
    alerts_by_home_team: dict[str, list[str]] = {}
    for row in weather_df.to_dict(orient="records"):
        flags: list[str] = []
        weather_code = row.get("weather_code")
        wind_speed = row.get("wind_speed_mph")
        try:
            if weather_code is not None and int(weather_code) in SEVERE_WEATHER_CODES:
                flags.append(ALERT_FLAG_WEATHER)
        except (TypeError, ValueError):
            pass
        try:
            if wind_speed is not None and float(wind_speed) >= 20:
                flags.append(ALERT_FLAG_WEATHER)
        except (TypeError, ValueError):
            pass
        if flags:
            alerts_by_home_team[str(row.get("team") or "")] = flags
    alert_lookup: dict[tuple[int, str], list[str]] = {}
    for entry in board_entries:
        game_pk = entry.get("game_pk")
        opponent_team = str(entry.get("opponent_team") or "")
        if game_pk in (None, "") or not opponent_team:
            continue
        try:
            normalized_game_pk = int(game_pk)
        except (TypeError, ValueError):
            continue
        flags = alerts_by_home_team.get(opponent_team)
        if flags:
            alert_lookup[(normalized_game_pk, opponent_team)] = list(flags)
    return alert_lookup


def apply_major_alerts(
    board_state: dict[str, Any],
    *,
    schedule_games: list[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    entries = [dict(entry) for entry in (board_state.get("entries") or []) if isinstance(entry, dict)]
    if not entries:
        updated = dict(board_state)
        updated["entries"] = []
        return updated, 0

    board_date = str(board_state.get("board_date") or entries[0].get("game_date") or "")
    lineup_lookup = _team_lineup_lookup(schedule_games)
    weather_alert_lookup = _weather_alert_lookup(entries, board_date)
    alert_count = 0
    updated_entries: list[dict[str, Any]] = []

    for entry in entries:
        row = dict(entry)
        current_flags = _coerce_flag_list(row.get("alert_flags"))
        flags = [flag for flag in current_flags if flag not in {ALERT_FLAG_LINEUP, ALERT_FLAG_WEATHER, ALERT_FLAG_PITCHER_CHANGE}]
        game_pk = row.get("game_pk")
        team = str(row.get("team") or "")
        lineup_context = None
        try:
            lineup_context = lineup_lookup.get((int(game_pk), team)) if game_pk not in (None, "") else None
        except (TypeError, ValueError):
            lineup_context = None
        lineup_source = str(lineup_context.get("lineup_source") or row.get("lineup_source") or "projected") if lineup_context else str(row.get("lineup_source") or "projected")
        row["lineup_source"] = lineup_source
        game_status_token = str(row.get("game_status") or "").strip().lower()
        if "postponed" in game_status_token or "cancelled" in game_status_token:
            row["result"] = "Postponed"
            row["result_label"] = "Postponed"
            row["current_status"] = BOARD_STATUS_POSTPONED
            row["display_style"] = DISPLAY_STYLE_MUTED
        if lineup_context and lineup_source == "confirmed":
            batter_id = row.get("batter_id")
            lineup_ids = lineup_context.get("lineup_ids") or set()
            is_active = batter_id in lineup_ids if batter_id not in (None, "") else False
            if not is_active:
                flags.append(ALERT_FLAG_LINEUP)
                row["inactive_flag"] = True
                row["display_style"] = DISPLAY_STYLE_STRIKETHROUGH
                row["result"] = "Inactive"
                row["result_label"] = "Inactive"
                row["current_status"] = BOARD_STATUS_INACTIVE
        if lineup_context:
            updated_pitcher_id = lineup_context.get("opponent_pitcher_id")
            updated_pitcher_name = lineup_context.get("opponent_pitcher_name")
            if updated_pitcher_id not in (None, "") and row.get("pitcher_id") not in (None, "") and int(updated_pitcher_id) != int(row.get("pitcher_id")):
                flags.append(ALERT_FLAG_PITCHER_CHANGE)
            if updated_pitcher_name and updated_pitcher_name != row.get("pitcher_name"):
                flags.append(ALERT_FLAG_PITCHER_CHANGE)
        try:
            weather_flags = weather_alert_lookup.get((int(game_pk), str(row.get("opponent_team") or ""))) if game_pk not in (None, "") else None
        except (TypeError, ValueError):
            weather_flags = None
        if weather_flags:
            flags.extend(weather_flags)
        deduped_flags: list[str] = []
        for flag in flags:
            if flag and flag not in deduped_flags:
                deduped_flags.append(flag)
        alert_count += len([flag for flag in deduped_flags if flag not in current_flags])
        row["alert_flags"] = deduped_flags
        updated_entries.append(row)

    updated = dict(board_state)
    updated["entries"] = sorted(updated_entries, key=_sort_entry_key)
    return updated, alert_count


def update_board_entry_status(
    board_state: dict[str, Any],
    dataset_df: pd.DataFrame,
    *,
    resolved_through_date: str,
    schedule_games: list[dict[str, Any]],
    reference_time: datetime | None = None,
) -> tuple[dict[str, Any], int]:
    entries = [dict(entry) for entry in (board_state.get("entries") or []) if isinstance(entry, dict)]
    if not entries:
        updated = dict(board_state)
        updated["entries"] = []
        return updated, 0

    settled_entries = settle_pick_records(
        entries,
        dataset_df,
        resolved_through_date=resolved_through_date,
        schedule_games=schedule_games,
        reference_time=reference_time,
    )
    updated_entries: list[dict[str, Any]] = []
    updated_count = 0
    for original_entry, settled_entry in zip(entries, settled_entries):
        row = dict(original_entry)
        row.update(dict(settled_entry))
        row["created_at"] = str(original_entry.get("created_at") or board_state.get("created_at") or datetime.now(timezone.utc).isoformat())
        row["board_date"] = str(board_state.get("board_date") or row.get("game_date") or "")
        row["original_rank"] = int(original_entry.get("original_rank") or original_entry.get("rank") or row.get("rank") or 999)
        row["original_score"] = original_entry.get("original_score", original_entry.get("predicted_hr_score"))
        row["original_tier"] = str(original_entry.get("original_tier") or original_entry.get("confidence_tier") or row.get("confidence_tier") or "watch")
        row["rank"] = int(row["original_rank"])
        row["predicted_hr_score"] = row.get("original_score", row.get("predicted_hr_score"))
        row["confidence_tier"] = str(row.get("original_tier") or row.get("confidence_tier") or "watch")
        row["alert_flags"] = _coerce_flag_list(original_entry.get("alert_flags"))
        row["inactive_flag"] = bool(original_entry.get("inactive_flag") or row.get("inactive_flag"))
        row["display_style"] = str(original_entry.get("display_style") or row.get("display_style") or DISPLAY_STYLE_DEFAULT)
        result_label = str(row.get("result_label") or row.get("result") or "Pending")
        row["current_status"] = derive_current_status(
            result_label=result_label,
            game_state=str(row.get("game_state") or "pregame"),
            game_status=str(row.get("game_status") or ""),
            inactive_flag=bool(row.get("inactive_flag")),
        )
        if row.get("current_status") == BOARD_STATUS_INACTIVE and row.get("display_style") == DISPLAY_STYLE_DEFAULT:
            row["display_style"] = DISPLAY_STYLE_STRIKETHROUGH
        if (
            str(original_entry.get("result_label") or original_entry.get("result") or "Pending") != result_label
            or str(original_entry.get("current_status") or "") != str(row.get("current_status") or "")
            or str(original_entry.get("game_state") or "") != str(row.get("game_state") or "")
        ):
            updated_count += 1
        updated_entries.append(row)

    updated = dict(board_state)
    updated["entries"] = sorted(updated_entries, key=_sort_entry_key)
    return updated, updated_count


def board_is_complete(
    board_state: dict[str, Any],
    *,
    schedule_games: list[dict[str, Any]],
    reference_time: datetime | None = None,
) -> bool:
    entries = [dict(entry) for entry in (board_state.get("entries") or []) if isinstance(entry, dict)]
    if not entries:
        return False
    slate_state = build_slate_state(schedule_games, reference_time=reference_time)
    relevant_game_pks = {
        int(entry["game_pk"])
        for entry in entries
        if entry.get("game_pk") not in (None, "")
    }
    relevant_games = [
        dict(game)
        for game_pk, game in (slate_state.get("games_by_pk") or {}).items()
        if int(game_pk) in relevant_game_pks
    ]
    games_terminal = bool(relevant_games) and all(bool(game.get("is_final")) for game in relevant_games)
    entries_terminal = all(str(entry.get("current_status") or "") in TERMINAL_BOARD_STATUSES for entry in entries)
    return entries_terminal and games_terminal


def finalize_daily_board(
    board_state: dict[str, Any],
    *,
    finalized_at: str | None = None,
) -> dict[str, Any]:
    updated = dict(board_state)
    updated["is_finalized"] = True
    updated["finalized_at"] = finalized_at or datetime.now(timezone.utc).isoformat()
    finalized_entries: list[dict[str, Any]] = []
    for entry in updated.get("entries") or []:
        row = dict(entry)
        if str(row.get("current_status") or "") == BOARD_STATUS_GAME_IN_PROGRESS:
            row["current_status"] = BOARD_STATUS_FINAL
        finalized_entries.append(row)
    updated["entries"] = sorted(finalized_entries, key=_sort_entry_key)
    return updated


def move_board_to_history(
    board_state: dict[str, Any],
    history_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {
        str(row.get("pick_id") or ""): dict(row)
        for row in history_rows
        if str(row.get("pick_id") or "")
    }
    for entry in board_entries_to_current_rows(board_state):
        pick_id = str(entry.get("pick_id") or "")
        if pick_id:
            by_id[pick_id] = dict(entry)
    return list(by_id.values())


def prepare_next_day_board(board_date: str) -> dict[str, Any]:
    next_day = (datetime.fromisoformat(board_date) + timedelta(days=1)).date().isoformat()
    return {
        "board_date": next_day,
        "created_at": None,
        "finalized_at": None,
        "is_finalized": False,
        "entries": [],
    }
