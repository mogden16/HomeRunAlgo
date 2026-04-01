"""Generate and publish today's live picks from the saved model bundle."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    LIVE_CURRENT_PICKS_PATH,
    LIVE_MODEL_BUNDLE_PATH,
    LIVE_MODEL_DATA_PATH,
    LIVE_MODEL_METADATA_PATH,
    LIVE_PICK_HISTORY_PATH,
    LIVE_TRACKING_START_DATE,
)
from scripts.build_dashboard_artifacts import DEFAULT_OUTPUT_DIR, build_dashboard_artifacts
from scripts.live_pipeline import (
    assert_live_publish_freshness,
    build_active_roster_map,
    build_live_candidate_frame,
    build_live_feature_frame,
    build_slate_state,
    CONFIDENCE_TIER_ORDER,
    default_publish_date,
    fetch_schedule_games,
    load_json_array,
    load_live_dataset,
    load_model_bundle,
    load_model_metadata,
    normalize_game_date,
    park_game_meta,
    score_live_candidates,
    weather_code_label,
    write_current_picks,
)

DEFAULT_MIN_CONFIDENCE_TIER = "strong"
DEFAULT_MAX_PICKS_PER_TEAM = None
DEFAULT_MAX_PICKS_PER_GAME = None

REQUIRED_PUBLISH_ARTIFACT_LABELS = {
    "dataset": "refreshed live training dataset",
    "bundle": "trained live model bundle",
    "metadata": "trained live model metadata",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(LIVE_MODEL_DATA_PATH), help="Path to the refreshed engineered dataset.")
    parser.add_argument("--bundle-path", default=str(LIVE_MODEL_BUNDLE_PATH), help="Path to the trained model bundle.")
    parser.add_argument("--metadata-path", default=str(LIVE_MODEL_METADATA_PATH), help="Path to the trained model metadata.")
    parser.add_argument("--output-path", default=str(LIVE_CURRENT_PICKS_PATH), help="Path to write the current published picks.")
    parser.add_argument("--history-path", default=str(LIVE_PICK_HISTORY_PATH), help=argparse.SUPPRESS)
    parser.add_argument("--dashboard-output-dir", default=str(DEFAULT_OUTPUT_DIR), help=argparse.SUPPRESS)
    parser.add_argument("--schedule-date", default=None, help="Official MLB date to publish. Defaults to today in ET.")
    parser.add_argument("--hitters-per-team", type=int, default=9, help="How many likely starters to consider for each team.")
    parser.add_argument("--max-picks", type=int, default=20, help="Maximum published picks across the slate.")
    parser.add_argument(
        "--min-confidence-tier",
        choices=tuple(CONFIDENCE_TIER_ORDER.keys()),
        default=DEFAULT_MIN_CONFIDENCE_TIER,
        help="Minimum confidence tier required for publication.",
    )
    parser.add_argument(
        "--max-picks-per-team",
        type=int,
        default=DEFAULT_MAX_PICKS_PER_TEAM,
        help="Maximum published picks allowed from the same team. Disabled by default.",
    )
    parser.add_argument(
        "--max-picks-per-game",
        type=int,
        default=DEFAULT_MAX_PICKS_PER_GAME,
        help="Maximum published picks allowed from the same game. Disabled by default.",
    )
    return parser.parse_args()


def refresh_cloudflare_dashboard(
    current_picks_path: Path,
    history_path: Path,
    dashboard_output_dir: Path,
    schedule_date: str,
    *,
    persist_history: bool = True,
) -> Path:
    dashboard_path = build_dashboard_artifacts(
        current_picks_path=current_picks_path,
        history_path=history_path,
        output_dir=dashboard_output_dir,
        tracking_start_date=LIVE_TRACKING_START_DATE,
        persist_history=persist_history,
        latest_available_date_override=schedule_date,
    )
    print(f"Refreshed Cloudflare dashboard artifact at {dashboard_path}")
    return dashboard_path


def persist_operational_alerts(
    metadata_path: Path,
    model_metadata: dict[str, Any],
    alerts: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    updated_metadata = dict(model_metadata)
    updated_metadata["operational_alerts"] = [dict(alert) for alert in (alerts or []) if isinstance(alert, dict)]
    metadata_path.write_text(json.dumps(updated_metadata, indent=2), encoding="utf-8")
    return updated_metadata


def _publish_reference_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_game_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _game_is_locked(game_like: dict[str, Any], publish_reference: datetime) -> bool:
    slate_state = build_slate_state([game_like], reference_time=publish_reference)
    if not slate_state["games"]:
        return False
    return bool(slate_state["games"][0]["is_locked"])


def _fill_missing_game_meta(
    row: dict[str, Any],
    *,
    schedule_game: dict[str, Any] | None,
    refreshed_game_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(row)
    home_team = str(schedule_game.get("home_team") or "") if schedule_game else ""
    park_meta = park_game_meta(home_team)
    candidates = [refreshed_game_meta or {}, park_meta]

    def _first_present(*values: Any) -> Any:
        for value in values:
            if value not in (None, ""):
                return value
        return None

    updated["ballpark_name"] = _first_present(
        updated.get("ballpark_name"),
        *(candidate.get("ballpark_name") for candidate in candidates),
    ) or ""
    updated["ballpark_region_abbr"] = _first_present(
        updated.get("ballpark_region_abbr"),
        *(candidate.get("ballpark_region_abbr") for candidate in candidates),
    ) or ""
    updated["field_bearing_deg"] = _first_present(
        updated.get("field_bearing_deg"),
        *(candidate.get("field_bearing_deg") for candidate in candidates),
    )
    updated["weather_code"] = _first_present(
        updated.get("weather_code"),
        *(candidate.get("weather_code") for candidate in candidates),
    )
    updated["weather_label"] = _first_present(
        updated.get("weather_label"),
        *(candidate.get("weather_label") for candidate in candidates),
    ) or weather_code_label(updated.get("weather_code"))
    updated["wind_speed_mph"] = _first_present(
        updated.get("wind_speed_mph"),
        *(candidate.get("wind_speed_mph") for candidate in candidates),
    )
    updated["wind_direction_deg"] = _first_present(
        updated.get("wind_direction_deg"),
        *(candidate.get("wind_direction_deg") for candidate in candidates),
    )
    return updated


def _merge_same_day_picks(
    existing_rows: list[dict[str, Any]],
    refreshed_rows: list[dict[str, Any]],
    schedule_games: list[dict[str, Any]],
    *,
    schedule_date: str,
    publish_reference: datetime,
    max_picks: int,
) -> list[dict[str, Any]]:
    other_dates = [dict(row) for row in existing_rows if normalize_game_date(row.get("game_date")) != schedule_date]
    same_day_rows = [dict(row) for row in existing_rows if normalize_game_date(row.get("game_date")) == schedule_date]
    if not same_day_rows:
        return [*other_dates, *refreshed_rows]

    slate_state = build_slate_state(schedule_games, reference_time=publish_reference)
    schedule_by_game_pk = slate_state["games_by_pk"]
    refreshed_game_meta_by_pk = {
        int(row["game_pk"]): dict(row)
        for row in refreshed_rows
        if row.get("game_pk") not in (None, "")
    }
    locked_rows: list[dict[str, Any]] = []
    locked_game_pks: set[int] = set()
    for row in same_day_rows:
        game_pk = row.get("game_pk")
        schedule_game = schedule_by_game_pk.get(int(game_pk)) if game_pk not in (None, "") else None
        if _game_is_locked(schedule_game or row, publish_reference):
            refreshed_game_meta = refreshed_game_meta_by_pk.get(int(game_pk)) if game_pk not in (None, "") else None
            locked_rows.append(
                _fill_missing_game_meta(
                    row,
                    schedule_game=schedule_game,
                    refreshed_game_meta=refreshed_game_meta,
                )
            )
            if game_pk not in (None, ""):
                locked_game_pks.add(int(game_pk))

    unlocked_refreshed = [
        dict(row)
        for row in refreshed_rows
        if row.get("game_pk") in (None, "") or int(row["game_pk"]) not in locked_game_pks
    ]
    if not locked_rows:
        return [*other_dates, *unlocked_refreshed]

    slot_ceiling = max(
        max_picks,
        max(
            (
                int(row.get("rank"))
                for row in locked_rows
                if row.get("rank") not in (None, "") and str(row.get("rank")).isdigit()
            ),
            default=0,
        ),
    )
    locked_rank_values = {
        int(row["rank"])
        for row in locked_rows
        if row.get("rank") not in (None, "") and str(row.get("rank")).isdigit() and 1 <= int(row["rank"]) <= slot_ceiling
    }
    available_ranks = [rank for rank in range(1, slot_ceiling + 1) if rank not in locked_rank_values]

    refreshed_reassigned: list[dict[str, Any]] = []
    for refreshed_row, rank in zip(unlocked_refreshed, available_ranks):
        updated = dict(refreshed_row)
        updated["rank"] = rank
        refreshed_reassigned.append(updated)

    merged_same_day = [*locked_rows, *refreshed_reassigned]
    return [*other_dates, *merged_same_day]


def publish_live_picks(
    *,
    dataset_path: Path = LIVE_MODEL_DATA_PATH,
    bundle_path: Path = LIVE_MODEL_BUNDLE_PATH,
    metadata_path: Path = LIVE_MODEL_METADATA_PATH,
    output_path: Path = LIVE_CURRENT_PICKS_PATH,
    history_path: Path = LIVE_PICK_HISTORY_PATH,
    dashboard_output_dir: Path = DEFAULT_OUTPUT_DIR,
    schedule_date: str | None = None,
    hitters_per_team: int = 9,
    max_picks: int = 20,
    min_confidence_tier: str | None = DEFAULT_MIN_CONFIDENCE_TIER,
    max_picks_per_team: int | None = DEFAULT_MAX_PICKS_PER_TEAM,
    max_picks_per_game: int | None = DEFAULT_MAX_PICKS_PER_GAME,
) -> list[dict[str, Any]]:
    resolved_schedule_date = schedule_date or default_publish_date()
    publish_reference = _publish_reference_now()
    try:
        picks = generate_live_picks(
            dataset_path=dataset_path,
            bundle_path=bundle_path,
            metadata_path=metadata_path,
            schedule_date=resolved_schedule_date,
            hitters_per_team=hitters_per_team,
            max_picks=max_picks,
            min_confidence_tier=min_confidence_tier,
            max_picks_per_team=max_picks_per_team,
            max_picks_per_game=max_picks_per_game,
            published_at=publish_reference.isoformat(),
        )
    except RuntimeError:
        write_current_picks([], output_path)
        print(f"Cleared stale picks at {output_path} because publish failed for {resolved_schedule_date}")
        refresh_cloudflare_dashboard(
            output_path,
            history_path,
            dashboard_output_dir,
            resolved_schedule_date,
            persist_history=False,
        )
        raise
    schedule_games = fetch_schedule_games(resolved_schedule_date)
    existing_rows = load_json_array(output_path)
    merged_picks = _merge_same_day_picks(
        existing_rows,
        picks,
        schedule_games,
        schedule_date=resolved_schedule_date,
        publish_reference=publish_reference,
        max_picks=max_picks,
    )
    write_current_picks(merged_picks, output_path)
    published_rows = load_json_array(output_path)
    refresh_cloudflare_dashboard(output_path, history_path, dashboard_output_dir, resolved_schedule_date)
    print(f"Published {len(published_rows)} picks to {output_path} for {resolved_schedule_date}")
    return published_rows


def generate_live_picks(
    *,
    dataset_path: Path = LIVE_MODEL_DATA_PATH,
    bundle_path: Path = LIVE_MODEL_BUNDLE_PATH,
    metadata_path: Path = LIVE_MODEL_METADATA_PATH,
    schedule_date: str | None = None,
    hitters_per_team: int = 9,
    max_picks: int = 20,
    min_confidence_tier: str | None = DEFAULT_MIN_CONFIDENCE_TIER,
    max_picks_per_team: int | None = DEFAULT_MAX_PICKS_PER_TEAM,
    max_picks_per_game: int | None = DEFAULT_MAX_PICKS_PER_GAME,
    published_at: str | None = None,
) -> list[dict[str, Any]]:
    resolved_schedule_date = schedule_date or default_publish_date()
    publish_reference = _parse_game_datetime(published_at) if published_at else _publish_reference_now()
    if publish_reference is None:
        publish_reference = _publish_reference_now()
    required_paths = {
        "dataset": Path(dataset_path),
        "bundle": Path(bundle_path),
        "metadata": Path(metadata_path),
    }
    missing_artifacts = [
        f"{REQUIRED_PUBLISH_ARTIFACT_LABELS[key]} ({path})"
        for key, path in required_paths.items()
        if not path.exists()
    ]
    if missing_artifacts:
        missing_summary = "; ".join(missing_artifacts)
        raise RuntimeError(
            "Live publish cannot run because required prepare artifacts are missing: "
            f"{missing_summary}. Run the prepare refresh successfully first so it can generate the "
            "live model bundle and metadata, and refresh the live dataset in this workspace before publishing."
        )
    dataset_df = load_live_dataset(Path(dataset_path))
    bundle = load_model_bundle(Path(bundle_path))
    model_metadata = load_model_metadata(Path(metadata_path))
    assert_live_publish_freshness(
        schedule_date=resolved_schedule_date,
        dataset_df=dataset_df,
        model_metadata=model_metadata,
    )
    schedule_games = fetch_schedule_games(resolved_schedule_date)
    active_roster_map = build_active_roster_map(schedule_games)
    candidates = build_live_candidate_frame(
        dataset_df,
        schedule_games,
        target_date=resolved_schedule_date,
        hitters_per_team=hitters_per_team,
        active_roster_map=active_roster_map,
    )
    operational_alerts = [dict(alert) for alert in (candidates.attrs.get("operational_alerts") or []) if isinstance(alert, dict)]
    if operational_alerts:
        for alert in operational_alerts:
            print(f"Operational alert [{alert.get('code', 'unknown')}]: {alert.get('message', '')}")
    model_metadata = persist_operational_alerts(Path(metadata_path), model_metadata, operational_alerts)
    featured = build_live_feature_frame(dataset_df, candidates)
    return score_live_candidates(
        featured,
        bundle,
        max_picks=max_picks,
        min_confidence_tier=min_confidence_tier,
        max_picks_per_team=max_picks_per_team,
        max_picks_per_game=max_picks_per_game,
        published_at=publish_reference.isoformat(),
    )


def main() -> None:
    args = parse_args()
    publish_live_picks(
        dataset_path=Path(args.dataset_path),
        bundle_path=Path(args.bundle_path),
        metadata_path=Path(args.metadata_path),
        output_path=Path(args.output_path),
        history_path=Path(args.history_path),
        dashboard_output_dir=Path(args.dashboard_output_dir),
        schedule_date=args.schedule_date,
        hitters_per_team=args.hitters_per_team,
        max_picks=args.max_picks,
        min_confidence_tier=args.min_confidence_tier,
        max_picks_per_team=args.max_picks_per_team,
        max_picks_per_game=args.max_picks_per_game,
    )


if __name__ == "__main__":
    main()
