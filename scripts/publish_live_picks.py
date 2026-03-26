"""Generate and publish today's live picks from the saved model bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    default_publish_date,
    fetch_schedule_games,
    load_live_dataset,
    load_model_bundle,
    load_model_metadata,
    score_live_candidates,
    write_current_picks,
)


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


def main() -> None:
    args = parse_args()
    schedule_date = args.schedule_date or default_publish_date()
    output_path = Path(args.output_path)
    history_path = Path(args.history_path)
    dashboard_output_dir = Path(args.dashboard_output_dir)
    dataset_df = load_live_dataset(Path(args.dataset_path))
    bundle = load_model_bundle(Path(args.bundle_path))
    model_metadata = load_model_metadata(Path(args.metadata_path))
    try:
        assert_live_publish_freshness(
            schedule_date=schedule_date,
            dataset_df=dataset_df,
            model_metadata=model_metadata,
        )
    except RuntimeError:
        write_current_picks([], output_path)
        print(f"Cleared stale picks at {output_path} because publish failed for {schedule_date}")
        refresh_cloudflare_dashboard(
            output_path,
            history_path,
            dashboard_output_dir,
            schedule_date,
            persist_history=False,
        )
        raise
    schedule_games = fetch_schedule_games(schedule_date)
    active_roster_map = build_active_roster_map(schedule_games)

    candidates = build_live_candidate_frame(
        dataset_df,
        schedule_games,
        target_date=schedule_date,
        hitters_per_team=args.hitters_per_team,
        active_roster_map=active_roster_map,
    )
    featured = build_live_feature_frame(dataset_df, candidates)
    picks = score_live_candidates(featured, bundle, max_picks=args.max_picks)
    write_current_picks(picks, output_path)
    refresh_cloudflare_dashboard(output_path, history_path, dashboard_output_dir, schedule_date)
    print(f"Published {len(picks)} picks to {args.output_path} for {schedule_date}")


if __name__ == "__main__":
    main()
