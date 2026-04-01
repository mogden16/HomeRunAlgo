"""CLI wrapper around the shared Prepare, Publish, Settle, and auto refresh mode functions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    LIVE_CURRENT_PICKS_PATH,
    LIVE_DRAFT_PICKS_PATH,
    LIVE_MODEL_BUNDLE_PATH,
    LIVE_MODEL_DATA_PATH,
    LIVE_MODEL_METADATA_PATH,
    LIVE_MODEL_START_DATE,
    LIVE_PICK_HISTORY_PATH,
)
from scripts.build_dashboard_artifacts import DEFAULT_OUTPUT_DIR
from scripts.refresh_modes import REFRESH_MODES, run_refresh_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=REFRESH_MODES, help="Refresh mode to run.")
    parser.add_argument("--dataset-path", default=str(LIVE_MODEL_DATA_PATH), help="Where to write or read the refreshed engineered dataset.")
    parser.add_argument("--bundle-path", default=str(LIVE_MODEL_BUNDLE_PATH), help="Path to the trained model bundle.")
    parser.add_argument("--metadata-path", default=str(LIVE_MODEL_METADATA_PATH), help="Path to the trained model metadata.")
    parser.add_argument("--current-picks-path", default=str(LIVE_CURRENT_PICKS_PATH), help="Path to the latest published picks.")
    parser.add_argument("--history-path", default=str(LIVE_PICK_HISTORY_PATH), help="Path to the forward-only pick ledger.")
    parser.add_argument("--draft-output-path", default=str(LIVE_DRAFT_PICKS_PATH), help="Path to the private draft slate.")
    parser.add_argument("--dashboard-output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where dashboard JSON will be written.")
    parser.add_argument("--start-date", default=LIVE_MODEL_START_DATE, help="Inclusive historical start date for settle and prepare.")
    parser.add_argument("--end-date", default=None, help="Inclusive historical end date for settle. Defaults to yesterday in ET.")
    parser.add_argument("--train-end-date", default=None, help="Inclusive training end date for prepare. Defaults to yesterday in ET.")
    parser.add_argument("--publish-date", default=None, help="Publish date for prepare.")
    parser.add_argument("--schedule-date", default=None, help="Schedule date for publish.")
    parser.add_argument("--refresh-results-before-publish", action="store_true", help="Refresh the historical results dataset before publish.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore caches and re-fetch remote data.")
    parser.add_argument("--model", default="logistic", choices=["logistic", "histgb", "xgboost", "all"], help="Model family search space for prepare.")
    parser.add_argument("--feature-profile", default="live_shrunk", choices=["stable", "live", "live_plus", "live_shrunk", "live_shrunk_precise", "expanded", "all"], help="Feature profile search space for prepare.")
    parser.add_argument("--calibration", default="sigmoid", choices=["disabled", "sigmoid", "isotonic"], help="Calibration mode for prepare.")
    parser.add_argument("--selection-metric", default="pr_auc", choices=["pr_auc", "roc_auc", "neg_log_loss", "neg_brier"], help="Primary CV metric for prepare.")
    parser.add_argument("--missingness-threshold", type=float, default=None, help="Optional fixed feature-missingness threshold for prepare.")
    parser.add_argument("--training-mode", default="fast_refit", choices=["search", "fast_refit"], help="Training mode for prepare.")
    parser.add_argument("--hitters-per-team", type=int, default=9, help="How many likely starters to consider for each team.")
    parser.add_argument("--max-picks", type=int, default=20, help="Maximum picks across the slate.")
    parser.add_argument("--min-confidence-tier", choices=("watch", "strong", "elite"), default="strong", help="Minimum confidence tier required for publish.")
    parser.add_argument("--max-picks-per-team", type=int, default=None, help="Maximum published picks allowed from the same team.")
    parser.add_argument("--max-picks-per-game", type=int, default=None, help="Maximum published picks allowed from the same game.")
    parser.add_argument("--skip-dashboard-build", action="store_true", help="Skip rebuilding dashboard artifacts after the mode-specific work.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip repo-side public/live artifact verification after the dashboard build.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_refresh_mode(
        args.mode,
        dataset_path=Path(args.dataset_path),
        bundle_path=Path(args.bundle_path),
        metadata_path=Path(args.metadata_path),
        current_picks_path=Path(args.current_picks_path),
        history_path=Path(args.history_path),
        draft_output_path=Path(args.draft_output_path),
        dashboard_output_dir=Path(args.dashboard_output_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        train_end_date=args.train_end_date,
        publish_date=args.publish_date,
        schedule_date=args.schedule_date,
        refresh_results_before_publish=args.refresh_results_before_publish,
        force_refresh=args.force_refresh,
        model=args.model,
        feature_profile=args.feature_profile,
        calibration=args.calibration,
        selection_metric=args.selection_metric,
        missingness_threshold=args.missingness_threshold,
        training_mode=args.training_mode,
        hitters_per_team=args.hitters_per_team,
        max_picks=args.max_picks,
        min_confidence_tier=args.min_confidence_tier,
        max_picks_per_team=args.max_picks_per_team,
        max_picks_per_game=args.max_picks_per_game,
        rebuild_dashboard=not args.skip_dashboard_build,
        verify_public_artifacts=not args.skip_verify,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
