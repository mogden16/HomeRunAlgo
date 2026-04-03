"""Refresh results, retrain the model, settle prior picks, and save a private draft slate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
from scripts.live_pipeline import (
    default_publish_date,
    default_training_end_date,
    load_json_array,
    load_live_dataset,
    refresh_live_dataset,
    settle_pick_records,
    train_live_model_bundle,
    write_current_picks,
    write_pick_history,
)
from scripts.publish_live_picks import generate_live_picks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(LIVE_MODEL_DATA_PATH), help="Where to write the refreshed engineered dataset.")
    parser.add_argument("--bundle-path", default=str(LIVE_MODEL_BUNDLE_PATH), help="Where to write the trained model bundle.")
    parser.add_argument("--metadata-path", default=str(LIVE_MODEL_METADATA_PATH), help="Where to write model metadata.")
    parser.add_argument("--current-picks-path", default=str(LIVE_CURRENT_PICKS_PATH), help="Path to the latest published picks.")
    parser.add_argument("--history-path", default=str(LIVE_PICK_HISTORY_PATH), help="Path to the forward-only pick ledger.")
    parser.add_argument("--draft-output-path", default=str(LIVE_DRAFT_PICKS_PATH), help="Path to write the private draft slate.")
    parser.add_argument("--start-date", default=LIVE_MODEL_START_DATE, help="Inclusive historical start date.")
    parser.add_argument("--train-end-date", default=None, help="Inclusive training end date. Defaults to yesterday in ET.")
    parser.add_argument("--publish-date", default=None, help="Draft date. Defaults to today in ET.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore caches and re-fetch remote data.")
    parser.add_argument("--model", default="histgb", choices=["logistic", "histgb", "xgboost", "all"], help="Model family search space.")
    parser.add_argument(
        "--feature-profile",
        default="live_usable_candidate_v1",
        choices=["stable", "live", "live_plus", "live_shrunk", "live_shrunk_precise", "live_usable_candidate_v1", "expanded", "all"],
        help="Feature profile search space.",
    )
    parser.add_argument("--calibration", default="sigmoid", choices=["disabled", "sigmoid", "isotonic"], help="Calibration mode.")
    parser.add_argument("--selection-metric", default="pr_auc", choices=["pr_auc", "roc_auc", "neg_log_loss", "neg_brier"], help="Primary CV metric for candidate selection.")
    parser.add_argument("--missingness-threshold", type=float, default=None, help="Optional fixed feature-missingness threshold.")
    parser.add_argument("--training-mode", default="fast_refit", choices=["search", "fast_refit"], help="Use fast_refit for morning production or search for slower full candidate selection.")
    parser.add_argument("--hitters-per-team", type=int, default=9, help="How many likely starters to consider for each team.")
    parser.add_argument("--max-picks", type=int, default=20, help="Maximum draft picks across the slate.")
    return parser.parse_args()


def _count_resolved(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if str(row.get("result") or row.get("result_label") or "Pending") in {"HR", "No HR"}
    )


def run_prepare_live_board(
    *,
    dataset_path: Path = LIVE_MODEL_DATA_PATH,
    bundle_path: Path = LIVE_MODEL_BUNDLE_PATH,
    metadata_path: Path = LIVE_MODEL_METADATA_PATH,
    current_picks_path: Path = LIVE_CURRENT_PICKS_PATH,
    history_path: Path = LIVE_PICK_HISTORY_PATH,
    draft_output_path: Path = LIVE_DRAFT_PICKS_PATH,
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
    max_picks: int = 20,
) -> list[dict[str, Any]]:
    resolved_train_end_date = train_end_date or default_training_end_date()
    resolved_publish_date = publish_date or default_publish_date()

    print("\nPrepare live board")
    print("-" * 60)
    print(f"Resolved ET train_end_date : {resolved_train_end_date}")
    print(f"Resolved ET publish_date   : {resolved_publish_date}")

    print("\n[1/4] Refreshing dataset and retraining live bundle")
    print(f"Training mode              : {training_mode}")
    refresh_live_dataset(
        output_path=dataset_path,
        start_date=start_date,
        end_date=resolved_train_end_date,
        force_refresh=force_refresh,
    )
    dataset_df = load_live_dataset(dataset_path)
    dataset_max_game_date = str(dataset_df["game_date"].max().date())
    print(f"Refreshed dataset max date : {dataset_max_game_date}")

    bundle = train_live_model_bundle(
        dataset_path=dataset_path,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        model_name=model,
        calibration=calibration,
        feature_profile=feature_profile,
        selection_metric=selection_metric,
        missingness_threshold=missingness_threshold,
        training_mode=training_mode,
    )
    print(f"Bundle trained through     : {bundle['trained_through']}")

    print("\n[2/4] Settling previously published picks")
    current_rows = load_json_array(current_picks_path)
    history_rows = load_json_array(history_path)
    settled_current = settle_pick_records(current_rows, dataset_df, resolved_through_date=dataset_max_game_date)
    settled_history = settle_pick_records(history_rows, dataset_df, resolved_through_date=dataset_max_game_date)
    write_current_picks(settled_current, current_picks_path)
    write_pick_history(settled_history, history_path)
    print(
        "Settled current/history    : "
        f"{_count_resolved(settled_current)}/{len(settled_current)} current, "
        f"{_count_resolved(settled_history)}/{len(settled_history)} history "
        f"through {dataset_max_game_date}"
    )

    print("\n[3/4] Generating private draft slate")
    draft_picks = generate_live_picks(
        dataset_path=dataset_path,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        schedule_date=resolved_publish_date,
        hitters_per_team=hitters_per_team,
        max_picks=max_picks,
    )
    write_current_picks(draft_picks, draft_output_path)
    print(f"Saved {len(draft_picks)} draft picks to {draft_output_path}")

    print("\n[4/4] Prepare workflow complete")
    return draft_picks


def main() -> None:
    args = parse_args()
    run_prepare_live_board(
        dataset_path=Path(args.dataset_path),
        bundle_path=Path(args.bundle_path),
        metadata_path=Path(args.metadata_path),
        current_picks_path=Path(args.current_picks_path),
        history_path=Path(args.history_path),
        draft_output_path=Path(args.draft_output_path),
        start_date=args.start_date,
        train_end_date=args.train_end_date,
        publish_date=args.publish_date,
        force_refresh=args.force_refresh,
        model=args.model,
        feature_profile=args.feature_profile,
        calibration=args.calibration,
        selection_metric=args.selection_metric,
        missingness_threshold=args.missingness_threshold,
        training_mode=args.training_mode,
        hitters_per_team=args.hitters_per_team,
        max_picks=args.max_picks,
    )


if __name__ == "__main__":
    main()
