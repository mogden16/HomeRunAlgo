"""Refresh the historical dataset and train the live model bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import LIVE_MODEL_BUNDLE_PATH, LIVE_MODEL_DATA_PATH, LIVE_MODEL_METADATA_PATH, LIVE_MODEL_START_DATE
from scripts.live_pipeline import default_training_end_date, refresh_live_dataset, train_live_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(LIVE_MODEL_DATA_PATH), help="Where to write the refreshed engineered dataset.")
    parser.add_argument("--bundle-path", default=str(LIVE_MODEL_BUNDLE_PATH), help="Where to write the trained model bundle.")
    parser.add_argument("--metadata-path", default=str(LIVE_MODEL_METADATA_PATH), help="Where to write model metadata.")
    parser.add_argument("--start-date", default=LIVE_MODEL_START_DATE, help="Inclusive historical start date.")
    parser.add_argument("--end-date", default=None, help="Inclusive historical end date. Defaults to yesterday in ET.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore caches and re-fetch remote data.")
    parser.add_argument("--model", default="logistic", choices=["logistic", "histgb", "xgboost", "all"], help="Model family search space.")
    parser.add_argument("--feature-profile", default="live", choices=["stable", "live", "expanded", "all"], help="Feature profile search space.")
    parser.add_argument("--calibration", default="sigmoid", choices=["disabled", "sigmoid", "isotonic"], help="Calibration mode.")
    parser.add_argument("--selection-metric", default="pr_auc", choices=["pr_auc", "roc_auc", "neg_log_loss", "neg_brier"], help="Primary CV metric for candidate selection.")
    parser.add_argument("--missingness-threshold", type=float, default=None, help="Optional fixed feature-missingness threshold. Defaults to the search set used by train_model.py.")
    parser.add_argument("--training-mode", default="search", choices=["search", "fast_refit"], help="Use the full search flow or reuse the approved live configuration for a fast refit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    bundle_path = Path(args.bundle_path)
    metadata_path = Path(args.metadata_path)
    end_date = args.end_date or default_training_end_date()

    refresh_live_dataset(
        output_path=dataset_path,
        start_date=args.start_date,
        end_date=end_date,
        force_refresh=args.force_refresh,
    )
    bundle = train_live_model_bundle(
        dataset_path=dataset_path,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        model_name=args.model,
        calibration=args.calibration,
        feature_profile=args.feature_profile,
        selection_metric=args.selection_metric,
        missingness_threshold=args.missingness_threshold,
        training_mode=args.training_mode,
    )
    print(f"Live model trained through {bundle['trained_through']} and written to {bundle_path}")


if __name__ == "__main__":
    main()
