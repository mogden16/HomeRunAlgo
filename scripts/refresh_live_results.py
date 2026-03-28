"""Refresh the historical live dataset through yesterday without retraining the model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import LIVE_MODEL_DATA_PATH, LIVE_MODEL_START_DATE
from scripts.live_pipeline import default_training_end_date, refresh_live_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(LIVE_MODEL_DATA_PATH), help="Where to write the refreshed engineered dataset.")
    parser.add_argument("--start-date", default=LIVE_MODEL_START_DATE, help="Inclusive historical start date.")
    parser.add_argument("--end-date", default=None, help="Inclusive historical end date. Defaults to yesterday in ET.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore caches and re-fetch remote data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end_date or default_training_end_date()
    refresh_live_dataset(
        output_path=Path(args.dataset_path),
        start_date=args.start_date,
        end_date=end_date,
        force_refresh=args.force_refresh,
    )
    print(f"Live result dataset refreshed through {end_date} at {args.dataset_path}")


if __name__ == "__main__":
    main()
