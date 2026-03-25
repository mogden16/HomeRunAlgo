"""Settle previously published picks using the refreshed engineered dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import LIVE_CURRENT_PICKS_PATH, LIVE_MODEL_DATA_PATH, LIVE_PICK_HISTORY_PATH
from scripts.live_pipeline import load_json_array, load_live_dataset, settle_pick_records, write_current_picks, write_pick_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(LIVE_MODEL_DATA_PATH), help="Path to the refreshed engineered dataset.")
    parser.add_argument("--current-picks-path", default=str(LIVE_CURRENT_PICKS_PATH), help="Path to the latest published picks.")
    parser.add_argument("--history-path", default=str(LIVE_PICK_HISTORY_PATH), help="Path to the forward-only pick ledger.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_df = load_live_dataset(Path(args.dataset_path))
    resolved_through_date = str(dataset_df["game_date"].max().date())

    current_rows = load_json_array(Path(args.current_picks_path))
    history_rows = load_json_array(Path(args.history_path))

    settled_current = settle_pick_records(current_rows, dataset_df, resolved_through_date=resolved_through_date)
    settled_history = settle_pick_records(history_rows, dataset_df, resolved_through_date=resolved_through_date)

    write_current_picks(settled_current, Path(args.current_picks_path))
    write_pick_history(settled_history, Path(args.history_path))
    print(f"Settled picks through {resolved_through_date}")


if __name__ == "__main__":
    main()
