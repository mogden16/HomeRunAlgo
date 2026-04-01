"""Settle previously published picks using the refreshed engineered dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import LIVE_CURRENT_PICKS_PATH, LIVE_MODEL_DATA_PATH, LIVE_PICK_HISTORY_PATH
from scripts.live_pipeline import (
    build_slate_state,
    fetch_schedule_games,
    load_json_array,
    load_live_dataset,
    normalize_game_date,
    settle_pick_records,
    write_current_picks,
    write_pick_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(LIVE_MODEL_DATA_PATH), help="Path to the refreshed engineered dataset.")
    parser.add_argument("--current-picks-path", default=str(LIVE_CURRENT_PICKS_PATH), help="Path to the latest published picks.")
    parser.add_argument("--history-path", default=str(LIVE_PICK_HISTORY_PATH), help="Path to the forward-only pick ledger.")
    return parser.parse_args()


def _upsert_history_rows(
    history_rows: list[dict[str, object]],
    archived_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_id = {
        str(row.get("pick_id") or ""): dict(row)
        for row in history_rows
        if str(row.get("pick_id") or "")
    }
    for row in archived_rows:
        key = str(row.get("pick_id") or "")
        if not key:
            continue
        by_id[key] = dict(row)
    return list(by_id.values())


def _recover_pending_history_rows(
    current_rows: list[dict[str, object]],
    history_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    recovered_rows = [
        dict(row)
        for row in history_rows
        if str(row.get("result_label") or row.get("result") or "Pending") == "Pending"
    ]
    if not recovered_rows:
        return current_rows, history_rows

    history_without_pending = [
        dict(row)
        for row in history_rows
        if str(row.get("result_label") or row.get("result") or "Pending") != "Pending"
    ]
    current_by_id = {
        str(row.get("pick_id") or ""): dict(row)
        for row in current_rows
        if str(row.get("pick_id") or "")
    }
    for row in recovered_rows:
        key = str(row.get("pick_id") or "")
        if not key:
            continue
        current_by_id[key] = dict(row)
    return list(current_by_id.values()), history_without_pending


def run_settle_live_results(
    *,
    dataset_path: Path = LIVE_MODEL_DATA_PATH,
    current_picks_path: Path = LIVE_CURRENT_PICKS_PATH,
    history_path: Path = LIVE_PICK_HISTORY_PATH,
) -> dict[str, object]:
    dataset_df = load_live_dataset(dataset_path)
    resolved_through_date = str(dataset_df["game_date"].max().date())

    current_rows = load_json_array(current_picks_path)
    history_rows = load_json_array(history_path)
    current_rows, history_rows = _recover_pending_history_rows(current_rows, history_rows)
    current_dates = sorted({normalize_game_date(row.get("game_date")) for row in current_rows if normalize_game_date(row.get("game_date"))})

    settled_current: list[dict[str, object]] = []
    archived_dates: list[str] = []
    for current_date in current_dates:
        date_rows = [dict(row) for row in current_rows if normalize_game_date(row.get("game_date")) == current_date]
        schedule_games = fetch_schedule_games(current_date)
        slate_state = build_slate_state(schedule_games)
        settled_rows = settle_pick_records(
            date_rows,
            dataset_df,
            resolved_through_date=resolved_through_date,
            schedule_games=schedule_games,
        )
        all_rows_terminal = all(
            str(row.get("result_label") or row.get("result") or "Pending") in {"HR", "No HR"}
            for row in settled_rows
        )
        if (slate_state["all_final"] and all_rows_terminal) or (not schedule_games and all_rows_terminal):
            history_rows = _upsert_history_rows(history_rows, settled_rows)
            archived_dates.append(current_date)
            continue
        settled_current.extend(settled_rows)

    write_current_picks(settled_current, current_picks_path)
    write_pick_history(history_rows, history_path)
    print(f"Settled picks through {resolved_through_date}")
    return {
        "resolved_through_date": resolved_through_date,
        "current_rows": settled_current,
        "history_rows": history_rows,
        "archived_dates": archived_dates,
    }


def main() -> None:
    args = parse_args()
    run_settle_live_results(
        dataset_path=Path(args.dataset_path),
        current_picks_path=Path(args.current_picks_path),
        history_path=Path(args.history_path),
    )


if __name__ == "__main__":
    main()
