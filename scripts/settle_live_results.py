"""Settle previously published picks using the refreshed engineered dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import LIVE_CURRENT_PICKS_PATH, LIVE_DAILY_BOARD_STATE_PATH, LIVE_MODEL_DATA_PATH, LIVE_PICK_HISTORY_PATH
from scripts.board_state import (
    apply_major_alerts,
    board_entries_to_current_rows,
    board_is_complete,
    create_daily_board_snapshot,
    finalize_daily_board,
    load_daily_board_state,
    move_board_to_history,
    prepare_next_day_board,
    resolve_board_state_path,
    update_board_entry_status,
    write_daily_board_state,
)
from scripts.live_pipeline import (
    fetch_schedule_games,
    load_json_array,
    load_live_dataset,
    normalize_game_date,
    write_current_picks,
    write_pick_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(LIVE_MODEL_DATA_PATH), help="Path to the refreshed engineered dataset.")
    parser.add_argument("--current-picks-path", default=str(LIVE_CURRENT_PICKS_PATH), help="Path to the latest published picks.")
    parser.add_argument("--history-path", default=str(LIVE_PICK_HISTORY_PATH), help="Path to the forward-only pick ledger.")
    parser.add_argument("--board-state-path", default=str(LIVE_DAILY_BOARD_STATE_PATH), help=argparse.SUPPRESS)
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


def _rows_reference_only_terminal_games(
    rows: list[dict[str, object]],
    slate_state: dict[str, object],
) -> bool:
    games_by_pk = slate_state.get("games_by_pk") or {}
    if not isinstance(games_by_pk, dict):
        return True
    relevant_games = []
    for row in rows:
        game_pk = row.get("game_pk")
        try:
            normalized_game_pk = int(game_pk) if game_pk not in (None, "") else None
        except (TypeError, ValueError):
            normalized_game_pk = None
        if normalized_game_pk is None:
            continue
        game = games_by_pk.get(normalized_game_pk)
        if isinstance(game, dict):
            relevant_games.append(game)
    if not relevant_games:
        return True
    return all(bool(game.get("is_final")) for game in relevant_games)


def run_settle_live_results(
    *,
    dataset_path: Path = LIVE_MODEL_DATA_PATH,
    current_picks_path: Path = LIVE_CURRENT_PICKS_PATH,
    history_path: Path = LIVE_PICK_HISTORY_PATH,
    board_state_path: Path | None = LIVE_DAILY_BOARD_STATE_PATH,
) -> dict[str, object]:
    dataset_df = load_live_dataset(dataset_path)
    resolved_through_date = str(dataset_df["game_date"].max().date())
    resolved_board_state_path = resolve_board_state_path(
        board_state_path=board_state_path,
        current_picks_path=current_picks_path,
    )

    current_rows = load_json_array(current_picks_path)
    history_rows = load_json_array(history_path)
    current_rows, history_rows = _recover_pending_history_rows(current_rows, history_rows)
    current_dates = sorted({normalize_game_date(row.get("game_date")) for row in current_rows if normalize_game_date(row.get("game_date"))})

    settled_current: list[dict[str, object]] = []
    archived_dates: list[str] = []
    for current_date in current_dates:
        date_rows = [dict(row) for row in current_rows if normalize_game_date(row.get("game_date")) == current_date]
        schedule_games = fetch_schedule_games(current_date)
        board_state = load_daily_board_state(resolved_board_state_path)
        if normalize_game_date(board_state.get("board_date")) != current_date or not board_state.get("entries"):
            board_state = create_daily_board_snapshot(
                date_rows,
                current_date,
                created_at=str(date_rows[0].get("created_at") or date_rows[0].get("published_at") or ""),
            )
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
        settled_rows = board_entries_to_current_rows(board_state)
        if board_is_complete(
            board_state,
            schedule_games=schedule_games,
        ):
            finalized_board = finalize_daily_board(board_state)
            history_rows = move_board_to_history(finalized_board, history_rows)
            archived_dates.append(current_date)
            write_daily_board_state(prepare_next_day_board(current_date), resolved_board_state_path)
            print(
                f"Board finalized for {current_date}: "
                f"{len(finalized_board.get('entries') or [])} entries moved to history"
            )
            continue
        settled_current.extend(settled_rows)
        write_daily_board_state(board_state, resolved_board_state_path)
        print(
            f"Updated board for {current_date}: "
            f"{status_updates} status changes, {alert_count} alerts"
        )

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
        board_state_path=Path(args.board_state_path),
    )


if __name__ == "__main__":
    main()
