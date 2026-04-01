from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.publish_live_picks import _merge_same_day_picks
from scripts import settle_live_results


class LiveSlateRepeatabilityTests(unittest.TestCase):
    def test_merge_same_day_picks_keeps_locked_games_and_replaces_unlocked_games(self) -> None:
        existing_rows = [
            {
                "pick_id": "locked-old",
                "game_pk": 1001,
                "game_date": "2026-03-31",
                "rank": 1,
                "batter_id": 10,
                "batter_name": "Locked Old",
                "team": "NYY",
                "opponent_team": "BOS",
                "pitcher_id": 20,
                "pitcher_name": "Pitcher A",
                "confidence_tier": "elite",
                "predicted_hr_probability": 0.25,
                "predicted_hr_score": 99.0,
                "top_reason_1": "reason",
                "result": "Pending",
            },
            {
                "pick_id": "unlocked-old",
                "game_pk": 1002,
                "game_date": "2026-03-31",
                "rank": 2,
                "batter_id": 11,
                "batter_name": "Unlocked Old",
                "team": "LAD",
                "opponent_team": "SD",
                "pitcher_id": 21,
                "pitcher_name": "Pitcher B",
                "confidence_tier": "strong",
                "predicted_hr_probability": 0.18,
                "predicted_hr_score": 88.0,
                "top_reason_1": "reason",
                "result": "Pending",
            },
        ]
        refreshed_rows = [
            {**existing_rows[0], "pick_id": "locked-new", "batter_name": "Locked New", "rank": 1},
            {**existing_rows[1], "pick_id": "unlocked-new", "batter_name": "Unlocked New", "rank": 1},
        ]
        merged = _merge_same_day_picks(
            existing_rows,
            refreshed_rows,
            [
                {"game_pk": 1001, "status": "Final", "game_datetime": "2026-03-31T19:05:00Z"},
                {"game_pk": 1002, "status": "Scheduled", "game_datetime": "2026-03-31T22:10:00Z"},
            ],
            schedule_date="2026-03-31",
            publish_reference=datetime(2026, 3, 31, 20, 0, tzinfo=timezone.utc),
            max_picks=2,
        )

        self.assertEqual({row["pick_id"] for row in merged}, {"locked-old", "unlocked-new"})
        unlocked_row = next(row for row in merged if row["pick_id"] == "unlocked-new")
        self.assertEqual(unlocked_row["rank"], 2)

    def test_settle_live_results_updates_live_statuses_and_archives_only_after_full_slate_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            current_path = base / "current.json"
            history_path = base / "history.json"

            pd.DataFrame(
                [
                    {"game_date": "2026-03-31", "batter_id": 10, "hit_hr": 1},
                    {"game_date": "2026-03-31", "batter_id": 11, "hit_hr": 0},
                ]
            ).to_csv(dataset_path, index=False)
            current_path.write_text(
                json.dumps(
                    [
                        self._current_pick("2026-03-31", 1001, 10, "Alpha"),
                        self._current_pick("2026-03-31", 1002, 11, "Bravo"),
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            history_path.write_text("[]", encoding="utf-8")

            with patch(
                "scripts.settle_live_results.fetch_schedule_games",
                return_value=[
                    {"game_pk": 1001, "game_date": "2026-03-31", "game_datetime": "2026-03-31T19:05:00Z", "status": "In Progress"},
                    {"game_pk": 1002, "game_date": "2026-03-31", "game_datetime": "2026-03-31T19:10:00Z", "status": "Final"},
                ],
            ):
                first_result = settle_live_results.run_settle_live_results(
                    dataset_path=dataset_path,
                    current_picks_path=current_path,
                    history_path=history_path,
                )

            current_rows = json.loads(current_path.read_text(encoding="utf-8"))
            self.assertEqual(first_result["archived_dates"], [])
            self.assertEqual([row["result"] for row in current_rows], ["HR", "No HR"])
            self.assertEqual([row["game_state"] for row in current_rows], ["live", "final"])
            self.assertEqual(json.loads(history_path.read_text(encoding="utf-8")), [])

            with patch(
                "scripts.settle_live_results.fetch_schedule_games",
                return_value=[
                    {"game_pk": 1001, "game_date": "2026-03-31", "game_datetime": "2026-03-31T19:05:00Z", "status": "Final"},
                    {"game_pk": 1002, "game_date": "2026-03-31", "game_datetime": "2026-03-31T19:10:00Z", "status": "Final"},
                ],
            ):
                second_result = settle_live_results.run_settle_live_results(
                    dataset_path=dataset_path,
                    current_picks_path=current_path,
                    history_path=history_path,
                )

            self.assertEqual(second_result["archived_dates"], ["2026-03-31"])
            self.assertEqual(json.loads(current_path.read_text(encoding="utf-8")), [])
            history_rows = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertEqual({row["pick_id"] for row in history_rows}, {"2026-03-31:1001:10:20", "2026-03-31:1002:11:20"})

    @staticmethod
    def _current_pick(game_date: str, game_pk: int, batter_id: int, batter_name: str) -> dict[str, object]:
        return {
            "game_date": game_date,
            "game_pk": game_pk,
            "game_datetime": "2026-03-31T19:05:00Z",
            "rank": 1,
            "batter_id": batter_id,
            "batter_name": batter_name,
            "team": "NYY",
            "opponent_team": "BOS",
            "pitcher_id": 20,
            "pitcher_name": "Pitcher",
            "confidence_tier": "elite",
            "predicted_hr_probability": 0.2,
            "predicted_hr_score": 90.0,
            "top_reason_1": "reason",
            "result": "Pending",
        }


if __name__ == "__main__":
    unittest.main()
