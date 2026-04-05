from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from scripts import board_state


class BoardStateTests(unittest.TestCase):
    def test_create_daily_board_snapshot_preserves_original_rank_score_and_tier(self) -> None:
        snapshot = board_state.create_daily_board_snapshot(
            [
                {
                    "game_date": "2026-04-05",
                    "game_pk": 1001,
                    "rank": 1,
                    "batter_id": 10,
                    "batter_name": "Alpha",
                    "team": "NYY",
                    "opponent_team": "BOS",
                    "pitcher_id": 20,
                    "pitcher_name": "Pitcher A",
                    "confidence_tier": "elite",
                    "predicted_hr_score": 98.2,
                    "predicted_hr_probability": 0.221,
                    "result": "Pending",
                }
            ],
            "2026-04-05",
            created_at="2026-04-05T10:00:00+00:00",
        )

        self.assertEqual(snapshot["board_date"], "2026-04-05")
        self.assertFalse(snapshot["is_finalized"])
        self.assertEqual(snapshot["created_at"], "2026-04-05T10:00:00+00:00")
        self.assertEqual(snapshot["entries"][0]["original_rank"], 1)
        self.assertEqual(snapshot["entries"][0]["original_score"], 98.2)
        self.assertEqual(snapshot["entries"][0]["original_tier"], "elite")
        self.assertEqual(snapshot["entries"][0]["current_status"], board_state.BOARD_STATUS_PENDING)

    def test_apply_major_alerts_marks_confirmed_lineup_scratch_inactive(self) -> None:
        snapshot = board_state.create_daily_board_snapshot(
            [
                {
                    "game_date": "2026-04-05",
                    "game_pk": 1001,
                    "rank": 1,
                    "batter_id": 10,
                    "batter_name": "Alpha",
                    "team": "NYY",
                    "opponent_team": "BOS",
                    "pitcher_id": 20,
                    "pitcher_name": "Pitcher A",
                    "confidence_tier": "elite",
                    "predicted_hr_score": 98.2,
                    "result": "Pending",
                }
            ],
            "2026-04-05",
            created_at="2026-04-05T10:00:00+00:00",
        )

        with patch("scripts.board_state.fetch_forecast_weather", return_value=pd.DataFrame()):
            updated, alert_count = board_state.apply_major_alerts(
                snapshot,
                schedule_games=[
                    {
                        "game_pk": 1001,
                        "home_team": "BOS",
                        "away_team": "NYY",
                        "home_projected_lineup": [],
                        "away_projected_lineup": [{"batter_id": 99, "batter_name": "Someone Else"}],
                        "home_lineup_source": "projected",
                        "away_lineup_source": "confirmed",
                        "home_pitcher_id": 21,
                        "home_pitcher_name": "Pitcher B",
                    }
                ],
            )

        entry = updated["entries"][0]
        self.assertEqual(alert_count, 2)
        self.assertTrue(entry["inactive_flag"])
        self.assertEqual(entry["display_style"], board_state.DISPLAY_STYLE_STRIKETHROUGH)
        self.assertIn(board_state.ALERT_FLAG_LINEUP, entry["alert_flags"])
        self.assertIn(board_state.ALERT_FLAG_PITCHER_CHANGE, entry["alert_flags"])

    def test_update_board_entry_status_and_weather_alerts_keep_original_order(self) -> None:
        snapshot = board_state.create_daily_board_snapshot(
            [
                {
                    "game_date": "2026-04-05",
                    "game_pk": 1002,
                    "rank": 2,
                    "batter_id": 11,
                    "batter_name": "Bravo",
                    "team": "ATL",
                    "opponent_team": "AZ",
                    "pitcher_id": 21,
                    "pitcher_name": "Pitcher B",
                    "confidence_tier": "strong",
                    "predicted_hr_score": 91.0,
                    "result": "Pending",
                },
                {
                    "game_date": "2026-04-05",
                    "game_pk": 1001,
                    "rank": 1,
                    "batter_id": 10,
                    "batter_name": "Alpha",
                    "team": "NYY",
                    "opponent_team": "BOS",
                    "pitcher_id": 20,
                    "pitcher_name": "Pitcher A",
                    "confidence_tier": "elite",
                    "predicted_hr_score": 98.2,
                    "result": "Pending",
                },
            ],
            "2026-04-05",
            created_at="2026-04-05T10:00:00+00:00",
        )
        dataset_df = pd.DataFrame(
            [
                {"game_date": "2026-04-05", "batter_id": 10, "hit_hr": 1},
                {"game_date": "2026-04-05", "batter_id": 11, "hit_hr": 0},
            ]
        )
        schedule_games = [
            {"game_pk": 1001, "game_date": "2026-04-05", "game_datetime": "2026-04-05T17:05:00Z", "status": "Final", "home_team": "BOS", "away_team": "NYY"},
            {"game_pk": 1002, "game_date": "2026-04-05", "game_datetime": "2026-04-05T19:05:00Z", "status": "In Progress", "home_team": "AZ", "away_team": "ATL"},
        ]

        updated, status_count = board_state.update_board_entry_status(
            snapshot,
            dataset_df,
            resolved_through_date="2026-04-05",
            schedule_games=schedule_games,
            reference_time=datetime(2026, 4, 5, 20, 0, tzinfo=timezone.utc),
        )
        with patch(
            "scripts.board_state.fetch_forecast_weather",
            return_value=pd.DataFrame([{"team": "AZ", "weather_code": 95, "wind_speed_mph": 22.0}]),
        ):
            alerted, _ = board_state.apply_major_alerts(updated, schedule_games=schedule_games)

        self.assertEqual(status_count, 2)
        self.assertEqual([entry["batter_name"] for entry in alerted["entries"]], ["Alpha", "Bravo"])
        self.assertEqual(alerted["entries"][0]["current_status"], board_state.BOARD_STATUS_HOME_RUN)
        self.assertEqual(alerted["entries"][1]["current_status"], board_state.BOARD_STATUS_NO_HOME_RUN)
        self.assertIn(board_state.ALERT_FLAG_WEATHER, alerted["entries"][1]["alert_flags"])


if __name__ == "__main__":
    unittest.main()
