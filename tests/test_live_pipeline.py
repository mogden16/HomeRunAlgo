from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import build_dashboard_artifacts
from scripts.live_pipeline import build_pick_id, score_live_candidates, select_probable_lineup_hitters, settle_pick_records


class LivePipelineTests(unittest.TestCase):
    def test_build_pick_id_uses_game_pk(self) -> None:
        first = build_pick_id("2026-03-25", 1001, 42, "Sample Hitter", 77, "Sample Pitcher")
        second = build_pick_id("2026-03-25", 1002, 42, "Sample Hitter", 77, "Sample Pitcher")
        self.assertNotEqual(first, second)

    def test_select_probable_lineup_hitters_is_deterministic(self) -> None:
        dataset = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": "2026-03-20", "batter_id": 10, "batter_name": "Alpha", "team": "NYY", "pa_count": 5, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 2, "game_date": "2026-03-21", "batter_id": 10, "batter_name": "Alpha", "team": "NYY", "pa_count": 4, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 3, "game_date": "2026-03-21", "batter_id": 11, "batter_name": "Bravo", "team": "NYY", "pa_count": 4, "bat_side": "L", "hr_per_pa_last_10d": 0.1, "hr_per_pa_last_30d": 0.08},
                {"game_pk": 4, "game_date": "2026-03-22", "batter_id": 12, "batter_name": "Charlie", "team": "NYY", "pa_count": 2, "bat_side": "R", "hr_per_pa_last_10d": 0.3, "hr_per_pa_last_30d": 0.12},
            ]
        )
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        hitters = select_probable_lineup_hitters(dataset, team_code="NYY", target_date="2026-03-25", hitters_per_team=2, lookback_days=10)
        self.assertEqual([row["batter_id"] for row in hitters], [10, 11])

    def test_select_probable_lineup_hitters_filters_to_active_roster(self) -> None:
        dataset = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": "2026-03-20", "batter_id": 10, "batter_name": "Alpha Old", "team": "NYY", "pa_count": 5, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 2, "game_date": "2026-03-21", "batter_id": 11, "batter_name": "Bravo Old", "team": "NYY", "pa_count": 4, "bat_side": "L", "hr_per_pa_last_10d": 0.1, "hr_per_pa_last_30d": 0.08},
                {"game_pk": 3, "game_date": "2026-03-22", "batter_id": 12, "batter_name": "Charlie Old", "team": "NYY", "pa_count": 4, "bat_side": "R", "hr_per_pa_last_10d": 0.3, "hr_per_pa_last_30d": 0.12},
            ]
        )
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        active_roster = pd.DataFrame(
            [
                {"batter_id": 10, "batter_name": "Alpha Current"},
                {"batter_id": 12, "batter_name": "Charlie Current"},
            ]
        )
        hitters = select_probable_lineup_hitters(
            dataset,
            team_code="NYY",
            target_date="2026-03-25",
            hitters_per_team=3,
            lookback_days=10,
            active_roster=active_roster,
        )
        self.assertEqual([row["batter_id"] for row in hitters], [10, 12])
        self.assertEqual([row["batter_name"] for row in hitters], ["Alpha Current", "Charlie Current"])

    def test_settle_pick_records_marks_hits_and_non_hits(self) -> None:
        picks = [
            {"game_date": "2026-03-25", "batter_id": 101, "result": "Pending"},
            {"game_date": "2026-03-25", "batter_id": 102, "result": "Pending"},
        ]
        dataset = pd.DataFrame(
            [
                {"game_date": pd.Timestamp("2026-03-25"), "batter_id": 101, "hit_hr": 1},
            ]
        )
        settled = settle_pick_records(picks, dataset, resolved_through_date="2026-03-25")
        self.assertEqual(settled[0]["result"], "HR")
        self.assertEqual(settled[0]["actual_hit_hr"], 1)
        self.assertEqual(settled[1]["result"], "No HR")
        self.assertEqual(settled[1]["actual_hit_hr"], 0)

    def test_dashboard_filters_out_pre_tracking_rows(self) -> None:
        old_row = build_dashboard_artifacts.normalize_pick(
            {
                "game_pk": 1,
                "game_date": "2026-03-24",
                "batter_id": 1,
                "batter_name": "Old Pick",
                "pitcher_id": 2,
                "pitcher_name": "Pitcher",
                "team": "NYY",
                "opponent_team": "BOS",
                "result": "pending",
            },
            tracking_start_date="2026-03-25",
        )
        self.assertIsNone(old_row)

    def test_score_live_candidates_ranks_by_highest_score(self) -> None:
        class FakeModel:
            def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
                probabilities = []
                for value in features["feature_a"].tolist():
                    probabilities.append([1.0 - float(value), float(value)])
                return pd.DataFrame(probabilities).to_numpy()

        candidate_df = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": pd.Timestamp("2026-03-25"), "batter_id": 10, "batter_name": "Alpha", "team": "NYY", "opponent_team": "BOS", "pitcher_id": 20, "pitcher_name": "Pitcher", "feature_a": 0.2},
                {"game_pk": 1, "game_date": pd.Timestamp("2026-03-25"), "batter_id": 11, "batter_name": "Bravo", "team": "NYY", "opponent_team": "BOS", "pitcher_id": 20, "pitcher_name": "Pitcher", "feature_a": 0.9},
                {"game_pk": 1, "game_date": pd.Timestamp("2026-03-25"), "batter_id": 12, "batter_name": "Charlie", "team": "NYY", "opponent_team": "BOS", "pitcher_id": 20, "pitcher_name": "Pitcher", "feature_a": 0.5},
            ]
        )
        bundle = {
            "model": FakeModel(),
            "feature_columns": ["feature_a"],
            "reference_df": pd.DataFrame({"feature_a": [0.2, 0.5, 0.9]}),
        }
        picks = score_live_candidates(candidate_df, bundle, max_picks=3, published_at="2026-03-25T12:00:00+00:00")
        self.assertEqual([row["batter_name"] for row in picks], ["Bravo", "Charlie", "Alpha"])
        self.assertEqual([row["rank"] for row in picks], [1, 2, 3])

    def test_dashboard_builder_merges_current_into_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "out"
            current_path.write_text(
                """
[
  {
    "game_pk": 1001,
    "game_date": "2026-03-25",
    "rank": 1,
    "batter_id": 10,
    "batter_name": "Alpha",
    "team": "NYY",
    "opponent_team": "BOS",
    "pitcher_id": 20,
    "pitcher_name": "Pitcher",
    "confidence_tier": "elite",
    "predicted_hr_probability": 0.22,
    "predicted_hr_score": 99.1,
    "top_reason_1": "strong recent HR form",
    "result": "Pending"
  }
]
                """.strip(),
                encoding="utf-8",
            )
            history_path.write_text("[]", encoding="utf-8")

            argv = [
                "build_dashboard_artifacts.py",
                "--current-picks-path",
                str(current_path),
                "--history-path",
                str(history_path),
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                build_dashboard_artifacts.main()

            payload = (output_dir / "dashboard.json").read_text(encoding="utf-8")
            self.assertIn("Alpha", payload)
            history_payload = history_path.read_text(encoding="utf-8")
            self.assertIn("2026-03-25", history_payload)

    def test_dashboard_builder_replaces_pending_same_day_rows(self) -> None:
        existing_rows = [
            {
                "pick_id": "old-pick",
                "game_date": "2026-03-25",
                "rank": 1,
                "batter_name": "Old",
                "team": "NYY",
                "result_label": "Pending",
                "predicted_hr_score": 99.0,
            }
        ]
        current_rows = [
            {
                "pick_id": "new-pick",
                "game_date": "2026-03-25",
                "rank": 1,
                "batter_name": "New",
                "team": "NYY",
                "result_label": "Pending",
                "predicted_hr_score": 88.0,
            }
        ]
        merged = build_dashboard_artifacts.upsert_history(existing_rows, current_rows)
        self.assertEqual([row["pick_id"] for row in merged], ["new-pick"])

    def test_dashboard_score_sort_value_orders_rows(self) -> None:
        rows = [
            {"pick_id": "a", "predicted_hr_score": 80.0, "rank": 3, "batter_name": "Alpha"},
            {"pick_id": "b", "predicted_hr_score": 95.0, "rank": 1, "batter_name": "Bravo"},
            {"pick_id": "c", "predicted_hr_score": 85.0, "rank": 2, "batter_name": "Charlie"},
        ]
        ordered = sorted(rows, key=lambda row: (-build_dashboard_artifacts.score_sort_value(row), row["rank"], row["batter_name"]))
        self.assertEqual([row["pick_id"] for row in ordered], ["b", "c", "a"])


if __name__ == "__main__":
    unittest.main()
