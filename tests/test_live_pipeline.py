from __future__ import annotations

import io
import json
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import build_dashboard_artifacts
from scripts import publish_live_picks
from scripts.live_pipeline import (
    assert_live_publish_freshness,
    build_live_feature_frame,
    build_pick_id,
    evaluate_live_publish_freshness,
    fetch_forecast_weather,
    load_model_metadata,
    score_live_candidates,
    select_probable_lineup_hitters,
    settle_pick_records,
)
from train_model import generate_reason_strings


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

    def test_select_probable_lineup_hitters_excludes_active_roster_hitter_without_recent_team_game(self) -> None:
        dataset = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": "2026-03-20", "batter_id": 10, "batter_name": "Alpha", "team": "NYY", "pa_count": 5, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 2, "game_date": "2026-03-21", "batter_id": 10, "batter_name": "Alpha", "team": "NYY", "pa_count": 4, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 3, "game_date": "2026-03-22", "batter_id": 10, "batter_name": "Alpha", "team": "NYY", "pa_count": 4, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 0, "game_date": "2026-03-10", "batter_id": 11, "batter_name": "Bravo", "team": "NYY", "pa_count": 4, "bat_side": "L", "hr_per_pa_last_10d": 0.1, "hr_per_pa_last_30d": 0.08},
            ]
        )
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        active_roster = pd.DataFrame(
            [
                {"batter_id": 10, "batter_name": "Alpha Current"},
                {"batter_id": 11, "batter_name": "Bravo Current"},
            ]
        )
        captured = io.StringIO()
        with redirect_stdout(captured):
            hitters = select_probable_lineup_hitters(
                dataset,
                team_code="NYY",
                target_date="2026-03-25",
                hitters_per_team=3,
                lookback_days=21,
                active_roster=active_roster,
            )
        self.assertEqual([row["batter_id"] for row in hitters], [10])
        self.assertIn("inactive_recent_games", captured.getvalue())
        self.assertIn("Underfilled team              : yes", captured.getvalue())

    def test_select_probable_lineup_hitters_does_not_backfill_when_underfilled(self) -> None:
        dataset = pd.DataFrame(
            [
                {"game_pk": 11, "game_date": "2026-03-20", "batter_id": 10, "batter_name": "Alpha", "team": "NYY", "pa_count": 5, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 12, "game_date": "2026-03-21", "batter_id": 11, "batter_name": "Bravo", "team": "NYY", "pa_count": 4, "bat_side": "L", "hr_per_pa_last_10d": 0.1, "hr_per_pa_last_30d": 0.08},
                {"game_pk": 13, "game_date": "2026-03-22", "batter_id": 12, "batter_name": "Charlie", "team": "NYY", "pa_count": 2, "bat_side": "R", "hr_per_pa_last_10d": 0.3, "hr_per_pa_last_30d": 0.12},
                {"game_pk": 5, "game_date": "2026-03-01", "batter_id": 13, "batter_name": "Delta", "team": "NYY", "pa_count": 5, "bat_side": "L", "hr_per_pa_last_10d": 0.4, "hr_per_pa_last_30d": 0.2},
            ]
        )
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        active_roster = pd.DataFrame(
            [
                {"batter_id": 10, "batter_name": "Alpha"},
                {"batter_id": 11, "batter_name": "Bravo"},
                {"batter_id": 13, "batter_name": "Delta"},
            ]
        )
        hitters = select_probable_lineup_hitters(
            dataset,
            team_code="NYY",
            target_date="2026-03-25",
            hitters_per_team=3,
            lookback_days=21,
            active_roster=active_roster,
        )
        self.assertEqual([row["batter_id"] for row in hitters], [10, 11])

    def test_select_probable_lineup_hitters_excludes_stale_active_roster_when_target_is_after_dataset_end(self) -> None:
        dataset = pd.DataFrame(
            [
                {"game_pk": 100, "game_date": "2024-09-28", "batter_id": 666464, "batter_name": "Jerar Encarnacion", "team": "SF", "pa_count": 4, "bat_side": "R", "hr_per_pa_last_10d": 0.05, "hr_per_pa_last_30d": 0.05},
                {"game_pk": 101, "game_date": "2024-09-29", "batter_id": 666464, "batter_name": "Jerar Encarnacion", "team": "SF", "pa_count": 3, "bat_side": "R", "hr_per_pa_last_10d": 0.05, "hr_per_pa_last_30d": 0.05},
            ]
        )
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        active_roster = pd.DataFrame([{"batter_id": 666464, "batter_name": "Jerar Encarnacion"}])
        captured = io.StringIO()
        with redirect_stdout(captured):
            hitters = select_probable_lineup_hitters(
                dataset,
                team_code="SF",
                target_date="2026-03-25",
                hitters_per_team=9,
                lookback_days=21,
                active_roster=active_roster,
            )
        self.assertEqual(hitters, [])
        self.assertIn("Recent team games considered  : 0", captured.getvalue())
        self.assertIn("Eligible after recent gate    : 0", captured.getvalue())

    def test_select_probable_lineup_hitters_keeps_bench_bat_if_he_appeared_within_last_three_team_games(self) -> None:
        dataset = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": "2026-03-20", "batter_id": 10, "batter_name": "Starter", "team": "NYY", "pa_count": 5, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 2, "game_date": "2026-03-21", "batter_id": 20, "batter_name": "Bench Bat", "team": "NYY", "pa_count": 1, "bat_side": "L", "hr_per_pa_last_10d": 0.08, "hr_per_pa_last_30d": 0.05},
                {"game_pk": 3, "game_date": "2026-03-22", "batter_id": 10, "batter_name": "Starter", "team": "NYY", "pa_count": 4, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 4, "game_date": "2026-03-23", "batter_id": 10, "batter_name": "Starter", "team": "NYY", "pa_count": 4, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
            ]
        )
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        active_roster = pd.DataFrame(
            [
                {"batter_id": 10, "batter_name": "Starter"},
                {"batter_id": 20, "batter_name": "Bench Bat"},
            ]
        )
        hitters = select_probable_lineup_hitters(
            dataset,
            team_code="NYY",
            target_date="2026-03-25",
            hitters_per_team=2,
            lookback_days=21,
            active_roster=active_roster,
        )
        self.assertEqual([row["batter_id"] for row in hitters], [10, 20])

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

    def test_fetch_forecast_weather_skips_api_for_historical_dates(self) -> None:
        with patch("scripts.live_pipeline.requests.get") as mock_get:
            weather = fetch_forecast_weather(["ATL"], "2024-09-29")
        self.assertFalse(mock_get.called)
        self.assertEqual(weather.to_dict(orient="records"), [{"game_date": "2024-09-29", "home_team": "ATL"}])

    def test_evaluate_live_publish_freshness_accepts_fresh_metadata_and_dataset(self) -> None:
        dataset_df = pd.DataFrame({"game_date": pd.to_datetime(["2026-03-24", "2026-03-25"])})
        diagnostics = evaluate_live_publish_freshness(
            schedule_date="2026-03-26",
            dataset_df=dataset_df,
            model_metadata={"trained_through": "2026-03-25"},
        )
        self.assertTrue(diagnostics["passed"])
        self.assertEqual(diagnostics["metadata_lag_days"], 1)
        self.assertEqual(diagnostics["dataset_lag_days"], 1)

    def test_assert_live_publish_freshness_rejects_stale_metadata(self) -> None:
        dataset_df = pd.DataFrame({"game_date": pd.to_datetime(["2026-03-24", "2026-03-25"])})
        with self.assertRaisesRegex(RuntimeError, "schedule_date=2026-03-26"):
            assert_live_publish_freshness(
                schedule_date="2026-03-26",
                dataset_df=dataset_df,
                model_metadata={"trained_through": "2026-03-10"},
            )

    def test_assert_live_publish_freshness_rejects_stale_dataset(self) -> None:
        dataset_df = pd.DataFrame({"game_date": pd.to_datetime(["2026-03-10", "2026-03-11"])})
        with self.assertRaisesRegex(RuntimeError, "dataset_max_game_date=2026-03-11"):
            assert_live_publish_freshness(
                schedule_date="2026-03-26",
                dataset_df=dataset_df,
                model_metadata={"trained_through": "2026-03-25"},
            )

    def test_load_model_metadata_reads_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "model_metadata.json"
            metadata_path.write_text(json.dumps({"trained_through": "2026-03-25"}), encoding="utf-8")
            metadata = load_model_metadata(metadata_path)
        self.assertEqual(metadata["trained_through"], "2026-03-25")

    def test_publish_live_picks_aborts_on_stale_training_data_before_lineup_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            output_path = base / "current.json"
            history_path = base / "pick_history.json"
            dashboard_dir = base / "dashboard"
            output_path.write_text(json.dumps([{"game_date": "2026-03-25", "batter_name": "Old Pick"}]), encoding="utf-8")
            history_path.write_text("[]", encoding="utf-8")
            dashboard_dir.mkdir(parents=True, exist_ok=True)
            (dashboard_dir / "dashboard.json").write_text(
                json.dumps({"tracking_start_date": "2026-03-25", "latest_picks": [{"game_date": "2026-03-25", "batter_name": "Old Pick"}]}),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "game_date": pd.to_datetime(["2024-09-29", "2024-09-30"]),
                    "game_pk": [1, 2],
                    "batter_id": [10, 11],
                    "hit_hr": [0, 1],
                }
            ).to_csv(dataset_path, index=False)
            with bundle_path.open("wb") as handle:
                pickle.dump({"trained_through": "2024-09-30", "model": object(), "feature_columns": [], "reference_df": pd.DataFrame()}, handle)
            metadata_path.write_text(json.dumps({"trained_through": "2024-09-30", "dataset_max_game_date": "2024-09-30"}), encoding="utf-8")

            argv = [
                "publish_live_picks.py",
                "--dataset-path",
                str(dataset_path),
                "--bundle-path",
                str(bundle_path),
                "--metadata-path",
                str(metadata_path),
                "--output-path",
                str(output_path),
                "--history-path",
                str(history_path),
                "--dashboard-output-dir",
                str(dashboard_dir),
                "--schedule-date",
                "2026-03-26",
            ]
            with patch.object(sys, "argv", argv):
                with patch("scripts.publish_live_picks.fetch_schedule_games") as mock_schedule:
                    with patch("scripts.publish_live_picks.build_live_candidate_frame") as mock_candidates:
                        with self.assertRaisesRegex(RuntimeError, "trained_through=2024-09-30"):
                            publish_live_picks.main()
            self.assertFalse(mock_schedule.called)
            self.assertFalse(mock_candidates.called)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), [])
            dashboard_payload = json.loads((dashboard_dir / "dashboard.json").read_text(encoding="utf-8"))
            self.assertEqual(dashboard_payload["latest_available_date"], "2026-03-26")
            self.assertEqual(dashboard_payload["latest_picks"], [])

    def test_publish_live_picks_fresh_path_writes_requested_schedule_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            output_path = base / "current.json"
            history_path = base / "pick_history.json"
            dashboard_dir = base / "dashboard"
            pd.DataFrame({"game_date": pd.to_datetime(["2026-03-24", "2026-03-25"])}).to_csv(dataset_path, index=False)
            with bundle_path.open("wb") as handle:
                pickle.dump({"trained_through": "2026-03-25", "model": object(), "feature_columns": [], "reference_df": pd.DataFrame()}, handle)
            metadata_path.write_text(json.dumps({"trained_through": "2026-03-25", "dataset_max_game_date": "2026-03-25"}), encoding="utf-8")
            history_path.write_text("[]", encoding="utf-8")

            picks = [{"game_date": "2026-03-26", "rank": 1, "batter_name": "Alpha"}]
            argv = [
                "publish_live_picks.py",
                "--dataset-path",
                str(dataset_path),
                "--bundle-path",
                str(bundle_path),
                "--metadata-path",
                str(metadata_path),
                "--output-path",
                str(output_path),
                "--history-path",
                str(history_path),
                "--dashboard-output-dir",
                str(dashboard_dir),
                "--schedule-date",
                "2026-03-26",
            ]
            with patch.object(sys, "argv", argv):
                with patch("scripts.publish_live_picks.fetch_schedule_games", return_value=[]):
                    with patch("scripts.publish_live_picks.build_active_roster_map", return_value={}):
                        with patch("scripts.publish_live_picks.build_live_candidate_frame", return_value=pd.DataFrame()):
                            with patch("scripts.publish_live_picks.build_live_feature_frame", return_value=pd.DataFrame()):
                                with patch("scripts.publish_live_picks.score_live_candidates", return_value=picks):
                                    publish_live_picks.main()
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written[0]["game_date"], "2026-03-26")
            dashboard_payload = json.loads((dashboard_dir / "dashboard.json").read_text(encoding="utf-8"))
            self.assertEqual(dashboard_payload["latest_available_date"], "2026-03-26")
            self.assertEqual(dashboard_payload["latest_picks"][0]["game_date"], "2026-03-26")

    def test_failed_publish_does_not_mutate_pick_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            output_path = base / "current.json"
            history_path = base / "pick_history.json"
            dashboard_dir = base / "dashboard"
            history_payload = [{"game_date": "2026-03-25", "batter_name": "Old Pick"}]
            history_path.write_text(json.dumps(history_payload), encoding="utf-8")
            output_path.write_text(json.dumps(history_payload), encoding="utf-8")
            pd.DataFrame(
                {
                    "game_date": pd.to_datetime(["2024-09-29", "2024-09-30"]),
                    "game_pk": [1, 2],
                    "batter_id": [10, 11],
                    "hit_hr": [0, 1],
                }
            ).to_csv(dataset_path, index=False)
            with bundle_path.open("wb") as handle:
                pickle.dump({"trained_through": "2024-09-30", "model": object(), "feature_columns": [], "reference_df": pd.DataFrame()}, handle)
            metadata_path.write_text(json.dumps({"trained_through": "2024-09-30", "dataset_max_game_date": "2024-09-30"}), encoding="utf-8")

            argv = [
                "publish_live_picks.py",
                "--dataset-path",
                str(dataset_path),
                "--bundle-path",
                str(bundle_path),
                "--metadata-path",
                str(metadata_path),
                "--output-path",
                str(output_path),
                "--history-path",
                str(history_path),
                "--dashboard-output-dir",
                str(dashboard_dir),
                "--schedule-date",
                "2026-03-26",
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(RuntimeError):
                    publish_live_picks.main()
            self.assertEqual(json.loads(history_path.read_text(encoding="utf-8")), history_payload)

    def test_build_live_feature_frame_backfills_latest_batter_and_pitcher_features(self) -> None:
        dataset_df = pd.DataFrame(
            [
                {
                    "game_pk": 9,
                    "game_date": "2024-09-20",
                    "batter_id": 101,
                    "batter_name": "Alpha",
                    "team": "NYY",
                    "opponent": "TOR",
                    "is_home": 0,
                    "bat_side": "R",
                    "pitcher_hand": "R",
                    "pitch_hand_primary": "R",
                    "pitcher_id": 202,
                    "opp_pitcher_name": "Max Fried",
                    "pa_count": 4,
                    "hr_count": 0,
                    "hit_hr": 0,
                    "bbe_count": 3,
                    "barrel_count": 1,
                    "hard_hit_bbe_count": 1,
                    "avg_exit_velocity": 91.0,
                    "max_exit_velocity": 105.0,
                    "ev_95plus_bbe_count": 1,
                    "fly_ball_bbe_count": 1,
                    "pull_air_bbe_count": 0,
                    "ballpark": "Rogers Centre",
                    "hard_hit_rate_last_10d": 0.41,
                    "max_exit_velocity_last_10d": 108.0,
                    "pitcher_hard_hit_allowed_rate_last_30d": 0.37,
                    "pitcher_hr_allowed_per_pa_last_30d": 0.05,
                },
                {
                    "game_pk": 10,
                    "game_date": "2024-09-30",
                    "batter_id": 101,
                    "batter_name": "Alpha",
                    "team": "NYY",
                    "opponent": "BOS",
                    "is_home": 1,
                    "bat_side": "R",
                    "pitcher_hand": "L",
                    "pitch_hand_primary": "L",
                    "pitcher_id": 202,
                    "opp_pitcher_name": "Max Fried",
                    "pa_count": 4,
                    "hr_count": 1,
                    "hit_hr": 1,
                    "bbe_count": 3,
                    "barrel_count": 1,
                    "hard_hit_bbe_count": 2,
                    "avg_exit_velocity": 95.0,
                    "max_exit_velocity": 108.0,
                    "ev_95plus_bbe_count": 2,
                    "fly_ball_bbe_count": 1,
                    "pull_air_bbe_count": 1,
                    "ballpark": "Yankee Stadium",
                    "hard_hit_rate_last_10d": 0.58,
                    "max_exit_velocity_last_10d": 112.0,
                    "pitcher_hard_hit_allowed_rate_last_30d": 0.44,
                    "pitcher_hr_allowed_per_pa_last_30d": 0.07,
                }
            ]
        )
        dataset_df["game_date"] = pd.to_datetime(dataset_df["game_date"])
        candidate_df = pd.DataFrame(
            [
                {
                    "game_pk": 999,
                    "game_date": "2026-03-25",
                    "batter_id": 101,
                    "batter_name": "Alpha",
                    "player_id": 101,
                    "player_name": "Alpha",
                    "team": "NYY",
                    "opponent": "BOS",
                    "is_home": 1,
                    "bat_side": "R",
                    "pitcher_hand": "L",
                    "pitch_hand_primary": "L",
                    "pitcher_id": 202,
                    "opp_pitcher_id": 202,
                    "pitcher_name": "Max Fried",
                    "opp_pitcher_name": "Max Fried",
                    "pa_count": 0,
                    "hr_count": 0,
                    "hit_hr": 0,
                    "bbe_count": 0,
                    "barrel_count": 0,
                    "hard_hit_bbe_count": 0,
                    "avg_exit_velocity": None,
                    "max_exit_velocity": None,
                    "ev_95plus_bbe_count": 0,
                    "fly_ball_bbe_count": 0,
                    "pull_air_bbe_count": 0,
                    "ballpark": "Yankee Stadium",
                    "temperature_f": 61.0,
                    "wind_speed_mph": 14.0,
                    "humidity_pct": 55.0,
                    "platoon_advantage": 1.0,
                }
            ]
        )
        featured = build_live_feature_frame(dataset_df, candidate_df)
        self.assertEqual(float(featured.iloc[0]["hard_hit_rate_last_10d"]), 0.58)
        self.assertEqual(float(featured.iloc[0]["max_exit_velocity_last_10d"]), 112.0)
        self.assertEqual(float(featured.iloc[0]["pitcher_hard_hit_allowed_rate_last_30d"]), 0.44)
        self.assertEqual(float(featured.iloc[0]["pitcher_hr_allowed_per_pa_last_30d"]), 0.07)

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

    def test_generate_reason_strings_returns_feature_specific_batter_reason(self) -> None:
        row = pd.Series(
            {
                "batter_name": "Alpha",
                "pitcher_name": "Pitcher",
                "hard_hit_rate_last_10d": 0.58,
                "platoon_advantage": 0.0,
            }
        )
        reference_df = pd.DataFrame({"hard_hit_rate_last_10d": [0.20, 0.25, 0.30, 0.35, 0.40, 0.58]})
        reasons = generate_reason_strings(
            row,
            reference_df=reference_df,
            positive_coef_map={"hard_hit_rate_last_10d": 0.8},
            max_reasons=3,
        )
        self.assertTrue(reasons)
        self.assertIn("Hard-hit rate over the last 10 days is 58.0%", reasons[0])

    def test_generate_reason_strings_returns_pitcher_risk_reason_with_name(self) -> None:
        row = pd.Series(
            {
                "batter_name": "Alpha",
                "pitcher_name": "Max Fried",
                "pitcher_hard_hit_allowed_rate_last_30d": 0.47,
            }
        )
        reference_df = pd.DataFrame({"pitcher_hard_hit_allowed_rate_last_30d": [0.21, 0.25, 0.29, 0.33, 0.38, 0.47]})
        reasons = generate_reason_strings(
            row,
            reference_df=reference_df,
            positive_coef_map={"pitcher_hard_hit_allowed_rate_last_30d": 0.9},
            max_reasons=3,
        )
        self.assertTrue(reasons)
        self.assertIn("Max Fried", reasons[0])
        self.assertIn("47.0%", reasons[0])

    def test_generate_reason_strings_returns_weather_reason(self) -> None:
        row = pd.Series(
            {
                "batter_name": "Alpha",
                "pitcher_name": "Pitcher",
                "wind_speed_mph": 14.0,
            }
        )
        reference_df = pd.DataFrame({"wind_speed_mph": [4.0, 5.0, 6.5, 7.0, 8.0, 14.0]})
        reasons = generate_reason_strings(
            row,
            reference_df=reference_df,
            positive_coef_map={"wind_speed_mph": 0.4},
            max_reasons=3,
        )
        self.assertTrue(reasons)
        self.assertIn("14.0 mph", reasons[0])

    def test_generate_reason_strings_prefers_batter_and_pitcher_before_weather(self) -> None:
        row = pd.Series(
            {
                "batter_name": "Alpha",
                "pitcher_name": "Max Fried",
                "hard_hit_rate_last_30d": 0.54,
                "pitcher_hard_hit_allowed_rate_last_30d": 0.46,
                "wind_speed_mph": 14.0,
            }
        )
        reference_df = pd.DataFrame(
            {
                "hard_hit_rate_last_30d": [0.28, 0.32, 0.35, 0.38, 0.42, 0.54],
                "pitcher_hard_hit_allowed_rate_last_30d": [0.24, 0.27, 0.30, 0.33, 0.36, 0.46],
                "wind_speed_mph": [4.0, 5.0, 6.5, 7.0, 8.0, 14.0],
            }
        )
        reasons = generate_reason_strings(
            row,
            reference_df=reference_df,
            positive_coef_map={
                "hard_hit_rate_last_30d": 0.7,
                "pitcher_hard_hit_allowed_rate_last_30d": 0.5,
                "wind_speed_mph": 0.4,
            },
            max_reasons=3,
        )
        self.assertGreaterEqual(len(reasons), 2)
        self.assertIn("Hard-hit rate over the last 30 days is 54.0%", reasons[0])
        self.assertIn("Max Fried", reasons[1])

    def test_generate_reason_strings_uses_generic_fallback_only_when_needed(self) -> None:
        row = pd.Series({"batter_name": "Alpha", "pitcher_name": "Pitcher", "hard_hit_rate_last_10d": 0.10})
        reference_df = pd.DataFrame({"hard_hit_rate_last_10d": [0.20, 0.25, 0.30, 0.35, 0.40]})
        reasons = generate_reason_strings(
            row,
            reference_df=reference_df,
            positive_coef_map={"hard_hit_rate_last_10d": 0.8},
            max_reasons=3,
        )
        self.assertEqual(
            reasons,
            ["The live model favors the overall recent-form and matchup profile, but no single feature cleared the explanation threshold."],
        )


if __name__ == "__main__":
    unittest.main()
