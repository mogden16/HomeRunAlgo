from __future__ import annotations

import io
import json
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import data_sources
import feature_engineering
from scripts import build_dashboard_artifacts
from scripts import live_pipeline
from scripts import publish_live_picks
from scripts import run_daily_live_refresh
from scripts.live_pipeline import (
    assert_live_publish_freshness,
    build_live_candidate_frame,
    build_live_feature_frame,
    build_pick_id,
    evaluate_live_publish_freshness,
    fetch_forecast_weather,
    load_model_metadata,
    score_live_candidates,
    select_probable_lineup_hitters,
    settle_pick_records,
)
from train_model import LIVE_PLUS_FEATURE_COLUMNS, generate_reason_strings


class LivePipelineTests(unittest.TestCase):
    def test_fetch_statcast_range_returns_empty_frame_for_offseason_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = Path(tmp_dir) / "statcast_empty.csv"
            with patch("data_sources._raw_chunk_path", return_value=chunk_path):
                with patch("data_sources.ensure_directories", side_effect=lambda: None):
                    with patch("data_sources.statcast", return_value=pd.DataFrame()):
                        frame = data_sources.fetch_statcast_range("2024-10-31", "2024-11-06", force_refresh=True)
            self.assertEqual(list(frame.columns), data_sources.STATCAST_COLUMNS)
            self.assertTrue(frame.empty)

    def test_build_weather_table_rebuilds_sparse_cache_and_normalizes_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            stale_cache = raw_dir / "weather_2024-03-28_2024-03-28.csv"
            pd.DataFrame(
                [
                    {
                        "game_date": "2024-03-28",
                        "home_team": "AZ",
                        "temperature_f": None,
                        "humidity_pct": None,
                        "wind_speed_mph": None,
                        "wind_direction_deg": None,
                        "pressure_hpa": None,
                    }
                ]
            ).to_csv(stale_cache, index=False)

            schedule = pd.DataFrame([{"game_date": "2024-03-28", "home_team": "ARI"}])

            def fake_fetch_open_meteo(lat: float, lon: float, start_date: str, end_date: str, tz: str) -> pd.DataFrame:
                hours = pd.date_range("2024-03-28 00:00", periods=24, freq="h", tz=tz)
                return pd.DataFrame(
                    {
                        "temperature_2m": [68.0] * len(hours),
                        "relative_humidity_2m": [55.0] * len(hours),
                        "wind_speed_10m": [9.0] * len(hours),
                        "wind_direction_10m": [180.0] * len(hours),
                        "surface_pressure": [1015.0] * len(hours),
                    },
                    index=hours,
                )

            with patch.object(data_sources, "DATA_DIR", tmp_path):
                with patch.object(data_sources, "RAW_DATA_DIR", raw_dir):
                    with patch.object(data_sources, "_fetch_open_meteo", side_effect=fake_fetch_open_meteo) as mock_fetch:
                        weather = data_sources.build_weather_table(schedule, force_refresh=False)

            self.assertEqual(mock_fetch.call_count, 1)
            self.assertEqual(len(weather), 1)
            self.assertEqual(weather.iloc[0]["home_team"], "AZ")
            self.assertAlmostEqual(float(weather.iloc[0]["temperature_f"]), 68.0)
            refreshed_cache = pd.read_csv(stale_cache, parse_dates=["game_date"])
            self.assertEqual(int(refreshed_cache["temperature_f"].notna().sum()), 1)

    def test_fetch_open_meteo_handles_dst_fall_back_day(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, dict[str, list[float | str]]]:
                times = [f"2024-11-03T{hour:02d}:00" for hour in range(24)]
                values = [float(hour) for hour in range(24)]
                return {
                    "hourly": {
                        "time": times,
                        "temperature_2m": values,
                        "relative_humidity_2m": values,
                        "wind_speed_10m": values,
                        "wind_direction_10m": values,
                        "surface_pressure": values,
                    }
                }

        with patch("data_sources.requests.get", return_value=FakeResponse()):
            weather = data_sources._fetch_open_meteo(40.8296, -73.9262, "2024-11-03", "2024-11-03", "America/New_York")

        self.assertEqual(len(weather), 24)
        self.assertEqual(str(weather.index.tz), "America/New_York")
        self.assertEqual(float(weather.iloc[0]["temperature_2m"]), 0.0)

    def test_extract_plate_appearances_computes_spray_angle_from_numeric_arrays(self) -> None:
        statcast_df = pd.DataFrame(
            [
                {
                    "game_date": "2026-03-25",
                    "game_pk": 1,
                    "at_bat_number": 1,
                    "pitch_number": 3,
                    "events": "field_out",
                    "inning_topbot": "Top",
                    "away_team": "NYY",
                    "home_team": "BOS",
                    "launch_speed": 101.2,
                    "launch_angle": 28.0,
                    "description": "hit_into_play",
                    "stand": "R",
                    "p_throws": "L",
                    "bb_type": "fly_ball",
                    "pitch_type": "FF",
                    "hc_x": "140.0",
                    "hc_y": "120.0",
                }
            ]
        )
        pa_df = feature_engineering.extract_plate_appearances(statcast_df)
        self.assertIn("spray_angle", pa_df.columns)
        self.assertTrue(pd.notna(pa_df.iloc[0]["spray_angle"]))

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

    def test_select_probable_lineup_hitters_filters_to_projected_lineup(self) -> None:
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
                {"batter_id": 11, "batter_name": "Bravo Current"},
                {"batter_id": 12, "batter_name": "Charlie Current"},
            ]
        )
        projected_lineup = pd.DataFrame(
            [
                {"batter_id": 10, "batter_name": "Alpha Projected"},
                {"batter_id": 12, "batter_name": "Charlie Projected"},
            ]
        )
        captured = io.StringIO()
        with redirect_stdout(captured):
            hitters = select_probable_lineup_hitters(
                dataset,
                team_code="NYY",
                target_date="2026-03-25",
                hitters_per_team=3,
                lookback_days=10,
                projected_lineup=projected_lineup,
                active_roster=active_roster,
            )
        self.assertEqual([row["batter_id"] for row in hitters], [10, 12])
        self.assertEqual([row["batter_name"] for row in hitters], ["Alpha Projected", "Charlie Projected"])
        self.assertIn("Projected lineup count       : 2", captured.getvalue())

    def test_select_probable_lineup_hitters_empty_projected_lineup_falls_back_to_active_roster(self) -> None:
        dataset = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": "2026-03-20", "batter_id": 10, "batter_name": "Alpha Old", "team": "NYY", "pa_count": 5, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 2, "game_date": "2026-03-21", "batter_id": 11, "batter_name": "Bravo Old", "team": "NYY", "pa_count": 4, "bat_side": "L", "hr_per_pa_last_10d": 0.1, "hr_per_pa_last_30d": 0.08},
            ]
        )
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        active_roster = pd.DataFrame(
            [
                {"batter_id": 10, "batter_name": "Alpha Current"},
                {"batter_id": 11, "batter_name": "Bravo Current"},
            ]
        )
        projected_lineup = pd.DataFrame(columns=["batter_id", "batter_name"])

        captured = io.StringIO()
        with redirect_stdout(captured):
            hitters = select_probable_lineup_hitters(
                dataset,
                team_code="NYY",
                target_date="2026-03-25",
                hitters_per_team=3,
                lookback_days=10,
                projected_lineup=projected_lineup,
                active_roster=active_roster,
            )

        self.assertEqual([row["batter_id"] for row in hitters], [10, 11])
        self.assertEqual([row["batter_name"] for row in hitters], ["Alpha Current", "Bravo Current"])
        self.assertIn("Projected lineup count       : 0", captured.getvalue())
        self.assertIn("Active roster count           : 2", captured.getvalue())

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
        self.assertIn("Eligibility mode             : stale_team_history", captured.getvalue())
        self.assertIn("Recent team games considered  : 0", captured.getvalue())
        self.assertIn("Eligible after recent gate    : 0", captured.getvalue())

    def test_select_probable_lineup_hitters_uses_offseason_fallback_for_season_opener(self) -> None:
        dataset = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": "2025-09-26", "batter_id": 10, "batter_name": "Starter", "team": "NYY", "pa_count": 5, "bat_side": "R", "hr_per_pa_last_10d": 0.20, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 1, "game_date": "2025-09-26", "batter_id": 13, "batter_name": "Depth Bat", "team": "NYY", "pa_count": 2, "bat_side": "L", "hr_per_pa_last_10d": 0.04, "hr_per_pa_last_30d": 0.03},
                {"game_pk": 2, "game_date": "2025-09-27", "batter_id": 10, "batter_name": "Starter", "team": "NYY", "pa_count": 4, "bat_side": "R", "hr_per_pa_last_10d": 0.20, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 2, "game_date": "2025-09-27", "batter_id": 12, "batter_name": "Second Starter", "team": "NYY", "pa_count": 4, "bat_side": "L", "hr_per_pa_last_10d": 0.09, "hr_per_pa_last_30d": 0.07},
                {"game_pk": 3, "game_date": "2025-09-28", "batter_id": 10, "batter_name": "Starter", "team": "NYY", "pa_count": 4, "bat_side": "R", "hr_per_pa_last_10d": 0.20, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 3, "game_date": "2025-09-28", "batter_id": 11, "batter_name": "Leadoff", "team": "NYY", "pa_count": 4, "bat_side": "L", "hr_per_pa_last_10d": 0.10, "hr_per_pa_last_30d": 0.08},
            ]
        )
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        active_roster = pd.DataFrame(
            [
                {"batter_id": 10, "batter_name": "Starter"},
                {"batter_id": 11, "batter_name": "Leadoff"},
                {"batter_id": 12, "batter_name": "Second Starter"},
                {"batter_id": 13, "batter_name": "Depth Bat"},
            ]
        )
        captured = io.StringIO()
        with redirect_stdout(captured):
            hitters = select_probable_lineup_hitters(
                dataset,
                team_code="NYY",
                target_date="2026-03-26",
                hitters_per_team=3,
                lookback_days=21,
                active_roster=active_roster,
            )
        self.assertEqual([row["batter_id"] for row in hitters], [10, 11, 12])
        self.assertIn("Eligibility mode             : season_opening_fallback", captured.getvalue())
        self.assertIn("Recent team games considered  : 3", captured.getvalue())

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

    def test_build_live_candidate_frame_uses_projected_lineups_not_full_roster(self) -> None:
        dataset = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": "2026-03-20", "batter_id": 10, "batter_name": "Alpha", "team": "NYY", "pa_count": 5, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 2, "game_date": "2026-03-21", "batter_id": 11, "batter_name": "Bench Bat", "team": "NYY", "pa_count": 4, "bat_side": "L", "hr_per_pa_last_10d": 0.1, "hr_per_pa_last_30d": 0.08},
                {"game_pk": 3, "game_date": "2026-03-22", "batter_id": 12, "batter_name": "Charlie", "team": "NYY", "pa_count": 4, "bat_side": "R", "hr_per_pa_last_10d": 0.3, "hr_per_pa_last_30d": 0.12},
            ]
        )
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        schedule_games = [
            {
                "game_pk": 999,
                "home_team": "BOS",
                "away_team": "NYY",
                "home_pitcher_id": 200,
                "home_pitcher_name": "Home Pitcher",
                "away_pitcher_id": 201,
                "away_pitcher_name": "Away Pitcher",
                "home_projected_lineup": [],
                "away_projected_lineup": [
                    {"batter_id": 10, "batter_name": "Alpha Projected"},
                    {"batter_id": 12, "batter_name": "Charlie Projected"},
                ],
            }
        ]
        active_roster_map = {
            "NYY": pd.DataFrame(
                [
                    {"batter_id": 10, "batter_name": "Alpha"},
                    {"batter_id": 11, "batter_name": "Bench Bat"},
                    {"batter_id": 12, "batter_name": "Charlie"},
                ]
            )
        }
        with patch(
            "scripts.live_pipeline.fetch_forecast_weather",
            return_value=pd.DataFrame(
                [
                    {
                        "game_date": "2026-03-26",
                        "home_team": "BOS",
                        "temperature_f": 60.0,
                        "wind_speed_mph": 8.0,
                        "humidity_pct": 45.0,
                    }
                ]
            ),
        ):
            with patch("scripts.live_pipeline.latest_pitcher_hand", return_value="R"):
                with patch("scripts.live_pipeline.fetch_player_handedness", return_value="R"):
                    frame = build_live_candidate_frame(
                        dataset,
                        schedule_games,
                        target_date="2026-03-26",
                        hitters_per_team=3,
                        active_roster_map=active_roster_map,
                    )
        self.assertEqual(frame["batter_id"].tolist(), [10, 12])
        self.assertEqual(frame["batter_name"].tolist(), ["Alpha Projected", "Charlie Projected"])

    def test_build_live_candidate_frame_falls_back_to_active_roster_when_lineups_missing(self) -> None:
        dataset = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": "2026-03-20", "batter_id": 10, "batter_name": "Alpha", "team": "NYY", "pa_count": 5, "bat_side": "R", "hr_per_pa_last_10d": 0.2, "hr_per_pa_last_30d": 0.15},
                {"game_pk": 2, "game_date": "2026-03-21", "batter_id": 11, "batter_name": "Bravo", "team": "NYY", "pa_count": 4, "bat_side": "L", "hr_per_pa_last_10d": 0.1, "hr_per_pa_last_30d": 0.08},
            ]
        )
        dataset["game_date"] = pd.to_datetime(dataset["game_date"])
        schedule_games = [
            {
                "game_pk": 999,
                "home_team": "BOS",
                "away_team": "NYY",
                "home_pitcher_id": 200,
                "home_pitcher_name": "Home Pitcher",
                "away_pitcher_id": 201,
                "away_pitcher_name": "Away Pitcher",
                "home_projected_lineup": [],
                "away_projected_lineup": [],
            }
        ]
        active_roster_map = {
            "NYY": pd.DataFrame(
                [
                    {"batter_id": 10, "batter_name": "Alpha"},
                    {"batter_id": 11, "batter_name": "Bravo"},
                ]
            )
        }
        with patch(
            "scripts.live_pipeline.fetch_forecast_weather",
            return_value=pd.DataFrame(
                [
                    {
                        "game_date": "2026-03-26",
                        "home_team": "BOS",
                        "temperature_f": 60.0,
                        "wind_speed_mph": 8.0,
                        "humidity_pct": 45.0,
                    }
                ]
            ),
        ):
            with patch("scripts.live_pipeline.latest_pitcher_hand", return_value="R"):
                with patch("scripts.live_pipeline.fetch_player_handedness", return_value="R"):
                    frame = build_live_candidate_frame(
                        dataset,
                        schedule_games,
                        target_date="2026-03-26",
                        hitters_per_team=3,
                        active_roster_map=active_roster_map,
                    )
        self.assertEqual(frame["batter_id"].tolist(), [10, 11])
        self.assertEqual(frame["batter_name"].tolist(), ["Alpha", "Bravo"])

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

    def test_score_live_candidates_applies_confidence_and_team_filters(self) -> None:
        class FakeModel:
            def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
                probabilities = []
                for value in features["feature_a"].tolist():
                    probabilities.append([1.0 - float(value), float(value)])
                return pd.DataFrame(probabilities).to_numpy()

        candidate_rows = []
        probability_map = {
            1: 0.99,
            2: 0.97,
            3: 0.98,
        }
        for batter_id in range(1, 21):
            candidate_rows.append(
                {
                    "game_pk": 100 + ((batter_id - 1) // 2),
                    "game_date": pd.Timestamp("2026-03-25"),
                    "batter_id": batter_id,
                    "batter_name": f"Batter {batter_id}",
                    "team": "NYY" if batter_id in {1, 2} else ("BOS" if batter_id == 3 else f"TEAM{batter_id}"),
                    "opponent_team": "OPP",
                    "pitcher_id": 200 + batter_id,
                    "pitcher_name": "Pitcher",
                    "feature_a": probability_map.get(batter_id, max(0.01, 0.50 - (batter_id * 0.01))),
                }
            )
        candidate_df = pd.DataFrame(candidate_rows)
        bundle = {
            "model": FakeModel(),
            "feature_columns": ["feature_a"],
            "reference_df": pd.DataFrame({"feature_a": [row["feature_a"] for row in candidate_rows]}),
        }

        picks = score_live_candidates(
            candidate_df,
            bundle,
            max_picks=20,
            min_confidence_tier="strong",
            max_picks_per_team=1,
            published_at="2026-03-25T12:00:00+00:00",
        )

        self.assertEqual([row["batter_name"] for row in picks], ["Batter 1", "Batter 3"])
        self.assertEqual([row["confidence_tier"] for row in picks], ["elite", "strong"])

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

    def test_publish_live_picks_preserves_started_same_day_rows_and_replaces_unstarted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            output_path = base / "current.json"
            history_path = base / "pick_history.json"
            dashboard_dir = base / "dashboard"
            pd.DataFrame({"game_date": pd.to_datetime(["2026-03-25"])}).to_csv(dataset_path, index=False)
            with bundle_path.open("wb") as handle:
                pickle.dump({"trained_through": "2026-03-25", "model": object(), "feature_columns": [], "reference_df": pd.DataFrame()}, handle)
            metadata_path.write_text(json.dumps({"trained_through": "2026-03-25", "dataset_max_game_date": "2026-03-25"}), encoding="utf-8")
            history_path.write_text("[]", encoding="utf-8")
            existing_rows = [
                {
                    "pick_id": "started-pick",
                    "published_at": "2026-03-26T17:00:00+00:00",
                    "game_pk": 1,
                    "game_date": "2026-03-26",
                    "game_datetime": "2026-03-26T18:00:00Z",
                    "rank": 2,
                    "batter_id": 10,
                    "batter_name": "Started Batter",
                    "team": "NYY",
                    "opponent_team": "BOS",
                    "pitcher_id": 20,
                    "pitcher_name": "Pitcher A",
                    "confidence_tier": "elite",
                    "predicted_hr_probability": 0.88,
                    "predicted_hr_score": 98.0,
                    "top_reason_1": "keep me",
                    "top_reason_2": "still live",
                    "top_reason_3": "",
                    "result": "Pending",
                },
                {
                    "pick_id": "future-pick",
                    "published_at": "2026-03-26T17:00:00+00:00",
                    "game_pk": 2,
                    "game_date": "2026-03-26",
                    "game_datetime": "2026-03-26T23:00:00Z",
                    "rank": 1,
                    "batter_id": 11,
                    "batter_name": "Replace Me",
                    "team": "LAD",
                    "opponent_team": "SF",
                    "pitcher_id": 21,
                    "pitcher_name": "Pitcher B",
                    "confidence_tier": "strong",
                    "predicted_hr_probability": 0.77,
                    "predicted_hr_score": 95.0,
                    "top_reason_1": "old future",
                    "top_reason_2": "",
                    "top_reason_3": "",
                    "result": "Pending",
                },
            ]
            output_path.write_text(json.dumps(existing_rows), encoding="utf-8")
            refreshed_rows = [
                {
                    "pick_id": "replacement-future",
                    "published_at": "2026-03-26T20:00:00+00:00",
                    "game_pk": 2,
                    "game_date": "2026-03-26",
                    "game_datetime": "2026-03-26T23:00:00Z",
                    "rank": 1,
                    "batter_id": 12,
                    "batter_name": "Future Refresh",
                    "team": "LAD",
                    "opponent_team": "SF",
                    "pitcher_id": 22,
                    "pitcher_name": "Pitcher C",
                    "confidence_tier": "elite",
                    "predicted_hr_probability": 0.91,
                    "predicted_hr_score": 99.0,
                    "top_reason_1": "new future",
                    "top_reason_2": "",
                    "top_reason_3": "",
                    "result": "Pending",
                },
                {
                    "pick_id": "replacement-later",
                    "published_at": "2026-03-26T20:00:00+00:00",
                    "game_pk": 3,
                    "game_date": "2026-03-26",
                    "game_datetime": "2026-03-26T23:30:00Z",
                    "rank": 2,
                    "batter_id": 13,
                    "batter_name": "Later Refresh",
                    "team": "SEA",
                    "opponent_team": "HOU",
                    "pitcher_id": 23,
                    "pitcher_name": "Pitcher D",
                    "confidence_tier": "strong",
                    "predicted_hr_probability": 0.83,
                    "predicted_hr_score": 96.0,
                    "top_reason_1": "later game",
                    "top_reason_2": "",
                    "top_reason_3": "",
                    "result": "Pending",
                },
            ]
            schedule_games = [
                {"game_pk": 1, "game_datetime": "2026-03-26T18:00:00Z", "status": "Scheduled"},
                {"game_pk": 2, "game_datetime": "2026-03-26T23:00:00Z", "status": "Scheduled"},
                {"game_pk": 3, "game_datetime": "2026-03-26T23:30:00Z", "status": "Scheduled"},
            ]

            with patch("scripts.publish_live_picks._publish_reference_now", return_value=datetime(2026, 3, 26, 20, 0, tzinfo=timezone.utc)):
                with patch("scripts.publish_live_picks.fetch_schedule_games", return_value=schedule_games):
                    with patch("scripts.publish_live_picks.build_active_roster_map", return_value={}):
                        with patch("scripts.publish_live_picks.build_live_candidate_frame", return_value=pd.DataFrame()):
                            with patch("scripts.publish_live_picks.build_live_feature_frame", return_value=pd.DataFrame()):
                                with patch("scripts.publish_live_picks.score_live_candidates", return_value=refreshed_rows):
                                    with patch("scripts.publish_live_picks.refresh_cloudflare_dashboard", return_value=dashboard_dir / "dashboard.json"):
                                        published = publish_live_picks.publish_live_picks(
                                            dataset_path=dataset_path,
                                            bundle_path=bundle_path,
                                            metadata_path=metadata_path,
                                            output_path=output_path,
                                            history_path=history_path,
                                            dashboard_output_dir=dashboard_dir,
                                            schedule_date="2026-03-26",
                                        )

            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([row["rank"] for row in written], [1, 2, 3])
            self.assertEqual([row["batter_name"] for row in written], ["Future Refresh", "Started Batter", "Later Refresh"])
            locked_row = written[1]
            self.assertEqual(locked_row["pick_id"], "started-pick")
            self.assertEqual(locked_row["published_at"], "2026-03-26T17:00:00+00:00")
            self.assertEqual(locked_row["top_reason_1"], "keep me")
            self.assertEqual(locked_row["predicted_hr_score"], 98.0)
            self.assertEqual(locked_row["game_datetime"], "2026-03-26T18:00:00Z")
            self.assertEqual([row["pick_id"] for row in published], ["replacement-future", "started-pick", "replacement-later"])

    def test_publish_live_picks_locks_in_progress_game_even_before_scheduled_first_pitch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            output_path = base / "current.json"
            history_path = base / "pick_history.json"
            dashboard_dir = base / "dashboard"
            pd.DataFrame({"game_date": pd.to_datetime(["2026-03-25"])}).to_csv(dataset_path, index=False)
            with bundle_path.open("wb") as handle:
                pickle.dump({"trained_through": "2026-03-25", "model": object(), "feature_columns": [], "reference_df": pd.DataFrame()}, handle)
            metadata_path.write_text(json.dumps({"trained_through": "2026-03-25", "dataset_max_game_date": "2026-03-25"}), encoding="utf-8")
            history_path.write_text("[]", encoding="utf-8")
            existing_rows = [
                {
                    "pick_id": "live-pick",
                    "published_at": "2026-03-26T17:00:00+00:00",
                    "game_pk": 10,
                    "game_date": "2026-03-26",
                    "game_datetime": "2026-03-26T23:00:00Z",
                    "rank": 1,
                    "batter_id": 30,
                    "batter_name": "Do Not Replace",
                    "team": "ATL",
                    "opponent_team": "PHI",
                    "pitcher_id": 40,
                    "pitcher_name": "Pitcher E",
                    "confidence_tier": "elite",
                    "predicted_hr_probability": 0.93,
                    "predicted_hr_score": 99.0,
                    "top_reason_1": "already live",
                    "top_reason_2": "",
                    "top_reason_3": "",
                    "result": "Pending",
                }
            ]
            output_path.write_text(json.dumps(existing_rows), encoding="utf-8")
            refreshed_rows = [
                {
                    "pick_id": "replacement-live",
                    "published_at": "2026-03-26T20:00:00+00:00",
                    "game_pk": 10,
                    "game_date": "2026-03-26",
                    "game_datetime": "2026-03-26T23:00:00Z",
                    "rank": 1,
                    "batter_id": 31,
                    "batter_name": "Should Not Appear",
                    "team": "ATL",
                    "opponent_team": "PHI",
                    "pitcher_id": 41,
                    "pitcher_name": "Pitcher F",
                    "confidence_tier": "strong",
                    "predicted_hr_probability": 0.81,
                    "predicted_hr_score": 97.0,
                    "top_reason_1": "replacement",
                    "top_reason_2": "",
                    "top_reason_3": "",
                    "result": "Pending",
                }
            ]
            schedule_games = [
                {"game_pk": 10, "game_datetime": "2026-03-26T23:00:00Z", "status": "In Progress"}
            ]

            with patch("scripts.publish_live_picks._publish_reference_now", return_value=datetime(2026, 3, 26, 20, 0, tzinfo=timezone.utc)):
                with patch("scripts.publish_live_picks.fetch_schedule_games", return_value=schedule_games):
                    with patch("scripts.publish_live_picks.build_active_roster_map", return_value={}):
                        with patch("scripts.publish_live_picks.build_live_candidate_frame", return_value=pd.DataFrame()):
                            with patch("scripts.publish_live_picks.build_live_feature_frame", return_value=pd.DataFrame()):
                                with patch("scripts.publish_live_picks.score_live_candidates", return_value=refreshed_rows):
                                    with patch("scripts.publish_live_picks.refresh_cloudflare_dashboard", return_value=dashboard_dir / "dashboard.json"):
                                        publish_live_picks.publish_live_picks(
                                            dataset_path=dataset_path,
                                            bundle_path=bundle_path,
                                            metadata_path=metadata_path,
                                            output_path=output_path,
                                            history_path=history_path,
                                            dashboard_output_dir=dashboard_dir,
                                            schedule_date="2026-03-26",
                                        )

            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["pick_id"], "live-pick")
            self.assertEqual(written[0]["batter_name"], "Do Not Replace")

    def test_train_live_model_bundle_fast_refit_reuses_existing_live_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            pd.DataFrame(
                {
                    "game_date": pd.to_datetime(["2026-03-24", "2026-03-25", "2026-03-25", "2026-03-25"]),
                    "game_pk": [1, 2, 3, 4],
                    "player_id": [10, 11, 12, 13],
                    "hit_hr": [0, 1, 0, 1],
                    "hr_per_pa_last_30d": [0.01, 0.02, 0.03, 0.04],
                    "hr_per_pa_last_10d": [0.02, 0.03, 0.04, 0.05],
                    "barrels_per_pa_last_30d": [0.01, 0.02, 0.03, 0.04],
                    "barrels_per_pa_last_10d": [0.01, 0.02, 0.03, 0.04],
                    "hard_hit_rate_last_30d": [0.3, 0.4, 0.5, 0.6],
                    "hard_hit_rate_last_10d": [0.3, 0.4, 0.5, 0.6],
                    "bbe_95plus_ev_rate_last_30d": [0.1, 0.2, 0.1, 0.2],
                    "bbe_95plus_ev_rate_last_10d": [0.1, 0.2, 0.1, 0.2],
                    "avg_exit_velocity_last_10d": [90.0, 92.0, 94.0, 96.0],
                    "max_exit_velocity_last_10d": [104.0, 106.0, 108.0, 110.0],
                    "pitcher_hr_allowed_per_pa_last_30d": [0.02, 0.03, 0.04, 0.05],
                    "pitcher_barrels_allowed_per_bbe_last_30d": [0.1, 0.2, 0.1, 0.2],
                    "pitcher_hard_hit_allowed_rate_last_30d": [0.3, 0.4, 0.5, 0.6],
                    "pitcher_avg_ev_allowed_last_30d": [88.0, 89.0, 90.0, 91.0],
                    "pitcher_95plus_ev_allowed_rate_last_30d": [0.1, 0.2, 0.1, 0.2],
                    "temperature_f": [65.0, 66.0, 67.0, 68.0],
                    "wind_speed_mph": [8.0, 9.0, 10.0, 11.0],
                    "humidity_pct": [40.0, 45.0, 50.0, 55.0],
                    "platoon_advantage": [0.0, 1.0, 0.0, 1.0],
                }
            ).to_csv(dataset_path, index=False)
            metadata_path.write_text(
                json.dumps(
                    {
                        "trained_through": "2024-09-30",
                        "model_family": "logistic",
                        "feature_profile": "live",
                        "missingness_threshold": 0.35,
                        "selection_metric": "pr_auc",
                        "best_params": {
                            "clf__C": 0.05,
                            "clf__class_weight": None,
                            "clf__l1_ratio": 1.0,
                            "clf__solver": "saga",
                        },
                        "calibration_status": {"used": "disabled"},
                        "training_cv_summary": {"mean_cv_pr_auc": 0.12},
                        "final_holdout_summary": {"pr_auc": 0.15},
                        "promotion_decision": {"used": "selected_candidate"},
                    }
                ),
                encoding="utf-8",
            )
            with patch("scripts.live_pipeline.run_backtest", side_effect=AssertionError("fast_refit should not run backtest")):
                bundle = live_pipeline.train_live_model_bundle(
                    dataset_path=dataset_path,
                    bundle_path=bundle_path,
                    metadata_path=metadata_path,
                    training_mode="fast_refit",
                )

            written_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["trained_through"], "2026-03-25")
            self.assertEqual(written_metadata["trained_through"], "2026-03-25")
            self.assertEqual(written_metadata["refit_strategy"], "fast_refit")
            self.assertEqual(written_metadata["model_family"], "logistic")
            self.assertTrue(bundle_path.exists())
            reloaded_bundle = live_pipeline.load_model_bundle(bundle_path)
            probabilities = reloaded_bundle["model"].predict_proba(
                live_pipeline.prepare_feature_matrix(pd.read_csv(dataset_path), reloaded_bundle["feature_columns"])
            )[:, 1]
            self.assertEqual(len(probabilities), 4)

    def test_train_live_model_bundle_fast_refit_accepts_legacy_metadata_feature_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            feature_columns = [
                "hr_per_pa_last_30d",
                "hr_per_pa_last_10d",
                "barrels_per_pa_last_30d",
                "barrels_per_pa_last_10d",
                "hard_hit_rate_last_30d",
                "hard_hit_rate_last_10d",
                "bbe_95plus_ev_rate_last_30d",
                "bbe_95plus_ev_rate_last_10d",
                "avg_exit_velocity_last_10d",
                "max_exit_velocity_last_10d",
                "pitcher_hr_allowed_per_pa_last_30d",
                "pitcher_barrels_allowed_per_bbe_last_30d",
                "pitcher_hard_hit_allowed_rate_last_30d",
                "pitcher_avg_ev_allowed_last_30d",
                "pitcher_95plus_ev_allowed_rate_last_30d",
                "temperature_f",
                "wind_speed_mph",
                "humidity_pct",
            ]
            pd.DataFrame(
                {
                    "game_date": pd.to_datetime(["2026-03-24", "2026-03-25", "2026-03-25", "2026-03-25"]),
                    "game_pk": [1, 2, 3, 4],
                    "player_id": [10, 11, 12, 13],
                    "hit_hr": [0, 1, 0, 1],
                    "hr_per_pa_last_30d": [0.01, 0.02, 0.03, 0.04],
                    "hr_per_pa_last_10d": [0.02, 0.03, 0.04, 0.05],
                    "barrels_per_pa_last_30d": [0.01, 0.02, 0.03, 0.04],
                    "barrels_per_pa_last_10d": [0.01, 0.02, 0.03, 0.04],
                    "hard_hit_rate_last_30d": [0.3, 0.4, 0.5, 0.6],
                    "hard_hit_rate_last_10d": [0.3, 0.4, 0.5, 0.6],
                    "bbe_95plus_ev_rate_last_30d": [0.1, 0.2, 0.1, 0.2],
                    "bbe_95plus_ev_rate_last_10d": [0.1, 0.2, 0.1, 0.2],
                    "avg_exit_velocity_last_10d": [90.0, 92.0, 94.0, 96.0],
                    "max_exit_velocity_last_10d": [104.0, 106.0, 108.0, 110.0],
                    "pitcher_hr_allowed_per_pa_last_30d": [0.02, 0.03, 0.04, 0.05],
                    "pitcher_barrels_allowed_per_bbe_last_30d": [0.1, 0.2, 0.1, 0.2],
                    "pitcher_hard_hit_allowed_rate_last_30d": [0.3, 0.4, 0.5, 0.6],
                    "pitcher_avg_ev_allowed_last_30d": [88.0, 89.0, 90.0, 91.0],
                    "pitcher_95plus_ev_allowed_rate_last_30d": [0.1, 0.2, 0.1, 0.2],
                    "temperature_f": [65.0, 66.0, 67.0, 68.0],
                    "wind_speed_mph": [8.0, 9.0, 10.0, 11.0],
                    "humidity_pct": [40.0, 45.0, 50.0, 55.0],
                }
            ).to_csv(dataset_path, index=False)
            metadata_path.write_text(
                json.dumps(
                    {
                        "trained_through": "2024-09-30",
                        "model_family": "logistic",
                        "feature_profile": "live_shrunk_precise",
                        "feature_columns": feature_columns,
                        "missingness_threshold": 0.35,
                        "selection_metric": "pr_auc",
                        "best_params": {
                            "clf__C": 0.05,
                            "clf__class_weight": None,
                            "clf__l1_ratio": 1.0,
                            "clf__solver": "saga",
                        },
                        "calibration_status": {"used": "disabled"},
                    }
                ),
                encoding="utf-8",
            )
            with patch("scripts.live_pipeline.run_backtest", side_effect=AssertionError("fast_refit should not run backtest")):
                bundle = live_pipeline.train_live_model_bundle(
                    dataset_path=dataset_path,
                    bundle_path=bundle_path,
                    metadata_path=metadata_path,
                    training_mode="fast_refit",
                )

            self.assertEqual(bundle["feature_profile"], "live_shrunk_precise")
            self.assertEqual(bundle["feature_columns"], feature_columns)

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

    def test_run_daily_live_refresh_runs_refresh_train_settle_publish_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            current_path = base / "current.json"
            history_path = base / "history.json"
            dashboard_dir = base / "dashboard"
            call_order: list[str] = []
            dataset_df = pd.DataFrame({"game_date": pd.to_datetime(["2026-03-25"]), "batter_id": [10], "hit_hr": [1]})

            def fake_load_json_array(_: Path) -> list[dict[str, object]]:
                call_order.append("load_json")
                return []

            def fake_settle_pick_records(records: list[dict[str, object]], _: pd.DataFrame, *, resolved_through_date: str) -> list[dict[str, object]]:
                call_order.append(f"settle:{resolved_through_date}")
                return records

            with patch("scripts.run_daily_live_refresh.refresh_live_dataset", side_effect=lambda **_: call_order.append("refresh")):
                with patch("scripts.run_daily_live_refresh.load_live_dataset", side_effect=lambda _: call_order.append("load_dataset") or dataset_df):
                    with patch(
                        "scripts.run_daily_live_refresh.train_live_model_bundle",
                        side_effect=lambda **kwargs: call_order.append(f"train:{kwargs['training_mode']}") or {"trained_through": "2026-03-25"},
                    ):
                        with patch("scripts.run_daily_live_refresh.load_json_array", side_effect=fake_load_json_array):
                            with patch("scripts.run_daily_live_refresh.settle_pick_records", side_effect=fake_settle_pick_records):
                                with patch("scripts.run_daily_live_refresh.write_current_picks", side_effect=lambda rows, path: call_order.append(f"write_current:{len(rows)}")):
                                    with patch("scripts.run_daily_live_refresh.write_pick_history", side_effect=lambda rows, path: call_order.append(f"write_history:{len(rows)}")):
                                        with patch(
                                            "scripts.run_daily_live_refresh.publish_live_picks",
                                            side_effect=lambda **_: call_order.append("publish") or [{"game_date": "2026-03-26"}],
                                        ):
                                            picks = run_daily_live_refresh.run_daily_live_refresh(
                                                dataset_path=dataset_path,
                                                bundle_path=bundle_path,
                                                metadata_path=metadata_path,
                                                current_picks_path=current_path,
                                                history_path=history_path,
                                                dashboard_output_dir=dashboard_dir,
                                                train_end_date="2026-03-25",
                                                publish_date="2026-03-26",
                                            )

            self.assertEqual(picks, [{"game_date": "2026-03-26"}])
            self.assertEqual(
                call_order,
                [
                    "refresh",
                    "load_dataset",
                    "train:fast_refit",
                    "load_json",
                    "load_json",
                    "settle:2026-03-25",
                    "settle:2026-03-25",
                    "write_current:0",
                    "write_history:0",
                    "publish",
                ],
            )

    def test_run_daily_live_refresh_refreshes_stale_inputs_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            current_path = base / "current.json"
            history_path = base / "history.json"
            dashboard_dir = base / "dashboard"
            pd.DataFrame({"game_date": pd.to_datetime(["2024-09-30"]), "batter_id": [10], "hit_hr": [0]}).to_csv(dataset_path, index=False)
            current_path.write_text("[]", encoding="utf-8")
            history_path.write_text("[]", encoding="utf-8")
            metadata_path.write_text(json.dumps({"trained_through": "2024-09-30"}), encoding="utf-8")

            def fake_refresh_live_dataset(**_: object) -> None:
                pd.DataFrame({"game_date": pd.to_datetime(["2026-03-25"]), "batter_id": [10], "hit_hr": [1]}).to_csv(dataset_path, index=False)

            def fake_train_live_model_bundle(**_: object) -> dict[str, object]:
                metadata_path.write_text(json.dumps({"trained_through": "2026-03-25", "dataset_max_game_date": "2026-03-25"}), encoding="utf-8")
                return {"trained_through": "2026-03-25"}

            def fake_publish_live_picks(**kwargs: object) -> list[dict[str, object]]:
                refreshed = pd.read_csv(dataset_path)
                self.assertEqual(str(pd.to_datetime(refreshed["game_date"]).max().date()), "2026-03-25")
                self.assertEqual(kwargs["schedule_date"], "2026-03-26")
                picks = [{"game_date": "2026-03-26", "batter_id": 22, "batter_name": "Beta", "rank": 1}]
                Path(kwargs["output_path"]).write_text(json.dumps(picks), encoding="utf-8")
                return picks

            with patch("scripts.run_daily_live_refresh.refresh_live_dataset", side_effect=fake_refresh_live_dataset):
                with patch("scripts.run_daily_live_refresh.train_live_model_bundle", side_effect=fake_train_live_model_bundle):
                    with patch("scripts.run_daily_live_refresh.publish_live_picks", side_effect=fake_publish_live_picks):
                        run_daily_live_refresh.run_daily_live_refresh(
                            dataset_path=dataset_path,
                            bundle_path=bundle_path,
                            metadata_path=metadata_path,
                            current_picks_path=current_path,
                            history_path=history_path,
                            dashboard_output_dir=dashboard_dir,
                            train_end_date="2026-03-25",
                            publish_date="2026-03-26",
                        )

            written = json.loads(current_path.read_text(encoding="utf-8"))
            self.assertEqual(written[0]["game_date"], "2026-03-26")

    def test_run_daily_live_refresh_settles_prior_rows_then_replaces_current_with_today(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            current_path = base / "current.json"
            history_path = base / "history.json"
            dashboard_dir = base / "dashboard"
            pd.DataFrame({"game_date": pd.to_datetime(["2026-03-25"]), "batter_id": [10], "hit_hr": [1]}).to_csv(dataset_path, index=False)
            pending_row = {
                "game_date": "2026-03-25",
                "batter_id": 10,
                "batter_name": "Alpha",
                "rank": 1,
                "predicted_hr_score": 95.0,
                "result": "Pending",
            }
            current_path.write_text(json.dumps([pending_row]), encoding="utf-8")
            history_path.write_text(json.dumps([dict(pending_row)]), encoding="utf-8")

            def fake_publish_live_picks(**kwargs: object) -> list[dict[str, object]]:
                picks = [{"game_date": "2026-03-26", "batter_id": 22, "batter_name": "Beta", "rank": 1, "predicted_hr_score": 88.0, "result": "Pending"}]
                Path(kwargs["output_path"]).write_text(json.dumps(picks), encoding="utf-8")
                return picks

            with patch("scripts.run_daily_live_refresh.refresh_live_dataset", side_effect=lambda **_: None):
                with patch("scripts.run_daily_live_refresh.train_live_model_bundle", side_effect=lambda **_: {"trained_through": "2026-03-25"}):
                    with patch("scripts.run_daily_live_refresh.publish_live_picks", side_effect=fake_publish_live_picks):
                        run_daily_live_refresh.run_daily_live_refresh(
                            dataset_path=dataset_path,
                            bundle_path=bundle_path,
                            metadata_path=metadata_path,
                            current_picks_path=current_path,
                            history_path=history_path,
                            dashboard_output_dir=dashboard_dir,
                            train_end_date="2026-03-25",
                            publish_date="2026-03-26",
                        )

            history_rows = json.loads(history_path.read_text(encoding="utf-8"))
            current_rows = json.loads(current_path.read_text(encoding="utf-8"))
            self.assertEqual(history_rows[0]["result_label"], "HR")
            self.assertEqual(history_rows[0]["actual_hit_hr"], 1)
            self.assertEqual(current_rows[0]["game_date"], "2026-03-26")

    def test_run_daily_live_refresh_preserves_settled_history_when_publish_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            current_path = base / "current.json"
            history_path = base / "history.json"
            dashboard_dir = base / "dashboard"
            pd.DataFrame({"game_date": pd.to_datetime(["2026-03-25"]), "batter_id": [10], "hit_hr": [0]}).to_csv(dataset_path, index=False)
            pending_row = {
                "game_date": "2026-03-25",
                "batter_id": 10,
                "batter_name": "Alpha",
                "rank": 1,
                "predicted_hr_score": 95.0,
                "result": "Pending",
            }
            current_path.write_text(json.dumps([pending_row]), encoding="utf-8")
            history_path.write_text(json.dumps([dict(pending_row)]), encoding="utf-8")

            def fake_publish_live_picks(**kwargs: object) -> list[dict[str, object]]:
                Path(kwargs["output_path"]).write_text("[]", encoding="utf-8")
                build_dashboard_artifacts.build_dashboard_artifacts(
                    current_picks_path=Path(kwargs["output_path"]),
                    history_path=Path(kwargs["history_path"]),
                    output_dir=Path(kwargs["dashboard_output_dir"]),
                    latest_available_date_override=str(kwargs["schedule_date"]),
                    persist_history=False,
                )
                raise RuntimeError("stale publish")

            with patch("scripts.run_daily_live_refresh.refresh_live_dataset", side_effect=lambda **_: None):
                with patch("scripts.run_daily_live_refresh.train_live_model_bundle", side_effect=lambda **_: {"trained_through": "2026-03-25"}):
                    with patch("scripts.run_daily_live_refresh.publish_live_picks", side_effect=fake_publish_live_picks):
                        with self.assertRaisesRegex(RuntimeError, "publish_date=2026-03-26"):
                            run_daily_live_refresh.run_daily_live_refresh(
                                dataset_path=dataset_path,
                                bundle_path=bundle_path,
                                metadata_path=metadata_path,
                                current_picks_path=current_path,
                                history_path=history_path,
                                dashboard_output_dir=dashboard_dir,
                                train_end_date="2026-03-25",
                                publish_date="2026-03-26",
                            )

            history_rows = json.loads(history_path.read_text(encoding="utf-8"))
            current_rows = json.loads(current_path.read_text(encoding="utf-8"))
            dashboard_payload = json.loads((dashboard_dir / "dashboard.json").read_text(encoding="utf-8"))
            self.assertEqual(history_rows[0]["result_label"], "No HR")
            self.assertEqual(history_rows[0]["actual_hit_hr"], 0)
            self.assertEqual(current_rows, [])
            self.assertEqual(dashboard_payload["latest_available_date"], "2026-03-26")
            self.assertEqual(dashboard_payload["latest_picks"], [])

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
                    "park_factor_hr_vs_lhb": 96.0,
                    "park_factor_hr_vs_rhb": 103.0,
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
                    "park_factor_hr_vs_lhb": 98.0,
                    "park_factor_hr_vs_rhb": 110.0,
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
        self.assertEqual(float(featured.iloc[0]["park_factor_hr_vs_batter_hand"]), 110.0)
        for feature in LIVE_PLUS_FEATURE_COLUMNS:
            self.assertIn(feature, featured.columns)
        self.assertTrue(pd.notna(featured.iloc[0]["batter_hr_per_pa_vs_pitcher_hand"]))
        self.assertTrue(pd.notna(featured.iloc[0]["pitcher_hr_allowed_per_pa_vs_batter_hand"]))
        self.assertTrue(pd.notna(featured.iloc[0]["split_matchup_hr"]))

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
    "result": "HR"
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

            payload = json.loads((output_dir / "dashboard.json").read_text(encoding="utf-8"))
            payload_text = json.dumps(payload)
            self.assertIn("Alpha", payload_text)
            self.assertEqual(
                payload["data_note"],
                "Public dashboard tracking begins on Opening Night, March 25, 2026. Trained on 2024 and 2025 season data.",
            )
            self.assertEqual(payload["model_family"], "2024-25 trained")
            self.assertEqual(payload["refresh_schedule"]["runs"][0]["time_et"], "2:00 AM ET")
            self.assertEqual(payload["refresh_schedule"]["runs"][0]["type"], "settle")
            self.assertEqual(payload["refresh_schedule"]["runs"][1]["time_et"], "4:00 AM ET")
            self.assertEqual(payload["refresh_schedule"]["runs"][1]["type"], "prepare")
            self.assertEqual(payload["refresh_schedule"]["runs"][2]["time_et"], "11:00 AM ET")
            elite_row = payload["confidence_summary"][0]
            self.assertEqual(elite_row["confidence_tier"], "elite")
            self.assertEqual(len(payload["history"]), 1)
            self.assertEqual(payload["history"][0]["batter_name"], "Alpha")
            self.assertNotIn("forward-only", payload_text)
            history_payload = history_path.read_text(encoding="utf-8")
            self.assertIn("2026-03-25", history_payload)
            self.assertEqual(json.loads(current_path.read_text(encoding="utf-8")), [])

    def test_dashboard_builder_keeps_only_latest_pending_slate_in_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "out"
            current_path.write_text(
                json.dumps(
                    [
                        {
                            "pick_id": "settled-old",
                            "game_pk": 1001,
                            "game_date": "2026-03-27",
                            "rank": 1,
                            "batter_id": 10,
                            "batter_name": "Settled Old",
                            "team": "NYY",
                            "opponent_team": "BOS",
                            "pitcher_id": 20,
                            "pitcher_name": "Pitcher A",
                            "confidence_tier": "elite",
                            "predicted_hr_probability": 0.22,
                            "predicted_hr_score": 99.1,
                            "top_reason_1": "reason",
                            "result": "HR",
                        },
                        {
                            "pick_id": "pending-old",
                            "game_pk": 1002,
                            "game_date": "2026-03-27",
                            "rank": 2,
                            "batter_id": 11,
                            "batter_name": "Pending Old",
                            "team": "NYY",
                            "opponent_team": "BOS",
                            "pitcher_id": 21,
                            "pitcher_name": "Pitcher B",
                            "confidence_tier": "strong",
                            "predicted_hr_probability": 0.18,
                            "predicted_hr_score": 88.0,
                            "top_reason_1": "reason",
                            "result": "Pending",
                        },
                        {
                            "pick_id": "pending-new",
                            "game_pk": 1003,
                            "game_date": "2026-03-28",
                            "rank": 1,
                            "batter_id": 12,
                            "batter_name": "Pending New",
                            "team": "LAD",
                            "opponent_team": "SD",
                            "pitcher_id": 22,
                            "pitcher_name": "Pitcher C",
                            "confidence_tier": "elite",
                            "predicted_hr_probability": 0.25,
                            "predicted_hr_score": 97.5,
                            "top_reason_1": "reason",
                            "result": "Pending",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            history_path.write_text("[]", encoding="utf-8")

            build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
            )

            current_rows = json.loads(current_path.read_text(encoding="utf-8"))
            history_rows = json.loads(history_path.read_text(encoding="utf-8"))
            payload = json.loads((output_dir / "dashboard.json").read_text(encoding="utf-8"))

            self.assertEqual([row["pick_id"] for row in current_rows], ["pending-new"])
            self.assertEqual([row["pick_id"] for row in payload["latest_picks"]], ["pending-new"])
            self.assertEqual({row["pick_id"] for row in history_rows}, {"settled-old", "pending-old", "pending-new"})

    def test_dashboard_builder_history_is_not_truncated_per_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "out"
            current_path.write_text("[]", encoding="utf-8")
            history_rows = [
                {
                    "pick_id": f"pick-{index}",
                    "game_pk": 1000 + index,
                    "game_date": "2026-03-25",
                    "rank": index + 1,
                    "batter_id": 10 + index,
                    "batter_name": f"Player {index + 1}",
                    "team": "NYY",
                    "opponent_team": "BOS",
                    "pitcher_id": 20,
                    "pitcher_name": "Pitcher",
                    "confidence_tier": "watch",
                    "predicted_hr_probability": 0.10,
                    "predicted_hr_score": 99.0 - index,
                    "top_reason_1": "reason",
                    "result_label": "No HR",
                    "actual_hit_hr": 0,
                }
                for index in range(12)
            ]
            history_path.write_text(json.dumps(history_rows), encoding="utf-8")

            build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
            )

            payload = json.loads((output_dir / "dashboard.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["history"]), 12)
            self.assertEqual(payload["history"][0]["batter_name"], "Player 1")
            self.assertEqual(payload["history"][-1]["batter_name"], "Player 12")

    def test_dashboard_builder_keeps_empty_player_leaderboard_when_no_settled_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "out"
            current_path.write_text("[]", encoding="utf-8")
            history_path.write_text("[]", encoding="utf-8")

            build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
            )

            payload = json.loads((output_dir / "dashboard.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["player_leaderboard"], [])

    def test_dashboard_builder_bootstraps_player_leaderboard_at_two_picks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "out"
            current_path.write_text("[]", encoding="utf-8")
            history_rows = [
                {
                    "pick_id": "pick-1",
                    "game_pk": 1001,
                    "game_date": "2026-03-25",
                    "rank": 1,
                    "batter_id": 10,
                    "batter_name": "Alpha",
                    "team": "NYY",
                    "opponent_team": "BOS",
                    "pitcher_id": 20,
                    "pitcher_name": "Pitcher A",
                    "confidence_tier": "elite",
                    "predicted_hr_probability": 0.22,
                    "predicted_hr_score": 98.0,
                    "top_reason_1": "reason",
                    "result_label": "HR",
                    "actual_hit_hr": 1,
                },
                {
                    "pick_id": "pick-2",
                    "game_pk": 1002,
                    "game_date": "2026-03-26",
                    "rank": 1,
                    "batter_id": 10,
                    "batter_name": "Alpha",
                    "team": "NYY",
                    "opponent_team": "TOR",
                    "pitcher_id": 21,
                    "pitcher_name": "Pitcher B",
                    "confidence_tier": "strong",
                    "predicted_hr_probability": 0.20,
                    "predicted_hr_score": 95.0,
                    "top_reason_1": "reason",
                    "result_label": "No HR",
                    "actual_hit_hr": 0,
                },
            ]
            history_path.write_text(json.dumps(history_rows), encoding="utf-8")

            build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
            )

            payload = json.loads((output_dir / "dashboard.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["player_leaderboard"][0]["batter_name"], "Alpha")
            self.assertEqual(payload["player_leaderboard"][0]["picks"], 2)

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
