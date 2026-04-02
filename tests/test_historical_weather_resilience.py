from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

import data_sources
import generate_data


class HistoricalWeatherResilienceTests(unittest.TestCase):
    def test_fetch_open_meteo_retries_then_succeeds(self) -> None:
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

        session = Mock()
        session.get.side_effect = [
            requests.exceptions.ReadTimeout("timed out"),
            FakeResponse(),
        ]
        with patch("data_sources.requests.Session", return_value=session):
            with patch("data_sources.time.sleep") as sleep_mock:
                weather = data_sources._fetch_open_meteo(40.8296, -73.9262, "2024-11-03", "2024-11-03", "America/New_York")

        self.assertEqual(session.get.call_count, 2)
        sleep_mock.assert_called_once()
        self.assertEqual(len(weather), 24)
        self.assertEqual(str(weather.index.tz), "America/New_York")
        self.assertEqual(float(weather.iloc[0]["temperature_2m"]), 0.0)

    def test_build_weather_table_reuses_cached_rows_when_fetch_fails(self) -> None:
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
                        "field_bearing_deg": 20.0,
                        "temperature_f": 72.0,
                        "humidity_pct": None,
                        "wind_speed_mph": 8.0,
                        "wind_direction_deg": 180.0,
                        "pressure_hpa": 1014.0,
                        "wind_out_to_cf_mph": None,
                        "crosswind_mph": None,
                        "air_density_index": None,
                    }
                ]
            ).to_csv(stale_cache, index=False)

            schedule = pd.DataFrame([{"game_date": "2024-03-28", "home_team": "ARI"}])

            with patch.object(data_sources, "DATA_DIR", tmp_path):
                with patch.object(data_sources, "RAW_DATA_DIR", raw_dir):
                    with patch.object(
                        data_sources,
                        "_fetch_open_meteo",
                        side_effect=requests.exceptions.ReadTimeout("timed out"),
                    ):
                        weather = data_sources.build_weather_table(schedule, force_refresh=False)

        self.assertEqual(len(weather), 1)
        self.assertEqual(weather.iloc[0]["home_team"], "AZ")
        self.assertAlmostEqual(float(weather.iloc[0]["temperature_f"]), 72.0)
        self.assertEqual(float(weather.iloc[0]["field_bearing_deg"]), 20.0)
        self.assertIn("wind_out_to_cf_mph", weather.columns)
        self.assertIn("crosswind_mph", weather.columns)
        self.assertIn("air_density_index", weather.columns)
        self.assertTrue(pd.notna(weather.iloc[0]["wind_out_to_cf_mph"]))
        self.assertTrue(pd.notna(weather.iloc[0]["crosswind_mph"]))
        self.assertTrue(pd.isna(weather.iloc[0]["air_density_index"]))
        self.assertEqual(weather.attrs["operational_alerts"][0]["code"], "historical_weather_cache_reused")

    def test_build_weather_table_falls_back_to_null_rows_when_fetch_fails_without_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            schedule = pd.DataFrame([{"game_date": "2024-03-28", "home_team": "ARI"}])

            with patch.object(data_sources, "DATA_DIR", tmp_path):
                with patch.object(data_sources, "RAW_DATA_DIR", raw_dir):
                    with patch.object(
                        data_sources,
                        "_fetch_open_meteo",
                        side_effect=requests.exceptions.ReadTimeout("timed out"),
                    ):
                        weather = data_sources.build_weather_table(schedule, force_refresh=False)

        self.assertEqual(len(weather), 1)
        self.assertEqual(weather.iloc[0]["home_team"], "AZ")
        self.assertTrue(pd.isna(weather.iloc[0]["temperature_f"]))
        self.assertEqual(weather.attrs["operational_alerts"][0]["code"], "historical_weather_null_fallback")

    def test_build_weather_table_neutralizes_missing_wind_direction(self) -> None:
        def fake_fetch_open_meteo(lat: float, lon: float, start_date: str, end_date: str, tz: str) -> pd.DataFrame:
            hours = pd.date_range("2024-03-28 00:00", periods=24, freq="h", tz=tz)
            return pd.DataFrame(
                {
                    "temperature_2m": [71.0] * len(hours),
                    "relative_humidity_2m": [52.0] * len(hours),
                    "wind_speed_10m": [11.0] * len(hours),
                    "wind_direction_10m": [None] * len(hours),
                    "surface_pressure": [1012.0] * len(hours),
                },
                index=hours,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            schedule = pd.DataFrame([{"game_date": "2024-03-28", "home_team": "ARI"}])

            with patch.object(data_sources, "DATA_DIR", tmp_path):
                with patch.object(data_sources, "RAW_DATA_DIR", raw_dir):
                    with patch.object(data_sources, "_fetch_open_meteo", side_effect=fake_fetch_open_meteo):
                        weather = data_sources.build_weather_table(schedule, force_refresh=False)

        self.assertEqual(len(weather), 1)
        self.assertEqual(weather.iloc[0]["home_team"], "AZ")
        self.assertEqual(float(weather.iloc[0]["field_bearing_deg"]), 20.0)
        self.assertEqual(weather.iloc[0]["wind_out_to_cf_mph"], 0.0)
        self.assertEqual(weather.iloc[0]["crosswind_mph"], 0.0)
        self.assertIsNotNone(weather.iloc[0]["air_density_index"])

    def test_generate_mlb_dataset_tolerates_null_weather_columns(self) -> None:
        statcast_df = pd.DataFrame(
            [
                {
                    "game_date": "2024-03-28",
                    "player_name": "Alpha",
                    "batter": 10,
                    "pitcher": 20,
                    "game_pk": 1,
                    "events": "single",
                }
            ]
        )
        dataset = pd.DataFrame(
            [
                {
                    "game_date": pd.Timestamp("2024-03-28"),
                    "game_pk": 1,
                    "batter_id": 10,
                    "team": "AZ",
                    "opponent": "LAD",
                    "is_home": 1,
                }
            ]
        )
        weather_df = pd.DataFrame(
            [
                {
                    "game_date": pd.Timestamp("2024-03-28"),
                    "home_team": "AZ",
                    "field_bearing_deg": 20.0,
                    "temperature_f": None,
                    "humidity_pct": None,
                    "wind_speed_mph": None,
                    "wind_direction_deg": None,
                    "pressure_hpa": None,
                    "wind_out_to_cf_mph": None,
                    "crosswind_mph": None,
                    "air_density_index": None,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "dataset.csv"
            with patch("generate_data.fetch_statcast_season", return_value=statcast_df):
                with patch("generate_data.print_source_summary"):
                    with patch("generate_data.build_player_game_dataset", return_value=(pd.DataFrame(), pd.DataFrame())):
                        with patch("generate_data.add_leakage_safe_features", return_value=dataset.copy()):
                            with patch("generate_data.build_weather_table", return_value=weather_df):
                                with patch("generate_data.audit_weather_feature_coverage"):
                                    with patch("generate_data.validate_final_model_df"):
                                        with patch("generate_data.validate_dataset", return_value=[]):
                                            result = generate_data.generate_mlb_dataset(
                                                output_path=str(output_path),
                                                start_date="2024-03-28",
                                                end_date="2024-03-28",
                                            )

            self.assertTrue(output_path.exists())
            self.assertIn("temperature_f", result.columns)
            self.assertIn("humidity_pct", result.columns)
            self.assertIn("wind_speed_mph", result.columns)
            self.assertIn("field_bearing_deg", result.columns)
            self.assertIn("wind_out_to_cf_mph", result.columns)
            self.assertIn("crosswind_mph", result.columns)
            self.assertIn("air_density_index", result.columns)
            self.assertTrue(pd.isna(result.iloc[0]["temperature_f"]))


if __name__ == "__main__":
    unittest.main()
