import unittest
from unittest.mock import patch

import requests

from scripts.live_pipeline import fetch_forecast_weather


class ForecastWeatherResilienceTests(unittest.TestCase):
    def test_fetch_forecast_weather_falls_back_to_null_weather_after_retries(self) -> None:
        with patch("scripts.live_pipeline.requests.get", side_effect=requests.exceptions.ReadTimeout("timed out")) as mock_get:
            weather = fetch_forecast_weather(["ATL"], "2026-04-02")

        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(
            weather.to_dict(orient="records"),
            [
                {
                    "game_date": "2026-04-02",
                    "home_team": "ATL",
                    "field_bearing_deg": 32.0,
                    "temperature_f": None,
                    "humidity_pct": None,
                    "wind_speed_mph": None,
                    "wind_direction_deg": None,
                    "weather_code": None,
                    "weather_label": "Unknown",
                    "pressure_hpa": None,
                    "wind_out_to_cf_mph": None,
                    "crosswind_mph": None,
                    "air_density_index": None,
                }
            ],
        )
        self.assertEqual(len(weather.attrs.get("operational_alerts", [])), 1)
        self.assertEqual(weather.attrs["operational_alerts"][0]["code"], "weather_forecast_unavailable")

    def test_fetch_forecast_weather_uses_neutral_carry_fallback_when_direction_missing(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, dict[str, list[float | None]]]:
                return {
                    "hourly": {
                        "time": [f"2026-04-02T{hour:02d}:00" for hour in range(24)],
                        "temperature_2m": [70.0] * 24,
                        "relative_humidity_2m": [50.0] * 24,
                        "wind_speed_10m": [10.0] * 24,
                        "wind_direction_10m": [None] * 24,
                        "weather_code": [2] * 24,
                        "surface_pressure": [1015.0] * 24,
                    }
                }

        with patch("scripts.live_pipeline.requests.get", return_value=FakeResponse()):
            weather = fetch_forecast_weather(["ATL"], "2026-04-02")

        row = weather.iloc[0].to_dict()
        self.assertEqual(row["field_bearing_deg"], 32.0)
        self.assertEqual(row["wind_speed_mph"], 10.0)
        self.assertIsNone(row["wind_direction_deg"])
        self.assertEqual(row["wind_out_to_cf_mph"], 0.0)
        self.assertEqual(row["crosswind_mph"], 0.0)
        self.assertIsNotNone(row["air_density_index"])


if __name__ == "__main__":
    unittest.main()
