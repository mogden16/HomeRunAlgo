import unittest
from unittest.mock import patch

import requests

from scripts.live_pipeline import fetch_forecast_weather


class ForecastWeatherResilienceTests(unittest.TestCase):
    def test_fetch_forecast_weather_falls_back_to_null_weather_after_retries(self) -> None:
        with patch("scripts.live_pipeline.requests.get", side_effect=requests.exceptions.ReadTimeout("timed out")) as mock_get:
            weather = fetch_forecast_weather(["ATL"], "2026-03-30")

        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(
            weather.to_dict(orient="records"),
            [
                {
                    "game_date": "2026-03-30",
                    "home_team": "ATL",
                    "temperature_f": None,
                    "humidity_pct": None,
                    "wind_speed_mph": None,
                    "wind_direction_deg": None,
                    "pressure_hpa": None,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
