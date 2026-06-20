import unittest
from unittest.mock import patch

import requests

from scripts.live_pipeline import fetch_forecast_weather


class ForecastWeatherResilienceTests(unittest.TestCase):
    def test_fetch_forecast_weather_falls_back_to_null_weather_after_retries(self) -> None:
        with patch("scripts.live_pipeline.eastern_today", return_value=__import__("pandas").Timestamp("2026-05-02", tz="America/New_York")):
            with patch("scripts.live_pipeline._load_forecast_weather_cache", return_value={}):
                with patch("scripts.live_pipeline._write_forecast_weather_cache"):
                    with patch("scripts.live_pipeline.requests.get", side_effect=requests.exceptions.ReadTimeout("timed out")) as mock_get:
                        weather = fetch_forecast_weather(["ATL"], "2026-05-03")

        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(
            weather.to_dict(orient="records"),
            [
                {
                    "game_date": "2026-05-03",
                    "home_team": "ATL",
                    "roof_type": "open_air",
                    "roof_label": "Open air",
                    "roofed_park": False,
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

    def test_fetch_forecast_weather_neutralizes_roofed_parks_without_requesting_forecast(self) -> None:
        with patch("scripts.live_pipeline.requests.get") as mock_get:
            weather = fetch_forecast_weather(["TB"], "2026-04-02")

        mock_get.assert_not_called()
        self.assertEqual(
            weather.to_dict(orient="records"),
            [
                {
                    "game_date": "2026-04-02",
                    "home_team": "TB",
                    "roof_type": "dome",
                    "roof_label": "Dome",
                    "roofed_park": True,
                    "field_bearing_deg": 40.0,
                    "temperature_f": None,
                    "humidity_pct": None,
                    "wind_speed_mph": None,
                    "wind_direction_deg": None,
                    "weather_code": None,
                    "weather_label": "Dome",
                    "pressure_hpa": None,
                    "wind_out_to_cf_mph": None,
                    "crosswind_mph": None,
                    "air_density_index": None,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
