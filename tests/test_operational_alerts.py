import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_dashboard_artifacts import build_dashboard_artifacts
from scripts.publish_live_picks import persist_operational_alerts


class OperationalAlertTests(unittest.TestCase):
    def test_persist_operational_alerts_overwrites_metadata_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "model_metadata.json"
            metadata_path.write_text(
                json.dumps({"trained_through": "2026-03-29", "operational_alerts": [{"code": "old"}]}),
                encoding="utf-8",
            )

            updated = persist_operational_alerts(
                metadata_path,
                {"trained_through": "2026-03-29", "operational_alerts": [{"code": "old"}]},
                [{"code": "weather_forecast_unavailable", "message": "rerun later"}],
            )

            self.assertEqual(updated["operational_alerts"], [{"code": "weather_forecast_unavailable", "message": "rerun later"}])
            written = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(written["operational_alerts"], [{"code": "weather_forecast_unavailable", "message": "rerun later"}])

    def test_dashboard_builder_exports_operational_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "out"
            metadata_path = base / "model_metadata.json"
            current_path.write_text("[]", encoding="utf-8")
            history_path.write_text("[]", encoding="utf-8")
            metadata_path.write_text(
                json.dumps(
                    {
                        "trained_through": "2026-03-29",
                        "model_family": "logistic",
                        "feature_profile": "live_plus",
                        "operational_alerts": [
                            {
                                "kind": "warning",
                                "code": "weather_forecast_unavailable",
                                "title": "Weather data incomplete",
                                "message": "Rerun later for weather-refreshed picks.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
                model_metadata_path=metadata_path,
            )

            payload = json.loads((output_dir / "dashboard.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["operational_alerts"]), 1)
            self.assertEqual(payload["operational_alerts"][0]["code"], "weather_forecast_unavailable")


if __name__ == "__main__":
    unittest.main()
