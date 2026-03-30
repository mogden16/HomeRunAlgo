from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import build_dashboard_artifacts


class DashboardArtifactTests(unittest.TestCase):
    def test_dashboard_payload_removes_top_k_and_adds_date_filtered_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "dashboard"
            model_data_path = base / "model_training_dataset.csv"
            metadata_path = base / "model_metadata.json"

            current_rows = [
                self._pending_pick("2026-03-30", 1, "Alpha", "elite", 72.0),
                self._pending_pick("2026-03-30", 2, "Bravo", "strong", 61.0),
                self._pending_pick("2026-03-30", 3, "Charlie", "watch", 54.0),
            ]
            history_rows = [
                self._settled_pick("2026-03-29", 1, "Yesterday HR", "elite", 80.0, "HR"),
                self._settled_pick("2026-03-29", 2, "Yesterday Miss", "strong", 58.0, "No HR"),
                self._settled_pick("2026-03-28", 1, "Earlier HR", "strong", 66.0, "HR"),
            ]
            current_path.write_text(json.dumps(current_rows, indent=2), encoding="utf-8")
            history_path.write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_family": "logistic",
                        "feature_profile": "live_shrunk_precise",
                        "feature_columns": ["hr_per_pa_last_30d_shrunk"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            dataset_df = pd.DataFrame(
                [
                    self._season_row("Slugger A", "NYY", 1, "2026-03-27", 2, 18, 101),
                    self._season_row("Slugger A", "NYY", 1, "2026-03-28", 1, 22, 102),
                    self._season_row("Slugger B", "LAD", 2, "2026-03-27", 2, 20, 201),
                    self._season_row("Slugger C", "ATL", 3, "2026-03-27", 1, 23, 301),
                    self._season_row("Slugger D", "PHI", 4, "2026-03-27", 1, 21, 401),
                    self._season_row("Slugger E", "CHC", 5, "2026-03-27", 1, 19, 501),
                    self._season_row("Slugger F", "SEA", 6, "2026-03-27", 0, 17, 601),
                    self._season_row("Prior Year", "BOS", 7, "2025-09-20", 9, 90, 701),
                ]
            )
            dataset_df.to_csv(model_data_path, index=False)

            with patch("scripts.build_dashboard_artifacts.eastern_yesterday", return_value="2026-03-29"):
                output_path = build_dashboard_artifacts.build_dashboard_artifacts(
                    current_picks_path=current_path,
                    history_path=history_path,
                    output_dir=output_dir,
                    model_data_path=model_data_path,
                    model_metadata_path=metadata_path,
                    persist_history=False,
                    latest_count=1,
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertNotIn("top_k_summary", payload)
            self.assertEqual([row["batter_name"] for row in payload["latest_picks"]], ["Alpha", "Bravo", "Charlie"])
            self.assertEqual(payload["history_dates"], ["2026-03-30", "2026-03-29", "2026-03-28"])
            self.assertEqual(payload["history_default_date"], "2026-03-29")
            self.assertEqual(payload["yesterday_homer_date"], "2026-03-29")
            self.assertEqual([row["batter_name"] for row in payload["recent_successes"]], ["Yesterday HR"])
            self.assertEqual([row["batter_name"] for row in payload["season_hr_leaders_2026"]], ["Slugger A", "Slugger B", "Slugger C", "Slugger D", "Slugger E"])
            self.assertEqual(payload["season_hr_leaders_2026"][0]["home_runs_2026"], 3)
            self.assertEqual(payload["season_hr_leaders_2026"][0]["plate_appearances_2026"], 40)

    def test_history_default_date_falls_back_to_latest_when_yesterday_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "dashboard"
            metadata_path = base / "model_metadata.json"
            model_data_path = base / "dataset.csv"

            current_path.write_text(json.dumps([], indent=2), encoding="utf-8")
            history_path.write_text(
                json.dumps(
                    [
                        self._settled_pick("2026-03-27", 1, "Alpha", "elite", 75.0, "HR"),
                        self._settled_pick("2026-03-26", 2, "Bravo", "strong", 60.0, "No HR"),
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            metadata_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
            pd.DataFrame([self._season_row("Slugger A", "NYY", 1, "2026-03-27", 1, 4, 1)]).to_csv(model_data_path, index=False)

            with patch("scripts.build_dashboard_artifacts.eastern_yesterday", return_value="2026-03-29"):
                output_path = build_dashboard_artifacts.build_dashboard_artifacts(
                    current_picks_path=current_path,
                    history_path=history_path,
                    output_dir=output_dir,
                    model_data_path=model_data_path,
                    model_metadata_path=metadata_path,
                    persist_history=False,
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["history_default_date"], "2026-03-27")
            self.assertEqual(payload["recent_successes"], [])

    @staticmethod
    def _pending_pick(game_date: str, rank: int, batter_name: str, tier: str, score: float) -> dict[str, object]:
        return {
            "game_date": game_date,
            "game_pk": 1000 + rank,
            "rank": rank,
            "batter_id": 2000 + rank,
            "batter_name": batter_name,
            "team": "NYY",
            "opponent_team": "BOS",
            "pitcher_id": 3000 + rank,
            "pitcher_name": f"Pitcher {rank}",
            "confidence_tier": tier,
            "predicted_hr_probability": round(score / 100.0, 3),
            "predicted_hr_score": score,
            "top_reason_1": "Recent power",
            "top_reason_2": "Favorable weather",
            "top_reason_3": "Platoon edge",
            "result": "Pending",
        }

    @staticmethod
    def _settled_pick(game_date: str, rank: int, batter_name: str, tier: str, score: float, result: str) -> dict[str, object]:
        row = DashboardArtifactTests._pending_pick(game_date, rank, batter_name, tier, score)
        row["result"] = result
        return row

    @staticmethod
    def _season_row(
        batter_name: str,
        team: str,
        batter_id: int,
        game_date: str,
        hr_count: int,
        pa_count: int,
        game_pk: int,
    ) -> dict[str, object]:
        return {
            "game_date": game_date,
            "game_pk": game_pk,
            "batter_id": batter_id,
            "batter_name": batter_name,
            "team": team,
            "hr_count": hr_count,
            "pa_count": pa_count,
        }


if __name__ == "__main__":
    unittest.main()
