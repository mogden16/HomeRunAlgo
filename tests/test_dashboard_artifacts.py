from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import build_dashboard_artifacts


class DashboardArtifactTests(unittest.TestCase):
    def test_refresh_schedule_matches_five_minute_runtime_cadence(self) -> None:
        schedule = build_dashboard_artifacts.build_refresh_schedule()
        run_labels = [run["time_et"] for run in schedule["runs"]]

        self.assertEqual(run_labels[1], "Every 5 minutes until last first pitch")
        self.assertEqual(run_labels[2], "Every 5 minutes in-game")

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
            self.assertEqual(payload["history_dates"], ["2026-03-29"])
            self.assertEqual(payload["history_default_date"], "2026-03-29")
            self.assertEqual([row["batter_name"] for row in payload["history"]], ["Yesterday HR", "Yesterday Miss"])
            self.assertEqual(payload["yesterday_homer_date"], "2026-03-29")
            self.assertEqual([row["batter_name"] for row in payload["recent_successes"]], ["Yesterday HR"])
            self.assertEqual([row["batter_name"] for row in payload["season_hr_leaders_2026"]], ["Slugger A", "Slugger B", "Slugger C", "Slugger D", "Slugger E"])
            self.assertEqual(payload["season_hr_leaders_2026"][0]["home_runs_2026"], 3)
            self.assertEqual(payload["season_hr_leaders_2026"][0]["plate_appearances_2026"], 40)

    def test_dashboard_payload_keeps_same_day_resolved_rows_in_latest_and_out_of_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "dashboard"
            metadata_path = base / "model_metadata.json"
            model_data_path = base / "dataset.csv"

            current_rows = [
                self._settled_pick("2026-03-31", 1, "Alpha", "elite", 72.0, "HR"),
                self._pending_pick("2026-03-31", 2, "Bravo", "strong", 61.0),
            ]
            history_rows = [
                self._settled_pick("2026-03-30", 1, "Yesterday HR", "elite", 80.0, "HR"),
                self._settled_pick("2026-03-31", 3, "Should Hide", "watch", 40.0, "No HR"),
            ]
            current_path.write_text(json.dumps(current_rows, indent=2), encoding="utf-8")
            history_path.write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
            metadata_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
            pd.DataFrame([self._season_row("Slugger A", "NYY", 1, "2026-03-27", 1, 4, 1)]).to_csv(model_data_path, index=False)

            output_path = build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
                model_data_path=model_data_path,
                model_metadata_path=metadata_path,
                persist_history=False,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([row["batter_name"] for row in payload["latest_picks"]], ["Alpha", "Bravo"])
            self.assertEqual([row["batter_name"] for row in payload["history"]], ["Yesterday HR"])
            self.assertEqual(payload["overview"]["settled_picks"], 2)
            self.assertEqual(payload["overview"]["open_picks"], 1)

    def test_dashboard_payload_recovers_pending_history_rows_into_latest(self) -> None:
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
                        self._pending_pick("2026-03-31", 1, "Alpha", "elite", 72.0),
                        self._settled_pick("2026-03-30", 1, "Yesterday HR", "elite", 80.0, "HR"),
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            metadata_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
            pd.DataFrame([self._season_row("Slugger A", "NYY", 1, "2026-03-27", 1, 4, 1)]).to_csv(model_data_path, index=False)

            output_path = build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
                model_data_path=model_data_path,
                model_metadata_path=metadata_path,
                persist_history=False,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([row["batter_name"] for row in payload["latest_picks"]], ["Alpha"])
            self.assertEqual([row["batter_name"] for row in payload["history"]], ["Yesterday HR"])
            self.assertEqual(payload["overview"]["open_picks"], 1)

    def test_dashboard_payload_preserves_latest_game_meta_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "dashboard"
            metadata_path = base / "model_metadata.json"
            model_data_path = base / "dataset.csv"

            current_row = self._pending_pick("2026-04-01", 1, "Alpha", "elite", 72.0)
            current_row.update(
                {
                    "game_datetime": "2026-04-01T19:07:00Z",
                    "ballpark_name": "Rogers Centre",
                    "ballpark_region_abbr": "ON",
                    "weather_code": 3,
                    "weather_label": "Cloudy",
                    "temperature_f": 57.0,
                    "wind_speed_mph": 12.0,
                    "wind_direction_deg": 210.0,
                    "field_bearing_deg": 30.0,
                }
            )
            current_path.write_text(json.dumps([current_row], indent=2), encoding="utf-8")
            history_path.write_text(json.dumps([], indent=2), encoding="utf-8")
            metadata_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
            pd.DataFrame([self._season_row("Slugger A", "NYY", 1, "2026-03-27", 1, 4, 1)]).to_csv(model_data_path, index=False)

            output_path = build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
                model_data_path=model_data_path,
                model_metadata_path=metadata_path,
                persist_history=False,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            latest = payload["latest_picks"][0]
            self.assertEqual(latest["ballpark_name"], "Rogers Centre")
            self.assertEqual(latest["ballpark_region_abbr"], "ON")
            self.assertEqual(latest["weather_code"], 3)
            self.assertEqual(latest["weather_label"], "Cloudy")
            self.assertEqual(latest["temperature_f"], 57.0)
            self.assertEqual(latest["wind_speed_mph"], 12.0)
            self.assertEqual(latest["wind_direction_deg"], 210.0)
            self.assertEqual(latest["field_bearing_deg"], 30.0)

    def test_dashboard_payload_infers_roofed_parks_from_ballpark_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "dashboard"
            metadata_path = base / "model_metadata.json"
            model_data_path = base / "dataset.csv"

            current_row = self._pending_pick("2026-05-02", 1, "Alpha", "elite", 72.0)
            current_row.update(
                {
                    "ballpark_name": "loanDepot park",
                    "ballpark_region_abbr": "FL",
                    "weather_code": 0,
                    "weather_label": "Clear",
                    "temperature_f": 89.1,
                    "wind_speed_mph": 13.2,
                    "wind_direction_deg": 242.0,
                    "field_bearing_deg": 37.0,
                }
            )
            current_path.write_text(json.dumps([current_row], indent=2), encoding="utf-8")
            history_path.write_text(json.dumps([], indent=2), encoding="utf-8")
            metadata_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
            pd.DataFrame([self._season_row("Slugger A", "MIA", 1, "2026-05-01", 1, 4, 1)]).to_csv(
                model_data_path,
                index=False,
            )

            output_path = build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
                model_data_path=model_data_path,
                model_metadata_path=metadata_path,
                persist_history=False,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            latest = payload["latest_picks"][0]
            self.assertTrue(latest["roofed_park"])
            self.assertEqual(latest["roof_type"], "retractable_roof")
            self.assertEqual(latest["roof_label"], "Retractable roof")
            self.assertIsNone(latest["weather_code"])
            self.assertEqual(latest["weather_label"], "Retractable roof")
            self.assertIsNone(latest["temperature_f"])
            self.assertIsNone(latest["wind_speed_mph"])
            self.assertIsNone(latest["wind_direction_deg"])

    @patch("scripts.build_dashboard_artifacts._fetch_current_season_hitting_totals_by_player_id")
    def test_dashboard_payload_uses_live_season_hr_source(self, mock_fetch: object) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "dashboard"
            metadata_path = base / "model_metadata.json"
            model_data_path = base / "dataset.csv"

            current_rows = [self._pending_pick("2026-04-29", 1, "Juan Soto", "elite", 98.0)]
            current_rows[0]["batter_id"] = 665742
            current_rows[0]["team"] = "NYM"
            current_rows[0]["opponent_team"] = "WSH"
            current_path.write_text(json.dumps(current_rows, indent=2), encoding="utf-8")
            history_path.write_text(json.dumps([], indent=2), encoding="utf-8")
            metadata_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
            pd.DataFrame([self._season_row("Juan Soto", "NYM", 665742, "2026-04-28", 1, 14, 9001)]).to_csv(
                model_data_path,
                index=False,
            )

            mock_fetch.return_value = {
                "665742": {
                    "batter_name": "Juan Soto",
                    "team": "NYM",
                    "home_runs_2026": 2,
                    "plate_appearances_2026": 59,
                    "games_2026": 14,
                }
            }

            output_path = build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
                model_data_path=model_data_path,
                model_metadata_path=metadata_path,
                persist_history=False,
                season_hr_source="live",
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["latest_picks"][0]["home_runs_2026"], 2)
            self.assertEqual(payload["season_hr_leaders_2026"][0]["home_runs_2026"], 2)

    @patch("scripts.build_dashboard_artifacts._fetch_current_season_hitting_totals_by_player_id")
    def test_live_season_hr_leaders_fall_back_to_dataset_when_lookup_empty(self, mock_fetch: object) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            model_data_path = base / "dataset.csv"
            pd.DataFrame(
                [
                    self._season_row("Slugger A", "NYY", 1, "2026-04-28", 3, 14, 9001),
                    self._season_row("Slugger B", "LAD", 2, "2026-04-28", 2, 12, 9002),
                ]
            ).to_csv(model_data_path, index=False)
            mock_fetch.return_value = {}

            leaders = build_dashboard_artifacts.build_season_hr_leaders_2026(
                model_data_path,
                source="live",
                season_year=2026,
            )

            self.assertEqual([row["batter_name"] for row in leaders], ["Slugger A", "Slugger B"])
            self.assertEqual(leaders[0]["home_runs_2026"], 3)

    @patch("scripts.build_dashboard_artifacts._fetch_current_season_hitting_totals_by_player_id")
    def test_dashboard_preserves_previous_season_leaders_when_live_lookup_empty(self, mock_fetch: object) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "dashboard"
            metadata_path = base / "model_metadata.json"
            model_data_path = base / "dataset.csv"

            current_path.write_text(json.dumps([self._pending_pick("2026-04-29", 1, "Alpha", "elite", 98.0)], indent=2), encoding="utf-8")
            history_path.write_text(json.dumps([], indent=2), encoding="utf-8")
            metadata_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
            pd.DataFrame([{"game_date": "2025-09-20", "batter_id": 1, "batter_name": "Old Row", "team": "NYY"}]).to_csv(model_data_path, index=False)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dashboard.json").write_text(
                json.dumps(
                    {
                        "season_hr_leaders_2026": [
                            {
                                "batter_name": "Existing Leader",
                                "team": "PHI",
                                "home_runs_2026": 20,
                                "plate_appearances_2026": 213,
                                "games_2026": 47,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            mock_fetch.return_value = {}

            output_path = build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
                model_data_path=model_data_path,
                model_metadata_path=metadata_path,
                persist_history=False,
                season_hr_source="live",
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["season_hr_leaders_2026"][0]["batter_name"], "Existing Leader")

    def test_dashboard_payload_uses_morning_snapshot_order_and_keeps_morning_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            draft_path = base / "draft.json"
            output_dir = base / "dashboard"
            metadata_path = base / "model_metadata.json"
            model_data_path = base / "dataset.csv"

            current_rows = [
                self._pending_pick("2026-04-01", 4, "Charlie", "strong", 92.0),
                self._pending_pick("2026-04-01", 2, "Alpha", "elite", 98.0),
                self._pending_pick("2026-04-01", 7, "Bravo", "elite", 95.0),
            ]
            draft_rows = [
                self._pending_pick("2026-04-01", 1, "Bravo", "elite", 95.0),
                self._pending_pick("2026-04-01", 2, "Alpha", "elite", 98.0),
                self._pending_pick("2026-04-01", 3, "Charlie", "strong", 92.0),
            ]
            draft_rows[0]["game_pk"] = current_rows[2]["game_pk"]
            draft_rows[0]["batter_id"] = current_rows[2]["batter_id"]
            draft_rows[0]["pitcher_id"] = current_rows[2]["pitcher_id"]
            draft_rows[1]["game_pk"] = current_rows[1]["game_pk"]
            draft_rows[1]["batter_id"] = current_rows[1]["batter_id"]
            draft_rows[1]["pitcher_id"] = current_rows[1]["pitcher_id"]
            draft_rows[2]["game_pk"] = current_rows[0]["game_pk"]
            draft_rows[2]["batter_id"] = current_rows[0]["batter_id"]
            draft_rows[2]["pitcher_id"] = current_rows[0]["pitcher_id"]
            current_path.write_text(json.dumps(current_rows, indent=2), encoding="utf-8")
            history_path.write_text(json.dumps([], indent=2), encoding="utf-8")
            draft_path.write_text(json.dumps(draft_rows, indent=2), encoding="utf-8")
            metadata_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
            pd.DataFrame([self._season_row("Slugger A", "NYY", 1, "2026-03-27", 1, 4, 1)]).to_csv(model_data_path, index=False)

            output_path = build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                draft_picks_path=draft_path,
                output_dir=output_dir,
                model_data_path=model_data_path,
                model_metadata_path=metadata_path,
                persist_history=False,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            latest = payload["latest_picks"]
            self.assertEqual([row["batter_name"] for row in latest], ["Bravo", "Alpha", "Charlie"])
            self.assertEqual([row["rank"] for row in latest], [1, 2, 3])
            self.assertEqual([row["original_rank"] for row in latest], [1, 2, 3])
            self.assertEqual([row["morning_rank"] for row in latest], [1, 2, 3])

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

    def test_dashboard_history_keeps_previous_day_public_payload_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "dashboard"
            metadata_path = base / "model_metadata.json"
            model_data_path = base / "dataset.csv"

            current_path.write_text(json.dumps([], indent=2), encoding="utf-8")
            history_rows = [
                self._settled_pick("2026-03-27", 1, "Alpha", "elite", 90.0, "HR"),
                self._settled_pick("2026-03-27", 2, "Bravo", "strong", 80.0, "No HR"),
                self._settled_pick("2026-03-27", 3, "Charlie", "watch", 70.0, "No HR"),
                self._settled_pick("2026-03-26", 1, "Delta", "elite", 85.0, "HR"),
                self._settled_pick("2026-03-26", 2, "Echo", "strong", 75.0, "No HR"),
                self._settled_pick("2026-03-26", 3, "Foxtrot", "watch", 65.0, "No HR"),
            ]
            history_path.write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
            metadata_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
            pd.DataFrame([self._season_row("Slugger A", "NYY", 1, "2026-03-27", 1, 4, 1)]).to_csv(model_data_path, index=False)

            output_path = build_dashboard_artifacts.build_dashboard_artifacts(
                current_picks_path=current_path,
                history_path=history_path,
                output_dir=output_dir,
                model_data_path=model_data_path,
                model_metadata_path=metadata_path,
                persist_history=False,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([row["batter_name"] for row in payload["history"]], ["Alpha", "Bravo"])
            self.assertEqual(payload["history_dates"], ["2026-03-27"])
            self.assertEqual(payload["previous_day_date"], "2026-03-27")
            self.assertEqual(payload["overview"]["tracked_picks"], 6)
            self.assertEqual(payload["overview"]["settled_picks"], 6)
            self.assertLess(output_path.read_text(encoding="utf-8").count("\n"), 2)

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
