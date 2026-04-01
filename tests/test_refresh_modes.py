from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from scripts import refresh_modes


class RefreshModesTests(unittest.TestCase):
    def test_resolve_auto_refresh_mode_is_idle_before_6am_without_active_slate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            metadata_path = base / "metadata.json"
            draft_path = base / "draft.json"
            current_path = base / "current.json"
            metadata_path.write_text('{"trained_through": "2026-03-29"}', encoding="utf-8")
            draft_path.write_text("[]", encoding="utf-8")
            current_path.write_text("[]", encoding="utf-8")

            mode = refresh_modes.resolve_auto_refresh_mode(
                current_picks_path=current_path,
                metadata_path=metadata_path,
                draft_output_path=draft_path,
                reference_time=datetime.fromisoformat("2026-03-31T05:45:00-04:00"),
            )

        self.assertEqual(mode, "idle")

    def test_resolve_auto_refresh_mode_runs_settle_before_6am_for_prior_active_slate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            metadata_path = base / "metadata.json"
            draft_path = base / "draft.json"
            current_path = base / "current.json"
            metadata_path.write_text('{"trained_through": "2026-03-30"}', encoding="utf-8")
            draft_path.write_text("[]", encoding="utf-8")
            current_path.write_text('[{"game_date": "2026-03-30", "pick_id": "x"}]', encoding="utf-8")

            with patch("scripts.refresh_modes.fetch_schedule_games", return_value=[{"game_pk": 77}]):
                with patch(
                    "scripts.refresh_modes.build_slate_state",
                    return_value={
                        "games": [{"game_pk": 77, "game_state": "live", "is_final": False}],
                        "all_final": False,
                        "has_live_games": True,
                    },
                ):
                    mode = refresh_modes.resolve_auto_refresh_mode(
                        current_picks_path=current_path,
                        metadata_path=metadata_path,
                        draft_output_path=draft_path,
                        reference_time=datetime.fromisoformat("2026-03-31T05:45:00-04:00"),
                    )

        self.assertEqual(mode, "settle")

    def test_resolve_auto_refresh_mode_runs_prepare_after_6am_when_daily_training_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            metadata_path = base / "metadata.json"
            draft_path = base / "draft.json"
            current_path = base / "current.json"
            metadata_path.write_text('{"trained_through": "2026-03-29"}', encoding="utf-8")
            draft_path.write_text("[]", encoding="utf-8")
            current_path.write_text("[]", encoding="utf-8")

            with patch("scripts.refresh_modes.fetch_schedule_games", return_value=[{"game_pk": 77}]):
                mode = refresh_modes.resolve_auto_refresh_mode(
                    current_picks_path=current_path,
                    metadata_path=metadata_path,
                    draft_output_path=draft_path,
                    reference_time=datetime.fromisoformat("2026-03-31T08:00:00-04:00"),
                )

        self.assertEqual(mode, "prepare")

    def test_resolve_auto_refresh_mode_runs_mixed_before_last_pitch_when_prepare_is_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            metadata_path = base / "metadata.json"
            draft_path = base / "draft.json"
            current_path = base / "current.json"
            metadata_path.write_text('{"trained_through": "2026-03-30"}', encoding="utf-8")
            draft_path.write_text("[]", encoding="utf-8")
            current_path.write_text("[]", encoding="utf-8")

            with patch("scripts.refresh_modes.fetch_schedule_games", return_value=[{"game_pk": 77}]):
                with patch(
                    "scripts.refresh_modes.build_slate_state",
                    return_value={
                        "first_game_datetime": datetime.fromisoformat("2026-03-31T23:10:00+00:00"),
                        "last_game_datetime": datetime.fromisoformat("2026-04-01T01:10:00+00:00"),
                        "games": [{"game_pk": 77, "game_state": "pregame", "is_final": False}],
                        "all_final": False,
                        "has_live_games": False,
                    },
                ):
                    mode = refresh_modes.resolve_auto_refresh_mode(
                        current_picks_path=current_path,
                        metadata_path=metadata_path,
                        draft_output_path=draft_path,
                        reference_time=datetime.fromisoformat("2026-03-31T15:00:00-04:00"),
                    )

        self.assertEqual(mode, "mixed")

    def test_resolve_auto_refresh_mode_runs_settle_after_last_first_pitch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            metadata_path = base / "metadata.json"
            draft_path = base / "draft.json"
            current_path = base / "current.json"
            metadata_path.write_text('{"trained_through": "2026-03-30"}', encoding="utf-8")
            draft_path.write_text('[{"game_date": "2026-03-31"}]', encoding="utf-8")
            current_path.write_text('[{"game_date": "2026-03-31", "pick_id": "x"}]', encoding="utf-8")

            with patch("scripts.refresh_modes.fetch_schedule_games", return_value=[{"game_pk": 77}]):
                with patch(
                    "scripts.refresh_modes.build_slate_state",
                    return_value={
                        "last_game_datetime": datetime.fromisoformat("2026-03-31T23:10:00+00:00"),
                        "games": [{"game_pk": 77, "game_state": "live", "is_final": False}],
                        "all_final": False,
                        "has_live_games": True,
                    },
                ):
                    mode = refresh_modes.resolve_auto_refresh_mode(
                        current_picks_path=current_path,
                        metadata_path=metadata_path,
                        draft_output_path=draft_path,
                        reference_time=datetime.fromisoformat("2026-03-31T20:00:00-04:00"),
                    )

        self.assertEqual(mode, "settle")

    def test_run_refresh_mode_auto_dispatches_resolved_mode(self) -> None:
        with patch("scripts.refresh_modes.resolve_auto_refresh_mode", return_value="settle"):
            with patch("scripts.refresh_modes.run_settle_refresh", return_value={"resolved_through_date": "2026-03-31"}) as settle_mock:
                result = refresh_modes.run_refresh_mode("auto")

        settle_mock.assert_called_once()
        self.assertEqual(result["mode"], "settle")
        self.assertEqual(result["result"]["resolved_through_date"], "2026-03-31")

    def test_run_refresh_mode_auto_returns_idle_without_dispatch(self) -> None:
        with patch("scripts.refresh_modes.resolve_auto_refresh_mode", return_value="idle"):
            result = refresh_modes.run_refresh_mode("auto")

        self.assertEqual(result["mode"], "idle")
        self.assertEqual(result["result"]["status"], "idle")

    def test_run_settle_refresh_refreshes_then_settles_then_builds_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "dashboard"

            call_order: list[str] = []

            with patch("scripts.refresh_modes.default_training_end_date", return_value="2026-03-30"):
                with patch("scripts.refresh_modes.refresh_live_dataset", side_effect=lambda **_: call_order.append("refresh")):
                    with patch(
                        "scripts.refresh_modes.run_settle_live_results",
                        side_effect=lambda **_: call_order.append("settle") or {"resolved_through_date": "2026-03-30"},
                    ):
                        with patch(
                            "scripts.refresh_modes.build_dashboard_artifacts",
                            side_effect=lambda **_: call_order.append("build") or (output_dir / "dashboard.json"),
                        ):
                            with patch("scripts.refresh_modes.verify_public_live_artifacts", side_effect=lambda **_: call_order.append("verify")):
                                result = refresh_modes.run_settle_refresh(
                                    dataset_path=dataset_path,
                                    current_picks_path=current_path,
                                    history_path=history_path,
                                    dashboard_output_dir=output_dir,
                                )

            self.assertEqual(call_order, ["refresh", "settle", "build", "verify"])
            self.assertEqual(result["resolved_through_date"], "2026-03-30")
            self.assertEqual(result["dashboard_path"], output_dir / "dashboard.json")

    def test_run_prepare_refresh_runs_prepare_then_builds_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            baseline_path = base / "baseline.json"
            output_dir = base / "dashboard"

            call_order: list[str] = []
            draft_rows = [{"batter_name": "Alpha", "rank": 1, "game_date": "2026-03-31"}]

            with patch(
                "scripts.refresh_modes.run_prepare_live_board",
                side_effect=lambda **_: call_order.append("prepare") or draft_rows,
            ):
                with patch(
                    "scripts.refresh_modes.build_dashboard_artifacts",
                    side_effect=lambda **_: call_order.append("build") or (output_dir / "dashboard.json"),
                ):
                    with patch("scripts.refresh_modes.verify_public_live_artifacts", side_effect=lambda **_: call_order.append("verify")):
                        result = refresh_modes.run_prepare_refresh(
                            current_picks_path=current_path,
                            history_path=history_path,
                            morning_baseline_path=baseline_path,
                            dashboard_output_dir=output_dir,
                            publish_date="2026-03-31",
                        )

            self.assertEqual(call_order, ["prepare", "build", "verify"])
            self.assertEqual(result, draft_rows)
            self.assertEqual(json.loads(baseline_path.read_text(encoding="utf-8"))[0]["batter_name"], "Alpha")

    def test_run_prepare_refresh_keeps_existing_same_day_morning_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            baseline_path = base / "baseline.json"
            baseline_path.write_text(
                json.dumps([{"batter_name": "Existing", "rank": 1, "game_date": "2026-03-31"}], indent=2),
                encoding="utf-8",
            )

            with patch("scripts.refresh_modes.run_prepare_live_board", return_value=[{"batter_name": "Fresh", "rank": 1, "game_date": "2026-03-31"}]):
                with patch("scripts.refresh_modes.build_dashboard_artifacts", return_value=base / "dashboard.json"):
                    with patch("scripts.refresh_modes.verify_public_live_artifacts"):
                        refresh_modes.run_prepare_refresh(
                            current_picks_path=base / "current.json",
                            history_path=base / "history.json",
                            morning_baseline_path=baseline_path,
                            dashboard_output_dir=base / "dashboard",
                            publish_date="2026-03-31",
                        )

            self.assertEqual(json.loads(baseline_path.read_text(encoding="utf-8"))[0]["batter_name"], "Existing")

    def test_run_mixed_refresh_refreshes_settles_publishes_then_builds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            current_path = base / "current.json"
            history_path = base / "history.json"
            baseline_path = base / "baseline.json"
            output_dir = base / "dashboard"

            call_order: list[str] = []

            with patch("scripts.refresh_modes.refresh_live_dataset", side_effect=lambda **_: call_order.append("refresh")):
                with patch(
                    "scripts.refresh_modes.run_settle_live_results",
                    side_effect=lambda **_: call_order.append("settle") or {"resolved_through_date": "2026-03-31"},
                ):
                    with patch(
                        "scripts.refresh_modes.publish_live_picks",
                        side_effect=lambda **_: call_order.append("publish") or [{"batter_name": "Alpha", "rank": 1}],
                    ):
                        with patch(
                            "scripts.refresh_modes.build_dashboard_artifacts",
                            side_effect=lambda **_: call_order.append("build") or (output_dir / "dashboard.json"),
                        ):
                            with patch("scripts.refresh_modes.verify_public_live_artifacts", side_effect=lambda **_: call_order.append("verify")):
                                result = refresh_modes.run_mixed_refresh(
                                    dataset_path=dataset_path,
                                    current_picks_path=current_path,
                                    history_path=history_path,
                                    morning_baseline_path=baseline_path,
                                    dashboard_output_dir=output_dir,
                                    schedule_date="2026-03-31",
                                )

            self.assertEqual(call_order, ["refresh", "settle", "publish", "build", "verify"])
            self.assertEqual(result["resolved_schedule_date"], "2026-03-31")
            self.assertEqual(result["dashboard_path"], output_dir / "dashboard.json")

    def test_run_publish_refresh_runs_publish_then_builds_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            current_path = base / "current.json"
            history_path = base / "history.json"
            output_dir = base / "dashboard"

            call_order: list[str] = []
            published_rows = [{"batter_name": "Alpha", "rank": 1}]

            with patch(
                "scripts.refresh_modes.publish_live_picks",
                side_effect=lambda **_: call_order.append("publish") or published_rows,
            ):
                with patch(
                    "scripts.refresh_modes.build_dashboard_artifacts",
                    side_effect=lambda **_: call_order.append("build") or (output_dir / "dashboard.json"),
                ):
                    with patch("scripts.refresh_modes.verify_public_live_artifacts", side_effect=lambda **_: call_order.append("verify")):
                        result = refresh_modes.run_publish_refresh(
                            current_picks_path=current_path,
                            history_path=history_path,
                            dashboard_output_dir=output_dir,
                        )

            self.assertEqual(call_order, ["publish", "build", "verify"])
            self.assertEqual(result, published_rows)


if __name__ == "__main__":
    unittest.main()
