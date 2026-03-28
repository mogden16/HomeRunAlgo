from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import prepare_live_board


class PrepareLiveBoardTests(unittest.TestCase):
    def test_run_prepare_live_board_refreshes_trains_settles_and_writes_private_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            current_path = base / "current.json"
            history_path = base / "history.json"
            draft_path = base / "draft.json"

            call_order: list[str] = []
            dataset_df = pd.DataFrame({"game_date": pd.to_datetime(["2026-03-27"])})

            def fake_load_json_array(path: Path) -> list[dict[str, object]]:
                if path == current_path:
                    call_order.append("load_current")
                else:
                    call_order.append("load_history")
                return []

            def fake_settle(records: list[dict[str, object]], _: pd.DataFrame, *, resolved_through_date: str) -> list[dict[str, object]]:
                call_order.append(f"settle:{resolved_through_date}")
                return records

            draft_rows = [{"game_date": "2026-03-28", "batter_name": "Alpha", "rank": 1}]

            with patch("scripts.prepare_live_board.refresh_live_dataset", side_effect=lambda **_: call_order.append("refresh")):
                with patch("scripts.prepare_live_board.load_live_dataset", side_effect=lambda _: call_order.append("load_dataset") or dataset_df):
                    with patch(
                        "scripts.prepare_live_board.train_live_model_bundle",
                        side_effect=lambda **kwargs: call_order.append(f"train:{kwargs['training_mode']}") or {"trained_through": "2026-03-27"},
                    ):
                        with patch("scripts.prepare_live_board.load_json_array", side_effect=fake_load_json_array):
                            with patch("scripts.prepare_live_board.settle_pick_records", side_effect=fake_settle):
                                with patch("scripts.prepare_live_board.write_current_picks", side_effect=lambda rows, path: call_order.append(f"write_current:{path.name}:{len(rows)}")):
                                    with patch("scripts.prepare_live_board.write_pick_history", side_effect=lambda rows, path: call_order.append(f"write_history:{path.name}:{len(rows)}")):
                                        with patch("scripts.prepare_live_board.generate_live_picks", side_effect=lambda **_: call_order.append("generate_draft") or draft_rows):
                                            picks = prepare_live_board.run_prepare_live_board(
                                                dataset_path=dataset_path,
                                                bundle_path=bundle_path,
                                                metadata_path=metadata_path,
                                                current_picks_path=current_path,
                                                history_path=history_path,
                                                draft_output_path=draft_path,
                                            )

            self.assertEqual(picks, draft_rows)
            self.assertEqual(
                call_order,
                [
                    "refresh",
                    "load_dataset",
                    "train:fast_refit",
                    "load_current",
                    "load_history",
                    "settle:2026-03-27",
                    "settle:2026-03-27",
                    "write_current:current.json:0",
                    "write_history:history.json:0",
                    "generate_draft",
                    "write_current:draft.json:1",
                ],
            )


if __name__ == "__main__":
    unittest.main()
