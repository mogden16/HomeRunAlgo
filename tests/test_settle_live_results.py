from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import settle_live_results


class SettleLiveResultsTests(unittest.TestCase):
    def test_archives_terminal_rows_even_if_unrelated_game_is_postponed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            current_path = base / "current.json"
            history_path = base / "history.json"

            pd.DataFrame(
                [
                    {"game_date": pd.Timestamp("2026-03-31"), "batter_id": 101, "hit_hr": 1},
                ]
            ).to_csv(dataset_path, index=False)
            current_path.write_text(
                json.dumps(
                    [
                        {
                            "pick_id": "pick-1",
                            "game_pk": 2001,
                            "game_date": "2026-04-03",
                            "batter_id": 102,
                            "batter_name": "Alpha",
                            "pitcher_id": 301,
                            "pitcher_name": "Pitcher A",
                            "result": "Pending",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            history_path.write_text("[]", encoding="utf-8")

            schedule_games = [
                {"game_pk": 2001, "status": "Final", "game_datetime": "2026-04-03T23:00:00Z"},
                {"game_pk": 2999, "status": "Postponed", "game_datetime": "2026-04-03T23:10:00Z"},
            ]

            with patch("scripts.settle_live_results.fetch_schedule_games", return_value=schedule_games):
                result = settle_live_results.run_settle_live_results(
                    dataset_path=dataset_path,
                    current_picks_path=current_path,
                    history_path=history_path,
                )

            self.assertEqual(result["archived_dates"], ["2026-04-03"])
            self.assertEqual(json.loads(current_path.read_text(encoding="utf-8")), [])
            history_rows = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertEqual(len(history_rows), 1)
            self.assertEqual(history_rows[0]["result_label"], "No HR")
            self.assertEqual(history_rows[0]["actual_hit_hr"], 0)


if __name__ == "__main__":
    unittest.main()
