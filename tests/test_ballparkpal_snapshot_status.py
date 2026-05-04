from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from scripts.live_pipeline import enrich_candidate_frame_with_ballparkpal


class BallparkPalSnapshotStatusTests(unittest.TestCase):
    def test_unavailable_snapshot_clears_stale_ballparkpal_fields(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "game_date": "2026-05-03",
                    "game_pk": 123,
                    "batter_id": 456,
                    "predicted_hr_score": 77.7,
                    "ballparkpal_home_run_probability": 0.21,
                    "ballparkpal_hit_probability": 0.64,
                    "ballparkpal_team_home_runs": 1.2,
                    "ballparkpal_runs_allowed": 3.4,
                    "ballparkpal_home_runs_allowed": 0.9,
                    "ballparkpal_overlay_raw_score": 21.0,
                }
            ]
        )
        with patch("scripts.live_pipeline.load_ballparkpal_snapshot", return_value=None):
            enriched = enrich_candidate_frame_with_ballparkpal(frame, schedule_date="2026-05-04")

        row = enriched.iloc[0]
        self.assertEqual(row["ballparkpal_snapshot_status"], "unavailable")
        self.assertTrue(pd.isna(row["ballparkpal_home_run_probability"]))
        self.assertTrue(pd.isna(row["ballparkpal_hit_probability"]))
        self.assertTrue(pd.isna(row["ballparkpal_team_home_runs"]))
        self.assertTrue(pd.isna(row["ballparkpal_runs_allowed"]))
        self.assertTrue(pd.isna(row["ballparkpal_home_runs_allowed"]))
        self.assertTrue(pd.isna(row["ballparkpal_overlay_raw_score"]))
        self.assertTrue(pd.isna(row["ballparkpal_overlay_display_score"]))
        self.assertAlmostEqual(float(row["ballparkpal_overlay_adjusted_score"]), 77.7)


if __name__ == "__main__":
    unittest.main()

