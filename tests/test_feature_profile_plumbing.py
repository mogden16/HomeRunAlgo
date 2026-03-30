from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import live_pipeline
from scripts import prepare_live_board, train_live_model
import train_model


class FeatureProfilePlumbingTests(unittest.TestCase):
    def test_train_model_profile_choices_include_shrunk_variants(self) -> None:
        self.assertIn("live_shrunk", train_model.FEATURE_PROFILE_CHOICES)
        self.assertIn("live_shrunk_precise", train_model.FEATURE_PROFILE_CHOICES)

    def test_feature_columns_for_profile_returns_shrunk_sets(self) -> None:
        self.assertEqual(
            train_model.feature_columns_for_profile("live_shrunk"),
            list(train_model.LIVE_SHRUNK_FEATURE_COLUMNS),
        )
        self.assertEqual(
            train_model.feature_columns_for_profile("live_shrunk_precise"),
            list(train_model.LIVE_SHRUNK_PRECISE_FEATURE_COLUMNS),
        )

    def test_resolve_feature_profiles_all_includes_shrunk_variants(self) -> None:
        self.assertEqual(
            train_model.resolve_feature_profiles("all"),
            ["stable", "live", "live_plus", "live_shrunk", "live_shrunk_precise", "expanded"],
        )

    def test_live_training_cli_accepts_live_shrunk_precise(self) -> None:
        with patch.object(sys, "argv", ["train_live_model.py", "--feature-profile", "live_shrunk_precise"]):
            args = train_live_model.parse_args()
        self.assertEqual(args.feature_profile, "live_shrunk_precise")

    def test_prepare_live_board_cli_accepts_live_shrunk(self) -> None:
        with patch.object(sys, "argv", ["prepare_live_board.py", "--feature-profile", "live_shrunk"]):
            args = prepare_live_board.parse_args()
        self.assertEqual(args.feature_profile, "live_shrunk")

    def test_live_entrypoints_default_to_live_shrunk(self) -> None:
        with patch.object(sys, "argv", ["train_live_model.py"]):
            train_args = train_live_model.parse_args()
        with patch.object(sys, "argv", ["prepare_live_board.py"]):
            prepare_args = prepare_live_board.parse_args()

        self.assertEqual(train_args.feature_profile, "live_shrunk")
        self.assertEqual(prepare_args.feature_profile, "live_shrunk")

    def test_fast_refit_prefers_requested_profile_over_existing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "dataset.csv"
            metadata_path = base / "metadata.json"
            bundle_path = base / "bundle.pkl"
            pd.DataFrame(
                [
                    {
                        "game_date": "2026-03-29",
                        "hit_hr": 0,
                        "temperature_f": 65.0,
                        "wind_speed_mph": 8.0,
                        "humidity_pct": 45.0,
                    },
                    {
                        "game_date": "2026-03-30",
                        "hit_hr": 1,
                        "temperature_f": 68.0,
                        "wind_speed_mph": 10.0,
                        "humidity_pct": 50.0,
                    },
                ]
            ).to_csv(dataset_path, index=False)
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_family": "xgboost",
                        "feature_profile": "live_plus",
                        "missingness_threshold": 0.5,
                        "selection_metric": "pr_auc",
                    }
                ),
                encoding="utf-8",
            )

            captured: dict[str, object] = {}

            def fake_fit(df: pd.DataFrame, **kwargs: object):
                captured.update(kwargs)
                return (
                    {
                        "trained_through": "2026-03-30",
                        "dataset_max_game_date": "2026-03-30",
                        "model_family": kwargs["model_family"],
                        "feature_profile": kwargs["feature_profile"],
                        "missingness_threshold": kwargs["missingness_threshold"],
                        "selection_metric": kwargs["selection_metric"],
                        "feature_columns": [],
                        "best_params": {},
                        "excluded_features": [],
                        "training_cv_summary": {},
                        "final_holdout_summary": {},
                        "promotion_decision": {},
                    },
                    {
                        "trained_through": "2026-03-30",
                        "dataset_max_game_date": "2026-03-30",
                        "model_family": kwargs["model_family"],
                        "feature_profile": kwargs["feature_profile"],
                    },
                )

            with patch("scripts.live_pipeline.print_weather_join_contract"):
                with patch("scripts.live_pipeline.audit_weather_feature_coverage", return_value={"summary": {}}):
                    with patch("scripts.live_pipeline._weather_coverage_summary_payload", return_value={"summary": {}}):
                        with patch("scripts.live_pipeline._live_publish_weather_contract_label", return_value="contract"):
                            with patch("scripts.live_pipeline._fit_live_bundle_fast_refit", side_effect=fake_fit):
                                with patch("scripts.live_pipeline._persist_live_bundle"):
                                    live_pipeline.train_live_model_bundle(
                                        dataset_path,
                                        bundle_path=bundle_path,
                                        metadata_path=metadata_path,
                                        model_name="logistic",
                                        feature_profile="live_shrunk",
                                        training_mode="fast_refit",
                                    )

            self.assertEqual(captured["model_family"], "logistic")
            self.assertEqual(captured["feature_profile"], "live_shrunk")


if __name__ == "__main__":
    unittest.main()
