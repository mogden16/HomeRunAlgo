from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
