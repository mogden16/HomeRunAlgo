from __future__ import annotations

import unittest
from pathlib import Path

from scripts.verify_public_live_artifacts import verify_migrating_columns


class PublicArtifactVerifierTests(unittest.TestCase):
    def test_allows_only_legacy_prediction_columns_to_be_missing(self) -> None:
        expected = {"pick_id", "positive_call_threshold", "positive_hr_call"}
        verify_migrating_columns({"pick_id"}, expected, Path("artifact.json"), "pick")

        with self.assertRaisesRegex(AssertionError, "missing required"):
            verify_migrating_columns(
                {"positive_hr_call"},
                expected,
                Path("artifact.json"),
                "pick",
            )

    def test_rejects_unknown_columns_during_migration(self) -> None:
        with self.assertRaisesRegex(AssertionError, "unexpected"):
            verify_migrating_columns(
                {"pick_id", "unknown"},
                {"pick_id"},
                Path("artifact.json"),
                "pick",
            )


if __name__ == "__main__":
    unittest.main()
