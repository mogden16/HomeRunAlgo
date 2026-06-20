from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import promote_prediction_audit as promotion


def valid_report() -> dict[str, object]:
    positive = {
        "threshold_source": "frozen_from_selection_period",
        "selected_threshold_strategy": "probability_at_least_0.20",
        "minimum_bets_required": 25,
        "minimum_slate_days_required": 10,
        "bets": 106,
        "slate_days": 32,
        "precision_confidence_interval": {"iterations": 1000, "lower": 0.19, "upper": 0.40},
    }
    return {
        "temporal_contract": {
            "model_and_threshold_selection_period": [2025],
            "untouched_final_evaluation_period": [2026],
            "selection_uses_final_evaluation": False,
            "threshold_is_frozen_before_final_evaluation": True,
        },
        "selected_model": {
            "model_name": "histgb",
            "feature_profile": "pregame_safe_v1",
            "release_eligible": True,
            "final_evaluation": {
                "probability_metrics": {"average_precision": 0.15},
                "positive_call_summary": positive,
                "ranking_summary": {"top_3_daily_hit_rate": 0.30},
            },
        },
    }


class PredictionPromotionTests(unittest.TestCase):
    def test_validate_audit_accepts_frozen_release_eligible_winner(self) -> None:
        approved = promotion.validate_audit_for_promotion(valid_report())
        self.assertEqual(approved["model_name"], "histgb")
        self.assertEqual(approved["feature_profile"], "pregame_safe_v1")
        self.assertEqual(approved["positive_call_threshold"], 0.20)

    def test_validate_audit_rejects_final_period_selection(self) -> None:
        report = valid_report()
        report["temporal_contract"]["selection_uses_final_evaluation"] = True
        with self.assertRaisesRegex(ValueError, "used the final evaluation"):
            promotion.validate_audit_for_promotion(report)

    def test_promote_persists_audit_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            audit_path = base / "audit.json"
            dataset_path = base / "dataset.csv"
            bundle_path = base / "bundle.pkl"
            metadata_path = base / "metadata.json"
            audit_path.write_text(json.dumps(valid_report()), encoding="utf-8")
            dataset_path.write_text("placeholder", encoding="utf-8")
            fake_dataset = pd.DataFrame({"hr_per_pa_last_30d": [0.1]})

            def fake_train(**kwargs):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                return {
                    "model_family": "histgb",
                    "feature_profile": "pregame_safe_v1",
                    "trained_through": "2026-05-02",
                }

            with patch.object(promotion, "load_data", return_value=fake_dataset):
                with patch.object(promotion, "train_live_model_bundle", side_effect=fake_train):
                    promotion.promote_prediction_audit(
                        audit_path=audit_path,
                        dataset_path=dataset_path,
                        bundle_path=bundle_path,
                        metadata_path=metadata_path,
                    )

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["positive_call_threshold"], 0.20)
            self.assertEqual(metadata["prediction_audit"]["selection_years"], [2025])


if __name__ == "__main__":
    unittest.main()
