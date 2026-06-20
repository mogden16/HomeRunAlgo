#!/usr/bin/env python
"""Promote a leakage-safe prediction-audit winner into a live model bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import LIVE_MODEL_BUNDLE_PATH, LIVE_MODEL_DATA_PATH, LIVE_MODEL_METADATA_PATH
from scripts.live_pipeline import train_live_model_bundle
from train_model import DEFAULT_CONFIDENCE_POLICY, feature_columns_for_profile, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=LIVE_MODEL_DATA_PATH)
    parser.add_argument("--bundle-path", type=Path, default=LIVE_MODEL_BUNDLE_PATH)
    parser.add_argument("--metadata-path", type=Path, default=LIVE_MODEL_METADATA_PATH)
    return parser.parse_args()


def validate_audit_for_promotion(report: dict[str, Any]) -> dict[str, Any]:
    selected = report.get("selected_model")
    temporal = report.get("temporal_contract")
    if not isinstance(selected, dict) or not isinstance(temporal, dict):
        raise ValueError("Audit report is missing selected_model or temporal_contract.")
    if not bool(selected.get("release_eligible")):
        raise ValueError("Selected model is not release-eligible under the as-of leakage audit.")
    if bool(temporal.get("selection_uses_final_evaluation")):
        raise ValueError("Audit model selection used the final evaluation period.")
    if not bool(temporal.get("threshold_is_frozen_before_final_evaluation")):
        raise ValueError("Audit threshold was not frozen before final evaluation.")
    final_evaluation = selected.get("final_evaluation")
    if not isinstance(final_evaluation, dict):
        raise ValueError("Selected model is missing final_evaluation.")
    positive = final_evaluation.get("positive_call_summary")
    if not isinstance(positive, dict) or positive.get("threshold_source") != "frozen_from_selection_period":
        raise ValueError("Final positive-call threshold is not frozen from the selection period.")
    threshold_name = str(positive.get("selected_threshold_strategy") or "")
    if not threshold_name.startswith("probability_at_least_"):
        raise ValueError("Selected positive-call threshold is missing or malformed.")
    threshold = float(threshold_name.removeprefix("probability_at_least_"))
    if int(positive.get("bets") or 0) < int(positive.get("minimum_bets_required") or 0):
        raise ValueError("Final positive-call result does not meet its minimum sample requirement.")
    if int(positive.get("slate_days") or 0) < int(positive.get("minimum_slate_days_required") or 0):
        raise ValueError("Final positive-call result does not meet its minimum slate-day requirement.")
    interval = positive.get("precision_confidence_interval")
    if not isinstance(interval, dict) or int(interval.get("iterations") or 0) <= 0:
        raise ValueError("Final positive-call precision is missing a bootstrap confidence interval.")
    return {
        "model_name": str(selected["model_name"]),
        "feature_profile": str(selected["feature_profile"]),
        "positive_call_threshold": threshold,
        "selection_years": list(temporal.get("model_and_threshold_selection_period") or []),
        "final_test_years": list(temporal.get("untouched_final_evaluation_period") or []),
        "final_metrics": final_evaluation.get("probability_metrics") or {},
        "positive_call_summary": positive,
        "ranking_summary": final_evaluation.get("ranking_summary") or {},
    }


def promote_prediction_audit(
    *,
    audit_path: Path,
    dataset_path: Path,
    bundle_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    approved = validate_audit_for_promotion(report)
    dataset = load_data(str(dataset_path))
    feature_columns = [column for column in feature_columns_for_profile(approved["feature_profile"]) if column in dataset.columns]
    if not feature_columns:
        raise ValueError("Approved feature profile has no columns in the training dataset.")

    confidence_policy = dict(DEFAULT_CONFIDENCE_POLICY)
    confidence_policy["elite_probability_floor"] = float(approved["positive_call_threshold"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "model_family": approved["model_name"],
                "feature_profile": approved["feature_profile"],
                "feature_profile_variant": approved["feature_profile"],
                "feature_columns": feature_columns,
                "missingness_threshold": 0.50,
                "selection_metric": "pr_auc",
                "best_params": {},
                "calibration_status": {"used": "not_applicable"},
                "confidence_policy": confidence_policy,
                "prediction_audit": approved,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle = train_live_model_bundle(
        dataset_path=dataset_path,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        model_name=approved["model_name"],
        calibration="disabled",
        feature_profile=approved["feature_profile"],
        selection_metric="pr_auc",
        missingness_threshold=0.50,
        training_mode="fast_refit",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["prediction_audit"] = approved
    metadata["positive_call_threshold"] = float(approved["positive_call_threshold"])
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return bundle


def main() -> None:
    args = parse_args()
    bundle = promote_prediction_audit(
        audit_path=args.audit_path,
        dataset_path=args.dataset_path,
        bundle_path=args.bundle_path,
        metadata_path=args.metadata_path,
    )
    print(
        "Promoted prediction audit winner: "
        f"{bundle['model_family']} / {bundle['feature_profile']} through {bundle['trained_through']}"
    )


if __name__ == "__main__":
    main()
