"""Compare the current model against Ballpark Pal-augmented feature sets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train_model import (
    CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS,
    compare_metric_direction,
    evaluate_predictions,
    fit_selected_model,
    prune_sparse_features,
    score_frame,
    chronological_split,
)
from tools.ballparkpal.feature_join import (
    BALLPARKPAL_BATTER_FEATURE_COLUMNS,
    BALLPARKPAL_FEATURE_COLUMNS,
    BALLPARKPAL_GAME_FEATURE_COLUMNS,
    BALLPARKPAL_PITCHER_FEATURE_COLUMNS,
    BALLPARKPAL_TEAM_FEATURE_COLUMNS,
    augment_model_dataset_with_ballparkpal,
)


DEFAULT_SOURCE_DATA_PATH = Path("data/live/model_training_dataset.csv")
DEFAULT_ARCHIVE_ROOT = Path("data/ballparkpal/raw")
DEFAULT_OUTPUT_DIR = Path("data/ballparkpal/analysis")

BP_CORE_FEATURE_COLUMNS = [
    "bp_batter_home_run_probability",
    "bp_batter_hit_probability",
    "bp_batter_points_dk",
    "bp_batter_points_fd",
    "bp_pitcher_points_dk",
    "bp_pitcher_runs_allowed",
    "bp_pitcher_strikeouts",
    "bp_team_runs",
    "bp_team_win_pct",
    "bp_game_total_runs",
    "bp_game_run_diff",
    "bp_game_win_pct_gap",
]


@dataclass
class VariantResult:
    name: str
    feature_count: int
    rows_train: int
    rows_test: int
    metrics: dict[str, float]
    compared_to_baseline: dict[str, str]
    ballparkpal_features_used: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-path", default=str(DEFAULT_SOURCE_DATA_PATH), help="Batter-game CSV to score.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT), help="Root of archived Ballpark Pal exports.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where to write analysis artifacts.")
    parser.add_argument("--model", choices=["logistic", "xgboost"], default="logistic", help="Model family to evaluate.")
    parser.add_argument("--max-report-coeffs", type=int, default=20, help="How many Ballpark Pal coefficients to report.")
    return parser.parse_args()


def load_source_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {path}. Generate a matching season dataset first "
            "with generate_data.py."
        )
    df = pd.read_csv(path, parse_dates=["game_date"])
    required = {"game_date", "game_pk", "player_id", "hit_hr", "team", "opponent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source dataset is missing required columns: {sorted(missing)}")
    return df.sort_values(["game_date", "game_pk", "player_id"]).reset_index(drop=True)


def extract_logistic_coefficients(model, feature_columns: list[str]) -> pd.DataFrame:
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
    if not isinstance(clf, LogisticRegression):
        return pd.DataFrame(columns=["feature_name", "coefficient", "abs_coefficient"])
    coefficient_df = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "coefficient": clf.coef_.ravel(),
        }
    )
    coefficient_df["abs_coefficient"] = coefficient_df["coefficient"].abs()
    return coefficient_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def run_variant(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str], model_name: str) -> tuple[dict[str, float], pd.DataFrame, Any]:
    model = fit_selected_model(train_df, feature_columns, model_name=model_name)
    y_prob = score_frame(model, test_df, feature_columns)
    metrics = evaluate_predictions(test_df["hit_hr"].to_numpy(), y_prob)
    return metrics, model, y_prob


def build_variant_feature_sets() -> dict[str, list[str]]:
    return {
        "baseline": CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS,
        "baseline_plus_bp_core": [*CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS, *BP_CORE_FEATURE_COLUMNS],
        "baseline_plus_bp_batters": [*CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS, *BALLPARKPAL_BATTER_FEATURE_COLUMNS],
        "baseline_plus_bp_pitchers": [*CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS, *BALLPARKPAL_PITCHER_FEATURE_COLUMNS],
        "baseline_plus_bp_teams": [*CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS, *BALLPARKPAL_TEAM_FEATURE_COLUMNS],
        "baseline_plus_bp_games": [*CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS, *BALLPARKPAL_GAME_FEATURE_COLUMNS],
        "baseline_plus_bp_all": [*CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS, *BALLPARKPAL_FEATURE_COLUMNS],
    }


def evaluate_variants(base_df: pd.DataFrame, model_name: str) -> tuple[list[VariantResult], pd.DataFrame, Any, list[str]]:
    train_df, test_df = chronological_split(base_df)
    variant_feature_sets = build_variant_feature_sets()
    results: list[VariantResult] = []
    best_model = None
    best_feature_columns: list[str] = []

    baseline_metrics: dict[str, float] | None = None
    for name, candidate_features in variant_feature_sets.items():
        usable_features, pruning_audit = prune_sparse_features(train_df, candidate_features)
        if not usable_features:
            continue
        metrics, model, _ = run_variant(train_df, test_df, usable_features, model_name=model_name)
        if name == "baseline":
            baseline_metrics = metrics
        compared = (
            {metric: compare_metric_direction(baseline_metrics[metric], metrics[metric]) for metric in metrics}
            if baseline_metrics is not None and name != "baseline"
            else {metric: "baseline" for metric in metrics}
        )
        bp_used = [feature for feature in usable_features if feature in BALLPARKPAL_FEATURE_COLUMNS]
        results.append(
            VariantResult(
                name=name,
                feature_count=len(usable_features),
                rows_train=len(train_df),
                rows_test=len(test_df),
                metrics=metrics,
                compared_to_baseline=compared,
                ballparkpal_features_used=bp_used,
            )
        )
        if name == "baseline_plus_bp_all":
            best_model = model
            best_feature_columns = usable_features

    if best_model is None:
        raise RuntimeError("No viable Ballpark Pal comparison variant could be fit.")
    return results, test_df, best_model, best_feature_columns


def print_summary(results: list[VariantResult]) -> None:
    baseline = next((result for result in results if result.name == "baseline"), None)
    print("\nBallpark Pal comparison summary")
    print("-" * 60)
    if baseline:
        print(f"Baseline PR-AUC: {baseline.metrics['pr_auc']:.4f}")
        print(f"Baseline ROC-AUC: {baseline.metrics['roc_auc']:.4f}")
    for result in results:
        print(
            f"{result.name}: features={result.feature_count}, "
            f"PR-AUC={result.metrics['pr_auc']:.4f}, ROC-AUC={result.metrics['roc_auc']:.4f}, "
            f"log_loss={result.metrics['log_loss']:.4f}"
        )
    print("\nVariants sorted by PR-AUC")
    for result in sorted(results, key=lambda item: item.metrics["pr_auc"], reverse=True):
        print(f"  {result.name}: {result.metrics['pr_auc']:.4f}")


def write_outputs(
    output_dir: Path,
    *,
    joined_df: pd.DataFrame,
    coverage: dict[str, Any],
    results: list[VariantResult],
    coefficient_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joined_path = output_dir / "ballparkpal_augmented_dataset.csv"
    report_path = output_dir / "ballparkpal_comparison_report.json"
    coeff_path = output_dir / "ballparkpal_feature_coefficients.csv"

    joined_df.to_csv(joined_path, index=False)
    report_payload = {
        "coverage": coverage,
        "variants": [result.to_dict() for result in results],
        "joined_rows": len(joined_df),
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    coefficient_df.to_csv(coeff_path, index=False)

    print(f"\nWrote augmented dataset: {joined_path}")
    print(f"Wrote comparison report: {report_path}")
    print(f"Wrote coefficient report: {coeff_path}")


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_data_path)
    archive_root = Path(args.archive_root)
    output_dir = Path(args.output_dir)

    source_df = load_source_dataset(source_path)
    joined_df, coverage = augment_model_dataset_with_ballparkpal(source_df, archive_root)
    print("\nBallpark Pal join coverage")
    print("-" * 60)
    print(json.dumps(coverage.to_dict(), indent=2))
    if max(coverage.batter_coverage, coverage.pitcher_coverage, coverage.team_coverage, coverage.game_coverage) == 0.0:
        raise RuntimeError(
            "Ballpark Pal exports did not align with the source season dataset. "
            "The archive looks exploratory rather than point-in-time, so the comparison would be misleading."
        )

    results, test_df, best_model, best_feature_columns = evaluate_variants(joined_df, model_name=args.model)
    print_summary(results)

    coefficient_df = extract_logistic_coefficients(best_model, best_feature_columns)
    if not coefficient_df.empty:
        bp_coeffs = coefficient_df[coefficient_df["feature_name"].isin(BALLPARKPAL_FEATURE_COLUMNS)].head(args.max_report_coeffs)
        print("\nTop Ballpark Pal coefficients from baseline_plus_bp_all")
        print("-" * 60)
        print(bp_coeffs.to_string(index=False) if not bp_coeffs.empty else "None")
    else:
        bp_coeffs = pd.DataFrame()

    write_outputs(
        output_dir,
        joined_df=joined_df,
        coverage=coverage.to_dict(),
        results=results,
        coefficient_df=coefficient_df,
    )


if __name__ == "__main__":
    main()
