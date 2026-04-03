from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit

import train_model
from scripts.live_pipeline import score_live_candidates, train_live_model_bundle
from weather_audit import summarize_weather_feature_coverage


def make_live_dataset(rows: int = 24) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for idx in range(rows):
        records.append(
            {
                "game_date": str(pd.Timestamp("2024-03-28") + pd.Timedelta(days=idx)),
                "game_pk": 1000 + idx,
                "player_id": 2000 + idx,
                "batter_id": 3000 + idx,
                "batter_name": f"Hitter {idx}",
                "team": "NYY",
                "opponent_team": "BOS",
                "pitcher_id": 4000 + idx,
                "pitcher_name": f"Pitcher {idx}",
                "hit_hr": int(idx % 5 == 0),
                "hr_per_pa_last_30d": 0.02 + idx * 0.001,
                "hr_per_pa_last_10d": 0.01 + idx * 0.001,
                "barrels_per_pa_last_30d": 0.03 + idx * 0.001,
                "barrels_per_pa_last_10d": 0.02 + idx * 0.001,
                "hard_hit_rate_last_30d": 0.30 + idx * 0.002,
                "hard_hit_rate_last_10d": 0.28 + idx * 0.002,
                "bbe_95plus_ev_rate_last_30d": 0.20 + idx * 0.001,
                "bbe_95plus_ev_rate_last_10d": 0.18 + idx * 0.001,
                "avg_exit_velocity_last_10d": 88.0 + idx * 0.1,
                "max_exit_velocity_last_10d": 103.0 + idx * 0.1,
                "pitcher_hr_allowed_per_pa_last_30d": 0.03 + idx * 0.001,
                "pitcher_barrels_allowed_per_bbe_last_30d": 0.07 + idx * 0.001,
                "pitcher_hard_hit_allowed_rate_last_30d": 0.34 + idx * 0.001,
                "pitcher_avg_ev_allowed_last_30d": 89.0 + idx * 0.1,
                "pitcher_95plus_ev_allowed_rate_last_30d": 0.22 + idx * 0.001,
                "park_factor_hr_vs_batter_hand": 101.0 + (idx % 4),
                "batter_hr_per_pa_vs_pitcher_hand": 0.015 + idx * 0.001,
                "batter_barrels_per_pa_vs_pitcher_hand": 0.025 + idx * 0.001,
                "pitcher_hr_allowed_per_pa_vs_batter_hand": 0.02 + idx * 0.001,
                "pitcher_barrels_allowed_per_bbe_vs_batter_hand": 0.06 + idx * 0.001,
                "split_matchup_hr": 0.0003 + idx * 0.00001,
                "split_matchup_barrel": 0.0015 + idx * 0.00005,
                "split_matchup_hard_hit": 0.08 + idx * 0.001,
                "temperature_f": 60.0 + (idx % 5),
                "wind_speed_mph": 10.0 + (idx % 4),
                "humidity_pct": 50.0 + (idx % 6),
                "platoon_advantage": float(idx % 2),
                "expected_pa_today": 3.8 + (idx % 4) * 0.1,
                "batting_order_slot": float((idx % 9) + 1),
                "lineup_confirmation_score": 1.0 if idx % 3 == 0 else 0.5,
                "projected_lineup_rank": float((idx % 9) + 1),
            }
        )
    return pd.DataFrame(records)


class RecordingTimeSeriesSplit:
    created: list[int] = []

    def __init__(self, n_splits: int):
        self.inner = SklearnTimeSeriesSplit(n_splits=n_splits)
        self.created.append(n_splits)

    def split(self, X, y=None, groups=None):
        return self.inner.split(X, y, groups)


class ModelSearchTests(unittest.TestCase):
    def test_representative_training_slice_has_material_weather_population(self) -> None:
        df = make_live_dataset()
        coverage = summarize_weather_feature_coverage(df)
        for feature, stats in coverage.items():
            self.assertLess(float(stats["null_rate"]), 0.35, feature)

    def test_cross_validate_probability_metrics_uses_time_series_split(self) -> None:
        df = make_live_dataset()
        X = train_model.prepare_feature_matrix(df, ["hr_per_pa_last_30d", "temperature_f"])
        y = df["hit_hr"].to_numpy()
        RecordingTimeSeriesSplit.created = []
        with patch.object(train_model, "TimeSeriesSplit", RecordingTimeSeriesSplit):
            summary = train_model.cross_validate_probability_metrics_time_series(
                train_model.build_logistic_pipeline(),
                X,
                y,
                feature_columns=["hr_per_pa_last_30d", "temperature_f"],
            )
        self.assertTrue(RecordingTimeSeriesSplit.created)
        self.assertIn("mean_cv_pr_auc", summary)

    def test_run_backtest_search_receives_training_rows_only(self) -> None:
        df = make_live_dataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.csv"
            df.to_csv(dataset_path, index=False)

            def fake_search_training_candidates(train_df: pd.DataFrame, **kwargs):
                self.assertLess(train_df["game_date"].max(), pd.to_datetime(df["game_date"]).max())
                return [
                    {
                        "summary_row": {
                            "model_family": "logistic",
                            "feature_profile": "live",
                            "missingness_threshold": 0.5,
                            "feature_count": 2,
                            "excluded_feature_count": 0,
                            "selection_metric": "pr_auc",
                            "selection_score": 0.2,
                            "mean_cv_pr_auc": 0.2,
                            "mean_cv_roc_auc": 0.6,
                            "mean_cv_log_loss": 0.5,
                            "mean_cv_brier_score": 0.2,
                            "search_best_score": 0.2,
                            "model_priority": 0,
                            "best_params": {},
                        },
                        "feature_columns": ["hr_per_pa_last_30d", "temperature_f"],
                        "excluded_features": [],
                        "search": SimpleNamespace(best_estimator_=train_model.build_logistic_pipeline()),
                    }
                ]

            fake_final_result = {
                "summary_row": {
                    "model_family": "logistic",
                    "feature_profile": "live",
                    "threshold": 0.2,
                    "precision": 0.2,
                    "recall": 0.1,
                    "f1": 0.13,
                    "f0.5": 0.16,
                    "balanced_accuracy": 0.55,
                    "positive_prediction_rate": 0.12,
                    "pr_auc": 0.21,
                    "roc_auc": 0.61,
                    "log_loss": 0.49,
                    "brier_score": 0.19,
                    "actual_hr_rate": 0.1,
                    "prediction_to_actual_rate_ratio": 1.2,
                    "model": "logistic::live",
                    "operationally_usable": "yes",
                    "operational_usability_reason": "ok",
                }
            }

            with patch.object(train_model, "search_training_candidates", side_effect=fake_search_training_candidates):
                with patch.object(train_model, "evaluate_model_run", return_value=fake_final_result):
                    result = train_model.run_backtest(str(dataset_path), report_path=None)
        self.assertEqual(result["final_result"]["summary_row"]["model_family"], "logistic")

    def test_rank_candidate_rows_uses_pr_auc_then_logloss_then_brier_then_priority(self) -> None:
        rows = [
            {"selection_score": 0.40, "mean_cv_log_loss": 0.50, "mean_cv_brier_score": 0.20, "model_priority": 2},
            {"selection_score": 0.40, "mean_cv_log_loss": 0.45, "mean_cv_brier_score": 0.25, "model_priority": 1},
            {"selection_score": 0.40, "mean_cv_log_loss": 0.45, "mean_cv_brier_score": 0.20, "model_priority": 0},
        ]
        ranked = train_model.rank_candidate_rows(rows)
        self.assertEqual(ranked[0]["model_priority"], 0)

    def test_evaluate_training_candidate_respects_missingness_threshold(self) -> None:
        df = make_live_dataset()
        df.loc[:13, "barrels_per_pa_last_30d"] = pd.NA

        fake_search = SimpleNamespace(best_estimator_=train_model.build_logistic_pipeline(), best_score_=0.2, best_params_={})
        fake_cv = {
            "mean_cv_pr_auc": 0.2,
            "mean_cv_roc_auc": 0.6,
            "mean_cv_log_loss": 0.5,
            "mean_cv_brier_score": 0.2,
        }
        with patch.object(train_model, "tune_model_family", return_value=fake_search):
            with patch.object(train_model, "cross_validate_probability_metrics_time_series", return_value=fake_cv):
                strict_result = train_model.evaluate_training_candidate(
                    df,
                    feature_profile="live",
                    model_name="logistic",
                    missingness_threshold=0.35,
                )
                loose_result = train_model.evaluate_training_candidate(
                    df,
                    feature_profile="live",
                    model_name="logistic",
                    missingness_threshold=0.65,
                )
        self.assertNotIn("barrels_per_pa_last_30d", strict_result["feature_columns"])
        self.assertIn("barrels_per_pa_last_30d", loose_result["feature_columns"])

    def test_non_logistic_bundle_round_trip_scores_live_candidates(self) -> None:
        model = HistGradientBoostingClassifier(random_state=42)
        train_X = pd.DataFrame({"feature_a": [0.1, 0.3, 0.6, 0.9, 0.95, 0.2]})
        train_y = [0, 0, 1, 1, 1, 0]
        model.fit(train_X, train_y)
        bundle = {
            "model": model,
            "feature_columns": ["feature_a"],
            "reference_df": train_X.copy(),
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "bundle.pkl"
            bundle_path.write_bytes(pickle.dumps(bundle))
            loaded_bundle = pickle.loads(bundle_path.read_bytes())
        candidate_df = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": pd.Timestamp("2026-03-25"), "batter_id": 1, "batter_name": "Alpha", "team": "NYY", "opponent_team": "BOS", "pitcher_id": 11, "pitcher_name": "Pitcher", "feature_a": 0.2},
                {"game_pk": 1, "game_date": pd.Timestamp("2026-03-25"), "batter_id": 2, "batter_name": "Bravo", "team": "NYY", "opponent_team": "BOS", "pitcher_id": 11, "pitcher_name": "Pitcher", "feature_a": 0.92},
            ]
        )
        picks = score_live_candidates(candidate_df, loaded_bundle, max_picks=2, published_at="2026-03-25T12:00:00+00:00")
        self.assertEqual(picks[0]["batter_name"], "Bravo")

    def test_train_live_model_bundle_writes_selection_metadata(self) -> None:
        df = make_live_dataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "dataset.csv"
            bundle_path = tmp_path / "bundle.pkl"
            metadata_path = tmp_path / "metadata.json"
            df.to_csv(dataset_path, index=False)

            summary_row = {
                "model_family": "logistic",
                "feature_profile": "live",
                "missingness_threshold": 0.5,
                "feature_count": 3,
                "excluded_feature_count": 0,
                "selection_metric": "pr_auc",
                "selection_score": 0.3,
                "mean_cv_pr_auc": 0.3,
                "mean_cv_roc_auc": 0.6,
                "mean_cv_log_loss": 0.5,
                "mean_cv_brier_score": 0.2,
                "search_best_score": 0.3,
                "model_priority": 0,
                "best_params": {},
            }
            fake_candidate = {
                "summary_row": summary_row,
                "feature_columns": list(train_model.LIVE_PRODUCTION_FEATURE_COLUMNS),
                "excluded_features": [],
                "search": SimpleNamespace(best_estimator_=train_model.build_logistic_pipeline()),
            }
            fake_backtest = {
                "selected_candidate": fake_candidate,
                "final_result": {"summary_row": {**summary_row, "pr_auc": 0.31}},
                "candidate_results": [fake_candidate],
                "train_df": df.iloc[:16].copy(),
                "test_df": df.iloc[16:].copy(),
            }

            with patch("scripts.live_pipeline.run_backtest", return_value=fake_backtest):
                with patch("scripts.live_pipeline.evaluate_training_candidate", return_value=fake_candidate):
                    bundle = train_live_model_bundle(
                        dataset_path,
                        bundle_path=bundle_path,
                        metadata_path=metadata_path,
                        calibration="disabled",
                    )

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(bundle["feature_profile"], "live")
        self.assertEqual(metadata["selection_metric"], "pr_auc")
        self.assertIn("dataset_max_game_date", metadata)
        self.assertIn("training_cv_summary", metadata)
        self.assertIn("final_holdout_summary", metadata)
        self.assertIn("promotion_decision", metadata)
        self.assertIn("weather_feature_coverage", metadata)

    def test_usability_gate_summary_matches_existing_gate_functions(self) -> None:
        metrics = {"precision": 0.19, "recall": 0.11, "positive_prediction_rate": 0.12}
        summary = train_model.usability_gate_summary(metrics, actual_rate=0.10)
        self.assertEqual(
            summary["passes_absolute_usability_gate"],
            train_model.is_operationally_usable(metrics, actual_rate=0.10),
        )
        self.assertEqual(
            summary["failure_reason"],
            train_model.usability_reason(metrics, actual_rate=0.10),
        )

    def test_usability_threshold_search_includes_conservative_recall_floor(self) -> None:
        self.assertIn(0.08, train_model.USABILITY_MIN_RECALL_CHOICES)

    def test_build_rolling_window_slices_is_deterministic(self) -> None:
        df = make_live_dataset(rows=90)
        df["game_date"] = pd.to_datetime(df["game_date"])
        first = train_model.build_rolling_window_slices(df)
        second = train_model.build_rolling_window_slices(df)
        self.assertEqual(
            [(row["holdout_start_date"], row["holdout_end_date"]) for row in first],
            [(row["holdout_start_date"], row["holdout_end_date"]) for row in second],
        )
        self.assertGreater(len(first), 0)
        self.assertLessEqual(len(first), train_model.USABILITY_ROLLING_WINDOW_COUNT)

    def test_build_rolling_window_slices_skips_empty_offseason_windows(self) -> None:
        first_block = make_live_dataset(rows=40)
        first_block["game_date"] = pd.to_datetime(first_block["game_date"])
        second_block = make_live_dataset(rows=20)
        second_block["game_date"] = pd.to_datetime("2025-03-05") + pd.to_timedelta(range(20), unit="D")
        df = pd.concat([first_block, second_block], ignore_index=True)
        windows = train_model.build_rolling_window_slices(df, window_days=14, window_count=3)
        self.assertEqual(len(windows), 3)
        self.assertTrue(all(window["holdout_rows"] > 0 for window in windows))

    def test_backward_elimination_search_keeps_recall_floor(self) -> None:
        df = make_live_dataset(rows=36)

        def fake_evaluate_profile_subset_candidate(
            train_df: pd.DataFrame,
            *,
            feature_columns: list[str],
            missingness_threshold: float,
            selection_metric: str,
            variant_label: str,
            best_params=None,
        ):
            feature_count = len(feature_columns)
            recall = 0.11 if feature_count >= 16 else 0.09
            precision = 0.14 + max(0, 24 - feature_count) * 0.001
            return {
                "summary_row": {
                    "feature_profile": train_model.LIVE_USABLE_CANDIDATE_PROFILE,
                    "feature_profile_variant": variant_label,
                    "mean_cv_pr_auc": 0.20 + feature_count * 0.0001,
                },
                "feature_columns": list(feature_columns),
                "profile_search_summary": {
                    "oof_precision": precision,
                    "oof_recall": recall,
                    "oof_positive_prediction_rate": 0.12 + feature_count * 0.0001,
                    "oof_pr_auc": 0.20 + feature_count * 0.0001,
                    "oof_threshold": 0.20,
                    "variant_label": variant_label,
                    "feature_count": feature_count,
                },
            }

        with patch.object(train_model, "evaluate_profile_subset_candidate", side_effect=fake_evaluate_profile_subset_candidate):
            search = train_model.generate_live_usable_profile_candidates(
                df,
                missingness_threshold=0.5,
                selection_metric="pr_auc",
            )
        accepted_recalls = [
            candidate["profile_search_summary"]["oof_recall"]
            for candidate in search["accepted_path"]
        ]
        self.assertTrue(all(recall >= 0.10 for recall in accepted_recalls))

    def test_train_live_model_bundle_live_plus_search_compares_against_live_and_serializes_profile(self) -> None:
        df = make_live_dataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "dataset.csv"
            bundle_path = tmp_path / "bundle.pkl"
            metadata_path = tmp_path / "metadata.json"
            df.to_csv(dataset_path, index=False)

            live_plus_summary = {
                "model_family": "logistic",
                "feature_profile": "live_plus",
                "missingness_threshold": 0.5,
                "feature_count": len(train_model.LIVE_PLUS_FEATURE_COLUMNS),
                "excluded_feature_count": 0,
                "selection_metric": "pr_auc",
                "selection_score": 0.31,
                "mean_cv_pr_auc": 0.31,
                "mean_cv_roc_auc": 0.62,
                "mean_cv_log_loss": 0.49,
                "mean_cv_brier_score": 0.19,
                "search_best_score": 0.31,
                "model_priority": 0,
                "best_params": {},
            }
            live_summary = {
                **live_plus_summary,
                "feature_profile": "live",
                "feature_count": len(train_model.LIVE_PRODUCTION_FEATURE_COLUMNS),
                "selection_score": 0.29,
                "mean_cv_pr_auc": 0.29,
            }
            live_plus_candidate = {
                "summary_row": live_plus_summary,
                "feature_columns": list(train_model.LIVE_PLUS_FEATURE_COLUMNS),
                "excluded_features": [],
                "search": SimpleNamespace(best_estimator_=train_model.build_logistic_pipeline()),
            }
            live_candidate = {
                "summary_row": live_summary,
                "feature_columns": list(train_model.LIVE_PRODUCTION_FEATURE_COLUMNS),
                "excluded_features": [],
                "search": SimpleNamespace(best_estimator_=train_model.build_logistic_pipeline()),
            }
            fake_backtest = {
                "selected_candidate": live_plus_candidate,
                "final_result": {"summary_row": {**live_plus_summary, "pr_auc": 0.32}},
                "candidate_results": [live_plus_candidate, live_candidate],
                "train_df": df.iloc[:16].copy(),
                "test_df": df.iloc[16:].copy(),
            }

            def fake_evaluate_training_candidate(train_df: pd.DataFrame, *, feature_profile: str, **_: object):
                return live_candidate if feature_profile == "live" else live_plus_candidate

            with patch("scripts.live_pipeline.run_backtest", return_value=fake_backtest) as mock_backtest:
                with patch("scripts.live_pipeline.evaluate_training_candidate", side_effect=fake_evaluate_training_candidate):
                    with patch(
                        "scripts.live_pipeline.evaluate_model_run",
                        return_value={"summary_row": {**live_summary, "pr_auc": 0.30}},
                    ):
                        bundle = train_live_model_bundle(
                            dataset_path,
                            bundle_path=bundle_path,
                            metadata_path=metadata_path,
                            calibration="disabled",
                            feature_profile="live_plus",
                        )

            self.assertEqual(mock_backtest.call_args.kwargs["compare_against"], "live")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["feature_profile"], "live_plus")
            self.assertEqual(metadata["feature_profile"], "live_plus")
            self.assertEqual(metadata["promotion_decision"]["used"], "selected_candidate")


if __name__ == "__main__":
    unittest.main()
