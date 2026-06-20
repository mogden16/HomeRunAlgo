from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

import feature_engineering
import train_model
from scripts import walk_forward_betting_backtest as audit


class RecordingClassifier(BaseEstimator, ClassifierMixin):
    fit_max_values: list[float] = []
    fit_row_counts: list[int] = []

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fit_max_values.append(float(pd.DataFrame(X).iloc[:, 0].max()))
        self.fit_row_counts.append(len(X))
        return self

    def predict_proba(self, X):
        values = pd.DataFrame(X).iloc[:, 0].astype(float).to_numpy()
        probs = np.clip(values / 100.0, 0.01, 0.99)
        return np.column_stack([1.0 - probs, probs])


def make_minimal_dataset(day_count: int = 8, rows_per_day: int = 4) -> pd.DataFrame:
    rows = []
    for day_idx in range(day_count):
        game_date = pd.Timestamp("2026-04-01") + pd.Timedelta(days=day_idx)
        for row_idx in range(rows_per_day):
            rows.append(
                {
                    "game_date": game_date,
                    "game_pk": 1000 + day_idx,
                    "player_id": 2000 + row_idx,
                    "batter_id": 2000 + row_idx,
                    "batter_name": f"Hitter {row_idx}",
                    "team": "NYY",
                    "opponent": "BOS",
                    "pitcher_id": 3000 + day_idx,
                    "opp_pitcher_id": 3000 + day_idx,
                    "hit_hr": int((day_idx + row_idx) % 5 == 0),
                    "hr_per_pa_last_30d": float(day_idx),
                }
            )
    return pd.DataFrame(rows)


class BettingAuditTests(unittest.TestCase):
    def test_default_walk_forward_window_uses_latest_year_as_test_set(self) -> None:
        frame = pd.DataFrame(
            [
                {"game_date": "2024-03-28", "game_pk": 1, "batter_id": 10, "hit_hr": 0},
                {"game_date": "2025-03-28", "game_pk": 2, "batter_id": 10, "hit_hr": 0},
                {"game_date": "2026-03-28", "game_pk": 3, "batter_id": 10, "hit_hr": 1},
            ]
        )

        source, test_dates, window_summary = audit.filter_source_for_walk_forward(frame)

        self.assertEqual(window_summary["train_years"], [2024, 2025])
        self.assertEqual(window_summary["test_years"], [2026])
        self.assertEqual(window_summary["window_mode"], "default_latest_year")
        self.assertEqual(sorted(pd.to_datetime(source["game_date"]).dt.year.unique().tolist()), [2024, 2025, 2026])
        self.assertEqual([str(pd.Timestamp(item).date()) for item in test_dates], ["2026-03-28"])

    def test_date_window_features_are_shifted_left(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "game_date": pd.Timestamp("2026-04-01"),
                    "game_pk": 1,
                    "hr_count": 1,
                    "pa_count": 4,
                    "barrel_count": 1,
                    "hard_hit_bbe_count": 2,
                    "bbe_count": 3,
                    "ev_95plus_bbe_count": 2,
                    "avg_exit_velocity": 90.0,
                    "max_exit_velocity": 103.0,
                },
                {
                    "game_date": pd.Timestamp("2026-04-02"),
                    "game_pk": 2,
                    "hr_count": 0,
                    "pa_count": 5,
                    "barrel_count": 0,
                    "hard_hit_bbe_count": 1,
                    "bbe_count": 4,
                    "ev_95plus_bbe_count": 1,
                    "avg_exit_velocity": 88.0,
                    "max_exit_velocity": 101.0,
                },
            ]
        )

        out = feature_engineering._append_date_window_features(
            frame,
            entity_id=1,
            entity_label="batter",
            configs=[("30D", "30d"), ("10D", "10d")],
        )

        self.assertTrue(pd.isna(out.loc[0, "hr_per_pa_last_30d"]))
        self.assertAlmostEqual(float(out.loc[1, "hr_per_pa_last_30d"]), 0.25)
        self.assertAlmostEqual(float(out.loc[1, "barrels_per_pa_last_30d"]), 0.25)

    def test_chronological_split_is_clean(self) -> None:
        df = make_minimal_dataset(day_count=10, rows_per_day=2)
        train_df, test_df = train_model.chronological_split(df, fraction=0.7)
        self.assertLess(train_df["game_date"].max(), test_df["game_date"].min())

    def test_logistic_calibration_search_receives_training_matrix_only(self) -> None:
        train_df = make_minimal_dataset(day_count=12, rows_per_day=3)
        X_train = train_model.prepare_feature_matrix(train_df, ["hr_per_pa_last_30d"])
        y_train = train_df["hit_hr"].to_numpy()
        estimator = train_model.build_logistic_pipeline()
        seen_row_counts: list[int] = []

        def fake_cv(model, X, y, *, feature_columns):
            seen_row_counts.append(len(X))
            return {
                "mean_cv_pr_auc": 0.2,
                "mean_cv_roc_auc": 0.6,
                "mean_cv_log_loss": 0.5,
                "mean_cv_brier_score": 0.2,
            }

        with patch.object(train_model, "cross_validate_probability_metrics_time_series", side_effect=fake_cv):
            with patch.object(train_model, "maybe_calibrate_logistic", side_effect=lambda model, X, y, mode, family: (model.fit(X, y), {"used": mode, "status": "ok", "message": "ok"})):
                train_model.choose_logistic_calibration(
                    estimator,
                    X_train,
                    y_train,
                    ["hr_per_pa_last_30d"],
                    "sigmoid",
                )

        self.assertTrue(seen_row_counts)
        self.assertEqual(set(seen_row_counts), {len(X_train)})

    def test_walk_forward_training_uses_only_prior_dates(self) -> None:
        df = make_minimal_dataset(day_count=7, rows_per_day=4)
        RecordingClassifier.fit_max_values = []
        RecordingClassifier.fit_row_counts = []

        predictions = audit.predict_walk_forward(
            df,
            profile="live",
            spec=audit.ModelSpec("recording", RecordingClassifier()),
            min_train_days=3,
            max_test_days=2,
        )

        self.assertEqual(sorted(predictions["game_date"].astype(str).unique()), ["2026-04-04", "2026-04-05"])
        self.assertEqual(RecordingClassifier.fit_max_values, [2.0, 3.0])
        self.assertEqual(RecordingClassifier.fit_row_counts, [12, 16])

    def test_walk_forward_honors_explicit_year_windows(self) -> None:
        frame_2024 = make_minimal_dataset(day_count=2, rows_per_day=2)
        frame_2024["game_date"] = pd.to_datetime(["2024-03-28", "2024-03-28", "2024-03-29", "2024-03-29"])
        frame_2025 = make_minimal_dataset(day_count=2, rows_per_day=2)
        frame_2025["game_date"] = pd.to_datetime(["2025-03-28", "2025-03-28", "2025-03-29", "2025-03-29"])
        frame_2026 = make_minimal_dataset(day_count=4, rows_per_day=2)
        frame_2026["game_date"] = pd.to_datetime(
            ["2026-04-01", "2026-04-01", "2026-04-02", "2026-04-02", "2026-04-03", "2026-04-03", "2026-04-04", "2026-04-04"]
        )
        df = pd.concat([frame_2024, frame_2025, frame_2026], ignore_index=True)
        RecordingClassifier.fit_max_values = []

        predictions = audit.predict_walk_forward(
            df,
            profile="live",
            spec=audit.ModelSpec("recording", RecordingClassifier()),
            min_train_days=2,
            max_test_days=None,
            train_years=[2024, 2025],
            test_years=[2026],
        )

        self.assertEqual(sorted(predictions["game_date"].astype(str).unique()), ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04"])
        self.assertGreaterEqual(min(RecordingClassifier.fit_max_values), 1.0)

    def test_no_odds_report_keeps_profitability_unavailable(self) -> None:
        frame = pd.DataFrame(
            [
                {"game_date": "2026-04-01", "slate_rank": 1, "batter_name": "A", "predicted_hr_probability": 0.3, "actual_hit_hr": 1, "confidence_tier": "elite", "american_odds": np.nan},
                {"game_date": "2026-04-01", "slate_rank": 2, "batter_name": "B", "predicted_hr_probability": 0.2, "actual_hit_hr": 0, "confidence_tier": "strong", "american_odds": np.nan},
            ]
        )

        result = audit.summarize_selection(
            frame,
            strategy_name="top_1_daily",
            odds_available=False,
            kelly_fraction=0.25,
            kelly_max_stake=0.02,
            total_actual_positives=1,
            total_population=2,
        )

        self.assertEqual(result["bets"], 2)
        self.assertIsNone(result["roi"])
        self.assertIsNone(result["kelly_roi"])
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["false_positives"], 1)

    def test_real_odds_import_enables_edge_and_profit_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            odds_path = Path(tmp_dir) / "odds.csv"
            pd.DataFrame(
                [
                    {"game_date": "2026-04-01", "game_pk": 1, "batter_id": 10, "american_odds": 300},
                    {"game_date": "2026-04-01", "game_pk": 1, "batter_id": 11, "american_odds": 200},
                ]
            ).to_csv(odds_path, index=False)
            odds = audit.load_odds_csv(odds_path)

        self.assertIsNotNone(odds)
        assert odds is not None
        self.assertIn("implied_probability", odds.columns)
        self.assertAlmostEqual(float(odds.iloc[0]["implied_probability"]), 0.25)

    def test_evaluation_selects_positive_call_threshold_by_precision(self) -> None:
        frame = pd.DataFrame(
            [
                {"game_date": "2026-04-01", "slate_rank": 1, "batter_name": "A", "predicted_hr_probability": 0.30, "actual_hit_hr": 1, "confidence_tier": "elite", "american_odds": np.nan},
                {"game_date": "2026-04-01", "slate_rank": 2, "batter_name": "B", "predicted_hr_probability": 0.26, "actual_hit_hr": 1, "confidence_tier": "elite", "american_odds": np.nan},
                {"game_date": "2026-04-01", "slate_rank": 3, "batter_name": "C", "predicted_hr_probability": 0.18, "actual_hit_hr": 0, "confidence_tier": "strong", "american_odds": np.nan},
                {"game_date": "2026-04-01", "slate_rank": 4, "batter_name": "D", "predicted_hr_probability": 0.12, "actual_hit_hr": 0, "confidence_tier": "watch", "american_odds": np.nan},
            ]
        )

        evaluation = audit.evaluate_predictions(
            frame,
            odds_available=False,
            kelly_fraction=0.25,
            kelly_max_stake=0.02,
            calibration_buckets=2,
            positive_call_min_bets=2,
            positive_call_min_days=1,
            bootstrap_iterations=100,
        )

        self.assertEqual(evaluation["positive_call_summary"]["selected_threshold_strategy"], "probability_at_least_0.25")
        self.assertEqual(evaluation["positive_call_summary"]["precision"], 1.0)
        self.assertEqual(evaluation["positive_call_summary"]["bets"], 2)
        self.assertEqual(evaluation["positive_call_summary"]["threshold_source"], "selected_on_selection_period")
        self.assertEqual(evaluation["positive_call_summary"]["precision_confidence_interval"]["iterations"], 100)

    def test_final_evaluation_uses_frozen_selection_threshold(self) -> None:
        frame = pd.DataFrame(
            [
                {"game_date": "2026-04-01", "slate_rank": 1, "batter_name": "A", "predicted_hr_probability": 0.30, "actual_hit_hr": 1, "confidence_tier": "elite", "american_odds": np.nan},
                {"game_date": "2026-04-01", "slate_rank": 2, "batter_name": "B", "predicted_hr_probability": 0.22, "actual_hit_hr": 0, "confidence_tier": "strong", "american_odds": np.nan},
                {"game_date": "2026-04-02", "slate_rank": 1, "batter_name": "C", "predicted_hr_probability": 0.18, "actual_hit_hr": 1, "confidence_tier": "elite", "american_odds": np.nan},
            ]
        )

        evaluation = audit.evaluate_predictions(
            frame,
            odds_available=False,
            kelly_fraction=0.25,
            kelly_max_stake=0.02,
            calibration_buckets=2,
            positive_call_min_bets=1,
            positive_call_min_days=1,
            fixed_positive_call_strategy="probability_at_least_0.20",
            bootstrap_iterations=50,
            bootstrap_seed=7,
        )

        self.assertEqual(evaluation["positive_call_summary"]["selected_threshold_strategy"], "probability_at_least_0.20")
        self.assertEqual(evaluation["positive_call_summary"]["threshold_source"], "frozen_from_selection_period")
        self.assertEqual(evaluation["positive_call_summary"]["bets"], 2)

    def test_selection_and_final_years_are_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "game_date": pd.to_datetime(["2024-04-01", "2025-04-01", "2026-04-01"]),
                "hit_hr": [0, 1, 0],
            }
        )

        initial, selection, history, final = audit.resolve_selection_and_final_years(
            frame,
            train_years=[2024, 2025],
            selection_years=None,
            test_years=[2026],
        )

        self.assertEqual(initial, [2024])
        self.assertEqual(selection, [2025])
        self.assertEqual(history, [2024, 2025])
        self.assertEqual(final, [2026])
        self.assertFalse(set(selection) & set(final))

    def test_pregame_safe_profile_passes_as_of_audit_while_live_profile_does_not(self) -> None:
        frame = make_minimal_dataset(day_count=3, rows_per_day=2)
        frame["pitcher_hr_allowed_per_pa_last_30d"] = 0.03
        frame["temperature_f"] = 72.0
        frame["wind_speed_mph"] = 8.0
        frame["humidity_pct"] = 50.0
        frame["roofed_park"] = False
        frame["platoon_advantage"] = 1

        safe_audit = audit.audit_profile_as_of_safety(frame, "pregame_safe_v1")
        live_audit = audit.audit_profile_as_of_safety(frame, "live")

        self.assertTrue(safe_audit["release_eligible"])
        self.assertFalse(live_audit["release_eligible"])
        self.assertIn("historical_actual_pitcher_without_as_of_snapshot", {row["code"] for row in live_audit["findings"]})
        self.assertIn("historical_observed_weather_without_forecast_snapshot", {row["code"] for row in live_audit["findings"]})

    def test_model_ranking_prefers_positive_call_precision_over_roi(self) -> None:
        reports = [
            {
                "model_name": "model_a",
                "feature_profile": "live",
                "evaluation": {
                    "positive_call_summary": {"precision": 0.40, "bets": 30, "selected_threshold_strategy": "probability_at_least_0.12"},
                    "ranking_summary": {"top_3_daily_hit_rate": 0.20, "top_3_daily_bets": 30},
                    "probability_metrics": {"average_precision": 0.12, "log_loss": 0.33, "brier_score": 0.09},
                    "strategy_results": [{"strategy": "top_3_daily", "roi": 1.0, "bets": 30}],
                },
            },
            {
                "model_name": "model_b",
                "feature_profile": "live",
                "evaluation": {
                    "positive_call_summary": {"precision": 0.55, "bets": 25, "selected_threshold_strategy": "probability_at_least_0.18"},
                    "ranking_summary": {"top_3_daily_hit_rate": 0.18, "top_3_daily_bets": 30},
                    "probability_metrics": {"average_precision": 0.11, "log_loss": 0.34, "brier_score": 0.10},
                    "strategy_results": [{"strategy": "top_3_daily", "roi": None, "bets": 30}],
                },
            },
        ]

        ranked = audit.rank_model_reports(reports, odds_available=True)

        self.assertEqual(ranked[0]["model_name"], "model_b")
        self.assertNotIn("best_strategy_roi", ranked[0])

    def test_live_prediction_features_match_bundle_training_features(self) -> None:
        candidate_df = pd.DataFrame(
            [
                {
                    "game_pk": 1,
                    "game_date": "2026-04-01",
                    "batter_id": 10,
                    "batter_name": "Alpha",
                    "team": "NYY",
                    "opponent_team": "BOS",
                    "pitcher_id": 20,
                    "pitcher_name": "Pitcher",
                    "hr_per_pa_last_30d": 0.1,
                    "ballparkpal_overlay_adjusted_score": np.nan,
                    "ballparkpal_overlay_raw_score": np.nan,
                    "ballparkpal_overlay_signed_score": np.nan,
                    "ballparkpal_overlay_display_score": np.nan,
                    "ballparkpal_overlay_direction": "neutral",
                }
            ]
        )
        bundle = {
            "model": SimpleNamespace(predict_proba=lambda X: np.array([[0.8, 0.2]])),
            "feature_columns": ["hr_per_pa_last_30d"],
            "reference_df": pd.DataFrame({"hr_per_pa_last_30d": [0.01, 0.1]}),
            "feature_profile": "pregame_safe_v1",
            "positive_call_threshold": 0.20,
        }

        picks = __import__("scripts.live_pipeline", fromlist=["score_live_candidates"]).score_live_candidates(
            candidate_df,
            bundle,
            max_picks=1,
            published_at="2026-04-01T12:00:00+00:00",
        )

        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0]["batter_name"], "Alpha")
        self.assertTrue(picks[0]["positive_hr_call"])
        self.assertEqual(picks[0]["positive_call_threshold"], 0.20)


if __name__ == "__main__":
    unittest.main()
