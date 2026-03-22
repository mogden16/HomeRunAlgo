"""
train_model.py

Time-aware backtest for MLB home run prediction.

WHY CHRONOLOGICAL SPLITTING IS REQUIRED
----------------------------------------
Sports prediction is a forecasting problem: at prediction time we only know
information that existed *before* the game being predicted. A random
train/test split (sklearn default) shuffles rows across time, so the model
could train on rows from late September while testing on rows from April.
This leaks future information into training (e.g. end-of-season hot streaks,
opponent fatigue) and produces optimistic evaluation metrics that do not
reflect real-world performance. Chronological splitting ensures every test
observation is strictly *after* every training observation, matching the
actual deployment scenario.

SPLIT STRATEGY
--------------
  - Sort all rows by game_date ascending.
  - Training set  : first 67% of rows (earlier games).
  - Holdout test  : final 33% of rows (later games).
  - Within training: walk-forward (expanding window) cross-validation via
    sklearn TimeSeriesSplit for hyperparameter tuning, so validation folds
    are always chronologically after the corresponding training folds.

LEAKAGE PREVENTION
------------------
  - All rolling / season-to-date features (iso_season_td, hr_last_30,
    at_bats_season, pitcher_hr9) are computed in generate_data.py using only
    information *before* each game row. No same-day or future values are used.
  - The StandardScaler and SimpleImputer are fit ONLY on the training set and
    then applied (transform-only) to the test set.
  - No feature selection or encoding step sees test-set labels.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "data/mlb_player_game.csv"
DATE_COL = "game_date"          # True game-date column used for chronological split
TARGET_COL = "hit_hr"           # Binary target: 1 if batter hit a HR in this game
TRAIN_FRACTION = 0.67           # First 67% of sorted rows → training set
TSCV_N_SPLITS = 5               # Walk-forward CV folds within the training set
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature columns (no target, no date, no high-cardinality IDs)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "hr_rate_career",    # career HR rate (static batter attribute, no leakage)
    "iso_season_td",     # isolated power season-to-date *before* this game
    "hr_last_30",        # HR count in last 30 games *before* this game
    "at_bats_season",    # season at-bats accumulated *before* this game
    "pitcher_hr9",       # opponent pitcher HR/9 season-to-date *before* this game
    "park_factor",       # ballpark HR park factor (static, no leakage)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df


def chronological_split(df: pd.DataFrame, fraction: float = TRAIN_FRACTION):
    """
    Split a dataframe into train and test sets using chronological ordering.

    All rows are sorted by DATE_COL ascending. The split is made at a clean
    *date boundary* so that no single game date straddles the train/test
    boundary (which would happen if multiple players share a game date and a
    row-index split lands mid-date). Specifically:

      1. Find the game date at the approximate `fraction` row index.
      2. Assign ALL rows with dates strictly before that cutoff date to train.
      3. Assign ALL rows with dates >= cutoff date to test.

    This guarantees max(train_date) < min(test_date), preventing any future
    information from leaking into the training set — a requirement for valid
    sports-prediction evaluation.

    Parameters
    ----------
    df       : DataFrame already containing DATE_COL
    fraction : proportion of rows (by time) assigned to training

    Returns
    -------
    train_df, test_df : DataFrames with no overlapping rows
    """
    # Sort by game date ascending so earlier games come first
    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)

    # Identify the cutoff date at the approximate fraction boundary.
    # We split on a clean date so no game date straddles the boundary.
    approx_index = int(len(df_sorted) * fraction)
    cutoff_date = df_sorted.iloc[approx_index][DATE_COL]

    # All rows strictly BEFORE the cutoff date go to train.
    # All rows ON OR AFTER the cutoff date go to test.
    # This ensures max(train_date) < min(test_date).
    train_df = df_sorted[df_sorted[DATE_COL] < cutoff_date].copy()
    test_df = df_sorted[df_sorted[DATE_COL] >= cutoff_date].copy()

    return train_df, test_df


def validate_temporal_integrity(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Assert that every test row is strictly after every training row.
    Raises ValueError if temporal leakage is detected.
    """
    max_train_date = train_df[DATE_COL].max()
    min_test_date = test_df[DATE_COL].min()

    if max_train_date >= min_test_date:
        raise ValueError(
            f"Temporal leakage detected: max train date ({max_train_date.date()}) "
            f">= min test date ({min_test_date.date()}). "
            "Ensure the data is split chronologically."
        )
    print(f"[OK] Temporal integrity check passed: "
          f"max(train_date)={max_train_date.date()} < min(test_date)={min_test_date.date()}")


def build_pipeline() -> Pipeline:
    """
    Build an sklearn Pipeline with:
      1. SimpleImputer  — handles any NaN values (fit on train only)
      2. StandardScaler — zero-mean unit-variance scaling (fit on train only)
      3. LogisticRegression — probabilistic classifier

    The Pipeline ensures that imputer and scaler are fit on training data and
    only applied (transform) to test data, preventing preprocessing leakage.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def tune_with_walk_forward_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """
    Hyperparameter tuning using walk-forward (expanding-window) cross-validation.

    TimeSeriesSplit creates folds where each validation fold is strictly after
    its corresponding training fold, preserving temporal order inside the
    training set. This mirrors real-world model selection where we only tune
    on data available at the time of each decision.

    The final model is re-trained on ALL training data after the best
    hyperparameters are selected.
    """
    tscv = TimeSeriesSplit(n_splits=TSCV_N_SPLITS)

    param_grid = {
        # Inverse regularization strength: smaller C = stronger regularization
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    }

    grid_search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=param_grid,
        cv=tscv,            # Walk-forward CV — no future leakage inside training
        scoring="neg_log_loss",
        refit=True,         # Re-fit best model on full training set
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best C: {grid_search.best_params_['clf__C']}  "
          f"(CV log-loss: {-grid_search.best_score_:.4f})")
    return grid_search.best_estimator_


def print_calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> None:
    """Print a simple calibration table: mean predicted prob vs. actual fraction positive."""
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    print("\n  Calibration summary (predicted prob → actual HR rate):")
    print(f"  {'Pred prob':>10}  {'Actual rate':>12}")
    for pred, actual in zip(mean_pred, fraction_pos):
        print(f"  {pred:>10.3f}  {actual:>12.3f}")


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(data_path: str = DATA_PATH) -> None:
    # ---- 1. Load data -------------------------------------------------------
    if not Path(data_path).exists():
        print(f"Data file not found at '{data_path}'. Generating it now...")
        from generate_data import generate_mlb_dataset
        generate_mlb_dataset(data_path)

    df = load_data(data_path)

    # ---- 2. Chronological split ---------------------------------------------
    #
    # WHY: We sort by game_date and take the first 67% as training data.
    # The final 33% serve as the holdout test set representing "future" games
    # the model has never seen. This mirrors production deployment where the
    # model is trained on historical data and evaluated on upcoming games.
    #
    train_df, test_df = chronological_split(df, fraction=TRAIN_FRACTION)

    # ---- 3. Temporal integrity check (lightweight validation) ---------------
    validate_temporal_integrity(train_df, test_df)

    # ---- 4. Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TIME-AWARE BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Total rows      : {len(df):,}")
    print(f"  Train rows      : {len(train_df):,}  ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test rows       : {len(test_df):,}  ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Train date range: {train_df[DATE_COL].min().date()} → {train_df[DATE_COL].max().date()}")
    print(f"  Test  date range: {test_df[DATE_COL].min().date()} → {test_df[DATE_COL].max().date()}")
    print(f"  Target column   : '{TARGET_COL}'")
    print(f"  HR rate (train) : {train_df[TARGET_COL].mean():.4f}")
    print(f"  HR rate (test)  : {test_df[TARGET_COL].mean():.4f}")

    # ---- 5. Prepare feature matrices ----------------------------------------
    X_train = train_df[NUMERIC_FEATURES].values
    y_train = train_df[TARGET_COL].values

    X_test = test_df[NUMERIC_FEATURES].values
    y_test = test_df[TARGET_COL].values

    # ---- 6. Tune + train (walk-forward CV inside training set) --------------
    print("\n  Tuning hyperparameters with walk-forward CV (TimeSeriesSplit)...")
    model = tune_with_walk_forward_cv(X_train, y_train)

    # ---- 7. Evaluate on holdout test set ------------------------------------
    #
    # The test set is evaluated ONCE after all training decisions are final.
    # Imputer and scaler were fit only on train data (via the Pipeline) and are
    # only applied (transform) to the test set here.
    #
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    ll = log_loss(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("  HOLDOUT TEST SET EVALUATION")
    print("=" * 60)
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Log Loss        : {ll:.4f}  (lower = better)")
    print(f"  Brier Score     : {brier:.4f}  (lower = better)")
    print(f"  ROC-AUC         : {auc:.4f}  (higher = better)")

    print_calibration_summary(y_test, y_prob)
    print("=" * 60)


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    run_backtest(data_path)
