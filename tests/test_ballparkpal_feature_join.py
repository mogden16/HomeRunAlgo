from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools.ballparkpal.feature_join import (
    BALLPARKPAL_BATTER_FEATURE_COLUMNS,
    BALLPARKPAL_GAME_FEATURE_COLUMNS,
    BALLPARKPAL_PITCHER_FEATURE_COLUMNS,
    BALLPARKPAL_TEAM_FEATURE_COLUMNS,
    BallparkPalJoinCoverage,
    augment_model_dataset_with_ballparkpal,
)


def test_augment_model_dataset_with_ballparkpal_joins_expected_columns() -> None:
    base_df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2025-03-18"]),
            "game_pk": [111],
            "player_id": [10],
            "opp_pitcher_id": [20],
            "team": ["NYM"],
            "opponent": ["ATL"],
            "is_home": [1],
        }
    )

    normalized_exports = {
        "batters": pd.DataFrame(
            {
                "game_date": ["2025-03-18"],
                "game_pk": [111],
                "player_id": [10],
                "team": ["NYM"],
                "opponent": ["ATL"],
                "is_home": [1],
                "bp_batter_home_run_probability": [0.12],
                "bp_batter_hit_probability": [0.64],
                "bp_batter_points_dk": [8.5],
                "bp_batter_points_fd": [10.1],
                "bp_batter_plate_appearances": [4.2],
                "bp_batter_at_bats": [3.7],
                "bp_batter_hits": [1.0],
                "bp_batter_bases": [1.6],
                "bp_batter_strikeouts": [0.9],
                "bp_batter_walks": [0.4],
                "bp_batter_runs": [0.5],
                "bp_batter_rbis": [0.6],
                "bp_batter_stolen_base_attempts": [0.1],
            }
        ),
        "pitchers": pd.DataFrame(
            {
                "game_date": ["2025-03-18"],
                "game_pk": [111],
                "player_id": [20],
                "team": ["NYM"],
                "opponent": ["ATL"],
                "is_home": [1],
                "bp_pitcher_points_dk": [9.0],
                "bp_pitcher_points_fd": [18.0],
                "bp_pitcher_batters_faced": [22.0],
                "bp_pitcher_innings": [5.2],
                "bp_pitcher_runs_allowed": [2.1],
                "bp_pitcher_hits_allowed": [4.2],
                "bp_pitcher_strikeouts": [6.1],
                "bp_pitcher_walks": [1.9],
                "bp_pitcher_home_runs_allowed": [0.6],
            }
        ),
        "teams": pd.DataFrame(
            {
                "game_date": ["2025-03-18"],
                "game_pk": [111],
                "team": ["NYM"],
                "opponent": ["ATL"],
                "is_home": [1],
                "bp_team_runs": [4.7],
                "bp_team_win_pct": [0.55],
                "bp_team_home_runs": [1.1],
                "bp_team_triples": [0.1],
                "bp_team_doubles": [1.4],
                "bp_team_singles": [5.2],
                "bp_team_walks": [4.8],
                "bp_team_strikeouts": [7.9],
            }
        ),
        "games": pd.DataFrame(
            {
                "game_date": ["2025-03-18"],
                "game_pk": [111],
                "bp_game_runs_away": [3.8],
                "bp_game_runs_home": [4.4],
                "bp_game_total_runs": [8.2],
                "bp_game_run_diff": [0.6],
                "bp_game_away_win_pct": [0.44],
                "bp_game_home_win_pct": [0.56],
                "bp_game_win_pct_gap": [0.12],
                "bp_game_runs_first_inning_pct": [0.5],
                "bp_game_runs_first5_away": [2.0],
                "bp_game_runs_first5_home": [2.4],
                "bp_game_first5_total_runs": [4.4],
                "bp_game_first5_run_diff": [0.4],
            }
        ),
    }

    with patch("tools.ballparkpal.feature_join.normalize_ballparkpal_exports", return_value=normalized_exports):
        joined, coverage = augment_model_dataset_with_ballparkpal(base_df, Path("unused"))

    assert isinstance(coverage, BallparkPalJoinCoverage)
    assert coverage.rows_in == 1
    assert coverage.rows_out == 1
    assert coverage.batter_coverage == 1.0
    assert coverage.pitcher_coverage == 1.0
    assert coverage.team_coverage == 1.0
    assert coverage.game_coverage == 1.0
    for column in (
        *BALLPARKPAL_BATTER_FEATURE_COLUMNS,
        *BALLPARKPAL_PITCHER_FEATURE_COLUMNS,
        *BALLPARKPAL_TEAM_FEATURE_COLUMNS,
        *BALLPARKPAL_GAME_FEATURE_COLUMNS,
    ):
        assert column in joined.columns

