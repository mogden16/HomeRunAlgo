from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.ballparkpal import feature_join


def test_pick_enrichment_joins_requested_fields(monkeypatch) -> None:
    picks = pd.DataFrame(
        [
            {
                "pick_id": "2026-04-16:1",
                "game_date": "2026-04-16",
                "game_pk": 123,
                "batter_id": 10,
                "pitcher_id": 20,
                "team": "ATL",
                "opponent_team": "NYM",
            }
        ]
    )

    exports = {
        "batters": pd.DataFrame(
            [
                {
                    "game_date": "2026-04-16",
                    "game_pk": 123,
                    "player_id": 10,
                    "team": "ATL",
                    "opponent": "NYM",
                    "bp_batter_home_run_probability": 0.2,
                    "bp_batter_hit_probability": 0.6,
                }
            ]
        ),
        "pitchers": pd.DataFrame(
            [
                {
                    "game_date": "2026-04-16",
                    "game_pk": 123,
                    "player_id": 20,
                    "team": "NYM",
                    "opponent": "ATL",
                    "bp_pitcher_runs_allowed": 3.1,
                    "bp_pitcher_home_runs_allowed": 0.4,
                }
            ]
        ),
        "teams": pd.DataFrame(),
        "games": pd.DataFrame(),
    }

    monkeypatch.setattr(feature_join, "load_ballparkpal_snapshot", lambda _: exports)

    enriched, coverage, overlay_summary = feature_join.enrich_picks_with_ballparkpal(picks, Path("unused"))

    assert enriched.loc[0, "bp_batter_home_run_probability"] == 0.2
    assert enriched.loc[0, "bp_batter_hit_probability"] == 0.6
    assert enriched.loc[0, "bp_pitcher_runs_allowed"] == 3.1
    assert enriched.loc[0, "bp_pitcher_home_runs_allowed"] == 0.4
    assert coverage.batter_coverage == 1.0
    assert coverage.pitcher_coverage == 1.0
    assert "ballparkpal_overlay_signed_score" in enriched.columns
    assert "ballparkpal_overlay_display_score" in enriched.columns
    assert "ballparkpal_overlay_adjusted_score" in enriched.columns
    assert enriched.loc[0, "ballparkpal_overlay_grade"] in {"against", "mixed", "supportive"}
    assert overlay_summary.rows_in == 1
