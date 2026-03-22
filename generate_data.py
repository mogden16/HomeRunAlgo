"""
generate_data.py

Generates a synthetic MLB player-game dataset for development and testing.
Each row represents one player-game appearance (plate appearance context).

Columns produced:
  game_date       - date of the game (used for chronological splitting)
  player_id       - integer player identifier
  team            - batter's team abbreviation
  opponent        - opponent team abbreviation
  ballpark        - ballpark name
  hr_rate_career  - simulated career HR rate for the batter (static, no leakage)
  iso_season_td   - isolated power, season-to-date *before* this game (rolling, no leakage)
  hr_last_30      - HR in the last 30 games *before* this game (rolling, no leakage)
  at_bats_season  - at-bats accumulated in the season *before* this game (rolling)
  pitcher_hr9     - opponent pitcher's season HR/9 *before* this game (rolling)
  park_factor     - HR park factor for the ballpark (static)
  hit_hr          - TARGET: 1 if the batter hit a home run in this game, else 0
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_PLAYERS = 60
TEAMS = [
    "NYY", "BOS", "LAD", "HOU", "ATL", "CHC", "STL", "SF",
    "PHI", "SD", "SEA", "TB", "MIN", "CLE", "MIL",
]
PARKS = {team: f"{team}_Park" for team in TEAMS}
# Simulated park HR factors (1.0 = neutral, >1 = hitter-friendly)
PARK_FACTORS = {team: round(np.random.default_rng(SEED).uniform(0.85, 1.20), 3)
                for team in TEAMS}


def generate_mlb_dataset(output_path: str = "data/mlb_player_game.csv") -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # Season: 2024 regular season (~April 1 – Sep 29, ~182 days)
    season_start = pd.Timestamp("2024-04-01")
    season_end = pd.Timestamp("2024-09-29")
    game_dates = pd.date_range(season_start, season_end, freq="D")
    # ~162 games but not every player plays every day
    game_dates = [d for d in game_dates if d.weekday() != 0 or rng.random() > 0.3]

    records = []

    # Assign each player a true latent HR talent (used to generate realistic probabilities)
    player_talents = {pid: rng.beta(1.5, 18) for pid in range(N_PLAYERS)}
    player_teams = {pid: TEAMS[pid % len(TEAMS)] for pid in range(N_PLAYERS)}

    for player_id in range(N_PLAYERS):
        talent = player_talents[player_id]
        team = player_teams[player_id]
        # season-level accumulators (reset each season, updated *before* each game row)
        hr_this_season = 0
        ab_this_season = 0
        hr_last30_window: list[int] = []

        for i, gdate in enumerate(game_dates):
            # Skip ~15% of games (days off, bench appearances)
            if rng.random() < 0.15:
                continue

            opponent = rng.choice([t for t in TEAMS if t != team])
            park = team if rng.random() > 0.5 else opponent  # home or away

            park_factor = PARK_FACTORS[park]

            # Season-to-date features built BEFORE this game (no leakage)
            games_played_so_far = i + 1
            iso_season_td = (hr_this_season / max(ab_this_season, 1)) * 4  # crude ISO proxy
            hr_last_30 = sum(hr_last30_window[-30:])
            at_bats_season = ab_this_season

            # Pitcher HR/9: simulate opponent pitcher quality (season-to-date before game)
            # Higher value = pitcher allows more HRs
            pitcher_hr9 = rng.gamma(shape=2.0, scale=0.6)

            # True probability of hitting HR this game
            p_hr = (
                0.5 * talent
                + 0.15 * park_factor * talent
                + 0.05 * (pitcher_hr9 / 5.0)
                + 0.02 * min(hr_last_30 / 5, 1.0)  # hot-hand signal (weak)
            )
            p_hr = float(np.clip(p_hr, 0.001, 0.25))

            hit_hr = int(rng.random() < p_hr)

            records.append({
                "game_date": gdate,
                "player_id": player_id,
                "team": team,
                "opponent": opponent,
                "ballpark": PARKS[park],
                "hr_rate_career": round(talent, 4),
                "iso_season_td": round(iso_season_td, 4),
                "hr_last_30": hr_last_30,
                "at_bats_season": at_bats_season,
                "pitcher_hr9": round(pitcher_hr9, 4),
                "park_factor": park_factor,
                "hit_hr": hit_hr,
            })

            # Update rolling accumulators AFTER the row is recorded
            ab_game = int(rng.integers(2, 5))
            hr_this_season += hit_hr
            ab_this_season += ab_game
            hr_last30_window.append(hit_hr)

    df = pd.DataFrame(records).sort_values("game_date").reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    generate_mlb_dataset()
