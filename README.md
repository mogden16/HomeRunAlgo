# HomeRunAlgo

Real-data MLB player-game home run prediction pipeline built from Statcast and Meteostat, with a strictly chronological backtest.

## What this version does

- Pulls real 2024 regular-season Statcast data in cached weekly chunks via `pybaseball`.
- Aggregates pitch-level Statcast into one batter-game row with `hit_hr` as the target.
- Builds leakage-safe batter and opposing-pitcher rolling features using only information from dates before each game.
- Joins historical hourly weather from Meteostat using MLB park coordinates and a documented nearest-to-7PM local-time proxy.
- Trains a chronological holdout backtest with logistic regression by default and optional XGBoost if installed.

## Install

```bash
pip install -r requirements.txt
```

## Generate the real dataset

```bash
python generate_data.py
```

This writes:

- raw Statcast cache files to `data/raw/statcast_2024_*.csv`
- raw weather cache to `data/raw/weather_2024.csv`
- final engineered dataset to `data/mlb_player_game_real.csv`

Optional refresh / smaller test window examples:

```bash
python generate_data.py --force-refresh
python generate_data.py --start-date 2024-03-28 --end-date 2024-04-15
```

## Train / backtest

```bash
python train_model.py data/mlb_player_game_real.csv
python train_model.py data/mlb_player_game_real.csv --model xgboost
```

The training script preserves a clean date-boundary split so every test row occurs strictly after every training row.

## Real sourced features currently included

### Batter features
- `hr_rate_season_to_date`
- `hr_per_pa_last_30d`
- `barrel_rate_last_50_bbe`
- `hard_hit_rate_last_50_bbe`
- `avg_launch_angle_last_50_bbe`
- `avg_exit_velocity_last_50_bbe`
- `fly_ball_rate_last_50_bbe`
- `pull_air_rate_last_50_bbe` using an approximate Statcast spray-angle heuristic
- `batter_k_rate_season_to_date`
- `batter_bb_rate_season_to_date`
- `recent_form_hr_last_7d`
- `recent_form_barrels_last_14d`
- `expected_pa_proxy` from prior 14-day PA average
- `days_since_last_game`

### Opposing pitcher features
- `opp_pitcher_id`
- `pitch_hand_primary`
- `pitcher_hr9_season_to_date`
- `pitcher_barrel_rate_allowed_last_50_bbe`
- `pitcher_hard_hit_rate_allowed_last_50_bbe`
- `pitcher_fb_rate_allowed_last_50_bbe`
- `pitcher_k_rate_season_to_date`
- `pitcher_bb_rate_season_to_date`
- `starter_or_bullpen_proxy` from batters faced by the primary opposing pitcher

### Context features
- `temperature_f`
- `humidity_pct`
- `wind_speed_mph`
- `wind_direction_deg`
- `pressure_hpa`
- `platoon_advantage`

## Features intentionally left null or partial for now

- `park_factor_hr` is left null because no trusted park-factor source is wired into this first real-data version.
- `opp_pitcher_name` is currently null because the first pass uses Statcast IDs directly and does not yet enrich pitcher names from an additional roster lookup.
- `batting_order` is inferred from each batter's first plate appearance order within a game; this is useful but should be treated as an approximation.

## Leakage controls

- Season-to-date rates use shifted cumulative totals.
- Time-window features use rolling windows with `closed='left'`, so same-day results are excluded.
- Last-50-BBE features are computed from prior daily batted-ball aggregates before the current game date is added to the rolling store.
- Train/test splitting uses a clean date boundary: `max(train_date) < min(test_date)`.
- Imputation and scaling are fit on training data only through the scikit-learn pipeline.

## Notes / limitations

- This first version targets the 2024 regular season only.
- Weather is matched using the nearest available hourly observation to 7 PM local park time when exact game time is not available from the base Statcast pull.
- If a remote source fails, the pipeline raises an error instead of falling back to synthetic or fabricated data.
