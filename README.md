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

- The historical dataset generator defaults to the 2024 season, but the live workflow can refresh through the current date by passing explicit start and end dates.
- Weather is matched using the nearest available hourly observation to 7 PM local park time when exact game time is not available from the base Statcast pull.
- If a remote source fails, the pipeline raises an error instead of falling back to synthetic or fabricated data.

## Cloudflare Pages dashboard

This repo now includes a static dashboard app in `cloudflare-app/` for:

- picks generated from March 25, 2026 onward
- forward-only public performance tracking
- confidence-tier and player-level results once picks settle

Build the dashboard artifact locally with:

```bash
python scripts/build_dashboard_artifacts.py --output-dir cloudflare-app/data
```

That command writes:

- `cloudflare-app/data/dashboard.json`
- updates `data/live/pick_history.json` as the forward-only source of truth

Public tracking starts on `2026-03-25`. Historical 2024 and 2025 picks are not published in the dashboard.

The dashboard reads its latest picks from:

- `data/live/current_picks.json`

Use `data/live/current_picks.sample.json` as the template for that file.

Local-only live model assets are written to:

- `data/live/model_training_dataset.csv`
- `data/live/model_bundle.pkl`
- `data/live/model_metadata.json`

Those files are not part of the public Cloudflare artifact.

## Deploy to Cloudflare Pages

1. Push this repo to GitHub.
2. In Cloudflare Pages, create a new project from that repo.
3. Set the Pages build command to blank and the output directory to `cloudflare-app`.
4. Each time `cloudflare-app/data/dashboard.json` changes and is pushed, Cloudflare Pages will redeploy the site on the free tier.

## Local refresh twice per day

The workflow is split into two modes:

- `settle` at `07:00` ET: refresh historical data through yesterday, retrain the live model bundle, settle prior published picks, rebuild the dashboard, and push updates
- `publish` at `15:00` ET: fetch today's MLB slate, generate probable-lineup picks from the saved bundle, rebuild the dashboard, and push updates

Use the PowerShell wrapper directly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\refresh_dashboard.ps1 -PythonPath .\.venv1\Scripts\python.exe -Mode settle
powershell -ExecutionPolicy Bypass -File .\scripts\refresh_dashboard.ps1 -PythonPath .\.venv1\Scripts\python.exe -Mode publish
```

Or run the Python entrypoints individually:

```powershell
.\.venv1\Scripts\python.exe .\scripts\train_live_model.py
.\.venv1\Scripts\python.exe .\scripts\settle_live_results.py
.\.venv1\Scripts\python.exe .\scripts\publish_live_picks.py
.\.venv1\Scripts\python.exe .\scripts\build_dashboard_artifacts.py --output-dir cloudflare-app\data
```

The publish flow uses:

- `data/live/model_bundle.pkl` from the morning training run
- the free MLB Stats API for today's schedule and probable pitchers
- a recent-starter heuristic to approximate probable lineups
- free Open-Meteo forecasts for same-day weather inputs

The wrapper script then:

- rebuilds `cloudflare-app/data/dashboard.json`
- merges the latest picks into `data/live/pick_history.json`
- commits the dashboard data if it changed
- pushes the commit so Cloudflare Pages redeploys automatically

To register two local scheduled tasks on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_dashboard_tasks.ps1
```

The default run times are `07:00` and `15:00` local time. Adjust them by passing a two-item `-RunTimes` array.
