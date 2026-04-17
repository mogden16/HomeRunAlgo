# Ballpark Pal Daily Archive and Pick Enrichment

## What this script does
- Logs into Ballpark Pal with credentials from environment variables.
- Downloads the four Export Center workbooks for one slate date into `data/ballparkpal/raw/YYYY-MM-DD/`.
- Validates the downloaded Excel files.
- Cross-references a picks file against the archived snapshot.
- Writes an enriched picks file with:
  - `bp_batter_home_run_probability`
  - `bp_batter_hit_probability`
  - `bp_pitcher_runs_allowed`
  - `bp_pitcher_home_runs_allowed`
- Adds a fixed-weight Ballpark Pal overlay:
  - signed score centered at zero
  - normalized `0-100` display score
  - adjusted ranking score that leaves the original model score intact
- Writes a run manifest and logs for auditing.

## Important caveat
- Historical Export Center pulls have already shown behavior consistent with current-slate data being served into old-date folders.
- Treat this as a daily forward archive and validation workflow, not proof of true historical point-in-time snapshots.
- For backtesting, only trust daily archives collected going forward after you verify the workbook dates inside the files.

## Environment variables
Required:
- `BALLPARKPAL_EMAIL`
- `BALLPARKPAL_PASSWORD`

Optional:
- `BALLPARKPAL_BASE_URL`
- `BALLPARKPAL_EXPORT_URL_TEMPLATE`

## Install
```powershell
pip install -r requirements.txt
python -m playwright install chromium
```

## Usage
Run for today by default:
```powershell
python tools/ballparkpal/daily_archive_and_enrich.py
```

Run for a specific date:
```powershell
python tools/ballparkpal/daily_archive_and_enrich.py --date 2026-04-16
```

Visible browser:
```powershell
python tools/ballparkpal/daily_archive_and_enrich.py --headless false
```

Use a specific picks file:
```powershell
python tools/ballparkpal/daily_archive_and_enrich.py --picks-path data/live/current_picks.json
```

Test selector/login flow only:
```powershell
python tools/ballparkpal/daily_archive_and_enrich.py --test-mode --headless false
```

## Overlay rules
- `bp_batter_home_run_probability` and `bp_batter_hit_probability` support the pick when they are above their neutral thresholds.
- `bp_pitcher_runs_allowed` and `bp_pitcher_home_runs_allowed` support the pick when they are above their neutral thresholds.
- The overlay is fixed-weight and inspectable.
- The original model score is preserved in the enriched output.
- The enriched file adds:
  - `ballparkpal_overlay_signed_score`
  - `ballparkpal_overlay_display_score`
  - `ballparkpal_overlay_adjusted_score`
  - `ballparkpal_overlay_adjusted_rank`
  - per-factor alignment columns for each core signal

Neutral thresholds and weights:
- Batter HR probability: neutral `0.10`, weight `12`
- Batter hit probability: neutral `0.70`, weight `6`
- Pitcher runs allowed: neutral `4.5`, weight `8`
- Pitcher home runs allowed: neutral `0.80`, weight `4`

## Output
Raw exports:
```text
data/ballparkpal/raw/YYYY-MM-DD/
  YYYY-MM-DD_HHMM_batters.xlsx
  YYYY-MM-DD_HHMM_pitchers.xlsx
  YYYY-MM-DD_HHMM_teams.xlsx
  YYYY-MM-DD_HHMM_games.xlsx
  download_manifest.json
```

Enriched picks:
```text
data/ballparkpal/enriched/YYYY-MM-DD/
  YYYY-MM-DD_ballparkpal_picks_enriched.csv
  YYYY-MM-DD_ballparkpal_picks_enriched.json
  picks_enrichment_manifest.json
```

Combined run manifest:
```text
data/ballparkpal/runs/ballparkpal_daily_run_<timestamp>.json
```

## Scheduling on Windows
Use Task Scheduler to run the daily script once per day after Ballpark Pal publishes the slate.
- Program: `python`
- Arguments: `tools/ballparkpal/daily_archive_and_enrich.py --headless true`
- Start in: repo root

## What to verify first
1. The raw XLSX files download into the expected date folder.
2. The workbook `GameDate` inside each file matches the requested date for the current slate.
3. The enriched picks file has non-null values for the four requested Ballpark Pal fields.
4. The run manifest records the file paths and coverage summary.
