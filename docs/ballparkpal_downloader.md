# Ballpark Pal Export Downloader

## What this does
- Automates browser login to Ballpark Pal using Playwright.
- Navigates to the Export Center for one date or a date range.
- Downloads the four export XLSX files: `batters`, `pitchers`, `teams`, `games`.
- Writes files to `data/ballparkpal/raw/YYYY-MM-DD/` with timestamped names.
- Validates each file and writes a per-date `download_manifest.json`.

## Required environment variables
- `BALLPARKPAL_EMAIL`
- `BALLPARKPAL_PASSWORD`

Optional:
- `BALLPARKPAL_BASE_URL` (default: `https://www.ballparkpal.com`)
- `BALLPARKPAL_EXPORT_URL_TEMPLATE` (default: `{base_url}/Export-Center.php`)

Credentials are never printed to logs.

## Install
```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## Usage
Single date:
```bash
python tools/ballparkpal/download_exports.py --date 2026-04-14
```

Date range:
```bash
python tools/ballparkpal/download_exports.py --start-date 2026-04-10 --end-date 2026-04-14
```

Visible browser:
```bash
python tools/ballparkpal/download_exports.py --date 2026-04-14 --headless false
```

Dry run / selector test (no files downloaded):
```bash
python tools/ballparkpal/download_exports.py --date 2026-04-14 --test-mode --headless false
```

Overwrite existing files:
```bash
python tools/ballparkpal/download_exports.py --date 2026-04-14 --overwrite
```

## Output structure
Example:
```text
data/
  ballparkpal/
    raw/
      2026-04-14/
        2026-04-14_1030_batters.xlsx
        2026-04-14_1030_pitchers.xlsx
        2026-04-14_1030_teams.xlsx
        2026-04-14_1030_games.xlsx
        download_manifest.json
    logs/
      ballparkpal_download_<timestamp>.log
    run_manifest_<timestamp>.json
```

## Validation behavior
Each downloaded file is checked for:
- file exists
- `.xlsx` extension
- binary signature is not obvious HTML/login redirect
- workbook opens with `openpyxl`
- first-sheet columns contain a tolerant minimum of expected hint tokens per export

Validation details are written into `download_manifest.json`.

## Inspect archived exports
```bash
python tools/ballparkpal/inspect_exports.py --root-dir data/ballparkpal/raw
```

Optional strict mode (non-zero exit on any invalid/missing set):
```bash
python tools/ballparkpal/inspect_exports.py --strict
```

## Known caveats and risks
- Historical date downloads may not represent original point-in-time pregame snapshots.
- Use historical pulls first for automation checks and schema validation.
- For trustworthy forward backtesting, rely on daily archived live pulls going forward.
- Selector drift is possible if Ballpark Pal UI changes; selectors are centralized in `tools/ballparkpal/selectors.py`.

## How this fits a HomeRunAlgo backtest workflow
- Step 1: run daily export pulls and archive raw XLSX snapshots.
- Step 2: validate schema/quality and track failures via manifests and logs.
- Step 3: ingest archived exports into feature engineering and model backtest joins.
- Step 4: treat old-date pulls as exploratory until point-in-time behavior is proven.

If you also want to archive the daily slate and enrich picks in one pass, use
[docs/ballparkpal_daily_archive.md](./ballparkpal_daily_archive.md) and
`tools/ballparkpal/daily_archive_and_enrich.py`.
