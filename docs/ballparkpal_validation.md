# Ballpark Pal Validation Workflow

This repo now includes a validation-first Ballpark Pal workflow that downloads the daily Export Center workbooks, checks that they are real Excel files, and refuses to treat stale or mismatched content as valid.

## What it does

The workflow:

1. Logs into Ballpark Pal with credentials from environment variables or a local `.env` file.
2. Opens the Export Center.
3. Downloads the four daily Excel exports:
   - Batters
   - Pitchers
   - Teams
   - Games
4. Validates each workbook with `openpyxl`.
5. Checks that each workbook contains the requested slate date.
6. Writes a manifest and a log file.
7. Optionally writes a small validation overlay JSON file if all exports validate.

The workflow is intentionally validation-first. It does not trust a historical request unless the workbook contents themselves prove that the requested date is present.

## Inputs

Set credentials in either environment variables or a local `.env` file:

```text
BALLPARKPAL_EMAIL=you@example.com
BALLPARKPAL_PASSWORD=your-password
```

Optional overrides:

- `BALLPARKPAL_LOGIN_URL`
- `BALLPARKPAL_EXPORT_CENTER_URL`

The code also accepts explicit URL overrides from the CLI.

## Outputs

By default, successful runs write files under:

- `data/ballparkpal/raw/YYYY-MM-DD/`
- `data/ballparkpal/logs/`

The manifest includes:

- requested date
- pulled-at timestamp
- original filename
- saved filename
- sha256 checksum
- file size
- validation result
- detected workbook date
- mismatch notes and errors

## Run it

Run the module directly:

```bash
python -m tools.ballparkpal --requested-date 2026-04-17
```

If Playwright is installed but the browser binary is not, install Chromium once:

```bash
python -m playwright install chromium
```

To run in a visible browser:

```bash
python -m tools.ballparkpal --requested-date 2026-04-17 --headed
```

To write a validation overlay JSON file:

```bash
python -m tools.ballparkpal --requested-date 2026-04-17 --overlay-output data/ballparkpal/overlay.json
```

## Validation rules

The validator rejects a file when any of the following is true:

- the file is missing
- the file does not end in `.xlsx`
- `openpyxl` cannot open it
- the bytes look like HTML or plain text instead of an Excel zip archive
- the workbook lacks the expected sheet/schema shape
- the workbook date fields do not match the requested slate date

If a historical request returns current-slate content, the date check fails and the run is marked invalid.

## Tests

The repository includes tests for:

- accepting a valid `.xlsx`
- rejecting HTML masquerading as Excel
- rejecting workbook date mismatches
- building the optional overlay output
