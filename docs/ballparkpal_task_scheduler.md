# Ballpark Pal Morning Task

Use a Windows Task Scheduler job to pull Ballpark Pal exports every morning and refresh the validated snapshot artifact.

## What it runs

- [scripts/run_ballparkpal_morning_refresh.ps1](../scripts/run_ballparkpal_morning_refresh.ps1)
- That wrapper:
  - downloads the four Ballpark Pal exports for today's slate
  - validates the workbooks
  - writes the normalized snapshot under `data/ballparkpal/validated/YYYY-MM-DD/`
  - updates `data/ballparkpal/validated/latest_snapshot.json`

## Default schedule

- Daily at `06:10` local time
- Uses the current machine user with an S4U logon token so it can run unattended

## Register the task

From the repo root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\register_ballparkpal_task.ps1
```

Optional overrides:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\register_ballparkpal_task.ps1 -RunTime 06:30
```

## Manual run

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_ballparkpal_morning_refresh.ps1
```

To pull only the exports and write the snapshot, skipping any dashboard publish step:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_ballparkpal_morning_refresh.ps1 -SkipPublish
```

## Requirements

- `.env` at the repo root with `BALLPARKPAL_EMAIL` and `BALLPARKPAL_PASSWORD`
- `.venv1` with project dependencies installed
- Playwright Chromium installed:

```powershell
python -m playwright install chromium
```

## Notes

- The task assumes the machine is on.
- Because the task uses S4U, it does not require an interactive desktop session.
- The task does not publish the dashboard directly. Live refresh paths read the latest validated snapshot throughout the day.
- The snapshot remains a daily forward archive. Historical date pulls are not assumed to be true point-in-time backfills unless the workbook content matches the requested slate date.
