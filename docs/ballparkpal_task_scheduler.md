# Ballpark Pal Morning Task

Use a Windows startup watcher to pull Ballpark Pal exports every morning and refresh the validated snapshot artifact.

## What it runs

- [scripts/run_ballparkpal_morning_refresh.ps1](../scripts/run_ballparkpal_morning_refresh.ps1)
- That wrapper:
  - downloads the four Ballpark Pal exports for today's slate
  - validates the workbooks
  - writes the normalized snapshot under `data/ballparkpal/validated/YYYY-MM-DD/`
  - updates `data/ballparkpal/validated/latest_snapshot.json`

## Default schedule

- Daily at `06:10` local time
- Starts when you log into Windows and then stays alive to run daily
- This is the no-password path for this machine

## Install the startup watcher

From the repo root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\install_ballparkpal_startup_watcher.ps1
```

That creates a `HomeRunAlgoBallparkPalMorning.cmd` file in your Windows Startup folder.

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

- The watcher assumes the machine is on and you log into Windows at least once.
- If you later want a true service-style unattended job, you will need an elevated Task Scheduler setup or a service account.
- The task does not publish the dashboard directly. Live refresh paths read the latest validated snapshot throughout the day.
- The snapshot remains a daily forward archive. Historical date pulls are not assumed to be true point-in-time backfills unless the workbook content matches the requested slate date.
