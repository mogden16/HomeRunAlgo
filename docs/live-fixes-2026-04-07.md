# Live Fix Notes - 2026-04-07

## Confirmed root causes

- Weather labels could stay stuck at `"Unknown"` even when real weather fields were present. The confirmed path was the same-day metadata merge in `scripts/publish_live_picks.py`, which preferred the existing placeholder label over refreshed weather data.
- Forecast requests had no persistent same-day cache, so repeated mixed/publish refreshes could hit Open-Meteo again for the same parks and date.
- Same-day pick storage was only keyed by `pick_id`, not by player/day. If the same slate was regenerated with a changed matchup or a different `pick_id`, the code path could keep a second row instead of updating the canonical player/day record and merging the reasoning.
- The full slate recap UI still rendered a long player table even though the summary cards already carried the useful recap totals.

## Fix summary

- Added persistent same-day forecast caching plus structured weather lookup debug logging in `scripts/live_pipeline.py`.
- Fixed weather merge behavior so refreshed weather labels replace placeholder `"Unknown"` values in `scripts/publish_live_picks.py`.
- Added same-day player/day consolidation and reason merging in `scripts/live_pipeline.py`, `scripts/board_state.py`, and `scripts/build_dashboard_artifacts.py`.
- Updated the dashboard UI in `cloudflare-app/app.js` so missing weather degrades to a clear fallback message and the yesterday/full-slate recap stays summary-only.
- Added regression coverage for cache reuse, weather label recovery, and same-day reason merging.

## Remaining risk / follow-up

- The installed Windows scheduled tasks currently point to an older repo path: `C:\\Users\\Mogde\\PycharmProjects\\HomeRunAlgo_masterpush\\scripts\\refresh_dashboard.ps1`. That is an operational risk outside the repo code. Re-registering the tasks from the current repo would remove that mismatch.
