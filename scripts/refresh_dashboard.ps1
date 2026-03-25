param(
    [string]$PythonPath = ".\.venv1\Scripts\python.exe",
    [ValidateSet("settle", "publish")]
    [string]$Mode = "publish",
    [switch]$SkipPush,
    [string]$CommitMessage = "chore: refresh dashboard data"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$resolvedPython = if ([System.IO.Path]::IsPathRooted($PythonPath)) { $PythonPath } else { Join-Path $repoRoot $PythonPath }
if (-not (Test-Path $resolvedPython)) {
    throw "Python executable not found at $resolvedPython"
}

$env:PYTHONUTF8 = "1"

if ($Mode -eq "settle") {
    & $resolvedPython scripts\train_live_model.py --dataset-path data\live\model_training_dataset.csv
    & $resolvedPython scripts\settle_live_results.py
}
else {
    & $resolvedPython scripts\publish_live_picks.py
}

& $resolvedPython scripts\build_dashboard_artifacts.py --output-dir cloudflare-app\data

$status = git status --porcelain -- data/live cloudflare-app\data
if (-not $status) {
    Write-Host "No dashboard changes detected."
    exit 0
}

git add data/live/current_picks.json
git add data/live/pick_history.json
git add cloudflare-app/data/dashboard.json
git commit -m "$CommitMessage ($Mode)"

if (-not $SkipPush) {
    git push
}
