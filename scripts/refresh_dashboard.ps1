param(
    [string]$PythonPath = ".\.venv1\Scripts\python.exe",
    [ValidateSet("settle", "prepare", "publish")]
    [string]$Mode = "prepare",
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
$trackedFiles = @(
    "data/live/current_picks.json",
    "data/live/pick_history.json",
    "cloudflare-app/data/dashboard.json"
)
$workflowFailed = $false

if ($Mode -eq "settle") {
    & $resolvedPython scripts\refresh_live_results.py --dataset-path data\live\model_training_dataset.csv
    & $resolvedPython scripts\settle_live_results.py
}
elseif ($Mode -eq "prepare") {
    & $resolvedPython scripts\prepare_live_board.py
    if ($LASTEXITCODE -ne 0) {
        $workflowFailed = $true
        Write-Warning "prepare_live_board.py failed, but tracked public artifacts may have been refreshed. Continuing to rebuild, verify, and push them."
    }
}
else {
    & $resolvedPython scripts\publish_live_picks.py
    if ($LASTEXITCODE -ne 0) {
        $workflowFailed = $true
        Write-Warning "publish_live_picks.py failed, but tracked public artifacts may have been refreshed. Continuing to rebuild, verify, and push them."
    }
}

& $resolvedPython scripts\build_dashboard_artifacts.py --output-dir cloudflare-app\data
& $resolvedPython scripts\verify_public_live_artifacts.py

$status = git status --porcelain -- $trackedFiles
if (-not $status) {
    Write-Host "No dashboard changes detected."
    exit 0
}

git add -- $trackedFiles
git diff --cached --quiet -- $trackedFiles
if ($LASTEXITCODE -eq 0) {
    Write-Host "No staged public/live artifact changes detected."
    exit 0
}
git commit -m "$CommitMessage ($Mode)"

if (-not $SkipPush) {
    git push
}

if ($workflowFailed) {
    exit 1
}
