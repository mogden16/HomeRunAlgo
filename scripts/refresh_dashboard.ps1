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

if ($Mode -eq "publish") {
    & $resolvedPython scripts\run_refresh_mode.py --mode $Mode --refresh-results-before-publish
} else {
    & $resolvedPython scripts\run_refresh_mode.py --mode $Mode
}
if ($LASTEXITCODE -ne 0) {
    $workflowFailed = $true
    Write-Warning "run_refresh_mode.py reported a non-zero exit code for mode=$Mode."
}

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
