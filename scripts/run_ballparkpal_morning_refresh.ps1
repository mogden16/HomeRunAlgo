param(
    [string]$PythonPath = "",
    [string]$RequestedDate = "",
    [string]$OutputRoot = "data/ballparkpal",
    [string]$EnvFile = ".env",
    [switch]$Headed,
    [switch]$SkipPublish
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedPython = if ($PythonPath) {
    if ([System.IO.Path]::IsPathRooted($PythonPath)) { $PythonPath } else { Join-Path $repoRoot $PythonPath }
} else {
    Join-Path $repoRoot ".venv1\Scripts\python.exe"
}

if (-not (Test-Path $resolvedPython)) {
    throw "Python executable not found at $resolvedPython"
}

$resolvedRequestedDate = if ($RequestedDate) {
    $RequestedDate
} else {
    (Get-Date).ToString("yyyy-MM-dd")
}

$resolvedEnvFile = if ($EnvFile) {
    if ([System.IO.Path]::IsPathRooted($EnvFile)) { $EnvFile } else { Join-Path $repoRoot $EnvFile }
} else {
    ""
}

$ballparkArgs = @(
    "-m", "tools.ballparkpal",
    "--requested-date", $resolvedRequestedDate,
    "--output-root", (Join-Path $repoRoot $OutputRoot)
)
if ($resolvedEnvFile -and (Test-Path $resolvedEnvFile)) {
    $ballparkArgs += @("--env-file", $resolvedEnvFile)
}
if ($Headed) {
    $ballparkArgs += "--headed"
}

Write-Host "Running Ballpark Pal validation for $resolvedRequestedDate"
& $resolvedPython @ballparkArgs
if ($LASTEXITCODE -ne 0) {
    throw "Ballpark Pal validation failed with exit code $LASTEXITCODE"
}

if (-not $SkipPublish) {
    $publishArgs = @(
        (Join-Path $repoRoot "scripts\publish_live_picks.py"),
        "--schedule-date", $resolvedRequestedDate,
        "--overlay-only"
    )
    Write-Host "Publishing overlay-only dashboard update for $resolvedRequestedDate"
    & $resolvedPython @publishArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Overlay-only publish failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Ballpark Pal morning refresh complete for $resolvedRequestedDate"
