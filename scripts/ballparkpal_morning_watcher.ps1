param(
    [string]$RunTime = "06:10",
    [string]$StateFile = "data/ballparkpal/runtime/morning_watcher_state.json",
    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$runScript = Join-Path $repoRoot "scripts\run_ballparkpal_morning_refresh.ps1"
$statePath = if ([System.IO.Path]::IsPathRooted($StateFile)) { $StateFile } else { Join-Path $repoRoot $StateFile }

if (-not (Test-Path $runScript)) {
    throw "Run script not found at $runScript"
}

$parts = $RunTime.Split(":")
if ($parts.Count -ne 2) {
    throw "Invalid RunTime '$RunTime'. Use HH:mm in 24-hour local time."
}
$hours = [int]$parts[0]
$minutes = [int]$parts[1]
if ($hours -lt 0 -or $hours -gt 23 -or $minutes -lt 0 -or $minutes -gt 59) {
    throw "Invalid RunTime '$RunTime'. Use HH:mm in 24-hour local time."
}

function Read-State {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return @{ last_run_date = $null }
    }
    try {
        $raw = Get-Content -LiteralPath $Path -Raw -ErrorAction Stop
        return ($raw | ConvertFrom-Json -ErrorAction Stop)
    } catch {
        return @{ last_run_date = $null }
    }
}

function Write-State {
    param([string]$Path, [string]$LastRunDate)
    $payload = @{
        last_run_date = $LastRunDate
        updated_at = (Get-Date).ToString("o")
    }
    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-MorningRefresh {
    param([string]$RequestedDate)
    & $runScript -RequestedDate $RequestedDate -EnvFile $EnvFile
    if ($LASTEXITCODE -ne 0) {
        throw "Morning refresh failed with exit code $LASTEXITCODE"
    }
}

$state = Read-State -Path $statePath
$lastRunDate = $state.last_run_date

Write-Host "Ballpark Pal watcher started. Run time: $RunTime. State: $statePath"

while ($true) {
    $now = Get-Date
    $today = $now.ToString("yyyy-MM-dd")
    $target = $now.Date.AddHours($hours).AddMinutes($minutes)

    if ($now -ge $target -and $lastRunDate -ne $today) {
        Write-Host "Running Ballpark Pal morning refresh for $today"
        Invoke-MorningRefresh -RequestedDate $today
        $lastRunDate = $today
        Write-State -Path $statePath -LastRunDate $today
        Write-Host "Completed Ballpark Pal morning refresh for $today"
    }

    Start-Sleep -Seconds 60
}
