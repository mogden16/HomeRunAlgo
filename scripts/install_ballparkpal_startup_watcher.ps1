param(
    [string]$StartupFolder = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$watcherScript = Join-Path $repoRoot "scripts\ballparkpal_morning_watcher.ps1"
$powershellExe = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"

if (-not (Test-Path $watcherScript)) {
    throw "Watcher script not found at $watcherScript"
}
if (-not (Test-Path $powershellExe)) {
    throw "PowerShell executable not found at $powershellExe"
}

$startupRoot = if ($StartupFolder) {
    $StartupFolder
} else {
    Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
}

if (-not (Test-Path $startupRoot)) {
    throw "Startup folder not found at $startupRoot"
}

$cmdPath = Join-Path $startupRoot "HomeRunAlgoBallparkPalMorning.cmd"
$cmdContent = @"
@echo off
start ""HomeRunAlgoBallparkPalMorning"" /min "$powershellExe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "$watcherScript"
"@

Set-Content -LiteralPath $cmdPath -Value $cmdContent -Encoding ASCII
Write-Host "Installed startup watcher to $cmdPath"
