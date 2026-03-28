param(
    [string]$TaskPrefix = "HomeRunAlgoDashboardRefresh",
    [string]$PythonPath = "",
    [string]$SettleRunTime = "02:00",
    [string]$PrepareRunTime = "04:00",
    [string[]]$PublishRunTimes = @("11:00", "13:00", "15:00", "18:00")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $PublishRunTimes -or $PublishRunTimes.Count -lt 1) {
    throw "Provide at least one publish run time, for example @('11:00', '13:00', '15:00', '18:00')."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$refreshScript = Join-Path $repoRoot "scripts\refresh_dashboard.ps1"
$resolvedPython = if ($PythonPath) {
    if ([System.IO.Path]::IsPathRooted($PythonPath)) { $PythonPath } else { Join-Path $repoRoot $PythonPath }
} else {
    Join-Path $repoRoot ".venv1\Scripts\python.exe"
}

if (-not (Test-Path $refreshScript)) {
    throw "Refresh script not found at $refreshScript"
}
if (-not (Test-Path $resolvedPython)) {
    throw "Python executable not found at $resolvedPython"
}

function New-TaskTriggerAtTime {
    param([Parameter(Mandatory = $true)][string]$TimeText)

    $parts = $TimeText.Split(":")
    if ($parts.Count -ne 2) {
        throw "Invalid time '$TimeText'. Use HH:mm in 24-hour local time."
    }

    $hours = [int]$parts[0]
    $minutes = [int]$parts[1]
    if ($hours -lt 0 -or $hours -gt 23 -or $minutes -lt 0 -or $minutes -gt 59) {
        throw "Invalid time '$TimeText'. Use HH:mm in 24-hour local time."
    }

    $start = (Get-Date).Date.AddHours($hours).AddMinutes($minutes)
    return New-ScheduledTaskTrigger -Daily -At $start
}

function Remove-TaskIfPresent {
    param([Parameter(Mandatory = $true)][string]$TaskName)

    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
}

$settleAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$refreshScript`" -PythonPath `"$resolvedPython`" -Mode settle"
$prepareAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$refreshScript`" -PythonPath `"$resolvedPython`" -Mode prepare"
$publishAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$refreshScript`" -PythonPath `"$resolvedPython`" -Mode publish"

$settleTrigger = New-TaskTriggerAtTime -TimeText $SettleRunTime
$prepareTrigger = New-TaskTriggerAtTime -TimeText $PrepareRunTime
$publishTriggers = @($PublishRunTimes | ForEach-Object { New-TaskTriggerAtTime -TimeText $_ })

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 72)

$settleTaskName = "${TaskPrefix}-Settle"
$prepareTaskName = "${TaskPrefix}-Prepare"
$publishTaskName = "${TaskPrefix}-Publish"

Remove-TaskIfPresent -TaskName "${TaskPrefix}-LateSettle"
Remove-TaskIfPresent -TaskName $settleTaskName
Remove-TaskIfPresent -TaskName $prepareTaskName
Remove-TaskIfPresent -TaskName $publishTaskName

Register-ScheduledTask -TaskName $settleTaskName -Action $settleAction -Trigger $settleTrigger -Principal $principal -Settings $settings -Force | Out-Null
Write-Host "Registered $settleTaskName at $SettleRunTime using -Mode settle"

Register-ScheduledTask -TaskName $prepareTaskName -Action $prepareAction -Trigger $prepareTrigger -Principal $principal -Settings $settings -Force | Out-Null
Write-Host "Registered $prepareTaskName at $PrepareRunTime using -Mode prepare"

Register-ScheduledTask -TaskName $publishTaskName -Action $publishAction -Trigger $publishTriggers -Principal $principal -Settings $settings -Force | Out-Null
Write-Host "Registered $publishTaskName at $($PublishRunTimes -join ', ') using -Mode publish"
