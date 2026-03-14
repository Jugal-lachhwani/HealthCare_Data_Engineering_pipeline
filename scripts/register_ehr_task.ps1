param(
    [string]$TaskName = "Healthcare-EHR-Simulator",
    [int]$EveryMinutes = 5
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$runnerPath = Join-Path $projectRoot "scripts\run_ehr_simulator.bat"

if (-not (Test-Path $runnerPath)) {
    Write-Error "Runner script not found: $runnerPath"
    exit 1
}

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$runnerPath`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$trigger.RepetitionInterval = (New-TimeSpan -Minutes $EveryMinutes)
$trigger.RepetitionDuration = (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -AllowStartIfOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Runs synthetic EHR simulator at fixed interval." -Force

Write-Host "Scheduled task '$TaskName' created. It runs every $EveryMinutes minute(s)."
