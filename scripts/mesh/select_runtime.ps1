# Detect active GH05T3 runtime (Windows vs WSL) for mesh probes.
# Usage: . .\scripts\mesh\select_runtime.ps1
param([switch]$Export)

function Test-Port([string]$HostName, [int]$Port) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $iar = $c.BeginConnect($HostName, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(800, $false)
        if ($ok -and $c.Connected) { $c.Close(); return $true }
        $c.Close()
    } catch {}
    return $false
}

$wslGw = Test-Port "127.0.0.1" 8002
$wslSup = Test-Port "127.0.0.1" 8090
# From Windows, WSL services are on localhost when using mirrored networking
$winGw = $wslGw
$winSup = $wslSup

# Also check if Windows supervisor is native (same ports)
if (Test-Port "127.0.0.1" 8002) { $active = "windows" }
elseif ($env:GH05T3_RUNTIME) { $active = $env:GH05T3_RUNTIME }
else { $active = "windows" }

# WSL stack detection: if only WSL launcher ran without supervisor
if (-not $wslSup -and $wslGw) { $active = "wsl" }
if ($wslSup -and $wslGw) {
    if ($active -eq "windows") {
        Write-Warning "Port 8002 in use — prefer Windows supervisor OR WSL launcher, not both."
        Write-Host "Stop WSL: wsl bash /mnt/c/Users/leer4/GH05T3/scripts/wsl_stop.sh"
        Write-Host "Stop Windows: python scripts\runtime\supervisor.py --stop"
    }
}

if ($Export) {
    "GH05T3_RUNTIME=$active"
    exit 0
}

Write-Host "GH05T3 runtime: $active"
Write-Host "  Gateway :8002  $(if ($wslGw) { 'UP' } else { 'down' })"
Write-Host "  Supervisor :8090 $(if ($wslSup) { 'UP' } else { 'down' })"
$env:GH05T3_RUNTIME = $active