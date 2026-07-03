# Load Pact Broker env vars from repo-root .env for local PowerShell testing.
# Usage:
#   .\scripts\set-pact-env.ps1
#   .\scripts\pact\run.ps1 consumer

$Root = Split-Path $PSScriptRoot -Parent
$EnvFile = Join-Path $Root ".env"

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            $name = $Matches[1].Trim()
            $value = $Matches[2].Trim().Trim('"').Trim("'")
            if ($name -match '^PACT_BROKER') {
                Set-Item -Path "env:$name" -Value $value
            }
        }
    }
}

if (-not $env:PACT_BROKER_BASE_URL -and -not $env:PACT_BROKER_URL) {
    Write-Host "No PACT_BROKER_* vars in .env — publish/can-i-deploy will no-op (by design)."
} else {
    Write-Host "Pact broker env loaded for this session."
}
Write-Host "Next: .\scripts\pact\run.ps1 consumer"