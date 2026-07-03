# Canonical Pact workflow entrypoint (Windows PowerShell).
# All broker logic lives in scripts/*.py — this script only orchestrates.
param(
    [Parameter(Position = 0)]
    [string]$Command = "help",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root

$Py = if (Test-Path "$Root\backend\.venv\Scripts\python.exe") {
    "$Root\backend\.venv\Scripts\python.exe"
} elseif (Test-Path "$Root\.venv\Scripts\python.exe") {
    "$Root\.venv\Scripts\python.exe"
} else {
    "python"
}

$env:PYTHONPATH = if ($env:PYTHONPATH) { "$($env:PYTHONPATH);$Root\backend" } else { "$Root\backend" }
$env:AETHYRO_SKIP_LICENSE = if ($env:AETHYRO_SKIP_LICENSE) { $env:AETHYRO_SKIP_LICENSE } else { "1" }
$env:PACT_DO_NOT_TRACK = if ($env:PACT_DO_NOT_TRACK) { $env:PACT_DO_NOT_TRACK } else { "true" }

function Get-GitSha {
    try { return (git -C $Root rev-parse HEAD).Trim() } catch { return "local" }
}

switch ($Command) {
    "consumer" {
        & $Py -m pytest tests/test_oss_pact.py -q --tb=line @Args
    }
    "provider" {
        & $Py -m pytest tests/test_oss_provider_verify.py -q --tb=line @Args
    }
    "publish" {
        $ver = if ($Args.Count -gt 0) { $Args[0] } else { Get-GitSha }
        $tag = if ($Args.Count -gt 1) { $Args[1] } else { "ci" }
        & $Py scripts/publish_pacts.py pacts/ $ver $tag
    }
    "can-i-deploy" {
        $ver = if ($Args.Count -gt 0) { $Args[0] } else { Get-GitSha }
        $tag = if ($Args.Count -gt 1) { $Args[1] } else { "ci" }
        & $Py scripts/can_i_deploy.py `
            --consumer gh05t3-gateway `
            --provider gh05t3-oss `
            --version $ver `
            --tag $tag
    }
    "health" {
        & $Py scripts/broker_health.py
    }
    "all" {
        & $PSScriptRoot\run.ps1 consumer
        & $PSScriptRoot\run.ps1 provider
        & $PSScriptRoot\run.ps1 publish
        & $PSScriptRoot\run.ps1 can-i-deploy
    }
    default {
        @"
Usage: scripts/pact/run.ps1 <command>

  consumer       Generate pacts (pytest tests/test_oss_pact.py)
  provider       Verify provider (pytest tests/test_oss_provider_verify.py)
  publish [ver] [tag]  Publish pacts/ to broker (no-op without secrets)
  can-i-deploy [ver] [tag]
  health         Broker health check
  all            consumer -> provider -> publish -> can-i-deploy

Env: PACT_BROKER_BASE_URL or PACT_BROKER_URL, PACT_BROKER_TOKEN
WSL:   bash scripts/pact/run.sh <command>
"@
    }
}