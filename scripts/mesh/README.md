# GH05T3 Dual Runtime (WSL + Windows)

One GH05T3 stack at a time on ports **8001**, **8002**, and **8090**.

## Which to use

| Goal | Runtime | Start |
|------|---------|-------|
| Full 16-service mesh (economy, sage, pipeline, NPU) | **Windows** | `native\windows\START_ALL.bat` |
| OSS/MVS dev, CUDA in WSL, sovereign-core in WSL | **WSL** | `bash scripts/wsl_start.sh` |
| Hybrid | Windows supervisor for economy/sage **only if** ports don't clash | Custom — avoid 8001/8002 overlap |

**Default probe env for sovereign-core (in WSL):** `GH05T3_RUNTIME=wsl`

**When GH05T3 runs via Windows supervisor and sovereign-core runs in WSL:**
```bash
export GH05T3_RUNTIME=windows
bash ~/sovereign-core/scripts/health_mesh.sh
```

## Detect active runtime

```bash
bash scripts/mesh/select_runtime.sh
# or export for child shells:
eval "$(bash scripts/mesh/select_runtime.sh --export)"
```

```powershell
.\scripts\mesh\select_runtime.ps1
```

## Stop before switching

```bash
# WSL
bash scripts/wsl_stop.sh

# Windows
python scripts/runtime/supervisor.py --stop
```