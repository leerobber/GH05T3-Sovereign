# GH05T3-Sovereign

A fully sovereign, local, multi-backend, streaming, evolving cognitive engine.
Built from the GH05T3 architecture, designed for complete independence from
corporate AI infrastructure.

## Core Components

- **gml_kernel/** — Rust cognitive kernel (glyphs, core loop, evolution)
- **backend/** — Python runtime (ghost_llm, backend registry, streaming, blending)
- **backend/runtime/** — Orchestrator loop (dependency sentinels, evolution fitness)
- **models/** — Local inference models (Ollama, custom backends)

## Sovereign Features

- Local-only inference
- Multi-backend registry (llama, mistral, phi, etc.)
- Real streaming (SSE)
- Multi-model blending
- Dependency sentinels (FS, NET, GPU, WSL)
- Evolution fitness scoring
- Autonomous orchestration loop

## Architecture decisions

See [docs/architecture/](docs/architecture/README.md) for a real, tested
record of what was built and why — quantization approach, Rust kernel
strategy, the genome/evolution subsystem, and proposals that were
evaluated and rejected (with the reasons).
