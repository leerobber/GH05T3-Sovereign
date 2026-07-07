# 0009: Add CI, declare the missing torch dependency

**Status:** Resolved

## Context

Two gaps found while auditing the repo for improvement work, both
confirmed by actually running the suites, not by inspection alone:

1. `torch` was never declared in `backend/requirements.txt`, despite
   being a hard, unconditional `import torch` in ~20 core files —
   `gh05t3_binary/core/{binary_layers,transformer,attention,stabilizers}.py`,
   `gh05t3_binary/hardware/{detector,dispatcher}.py`,
   `gh05t3_binary/inference/*.py`, `backend/oss/swarm/*.py`, and
   `backend/integration/{gml_kernel_bridge,binary_backend}.py`. A fresh
   `pip install -r backend/requirements.txt` followed by anything that
   touches real inference/swarm code hit `ModuleNotFoundError: No module
   named 'torch'` immediately. `numpy` was already declared; `torch` was
   not.
2. No CI existed (`.github/workflows` was absent) despite a real, fast,
   fully-green test suite: 21/21 `cargo test` in `gml_kernel`, and
   116-119/119 `pytest` across `backend/oss/tests`, `backend/tests`, and
   `gh05t3_binary/tests` (the remainder skip cleanly and correctly —
   they need a live server or a trained checkpoint that CI won't have).
   Regressions could only be caught if someone remembered to run both
   suites locally before pushing.

## Decision

- Added `torch==2.12.1` to `backend/requirements.txt`, next to the other
  pinned ML deps, with a comment pointing Windows/Blackwell (RTX 5050)
  hardware at GH05T3's documented `cu128` index instead of the generic
  PyPI wheel.
- Added `.github/workflows/ci.yml` with two jobs: `rust` runs
  `cargo test` then `cargo build --release` in `gml_kernel/` and uploads
  `libgml_kernel.so` as an artifact; `python` downloads that artifact
  (so the FFI-dependent tests in `gh05t3_binary/tests` run for real
  instead of skipping), installs `backend/requirements.txt`, and runs
  `pytest` across the three test directories.

## Consequences

- `pip install -r backend/requirements.txt` now actually produces a
  working environment for the core binary-inference and swarm code, not
  just the FastAPI/gateway surface.
- CI now gates `main` and PRs on both suites. Building `libgml_kernel.so`
  in CI also means the FFI tests (`test_ternary_inference_ffi.py` and
  friends) run against the real kernel in CI, not just locally when a
  contributor happens to have built it — verified locally: building the
  `.so` took the Python suite from 116 passed / 97 skipped to 119 passed
  / 94 skipped.
- Not addressed here: the checkpoint-and-live-server-gated tests still
  skip in CI (no trained `binary_v2.pt`, no running gateway) — that's
  correct behavior, not a gap this change tries to close.
