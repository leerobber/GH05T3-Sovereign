# Claude ↔ Gemini Bridge

Lets Claude and Gemini hand work back and forth on Aethyro — Claude drives
architecture/code, Gemini adds vision and review. Keys are read from
`GH05T3/.env` (`GOOGLE_AI_KEY` for Gemini, `ANTHROPIC_API_KEY` for Claude),
matching the rest of the stack. Uses the new `google-genai` SDK.

## Phase 1 — on-demand tool (`gemini_bridge.py`)

Claude Code calls this via Bash to borrow Gemini mid-task (text + vision + files):

```bash
cd C:\Users\leer4\GH05T3\bridge
python gemini_bridge.py ask "What's broken in this UI?" --image shot.png
python gemini_bridge.py ask "Review for bugs" --file app.py --file utils.py
python gemini_bridge.py ask "Summarize" --system "Be terse." --json
echo "long prompt" | python gemini_bridge.py ask -
```

Flags: `--image`/`--file` (repeatable), `--system`, `--model` (default
`gemini-2.0-flash`), `--json`. API/billing errors print one clean line to
stderr with exit code 2.

Importable too:

```python
from gemini_bridge import GeminiBridge
print(GeminiBridge().ask("hi", images=["shot.png"]))
```

## Phase 2 — autonomous loop (`mesh_loop.py`)

Runs N rounds of Claude ↔ Gemini on one task. Each side sees the running
transcript; images go to Gemini on round 1. Ends at `--turns` or early when both
emit `[[DONE]]`. Claude writes a final synthesis. Transcript saved to
`bridge/transcripts/mesh_<timestamp>.md`.

```bash
python mesh_loop.py --task "Design the Aethyro onboarding flow" --turns 6
python mesh_loop.py --task "Critique this dashboard" --image dash.png --turns 4
```

## Phase 3 — judicial mesh (`judicial_mesh.py`)

Local, no Docker. Gemini scans a repo and proposes a concrete fix; Claude reviews
it; they negotiate one candidate change. A patch is **signed only when both
conditions hold**: (1) consensus — both approve the same candidate — and (2)
evidence — the patched code passes the repo's tests in a throwaway sandbox copy.
A failing test is fed back so the agents must revise. Neither agreement nor a
green test alone is enough.

```bash
python judicial_mesh.py --path C:\Users\leer4\GH05T3\sovereignnation
python judicial_mesh.py --path . --focus "fix the broken voiceover script" --turns 8
python judicial_mesh.py --path ..\aethyro_launch --test-cmd "python -m pytest -q"
```

Flags: `--path` (default GH05T3 root), `--focus` (steer at a specific issue),
`--turns` (default 8), `--test-cmd` (auto-detected: pytest / npm test),
`--max-files` (scan budget, default 25), `--workspace` (where `agent_patch.md`
lands, default `bridge/workspace`).

**Both seats are swappable** via `providers.py` — proposer = `--provider`,
reviewer = `--reviewer`, each one of `groq | gemini | ollama | claude`
(`--model` / `--reviewer-model` to override the model). If both seats use the
same provider they're labelled `NAME-A` / `NAME-B`. Only `gemini` has vision.

```bash
# Works today with zero billing (verified end-to-end):
python judicial_mesh.py --path <dir> --provider groq --reviewer groq
# The intended config, once Anthropic + Gemini billing is restored:
python judicial_mesh.py --path <dir> --provider gemini --reviewer claude
# Fully local / offline:
python judicial_mesh.py --path <dir> --provider ollama --reviewer ollama --model llama3.2
```

Output `agent_patch.md` is marked **SIGNED** (consensus + tests green),
**UNVERIFIED** (consensus but no test suite found), or **NOT SIGNED** (no
consensus within the turn limit) — with the unified diff, last test output, and
full transcript. Aim it at a focused subproject, not the whole 50 GB tree; the
sandbox copy skips `models/`, `*.gguf`, `venv/`, `.git/`, etc.

## Phase 4 — long-horizon memory (`memory_store.py`)

Closes the cross-run learning loop. Each judicial run records its outcome; the
next run on a similar issue retrieves that experience and injects it into the
proposer's context.

- **Episodic** — `bridge/memory/ledger.jsonl`: one record/run
  `{ts, repo, issue, paths, test_outcome, verdict, models, n_rounds, comment, vec}`.
- **Semantic** — `bridge/memory/learnings.json`: distilled always-injected notes.
- **Embedder chain** — NPU `:8111` → Ollama `nomic-embed-text` → lexical fallback
  (works offline; records tagged so only like-embedder vectors are compared).
- **Privacy** — secrets/keys redacted on write; full source never stored (only
  issue text, paths, outcomes).
- **Forgetting** — ledger capped at 3000, keeps all SIGNED + most-recent others.
- `--no-memory` disables it for clean A/B baselines.

**Verified:** record → embed (NPU 1024-dim) → retrieve → inject all work; a
similar query correctly surfaces the right prior case. **Caveat:** memory is
infrastructure, not a behavior fix — raw episodic recall didn't change
Groq-on-both-seats gold-plating in testing. Behavioral lift needs (a) distilled
"don't do X" learnings and (b) a stronger/diverse reviewer seat that heeds the
recalled cases. Both are next steps, gated on Gemini/Anthropic billing.

## ⭐ Recommended configs

```bash
# BEST RESULTS — frontier cross-vendor (one OpenRouter key reaches both). VERIFIED.
python judicial_mesh.py --path <dir> \
  --provider openrouter --model openai/gpt-4o-mini \
  --reviewer openrouter --reviewer-model google/gemini-2.0-flash-001
#   upgrade the proposer to openai/gpt-4o or deepseek/deepseek-chat for harder code.

# FREE / cross-vendor — two different live keys, no per-token cost. VERIFIED.
python judicial_mesh.py --path <dir> --provider mistral --reviewer groq

# FULLY LOCAL / offline — zero cost.
python judicial_mesh.py --path <dir> --provider ollama --reviewer ollama --model llama3.2
```

Why cross-vendor: two *different* model families are a real second opinion;
two seats of the same model just rubber-stamp each other.

## Provider status (2026-05-29)

| Provider | `.env` key | Status |
|----------|-----------|--------|
| **OpenRouter** | `OPENROUTER_API_KEY` | ✅ works — reaches GPT-4o, Gemini 2.0, DeepSeek via one key (Claude slug 404s) |
| **Mistral** | `MISTRAL_API_KEY` | ✅ works — large + codestral (code) + pixtral (vision) |
| **Groq** | `GROQ_API_KEY` | ✅ works — fastest free seat |
| **Ollama** | local :11434 | ✅ free/offline |
| **HuggingFace** | `HF_TOKEN` | ✅ works (write) — data/model hosting + training, not a live seat here |
| **Gemini** | `GOOGLE_AI_KEY` | ❌ `429` no quota (reach it via OpenRouter instead) |
| **Anthropic** | `ANTHROPIC_API_KEY` | ❌ `400` no credits (reach Claude via OpenRouter instead) |
| **Cerebras** | `CEREBRAS_API_KEY` | ❌ `401` dead key — get a fresh one at cloud.cerebras.ai |
