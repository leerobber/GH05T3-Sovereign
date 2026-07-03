"""
sovereign_trainer_v5.py — SovereignNation AI University Training Accelerator v5

Curriculum-driven ORPO/DPO fine-tuning pipeline.
4 domains × 95 tasks → genius/mediocre pairs → LoRA fine-tune → HF push.

Fits in 8GB VRAM (RTX 5050) via 4-bit quantization + LoRA rank-16.

Usage:
  python sovereign_trainer_v5.py                          # all domains, orpo
  python sovereign_trainer_v5.py --domain engineering
  python sovereign_trainer_v5.py --domain all --mode dpo --push
  python sovereign_trainer_v5.py --generate-only          # build dataset, skip training
  HF_TOKEN=hf_... python sovereign_trainer_v5.py --push

Pair generation priority:
  1. Ollama local (free, best quality)
  2. Claude API via ANTHROPIC_API_KEY (best quality)
  3. Built-in genius templates (deterministic, always works)
"""
import os, sys, json, random, textwrap, argparse
from pathlib import Path
from typing import Optional

# ── Args ─────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser(description="SovereignNation Training Accelerator v5")
ap.add_argument("--domain",        default="all",
                choices=["engineering", "business", "product", "university", "all"])
ap.add_argument("--mode",          default="orpo", choices=["orpo", "dpo", "sft"])
ap.add_argument("--epochs",        type=int, default=3)
ap.add_argument("--push",          action="store_true", help="Push adapter to HuggingFace")
ap.add_argument("--generate-only", action="store_true", help="Build dataset, skip training")
ap.add_argument("--output-dir",    default="training_data/sovereign_v5")
ap.add_argument("--ollama-model",  default="qwen2.5:7b-instruct")
ap.add_argument("--base-model",    default="unsloth/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--hf-repo",       default="tastytator/sovereign-economy")
args, _ = ap.parse_known_args()

HF_TOKEN       = os.environ.get("HF_TOKEN", "")
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
OUTPUT_DIR     = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Curriculum ────────────────────────────────────────────────────────────────

CURRICULUM = {
    "engineering": [
        # Systems & Infrastructure (10)
        "Design a token-bucket rate limiter for a FastAPI microservice with Redis backing.",
        "Implement async retry with exponential backoff, jitter, and a circuit breaker.",
        "Build a WebSocket broadcast hub with per-connection backpressure and graceful shutdown.",
        "Design a SQLite WAL-mode connection pool safe for concurrent FastAPI workers.",
        "Create a thread-safe LRU cache for transformer embedding lookups (max 10K entries).",
        "Implement HMAC-SHA256 webhook signature verification resistant to timing attacks.",
        "Design a hot-reload configuration system that re-reads .env without service restart.",
        "Build a self-healing service mesh: health checks, auto-restart, dependency ordering.",
        "Implement a streaming SSE handler for LLM token output with client reconnect.",
        "Design a cost-aware LLM router: cheap model → expensive model fallback with budget cap.",
        # AI Infrastructure (10)
        "Build a vector index with HNSW-style approximate nearest-neighbour search.",
        "Implement 4-bit quantization loading for a 7B model on 8GB VRAM with unsloth.",
        "Design a LoRA adapter merge pipeline that produces a quantized GGUF for llama.cpp.",
        "Build a checkpoint resume system: detect latest step, reload optimizer state.",
        "Implement adaptive batch sizing that auto-reduces on CUDA OOM and retries.",
        "Design a multi-process vLLM inference server with IPC message routing.",
        "Implement gradient checkpointing trade-off: memory vs. compute for 8GB cards.",
        "Build an async inference queue with priority levels and timeout eviction.",
        "Design a semantic deduplication pass before training to remove near-duplicate rows.",
        "Implement differential privacy noise injection for training data anonymisation.",
        # Performance & Reliability (10)
        "Profile a FastAPI endpoint with cProfile, identify the bottleneck, and fix it.",
        "Design a dead-letter queue for failed background tasks with replay capability.",
        "Build a metrics collector: P50/P95/P99 latency for LLM inference with Prometheus.",
        "Implement connection draining for zero-downtime rolling deployments.",
        "Design a distributed rate limiter using Redis INCR + TTL (no Lua scripts needed).",
        "Build an adaptive prefetch window for retrieval-augmented generation pipelines.",
        "Implement memory-mapped JSONL reading for datasets too large for RAM.",
        "Design a multi-tenant isolation layer: per-org rate limits, data partitioning.",
        "Build a canary deployment controller that auto-rolls back on error-rate spike.",
        "Implement structured logging with correlation IDs across async task boundaries.",
    ],
    "business": [
        "Define a three-tier SaaS pricing model for an AI API (free/pro/enterprise).",
        "Build a unit economics model: CAC, LTV, payback period for a developer AI tool.",
        "Design a freemium-to-paid conversion funnel with in-app upgrade triggers.",
        "Create a competitive moat analysis: what prevents a well-funded competitor from cloning this?",
        "Design a 90-day GTM plan targeting indie developers and small engineering teams.",
        "Build an MRR growth model: inputs are conversion rate, churn rate, expansion revenue.",
        "Create a case study template that converts a technical win into a sales asset.",
        "Design a partner and reseller channel strategy for reaching enterprise buyers.",
        "Build a churn prediction signal: which product behaviours predict cancellation?",
        "Design a referral programme mechanics: incentive structure and fraud prevention.",
        "Create a content marketing flywheel: blog → newsletter → developer community.",
        "Build a financial runway model: burn rate, hiring triggers, fundraising milestones.",
        "Design a customer success playbook for onboarding new API customers in < 30 min.",
        "Create a board-ready monthly business review template for a bootstrapped AI company.",
        "Design an advisory board recruitment strategy: who to target and what to offer.",
    ],
    "product": [
        # AI University (10)
        "Design the multi-agent AI University architecture: professor, student, and evaluator agents.",
        "Build a course curriculum generator that produces syllabus, lessons, and assessments.",
        "Create a student progress tracker: mastery scores, spaced repetition schedule, certificates.",
        "Design a professor agent system prompt that adapts explanation depth to student mastery.",
        "Build an AI-graded short-answer assignment system with rubric-based scoring.",
        "Design a Socratic dialogue loop where the professor never gives direct answers.",
        "Build a prerequisite dependency graph that gates advanced modules on proven mastery.",
        "Create an adaptive difficulty scaler that adjusts question complexity in real time.",
        "Design a peer-review system where student agents critique each other's answers.",
        "Build a live coding workshop format: professor streams code, student follows along.",
        # Agent Marketplace (8)
        "Design an agent marketplace: listing schema, quality scoring, discovery search.",
        "Build a sandbox environment for testing an agent before purchase/deployment.",
        "Create a one-click agent deployment pipeline: HuggingFace → SwarmBus agent slot.",
        "Design a billing system for per-call API usage with metered overage charges.",
        "Build a developer portal: API key management, usage dashboards, webhook config.",
        "Create an agent version control system: deploy, rollback, A/B test two versions.",
        "Design a review and reputation system for marketplace agent listings.",
        "Build a white-label deployment: operator customises branding, model, and system prompt.",
        # Sovereign Core Product (7)
        "Design a multi-tenant SovereignNation deployment: data isolation, per-org config.",
        "Build an admin dashboard: user management, usage analytics, billing overrides.",
        "Create a GDPR-compliant data retention policy with automated deletion schedules.",
        "Design a disaster recovery system: backup strategy, RTO/RPO targets, failover runbook.",
        "Build an automated QA pipeline that catches agent regressions before deploy.",
        "Design a feedback loop: user thumbs up/down → re-ranking training pairs.",
        "Create an accessibility compliance layer: WCAG 2.1 AA for the AI University UI.",
    ],
    "university": [
        # Agent Specialisation Courses (10)
        "Write the ORACLE memory agent masterclass: recall mechanics, HCM vectors, dream cycle.",
        "Write the FORGE code generation course: async patterns, injection safety, testability.",
        "Write the CODEX documentation course: API docs, PR review, changelog standards.",
        "Write the SENTINEL security course: Ghost Protocol scanner, KillSwitch, threat models.",
        "Write the NEXUS orchestration course: task decomposition, routing, synthesis.",
        "Design a prompt engineering masterclass: chain-of-thought, few-shot, self-consistency.",
        "Build a system design interview prep course tailored to AI infrastructure roles.",
        "Create a red team / blue team exercise curriculum for security-aware engineers.",
        "Design an ethical AI decision-making module: bias, fairness, harm avoidance.",
        "Build a research paper dissection course: how to read, critique, and implement papers.",
        # Feynman & Deep Learning Pedagogy (8)
        "Implement the Feynman technique as an AI tutor loop: explain → simplify → test gaps.",
        "Design a spaced-repetition scheduler for AI concepts (SM-2 algorithm variant).",
        "Build a concept map generator that visualises knowledge graph gaps for a student.",
        "Create a mentorship matching system: expert agent paired to struggling student.",
        "Design a debate framework: two agents argue opposing positions, student adjudicates.",
        "Build a Socratic seminar format for group AI training sessions.",
        "Create a writing feedback loop: student drafts, AI critiques, student revises.",
        "Design an exam proctoring system: question randomisation, time pressure, plagiarism check.",
        # Leadership & Business Skills for Builders (7)
        "Design a financial literacy module for technical founders: revenue models, cap tables.",
        "Build a negotiation and persuasion course using LLM role-play scenarios.",
        "Create a public speaking and technical presentation curriculum.",
        "Design a leadership development programme: 1:1s, feedback delivery, team culture.",
        "Build a product management curriculum: discovery, prioritisation, roadmap communication.",
        "Create a data analysis and visualisation course for non-data-scientist engineers.",
        "Design a cold outreach and sales fundamentals module for developer-founders.",
    ],
}

# Flatten for "all" mode
ALL_TASKS = [(domain, task)
             for domain, tasks in CURRICULUM.items()
             for task in tasks]

assert sum(len(v) for v in CURRICULUM.values()) == 95, \
    f"Curriculum has {sum(len(v) for v in CURRICULUM.values())} tasks, expected 95"

# ── Pair Generation ───────────────────────────────────────────────────────────

MEDIOCRE_TEMPLATES = [
    "That's a good question. There are many ways to approach this. You could use a library or framework that handles this for you. It really depends on your specific use case and requirements.",
    "You should look into existing solutions before building your own. Consider the trade-offs carefully. There are pros and cons to each approach.",
    "This is a complex topic. I'd recommend reading the documentation for your chosen framework. Stack Overflow has many answers on this subject.",
    "It depends on your scale. At small scale, a simple solution works fine. At large scale you'd need something more sophisticated. Start simple and iterate.",
    "There are several options available. Pick the one that fits your team's expertise. Make sure to write tests and document your decision.",
]

_GENIUS_STRUCTURE = """\
{domain_intro}

**Design:**
{design}

**Implementation sketch:**
```python
{code}
```

**Edge cases and hardening:**
{edge_cases}

**Trade-offs:**
{tradeoffs}"""


def _mediocre(task: str) -> str:
    return random.choice(MEDIOCRE_TEMPLATES)


def _genius_via_ollama(task: str, model: str) -> Optional[str]:
    try:
        import httpx, json as _json
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": (
                    f"You are a senior engineer and expert educator. "
                    f"Give a comprehensive, expert-level answer to:\n\n{task}\n\n"
                    f"Include: design rationale, a concrete Python code example, "
                    f"edge cases, and trade-offs. Be specific and actionable."
                ),
                "stream": False,
            },
            timeout=60.0,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip() or None
    except Exception:
        pass
    return None


def _genius_via_anthropic(task: str) -> Optional[str]:
    if not ANTHROPIC_KEY:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a senior engineer and expert educator. "
                    f"Give a comprehensive, expert-level answer to:\n\n{task}\n\n"
                    f"Include: design rationale, a concrete Python code example, "
                    f"edge cases, and trade-offs."
                ),
            }],
        )
        return msg.content[0].text.strip() or None
    except Exception:
        pass
    return None


# Deterministic genius templates keyed by domain
_GENIUS_INTROS = {
    "engineering": "Solid systems engineering requires explicit contracts at every boundary.",
    "business":    "Strong business decisions are quantified, time-boxed, and reversible.",
    "product":     "Good product design starts with the user's job-to-be-done, not the feature.",
    "university":  "Effective teaching requires diagnosing the gap before prescribing the remedy.",
}

_GENIUS_STUBS = {
    "engineering": (
        "Define the interface first, then implement. Use dataclasses for config, "
        "type hints everywhere, dependency injection for testability.",
        (
            "import asyncio\nfrom dataclasses import dataclass, field\n"
            "from typing import Optional\n\n\n@dataclass\nclass Config:\n"
            "    rate: float = 10.0\n    burst: int = 20\n    timeout: float = 30.0\n\n\n"
            "class TokenBucket:\n    def __init__(self, cfg: Config) -> None:\n"
            "        self._tokens = float(cfg.burst)\n        self._rate = cfg.rate\n"
            "        self._burst = cfg.burst\n        self._lock = asyncio.Lock()\n"
            "        import time; self._last = time.monotonic()\n\n"
            "    async def acquire(self) -> bool:\n        async with self._lock:\n"
            "            import time; now = time.monotonic()\n"
            "            self._tokens = min(\n                self._burst,\n"
            "                self._tokens + (now - self._last) * self._rate,\n"
            "            )\n            self._last = now\n"
            "            if self._tokens >= 1:\n                self._tokens -= 1\n"
            "                return True\n            return False"
        ),
        "Handle clock skew on distributed hosts, token stealing under high concurrency, "
        "and overly generous burst that masks quota violations.",
        "Single-process in-memory: simple but lost on restart. "
        "Redis INCR+TTL: durable, adds network hop. "
        "Sliding window log: fairest, most memory-intensive.",
    ),
    "business": (
        "Quantify every assumption. A pricing model without elasticity estimates is a guess.",
        (
            "# Unit economics model\nCAC = 120          # $ to acquire one paid customer\n"
            "ARPU = 49          # avg monthly revenue per user\n"
            "CHURN = 0.03       # monthly churn rate\n"
            "MARGIN = 0.82      # gross margin\n\n"
            "LTV = ARPU * MARGIN / CHURN    # $1,339\n"
            "LTV_CAC = LTV / CAC             # 11.2x  (target: >3x)\n"
            "payback_months = CAC / (ARPU * MARGIN)  # 2.98 months"
        ),
        "Watch for CAC creep as you exhaust cheap acquisition channels. "
        "Segment LTV by acquisition source — paid vs organic can differ 3x.",
        "Bottom-up model: conservative but falsifiable. "
        "Top-down market share: impressive in pitches, useless for operations.",
    ),
    "product": (
        "Design the data model first. Every feature decision flows from schema constraints.",
        (
            "from dataclasses import dataclass, field\nfrom datetime import datetime\n"
            "from enum import Enum\n\nclass Status(Enum):\n    DRAFT = 'draft'\n"
            "    ACTIVE = 'active'\n    ARCHIVED = 'archived'\n\n\n@dataclass\n"
            "class Course:\n    id: str\n    title: str\n    domain: str\n"
            "    prerequisites: list[str] = field(default_factory=list)\n"
            "    mastery_threshold: float = 0.80\n    created: datetime = field(\n"
            "        default_factory=datetime.utcnow\n    )\n    status: Status = Status.DRAFT"
        ),
        "Circular prerequisites cause infinite dependency resolution. "
        "Mastery thresholds that are too high frustrate learners; too low produces false confidence.",
        "Rigid prerequisite graph: safe but inflexible. "
        "Soft recommendations: learner autonomy but risk knowledge gaps.",
    ),
    "university": (
        "The best tutors diagnose the misconception before correcting it.",
        (
            "import re\n\nSOCRATIC_PROMPT = (\n"
            '    "The student said: \\"{answer}\\". '\
            "Do NOT give the correct answer. \"\n"
            '    "Ask one targeted question that exposes the gap in their reasoning."\n'
            ")\n\n\ndef socratic_followup(student_answer: str, llm_fn) -> str:\n"
            '    prompt = SOCRATIC_PROMPT.format(answer=student_answer[:400])\n'
            "    return llm_fn(prompt)"
        ),
        "Avoid leading questions that telegraph the answer. "
        "Track consecutive wrong answers — three in a row signals a deeper conceptual gap.",
        "Pure Socratic: maximises deep learning but frustrating for beginners. "
        "Blended (hint after 2 failures): balances depth with retention.",
    ),
}


def _genius_template(domain: str, task: str) -> str:
    intro, design, code, edge_cases, tradeoffs = (
        _GENIUS_INTROS.get(domain, ""),
        *_GENIUS_STUBS.get(domain, ("", "", "", ""))
    )
    code_lines = textwrap.indent(code, "    ")
    # Weave task into the response header
    task_short = task[:80].rstrip(".") if len(task) > 80 else task.rstrip(".")
    return (
        f"**{task_short}**\n\n"
        f"{intro}\n\n"
        f"**Design:** {design}\n\n"
        f"**Implementation sketch:**\n```python\n{code}\n```\n\n"
        f"**Edge cases:** {edge_cases}\n\n"
        f"**Trade-offs:** {tradeoffs}"
    )


def make_pair(domain: str, task: str) -> dict:
    # Priority: Ollama → Anthropic → template
    genius = (
        _genius_via_ollama(task, args.ollama_model) or
        _genius_via_anthropic(task) or
        _genius_template(domain, task)
    )
    mediocre = _mediocre(task)
    return {
        "domain":   domain,
        "task":     task,
        "chosen":   genius,
        "rejected": mediocre,
        "system": (
            "You are a world-class engineer, educator, and strategist specialising in "
            "AI systems, sovereign infrastructure, and developer education. "
            "Your answers are specific, structured, and immediately actionable."
        ),
    }


# ── Dataset Builder ───────────────────────────────────────────────────────────

def build_dataset(domains: list[str]) -> Path:
    out_path = OUTPUT_DIR / "sovereign_v5_pairs.jsonl"
    tasks = []
    for domain in domains:
        for task in CURRICULUM.get(domain, []):
            tasks.append((domain, task))

    print(f"\n[1/4] Generating {len(tasks)} pairs across {len(domains)} domain(s)...")
    rows = []
    for i, (domain, task) in enumerate(tasks, 1):
        row = make_pair(domain, task)
        rows.append(row)
        if i % 10 == 0 or i == len(tasks):
            print(f"      {i}/{len(tasks)} pairs generated", end="\r")
    print()

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"      Saved {len(rows)} pairs → {out_path}")
    return out_path


# ── HuggingFace Push ──────────────────────────────────────────────────────────

def push_to_hf(jsonl_path: Path, config_name: str = "sovereign_v5") -> None:
    if not HF_TOKEN:
        print("      SKIP: HF_TOKEN not set")
        return
    try:
        from datasets import Dataset
    except ImportError:
        print("      SKIP: pip install datasets first")
        return

    rows = []
    with open(jsonl_path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))

    ds = Dataset.from_dict({
        "domain":   [r["domain"]   for r in rows],
        "task":     [r["task"]     for r in rows],
        "chosen":   [r["chosen"]   for r in rows],
        "rejected": [r["rejected"] for r in rows],
        "system":   [r["system"]   for r in rows],
    })
    print(f"\n[*] Pushing {len(ds)} rows to {args.hf_repo} (config={config_name})...")
    ds.push_to_hub(
        args.hf_repo,
        config_name=config_name,
        split="train",
        token=HF_TOKEN,
        private=False,
    )
    print(f"      Pushed → huggingface.co/datasets/{args.hf_repo} (config={config_name})")


# ── Trainer ───────────────────────────────────────────────────────────────────

def install_deps() -> None:
    import subprocess
    print("[2/4] Installing training dependencies...")
    pkgs = [
        "torch>=2.1.0", "transformers>=4.40.0", "datasets",
        "trl==0.12.2", "peft==0.14.0", "bitsandbytes>=0.43.0",
        "accelerate==1.2.1", "unsloth",
    ]
    ret = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + pkgs,
        capture_output=True,
    )
    if ret.returncode != 0:
        print("      WARNING: some packages failed to install — attempting anyway")


def run_training(jsonl_path: Path) -> None:
    install_deps()

    import torch
    from datasets import Dataset
    from transformers import TrainingArguments

    # Load dataset
    print("[3/4] Loading dataset...")
    rows = []
    with open(jsonl_path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    ds = Dataset.from_dict({
        "chosen":   [r["chosen"]   for r in rows],
        "rejected": [r["rejected"] for r in rows],
        "system":   [r["system"]   for r in rows],
        "prompt":   [r["task"]     for r in rows],
    })
    print(f"      {len(ds)} training pairs loaded")

    # Load model (4-bit for 8GB VRAM)
    print(f"      Loading base model: {args.base_model} (4-bit)...")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing=True,
        )
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb_cfg, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        lora_cfg = LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_cfg)

    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = str(ckpt_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        learning_rate               = 2e-4,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        warmup_ratio                = 0.1,
        lr_scheduler_type           = "cosine",
        logging_steps               = 10,
        save_strategy               = "steps",
        save_steps                  = 50,
        save_total_limit            = 2,
        report_to                   = "none",
        optim                       = "adamw_8bit",
    )

    # ORPO (odds-ratio preference optimization) — needs only chosen/rejected, no SFT warmup
    if args.mode in ("orpo", "dpo"):
        try:
            from trl import ORPOTrainer, ORPOConfig, DPOTrainer, DPOConfig

            def _fmt(row):
                sys_ = row["system"]
                prompt = row["prompt"]
                chosen = row["chosen"]
                rejected = row["rejected"]
                return {
                    "prompt":   f"<|im_start|>system\n{sys_}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                    "chosen":   chosen + "<|im_end|>",
                    "rejected": rejected + "<|im_end|>",
                }

            ds_fmt = ds.map(_fmt, remove_columns=ds.column_names)

            print(f"[4/4] Training ({args.mode.upper()}, {args.epochs} epochs, {len(ds_fmt)} pairs)...")
            if args.mode == "orpo":
                trainer = ORPOTrainer(
                    model=model, tokenizer=tokenizer,
                    args=ORPOConfig(**vars(training_args), beta=0.1, max_length=2048),
                    train_dataset=ds_fmt,
                )
            else:
                trainer = DPOTrainer(
                    model=model, tokenizer=tokenizer,
                    args=DPOConfig(**vars(training_args), beta=0.1, max_length=2048),
                    train_dataset=ds_fmt,
                )
            trainer.train()
        except ImportError as e:
            print(f"      WARNING: {e} — falling back to SFT on chosen responses")
            args.mode = "sft"

    if args.mode == "sft":
        from trl import SFTTrainer

        def _fmt_sft(row):
            sys_ = row["system"]
            return {
                "text": (
                    f"<|im_start|>system\n{sys_}<|im_end|>\n"
                    f"<|im_start|>user\n{row['prompt']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{row['chosen']}<|im_end|>"
                )
            }

        ds_sft = ds.map(_fmt_sft, remove_columns=ds.column_names)
        print(f"[4/4] Training (SFT, {args.epochs} epochs, {len(ds_sft)} examples)...")
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=ds_sft,
            dataset_text_field="text",
            max_seq_length=2048,
            args=training_args,
        )
        trainer.train()

    # Save adapter
    adapter_path = OUTPUT_DIR / "sovereign_v5_adapter"
    print(f"\n      Saving adapter → {adapter_path}")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    if args.push and HF_TOKEN:
        hf_repo = "tastytator/sovereign-university-lora"
        print(f"      Pushing adapter → huggingface.co/{hf_repo}")
        model.push_to_hub(hf_repo, token=HF_TOKEN)
        tokenizer.push_to_hub(hf_repo, token=HF_TOKEN)
        print("      Done.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== SovereignNation Training Accelerator v5 ===")
    print(f"    domain={args.domain}  mode={args.mode}  epochs={args.epochs}  push={args.push}")

    domains = list(CURRICULUM.keys()) if args.domain == "all" else [args.domain]
    total = sum(len(CURRICULUM[d]) for d in domains)
    print(f"    {total} curriculum tasks across {len(domains)} domain(s)\n")

    jsonl_path = build_dataset(domains)

    if args.push:
        push_to_hf(jsonl_path, config_name="sovereign_v5")

    if args.generate_only:
        print("\n--generate-only: skipping training.")
        print(f"Dataset written to: {jsonl_path}")
        sys.exit(0)

    run_training(jsonl_path)

    print("\n=== Training complete ===")
    print(f"  Adapter:  {OUTPUT_DIR}/sovereign_v5_adapter/")
    print(f"  Dataset:  {jsonl_path}")
    if args.push:
        print(f"  HF:       huggingface.co/tastytator/sovereign-university-lora")
    print()
