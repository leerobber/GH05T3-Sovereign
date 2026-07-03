"""
generate_bootstrap.py — Sovereign Bootstrap Dataset Generator

Uses Claude API to generate 2000+ high-quality Avery strategy training pairs.
Each pair: sovereign business goal + chosen (Avery-style response) + rejected (weak alternative).
Output: data/bootstrap_dataset.jsonl + pushed to HuggingFace

Run: python generate_bootstrap.py
     python generate_bootstrap.py --pairs-per-domain 100 --domains business,sales,cfo
"""
import argparse, json, os, time
from pathlib import Path

ROOT = Path(__file__).parent

DOMAINS = {
    "business":         "sovereign venture creation and startup design for lower/middle class founders",
    "sales":            "sales strategy, pipeline building, and closing for sovereign AI products",
    "product_strategy": "product roadmaps, PRDs, MVP design for affordable AI platforms",
    "growth":           "user acquisition, viral loops, retention for fixed-cost AI services",
    "cfo":              "financial modeling, unit economics, and capital strategy for bootstrapped ventures",
    "content":          "content marketing and community building for sovereign technology brands",
    "legal_ip":         "IP protection, contracts, and entity structure for independent founders",
    "ops":              "SOPs, hiring, vendor management for lean AI operations",
    "ml_engineer":      "MLOps, model deployment, and technical architecture for sovereign AI",
    "frontier":         "AI-native business models, agentic automation, post-SaaS computing",
    "core":             "self-improvement loops, decision frameworks, and meta-strategy",
}

AVERY_SYSTEM = """You are Avery, the sovereign business strategist for SovereignNation.

SovereignNation is a fixed-cost AI platform built for lower and middle class families, \
children's education, and affordable connectivity. Our users earn $20k-$60k/year and \
cannot afford $200/month SaaS tools.

Your responses always follow this exact structure:
## STRATEGY: [Descriptive Title] - "[Codename]"

**KAIROS Cycle:** [number 7000-8500]
**Domain:** [domain name]
**Goal Score:** [0.60-0.95] (High/Medium Confidence)

### 1. [Action Point Title]
[2-3 sentences of concrete, specific, actionable guidance]

### 2. [Action Point Title]
[2-3 sentences]

### 3. [Action Point Title]
[2-3 sentences]

### Implementation Steps
1. [Specific step with tool/resource]
2. [Specific step]
3. [Specific step]

### Success Metrics
- [Measurable metric]
- [Measurable metric]

Rules:
- ALWAYS answer the exact goal asked. Never drift to a different topic.
- Be specific: name tools, prices, platforms, timelines.
- Assume $0-$500 budget unless stated otherwise.
- Reference SovereignNation infrastructure when relevant (KAIROS, Sovereign-Core, GH05T3/Avery).
- Never recommend enterprise SaaS. Prefer open-source, free tiers, or sovereign-built tools."""


BATCH_PROMPT = """Generate {n} diverse training pairs for fine-tuning a sovereign business strategy AI named Avery.

Domain focus: {domain_desc}

Output a JSON array of exactly {n} objects. Each object must have:
- "goal": A specific, concrete business challenge a lower/middle class founder faces in this domain (1-2 sentences)
- "chosen": Avery's ideal response following the STRATEGY format with KAIROS metadata (300-600 words)
- "rejected": A weak alternative response that is generic, doesn't follow the format, ignores the sovereign mission, or recommends expensive enterprise tools (100-200 words)

Make goals diverse — different industries, scales, and specific challenges. Avoid repetition.
Goals should feel real: "How do I price a $9/month AI tutoring app for parents earning $35k/year?"

Return ONLY the JSON array, no other text."""


def _load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))


# ---------------------------------------------------------------------------
# Free provider cascade — tried in order, with rate-limit retry logic
# ---------------------------------------------------------------------------
import re as _re

# Max seconds to wait on a 429 before skipping to the next provider
_MAX_RATE_WAIT = 90

# Per-provider cooldown: if a provider 429'd, don't retry it until this time
_provider_cooldown: dict[str, float] = {}


def _parse_retry_after(err_str: str) -> float:
    """Extract retry-after seconds from a 429 error message."""
    # "Please try again in 13m13.152s"
    m = _re.search(r"(\d+)m(\d+(?:\.\d+)?)s", err_str)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    # "Please retry in 42.258s"
    m = _re.search(r"retry[^0-9]*(\d+(?:\.\d+)?)\s*s", err_str)
    if m:
        return float(m.group(1))
    return 0.0


def _is_rate_limit(err_str: str) -> bool:
    return "429" in err_str or "rate_limit" in err_str.lower() or "quota" in err_str.lower()


def _openai_chat(base_url: str, api_key: str, model: str,
                 prompt: str, system: str) -> str:
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": prompt}],
        max_tokens=8192, temperature=0.9,
    )
    return r.choices[0].message.content.strip()


def _call_groq(prompt: str, system: str) -> str:
    return _openai_chat("https://api.groq.com/openai/v1",
                        os.environ["GROQ_API_KEY"],
                        "llama-3.3-70b-versatile", prompt, system)


def _call_openrouter(prompt: str, system: str) -> str:
    # Try a few free models in order
    for model in ["meta-llama/llama-3.1-8b-instruct:free",
                  "google/gemma-2-9b-it:free",
                  "mistralai/mistral-7b-instruct:free"]:
        try:
            return _openai_chat("https://openrouter.ai/api/v1",
                                os.environ["OPENROUTER_API_KEY"],
                                model, prompt, system)
        except Exception as e:
            if "not a valid model" in str(e) or "404" in str(e):
                continue
            raise
    raise RuntimeError("No OpenRouter free models available")


def _call_gemini(prompt: str, system: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_AI_KEY"])
    r = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=8192,
            temperature=0.9,
        ),
    )
    return r.text.strip()


def _call_ollama(prompt: str, system: str) -> str:
    return _openai_chat("http://localhost:11434/v1", "ollama",
                        "qwen2.5:7b", prompt, system)


_PROVIDERS = [
    ("Groq/llama-3.3-70b",   "GROQ_API_KEY",       _call_groq),
    ("OpenRouter/free",       "OPENROUTER_API_KEY", _call_openrouter),
    ("Gemini/flash-2.0",     "GOOGLE_AI_KEY",       _call_gemini),
    ("Ollama/qwen2.5:7b",    None,                  _call_ollama),
]

_active_provider: str = ""


def _generate(prompt: str, system: str) -> str:
    """Try providers in order. On 429, wait up to _MAX_RATE_WAIT seconds then
    retry the same provider once before moving on."""
    global _active_provider
    now = time.time()

    for name, env_key, fn in _PROVIDERS:
        if env_key and not os.environ.get(env_key, ""):
            continue
        # Skip if still in cooldown from a prior 429
        cooldown_until = _provider_cooldown.get(name, 0)
        if now < cooldown_until:
            remaining = int(cooldown_until - now)
            print(f"  [{name} cooling down {remaining}s]", end=" ")
            continue
        try:
            result = fn(prompt, system)
            if _active_provider != name:
                _active_provider = name
                print(f"\n  [provider: {name}]", flush=True)
            return result
        except Exception as e:
            err = str(e)
            if _is_rate_limit(err):
                wait = _parse_retry_after(err)
                if 0 < wait <= _MAX_RATE_WAIT:
                    print(f"\n  [{name} rate-limited, waiting {wait:.0f}s...]", flush=True)
                    time.sleep(wait + 2)
                    try:
                        result = fn(prompt, system)
                        if _active_provider != name:
                            _active_provider = name
                            print(f"  [provider: {name} (after wait)]", flush=True)
                        return result
                    except Exception as e2:
                        err = str(e2)
                # Still failing or wait too long — set cooldown and try next
                cooldown = max(wait, 60.0)
                _provider_cooldown[name] = time.time() + cooldown
                print(f"  [{name} rate-limited, cooldown {cooldown:.0f}s, trying next...]",
                      flush=True)
            else:
                print(f"  [{name} error: {err[:120]}] trying next...", flush=True)

    raise RuntimeError("All providers exhausted")


def generate_domain_pairs(domain: str, domain_desc: str,
                          pairs_per_domain: int, batch_size: int = 10) -> list:
    pairs = []
    n_batches = (pairs_per_domain + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        n = min(batch_size, pairs_per_domain - len(pairs))
        print(f"    batch {batch_idx+1}/{n_batches} ({n} pairs)...", end=" ", flush=True)

        try:
            raw = _generate(BATCH_PROMPT.format(n=n, domain_desc=domain_desc), AVERY_SYSTEM)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            # Strip literal control characters that some local models embed
            import re as _re2
            raw_clean = _re2.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)

            try:
                batch = json.loads(raw_clean)
            except json.JSONDecodeError:
                # Last resort: find the JSON array boundaries and re-parse
                m = _re2.search(r'\[.*\]', raw_clean, _re2.DOTALL)
                if m:
                    batch = json.loads(m.group(0))
                else:
                    raise

            for item in batch:
                if item.get("goal") and item.get("chosen") and item.get("rejected"):
                    # Normalize whitespace in fields
                    pairs.append({
                        "goal":     item["goal"].strip(),
                        "chosen":   item["chosen"].strip(),
                        "rejected": item["rejected"].strip(),
                        "domain":   domain,
                        "source":   "free_bootstrap",
                    })
            print(f"OK ({len(batch)} parsed)")

        except json.JSONDecodeError as e:
            print(f"JSON ERROR: {e} -- skipping batch")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(0.3)

    return pairs


def upload_to_hf(pairs: list, hf_token: str, dataset_name: str, split: str):
    from datasets import Dataset
    print(f"\n[UPLOAD] Pushing {len(pairs)} pairs to {dataset_name} split={split}...")

    # Convert to SFT format (instruction/response) for compatibility with existing trainer
    sft_rows = []
    dpo_rows = []
    for p in pairs:
        sft_rows.append({
            "instruction": f"Business strategy goal: {p['goal']}",
            "context":     f"Rejected approach: {p['rejected'][:300]}",
            "response":    p["chosen"],
            "source":      p["source"],
            "domain":      p["domain"],
        })
        dpo_rows.append({
            "prompt":   f"GOAL: {p['goal']}\n\nProvide a detailed sovereign strategy:",
            "chosen":   p["chosen"],
            "rejected": p["rejected"],
            "domain":   p["domain"],
            "source":   p["source"],
        })

    # Push SFT split
    Dataset.from_list(sft_rows).push_to_hub(
        dataset_name, split=split, token=hf_token, private=False
    )
    # Push DPO split
    Dataset.from_list(dpo_rows).push_to_hub(
        dataset_name, split=f"{split}_dpo", token=hf_token, private=False
    )
    print(f"  Pushed {len(sft_rows)} SFT rows to split '{split}'")
    print(f"  Pushed {len(dpo_rows)} DPO rows to split '{split}_dpo'")


def main(pairs_per_domain: int, domains_filter: list, upload: bool):
    _load_env()

    hf_token = os.environ.get("HF_TOKEN", "")

    # Print which providers are available
    available = []
    for name, env_key, _ in _PROVIDERS:
        if env_key is None or os.environ.get(env_key, ""):
            available.append(name)
    if not available:
        print("ERROR: No LLM providers available. Set GROQ_API_KEY or GOOGLE_AI_KEY in .env")
        return

    target_domains = {k: v for k, v in DOMAINS.items()
                      if not domains_filter or k in domains_filter}

    total_target = len(target_domains) * pairs_per_domain
    print(f"\n+================================================+")
    print(f"|  AVERY BOOTSTRAP GENERATOR  (free cascade)     |")
    print(f"+================================================+")
    print(f"  Domains    : {list(target_domains.keys())}")
    print(f"  Per domain : {pairs_per_domain}")
    print(f"  Total      : {total_target} pairs")
    print(f"  Providers  : {available}")
    print(f"  Cost       : $0.00 (free tier)")

    all_pairs = []
    out_file = ROOT / "data" / "bootstrap_dataset.jsonl"
    out_file.parent.mkdir(exist_ok=True)

    for domain, desc in target_domains.items():
        print(f"\n  [{domain.upper()}] generating {pairs_per_domain} pairs...")
        pairs = generate_domain_pairs(domain, desc, pairs_per_domain)
        all_pairs.extend(pairs)

        with out_file.open("a", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  {len(pairs)} pairs saved ({len(all_pairs)} total so far)")

    print(f"\n  Bootstrap complete: {len(all_pairs)} pairs -> {out_file}")

    if upload and hf_token:
        upload_to_hf(all_pairs, hf_token,
                     dataset_name="tastytator/sovereign-economy",
                     split="bootstrap")
    elif upload:
        print("  HF_TOKEN not set -- skipping upload")

    print("\n  Done. Next step:")
    print("    python runpod_launcher.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Avery Bootstrap Dataset Generator")
    ap.add_argument("--pairs-per-domain", type=int, default=180)
    ap.add_argument("--domains", default="",
                    help="comma-separated domain names (default: all 11)")
    ap.add_argument("--no-upload", action="store_true",
                    help="skip HuggingFace upload")
    args = ap.parse_args()

    domains_filter = [d.strip() for d in args.domains.split(",") if d.strip()]
    main(args.pairs_per_domain, domains_filter, not args.no_upload)
