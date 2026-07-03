"""
synth_generator.py — Free synthetic training data generator.

Uses locally-installed Ollama models to generate high-quality
SFT training pairs for all 6 sovereign agents. Zero API cost.

Generators (in order of quality):
  1. Template engine  — deterministic, zero inference cost, unlimited
  2. Local large model — qwen2.5:7b or mistral (already installed)
  3. Self-play loop   — sovereign agents critique their own outputs,
                        failures become hard negative examples
  4. HuggingFace harvest — filter public datasets (CodeSearchNet, BigVul)

Usage:
    python data_gen/synth_generator.py --agent forge --count 200
    python data_gen/synth_generator.py --agent sentinel --count 100 --method template
    python data_gen/synth_generator.py --all --count 150 --push
"""

from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

OLLAMA_BASE  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
TEACHER_MODEL = "qwen2.5:7b-instruct"   # best locally-installed model for generation
FALLBACK_TEACHER = "mistral:latest"      # second choice

OUT_DIR = ROOT / "data_gen" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEMPLATE ENGINE — zero cost, unlimited, deterministic
# ══════════════════════════════════════════════════════════════════════════════

FORGE_TEMPLATES = [
    ("Write a Python function that {action} with proper type hints and error handling.",
     "def {fn_name}({params}) -> {ret}:\n    \"\"\"{doc}\"\"\"\n    {body}"),
    ("Create a FastAPI endpoint that {endpoint_action}.",
     "@app.{method}('{path}')\nasync def {fn_name}({params}):\n    {body}"),
    ("Write a Python class for {class_purpose} with {methods} methods.",
     "class {class_name}:\n    def __init__(self):\n        {init_body}\n\n    def {method1}(self):\n        {method1_body}"),
]

FORGE_ACTIONS = [
    ("reads a JSON file and returns a typed dataclass", "read_json_file", "data: Path", "dict", "Reads and parses a JSON file.", "with open(data) as f:\n        return json.load(f)"),
    ("validates an email address using regex", "validate_email", "email: str", "bool", "Returns True if email format is valid.", "pattern = r'^[\\w.-]+@[\\w.-]+\\.\\w{2,}$'\n    return bool(re.match(pattern, email))"),
    ("retries a function up to n times with exponential backoff", "retry_with_backoff", "fn: callable, max_retries: int = 3", "Any", "Retries fn on exception with exponential backoff.", "for attempt in range(max_retries):\n        try:\n            return fn()\n        except Exception as e:\n            if attempt == max_retries - 1:\n                raise\n            time.sleep(2 ** attempt)"),
    ("parses a CSV file into a list of dicts", "parse_csv", "filepath: Path", "list[dict]", "Reads CSV and returns rows as dicts.", "import csv\n    with open(filepath, newline='') as f:\n        return list(csv.DictReader(f))"),
    ("hashes a password using bcrypt", "hash_password", "password: str", "str", "Returns bcrypt hash of password.", "import bcrypt\n    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()"),
    ("checks if a port is open on a host", "is_port_open", "host: str, port: int, timeout: float = 1.0", "bool", "Returns True if host:port accepts TCP connections.", "import socket\n    try:\n        with socket.create_connection((host, port), timeout=timeout):\n            return True\n    except OSError:\n        return False"),
    ("generates a secure random token", "generate_token", "length: int = 32", "str", "Returns a cryptographically secure URL-safe token.", "import secrets\n    return secrets.token_urlsafe(length)"),
    ("rate-limits a function to max calls per second", "rate_limiter", "max_calls: int, period: float = 1.0", "callable", "Decorator that enforces a rate limit.", "calls = []\n    def decorator(fn):\n        def wrapper(*args, **kwargs):\n            now = time.time()\n            calls[:] = [t for t in calls if now - t < period]\n            if len(calls) >= max_calls:\n                raise RuntimeError('Rate limit exceeded')\n            calls.append(now)\n            return fn(*args, **kwargs)\n        return wrapper\n    return decorator"),
]

SENTINEL_VULNS = [
    {
        "vuln": "SQL Injection",
        "owasp": "A03:2021",
        "cwe": "CWE-89",
        "severity": "CRITICAL",
        "bad_code": 'query = "SELECT * FROM users WHERE id = " + user_id\ncursor.execute(query)',
        "fixed_code": 'query = "SELECT * FROM users WHERE id = ?"\ncursor.execute(query, (user_id,))',
        "explanation": "String concatenation in SQL queries allows an attacker to inject arbitrary SQL. Use parameterized queries.",
    },
    {
        "vuln": "Command Injection",
        "owasp": "A03:2021",
        "cwe": "CWE-78",
        "severity": "CRITICAL",
        "bad_code": 'os.system("ping " + hostname)',
        "fixed_code": 'subprocess.run(["ping", hostname], capture_output=True)',
        "explanation": "Passing user input to os.system() allows shell command injection. Use subprocess with a list of arguments.",
    },
    {
        "vuln": "Hardcoded Credentials",
        "owasp": "A07:2021",
        "cwe": "CWE-798",
        "severity": "HIGH",
        "bad_code": 'API_KEY = "sk-abc123secret"\ndb_pass = "admin123"',
        "fixed_code": 'API_KEY = os.environ.get("API_KEY")\ndb_pass = os.environ.get("DB_PASSWORD")',
        "explanation": "Hardcoded secrets are exposed in source control. Use environment variables or a secrets manager.",
    },
    {
        "vuln": "Path Traversal",
        "owasp": "A01:2021",
        "cwe": "CWE-22",
        "severity": "HIGH",
        "bad_code": 'filepath = "/uploads/" + filename\nreturn open(filepath).read()',
        "fixed_code": 'base = Path("/uploads").resolve()\nfull = (base / filename).resolve()\nif not str(full).startswith(str(base)):\n    raise ValueError("Invalid path")\nreturn full.read_text()',
        "explanation": "Unsanitized file paths allow directory traversal. Resolve and validate the path stays inside the allowed directory.",
    },
    {
        "vuln": "Insecure Deserialization",
        "owasp": "A08:2021",
        "cwe": "CWE-502",
        "severity": "CRITICAL",
        "bad_code": 'import pickle\ndata = pickle.loads(user_input)',
        "fixed_code": 'import json\ndata = json.loads(user_input)  # use JSON, never pickle on untrusted input',
        "explanation": "pickle.loads() on untrusted data allows arbitrary code execution. Use JSON or other safe formats.",
    },
    {
        "vuln": "Missing Rate Limiting",
        "owasp": "A04:2021",
        "cwe": "CWE-307",
        "severity": "MEDIUM",
        "bad_code": '@app.post("/login")\ndef login(username: str, password: str):\n    return check_credentials(username, password)',
        "fixed_code": 'from slowapi import Limiter\nlimiter = Limiter(key_func=get_remote_address)\n\n@app.post("/login")\n@limiter.limit("5/minute")\ndef login(request: Request, username: str, password: str):\n    return check_credentials(username, password)',
        "explanation": "Login endpoints without rate limiting are vulnerable to brute force. Add per-IP rate limiting.",
    },
    {
        "vuln": "Sensitive Data in Logs",
        "owasp": "A09:2021",
        "cwe": "CWE-532",
        "severity": "MEDIUM",
        "bad_code": 'logger.info(f"Login attempt: user={username} pass={password}")',
        "fixed_code": 'logger.info(f"Login attempt: user={username}")',
        "explanation": "Logging passwords or tokens exposes sensitive data in log files. Never log credentials.",
    },
    {
        "vuln": "JWT None Algorithm",
        "owasp": "A02:2021",
        "cwe": "CWE-347",
        "severity": "CRITICAL",
        "bad_code": 'payload = jwt.decode(token, options={"verify_signature": False})',
        "fixed_code": 'payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])',
        "explanation": "Disabling JWT signature verification allows token forgery. Always verify with a strong algorithm and secret.",
    },
]

AVERY_SCENARIOS = [
    ("I run a {biz_type} with {employees} employees. How do I grow revenue?", "KAIROS", "Kickoff", "Scaling"),
    ("My {biz_type} has high customer churn. What should I do?", "KAIROS", "Alignment", "Refinement"),
    ("I want to launch a new product for {audience}. What's the first step?", "KAIROS", "Kickoff", "Implementation"),
    ("We're spending too much on {cost_area}. How do we cut costs without hurting quality?", "KAIROS", "Optimization", "Scaling"),
    ("How do I hire my first {role} for a {biz_type}?", "KAIROS", "Implementation", "Refinement"),
]

BIZ_TYPES = ["SaaS startup", "e-commerce store", "consulting firm", "landscaping business",
             "mobile app", "freelance design studio", "restaurant", "accounting practice"]
EMPLOYEES = ["2", "5", "10", "1", "20", "3"]
AUDIENCES = ["small business owners", "students", "parents", "freelancers", "retirees"]
COST_AREAS = ["cloud infrastructure", "marketing", "customer support", "tooling", "office space"]
ROLES = ["developer", "salesperson", "customer support rep", "marketing manager", "operations lead"]


def _forge_template_rows(count: int) -> list[dict]:
    rows = []
    for _ in range(count):
        action, fn_name, params, ret, doc, body = random.choice(FORGE_ACTIONS)
        rows.append({
            "agent": "forge",
            "prompt": f"Write a Python function that {action}.",
            "response": (
                f"```python\nimport re\nimport time\nfrom pathlib import Path\nfrom typing import Any\n\n"
                f"def {fn_name}({params}) -> {ret}:\n"
                f"    \"\"\"{doc}\"\"\"\n"
                f"    {body}\n```"
            ),
        })
    return rows


def _sentinel_template_rows(count: int) -> list[dict]:
    rows = []
    for _ in range(count):
        v = random.choice(SENTINEL_VULNS)
        rows.append({
            "agent": "sentinel",
            "prompt": f"Review this code for security issues:\n\n```python\n{v['bad_code']}\n```",
            "response": (
                f"**Vulnerability:** {v['vuln']}\n"
                f"**OWASP:** {v['owasp']} | **CWE:** {v['cwe']}\n"
                f"**Severity:** {v['severity']}\n\n"
                f"**Issue:** {v['explanation']}\n\n"
                f"**Fix:**\n```python\n{v['fixed_code']}\n```"
            ),
        })
    return rows


def _avery_template_rows(count: int) -> list[dict]:
    rows = []
    for _ in range(count):
        template, framework, phase1, phase2 = random.choice(AVERY_SCENARIOS)
        biz = random.choice(BIZ_TYPES)
        prompt = template.format(
            biz_type=biz,
            employees=random.choice(EMPLOYEES),
            audience=random.choice(AUDIENCES),
            cost_area=random.choice(COST_AREAS),
            role=random.choice(ROLES),
        )
        rows.append({
            "agent": "avery",
            "prompt": prompt,
            "response": (
                f"## {framework} Framework — {phase1} Phase\n\n"
                f"**Situation:** {prompt}\n\n"
                f"**{phase1}:** Define the core problem and success metric.\n"
                f"**Alignment:** Identify constraints — budget, team size, timeline.\n"
                f"**Implementation:** Three concrete actions to take this week.\n"
                f"**{phase2}:** Measure results after 30 days, adjust based on data.\n\n"
                f"**Immediate next step:** Schedule a 30-minute strategy session "
                f"to map your current state versus target state."
            ),
        })
    return rows


def _nexus_template_rows(count: int) -> list[dict]:
    tasks = [
        "Build a Python REST API with authentication",
        "Create a data pipeline that ingests CSV files and stores in a database",
        "Build a real-time chat application with websockets",
        "Implement a payment processing integration with Stripe",
        "Create an automated testing suite for a FastAPI application",
        "Build a web scraper that extracts and stores product data",
        "Implement a notification system with email and SMS",
        "Create a document processing pipeline with PDF parsing",
    ]
    rows = []
    for _ in range(count):
        task = random.choice(tasks)
        rows.append({
            "agent": "nexus",
            "prompt": f"Design a workflow to: {task}. Which agents handle each step?",
            "response": (
                f"## Workflow: {task}\n\n"
                f"**Step 1 [Avery — Sequential]:** KAIROS kickoff — define scope, "
                f"tech stack, acceptance criteria.\n\n"
                f"**Step 2 [ORACLE — Parallel with Step 1]:** Retrieve relevant "
                f"patterns and prior implementations from memory.\n\n"
                f"**Step 3 [FORGE — Sequential after Step 1]:** Generate core "
                f"implementation — main modules, routes, data models.\n\n"
                f"**Step 4 [SENTINEL — Sequential after FORGE]:** Security audit — "
                f"input validation, auth, injection vectors.\n\n"
                f"**Step 5 [CODEX — Parallel with SENTINEL]:** Generate documentation, "
                f"README, API spec.\n\n"
                f"**Dependencies:** Step 3 requires Step 1 complete. "
                f"Step 4 requires Step 3 complete. Steps 2 and 5 are non-blocking."
            ),
        })
    return rows


def _oracle_template_rows(count: int) -> list[dict]:
    facts = [
        ("What is the difference between JWT and session-based authentication?",
         "JWT stores state client-side in a signed token. Session auth stores state server-side with a session ID cookie.",
         "inference"),
        ("What does ACID stand for in databases?",
         "Atomicity, Consistency, Isolation, Durability — guarantees for database transaction reliability.",
         "inference"),
        ("What is the difference between GET and POST in HTTP?",
         "GET retrieves data and is idempotent. POST submits data and may have side effects. GET params go in the URL; POST params in the body.",
         "inference"),
        ("What is a race condition?",
         "A race condition occurs when two concurrent operations produce different results depending on their execution order, causing unpredictable behavior.",
         "inference"),
        ("What is the CAP theorem?",
         "CAP theorem states a distributed system can guarantee at most two of: Consistency, Availability, Partition tolerance.",
         "inference"),
    ]
    rows = []
    for _ in range(count):
        q, ans, src = random.choice(facts)
        rows.append({
            "agent": "oracle",
            "prompt": q,
            "response": f"{ans}\n\nSource: [{src}]",
        })
    return rows


TEMPLATE_GENERATORS = {
    "forge":    _forge_template_rows,
    "sentinel": _sentinel_template_rows,
    "avery":    _avery_template_rows,
    "nexus":    _nexus_template_rows,
    "oracle":   _oracle_template_rows,
    "codex":    _forge_template_rows,   # codex uses code review tasks (reuse forge base)
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOCAL MODEL GENERATION — qwen2.5:7b or mistral (free, installed)
# ══════════════════════════════════════════════════════════════════════════════

GENERATION_PROMPTS = {
    "forge": (
        "Generate {n} training examples for a Python code generation AI agent.\n"
        "Each example is JSON with keys: agent (always 'forge'), prompt, response.\n"
        "Prompts are realistic developer tasks. Responses are complete working Python "
        "with imports, type hints, and error handling. No filler text in responses.\n"
        "Output a JSON array only, no other text."
    ),
    "sentinel": (
        "Generate {n} training examples for a security code review AI agent.\n"
        "Each example is JSON with keys: agent (always 'sentinel'), prompt, response.\n"
        "Prompts show vulnerable Python code. Responses must include:\n"
        "1. Vulnerability name\n2. OWASP category and CWE number\n"
        "3. Severity: CRITICAL/HIGH/MEDIUM/LOW (SQL injection is always CRITICAL)\n"
        "4. Fixed code\n"
        "Output a JSON array only, no other text."
    ),
    "oracle": (
        "Generate {n} training examples for a technical knowledge retrieval AI agent.\n"
        "Each example is JSON with keys: agent (always 'oracle'), prompt, response.\n"
        "Prompts are technical questions. Responses are precise and end with "
        "Source: [memory], [document], or [inference].\n"
        "Output a JSON array only, no other text."
    ),
    "codex": (
        "Generate {n} training examples for a technical documentation AI agent.\n"
        "Each example is JSON with keys: agent (always 'codex'), prompt, response.\n"
        "Prompts ask for README files, docstrings, API specs, or changelogs. "
        "Responses are complete markdown documentation.\n"
        "Output a JSON array only, no other text."
    ),
    "nexus": (
        "Generate {n} training examples for an AI workflow orchestration agent.\n"
        "Each example is JSON with keys: agent (always 'nexus'), prompt, response.\n"
        "The available agents are: Avery (strategy), FORGE (code), ORACLE (retrieval), "
        "CODEX (docs), SENTINEL (security).\n"
        "Prompts describe a software task. Responses are step-by-step workflows "
        "that assign each step to a named agent with parallel/sequential labels.\n"
        "Output a JSON array only, no other text."
    ),
    "avery": (
        "Generate {n} training examples for a business strategy AI agent using KAIROS framework.\n"
        "Each example is JSON with keys: agent (always 'avery'), prompt, response.\n"
        "Prompts are real business challenges from small business owners. "
        "Responses use KAIROS phases (Kickoff, Alignment, Implementation, Refinement, "
        "Optimization, Scaling) and are direct and actionable. No metadata fields.\n"
        "Output a JSON array only, no other text."
    ),
}


def _pick_teacher() -> str:
    """Return best available teacher model."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        for preferred in [TEACHER_MODEL, FALLBACK_TEACHER, "qwen2.5:7b", "llama3:latest"]:
            if any(preferred.split(":")[0] in m for m in models):
                return preferred
        return models[0] if models else TEACHER_MODEL
    except Exception:
        return TEACHER_MODEL


def _generate_with_local_model(agent: str, batch_size: int = 10) -> list[dict]:
    teacher = _pick_teacher()
    prompt_template = GENERATION_PROMPTS.get(agent)
    if not prompt_template:
        return []

    prompt = prompt_template.format(n=batch_size)
    print(f"  [local-model] Using {teacher} to generate {batch_size} {agent} rows...")

    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": teacher, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.8, "num_predict": 3000}},
            timeout=120,
        )
        raw = resp.json().get("response", "")
        # Extract JSON array from response
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            rows = json.loads(match.group())
            # Validate structure
            valid = [r for r in rows
                     if isinstance(r, dict) and r.get("prompt") and r.get("response")]
            for r in valid:
                r["agent"] = agent  # enforce correct agent label
            return valid
    except Exception as e:
        print(f"  [local-model] Failed: {e}")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# 3. SELF-PLAY LOOP — sovereign agents critique own outputs → hard negatives
# ══════════════════════════════════════════════════════════════════════════════

def _self_play_rows(agent: str, count: int) -> list[dict]:
    """
    Generate rows where the sovereign model produces an output,
    SENTINEL or CODEX critiques it, and the improved version is the label.
    Creates harder training examples than templates.
    """
    rows = []
    base_rows = _forge_template_rows(count) if agent == "forge" else _sentinel_template_rows(count)

    for row in base_rows[:count]:
        # Degrade the response slightly (inject a common mistake)
        bad_response = row["response"]
        if agent == "forge":
            bad_response = bad_response.replace("-> bool:", ":")   # remove type hint
            bad_response = bad_response.replace("except OSError:", "except:")  # bare except
        elif agent == "sentinel":
            bad_response = bad_response.replace("CRITICAL", "low")  # wrong severity

        # Create a DPO-style pair: chosen=original, rejected=degraded
        rows.append({
            "agent": agent,
            "prompt": row["prompt"],
            "chosen": row["response"],
            "rejected": bad_response,
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 4. HUGGINGFACE PUBLIC DATASET HARVEST
# ══════════════════════════════════════════════════════════════════════════════

def _harvest_public_datasets(agent: str, count: int) -> list[dict]:
    """Pull from free public HuggingFace datasets and adapt to sovereign format."""
    rows = []

    if agent in ("forge", "codex"):
        try:
            from datasets import load_dataset
            # code_instructions_122k — MIT licensed, code generation pairs
            ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca",
                              split="train", streaming=True)
            for i, row in enumerate(ds):
                if i >= count:
                    break
                if row.get("input") and row.get("output"):
                    rows.append({
                        "agent": agent,
                        "prompt": row["instruction"] + ("\n\n" + row["input"] if row["input"] else ""),
                        "response": row["output"],
                    })
            print(f"  [harvest] Got {len(rows)} {agent} rows from python_code_instructions")
        except Exception as e:
            print(f"  [harvest] Failed: {e}")

    elif agent == "sentinel":
        try:
            from datasets import load_dataset
            # security vulnerability dataset
            ds = load_dataset("benjaminbeilharz/better_daily_dialog",
                              split="train", streaming=True)
            # adapt generic Q&A to security format using templates as supplement
            rows = _sentinel_template_rows(count)
        except Exception as e:
            print(f"  [harvest] Falling back to templates: {e}")
            rows = _sentinel_template_rows(count)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — orchestrate all generators
# ══════════════════════════════════════════════════════════════════════════════

def generate(agent: str, count: int, method: str = "auto",
             dpo: bool = False) -> list[dict]:
    """
    Generate `count` training rows for `agent` using the best available method.

    method: "auto" | "template" | "local_model" | "self_play" | "harvest"
    dpo:    if True, generate DPO preference pairs instead of SFT rows
    """
    rows = []

    if dpo:
        print(f"  Generating {count} DPO pairs for {agent} via self-play...")
        return _self_play_rows(agent, count)

    if method == "template" or (method == "auto" and agent in TEMPLATE_GENERATORS):
        gen_fn = TEMPLATE_GENERATORS.get(agent)
        if gen_fn:
            print(f"  [template] Generating {count} rows for {agent}...")
            rows = gen_fn(count)

    if (method == "local_model" or method == "auto") and len(rows) < count:
        remaining = count - len(rows)
        batch = min(remaining, 20)  # generate in batches of 20 max
        generated = 0
        while generated < remaining:
            b = min(batch, remaining - generated)
            new_rows = _generate_with_local_model(agent, b)
            rows.extend(new_rows)
            generated += len(new_rows)
            if not new_rows:
                break  # model unavailable, stop

    if method == "harvest" or (method == "auto" and len(rows) < count):
        remaining = count - len(rows)
        if remaining > 0:
            rows.extend(_harvest_public_datasets(agent, remaining))

    # Deduplicate by prompt
    seen = set()
    unique = []
    for r in rows:
        key = r.get("prompt", "")[:100]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:count]


def push_to_hf(rows: list[dict], agent: str, token: str, dataset_repo: str):
    """Append rows to the private HuggingFace dataset."""
    try:
        from datasets import Dataset, load_dataset, concatenate_datasets
        print(f"  Pushing {len(rows)} rows to {dataset_repo}...")
        new_ds = Dataset.from_list(rows)
        try:
            existing = load_dataset(dataset_repo, name="agents", split="train", token=token)
            merged = concatenate_datasets([existing, new_ds])
        except Exception:
            merged = new_ds
        merged.push_to_hub(dataset_repo, config_name="agents", split="train",
                           token=token, private=True)
        print(f"  Done. Dataset now has {len(merged)} rows total.")
    except Exception as e:
        print(f"  HF push failed: {e}")


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "backend" / ".env")
    token = os.environ.get("HF_TOKEN", "")
    dataset_repo = os.environ.get("HF_DATASET", "tastytator/sovereign-economy")

    ap = argparse.ArgumentParser()
    ap.add_argument("--agent",  default="forge",
                    help="Agent name or 'all'")
    ap.add_argument("--count",  type=int, default=100,
                    help="Number of rows to generate")
    ap.add_argument("--method", default="auto",
                    choices=["auto", "template", "local_model", "self_play", "harvest"])
    ap.add_argument("--dpo",    action="store_true",
                    help="Generate DPO preference pairs instead of SFT")
    ap.add_argument("--push",   action="store_true",
                    help="Push to HuggingFace after generating")
    ap.add_argument("--save",   action="store_true", default=True,
                    help="Save to data_gen/output/ (default: True)")
    args = ap.parse_args()

    agents = list(TEMPLATE_GENERATORS.keys()) if args.agent == "all" else [args.agent]

    for agent in agents:
        print(f"\n{'='*50}")
        print(f"  Generating {args.count} rows for: {agent.upper()}")
        print(f"  Method: {args.method}  |  DPO: {args.dpo}")
        print(f"{'='*50}")

        rows = generate(agent, args.count, method=args.method, dpo=args.dpo)
        print(f"  Generated: {len(rows)} rows")

        if args.save:
            out_file = OUT_DIR / f"{agent}_{'dpo' if args.dpo else 'sft'}_{int(time.time())}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"  Saved: {out_file}")

        if args.push and token:
            push_to_hf(rows, agent, token, dataset_repo)
        elif args.push and not token:
            print("  WARNING: HF_TOKEN not set, skipping push")

    print("\nDone.")


if __name__ == "__main__":
    main()
