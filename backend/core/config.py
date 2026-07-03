"""GH05T3 core configuration — reads environment variables."""
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

BACKENDS = {
    "primary":  os.environ.get("VLLM_PRIMARY_URL",    "http://localhost:8010"),
    "verifier": os.environ.get("LLAMA_VERIFIER_URL",  "http://localhost:8011"),
    "fallback":  os.environ.get("LLAMA_FALLBACK_URL", "http://localhost:8012"),
}

# Ollama sovereign agents (fine-tuned, always running)
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
SOVEREIGN_MODELS = {
    "oracle":   os.environ.get("MODEL_ORACLE",   "oracle-sovereign"),
    "forge":    os.environ.get("MODEL_FORGE",    "forge-sovereign"),
    "codex":    os.environ.get("MODEL_CODEX",    "codex-sovereign"),
    "sentinel": os.environ.get("MODEL_SENTINEL", "sentinel-sovereign"),
    "nexus":    os.environ.get("MODEL_NEXUS",    "nexus-sovereign"),
    "avery":    os.environ.get("MODEL_AVERY",    "avery-sovereign"),
}

# GH05T3 fine-tuned inference server (gh05t3_inference.py — port 8010)
GH05T3_MODEL_URL = os.environ.get("GH05T3_MODEL_URL", "http://localhost:8010")

GATEWAY_HOST  = os.environ.get("GATEWAY_HOST",  "0.0.0.0")
GATEWAY_PORT  = int(os.environ.get("GATEWAY_PORT", "8002"))
GITHUB_PAT    = os.environ.get("GITHUB_PAT",    "")
GITHUB_REPO   = os.environ.get("GITHUB_REPO",   "leerobber/GH05T3")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

# Stripe — billing & subscriptions
STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY",     "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
