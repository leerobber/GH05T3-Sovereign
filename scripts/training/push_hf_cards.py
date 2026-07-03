"""
push_hf_cards.py — Push updated model cards to HuggingFace.
Run with system Python: python push_hf_cards.py
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).parent
ENV  = ROOT / ".env"

# Load .env
if ENV.exists():
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set in .env"); sys.exit(1)

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

cards = [
    ("tastytator/sovereign-economy",    "dataset", ROOT / "hf_cards" / "dataset_README.md"),
    ("tastytator/avery-sovereign-lora", "model",   ROOT / "hf_cards" / "model_README.md"),
]

for repo_id, repo_type, card_path in cards:
    if not card_path.exists():
        print(f"SKIP {repo_id} — card file not found: {card_path}")
        continue
    print(f"Pushing README to {repo_id} ({repo_type})...")
    try:
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Update model/dataset card to v3 stats",
        )
        print(f"  OK — https://huggingface.co/{'datasets/' if repo_type=='dataset' else ''}{repo_id}")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nDone.")
