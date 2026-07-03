"""Upload current bootstrap_dataset.jsonl to HF as bootstrap_dpo split."""
import json, os
from pathlib import Path

env_path = Path('.env')
for line in env_path.read_text(encoding='utf-8').splitlines():
    if '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"\''))

hf_token = os.environ.get('HF_TOKEN', '')
if not hf_token:
    print("ERROR: HF_TOKEN not set"); raise SystemExit(1)

data_file = Path('data/bootstrap_dataset.jsonl')
rows = [json.loads(l) for l in data_file.open(encoding='utf-8') if l.strip()]
print(f"Loaded {len(rows)} DPO pairs from {data_file}")
print(f"Sample keys: {list(rows[0].keys())}")

from datasets import Dataset

# DPO split: keep goal->prompt mapping, chosen, rejected
dpo_rows = []
skipped = 0
for r in rows:
    goal = str(r.get('goal') or r.get('instruction') or '')
    prompt = str(r.get('prompt') or f"GOAL: {goal}\n\nProvide a detailed sovereign strategy:")
    chosen   = r.get('chosen', '')
    rejected = r.get('rejected', '')
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        skipped += 1
        continue
    if not chosen.strip() or not rejected.strip():
        skipped += 1
        continue
    dpo_rows.append({
        'prompt':   prompt,
        'chosen':   chosen,
        'rejected': rejected,
        'domain':   str(r.get('domain', '')),
    })
if skipped:
    print(f"Skipped {skipped} malformed rows")

ds = Dataset.from_list(dpo_rows)
print(f"Pushing {len(ds)} rows to tastytator/sovereign-economy split=bootstrap_dpo ...")
ds.push_to_hub('tastytator/sovereign-economy', split='bootstrap_dpo',
               token=hf_token, private=False)
print("Done. bootstrap_dpo is live on HuggingFace.")
