"""
preflight_avery.py
==================
Validates every component of the Avery 3B pipeline before Kaggle run.
Run: python preflight_avery.py
"""

import json
import sys
import urllib.request
from pathlib import Path

OK = True

def check(label, passed, detail=""):
    global OK
    status = "PASS" if passed else "FAIL"
    if not passed:
        OK = False
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))


print()
print("=" * 55)
print("  Avery 3B Pipeline Pre-Flight Check")
print("=" * 55)
print()

# 1. HF Token
tok_path = Path("C:/Users/leer4/.cache/huggingface/token")
tok_ok = tok_path.exists() and len(tok_path.read_text().strip()) > 10
check("HF Token", tok_ok)

# 2. llama.cpp
llama = Path("C:/llama.cpp/convert_hf_to_gguf.py")
check("llama.cpp convert_hf_to_gguf.py", llama.exists(), str(llama))

# 3. Economy dataset on HuggingFace
try:
    from datasets import load_dataset
    tok = tok_path.read_text().strip()
    ds = load_dataset("tastytator/sovereign-economy", name="economy", split="train", token=tok)
    check("HF economy/train dataset", True, f"{len(ds):,} rows")
except Exception as e:
    check("HF economy/train dataset", False, str(e)[:80])

# 4. Notebook valid + all changes applied
nb_path = Path("C:/Users/leer4/GH05T3/kaggle_train.ipynb")
try:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    src = " ".join("".join(c["source"]) for c in cells)
    check("Notebook BASE_MODEL = Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct" in src)
    check("Notebook MODE = sft",                        "MODE   = 'sft'" in src)
    check("Notebook economy data merge",                "Economy data boost" in src)
    check("Notebook EPOCHS = 1",                        "EPOCHS = 1" in src)
except Exception as e:
    check("Notebook valid JSON", False, str(e))

# 5. Python packages
for pkg, min_ok in [
    ("transformers", None),
    ("peft", None),
    ("torch", None),
    ("datasets", None),
    ("huggingface_hub", None),
]:
    try:
        m = __import__(pkg)
        ver = getattr(m, "__version__", "ok")
        check(f"Package: {pkg}", True, ver)
    except ImportError:
        check(f"Package: {pkg}", False, f"pip install {pkg}")

# 6. Scripts exist
check("merge_avery_3b.py",  Path("C:/Users/leer4/GH05T3/merge_avery_3b.py").exists())
check("rebuild_avery.bat",  Path("C:/Users/leer4/GH05T3/rebuild_avery.bat").exists())
check("upload_economy_data.py", Path("C:/Users/leer4/GH05T3/upload_economy_data.py").exists())

# 7. Ollama running + avery present
try:
    resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
    data = json.loads(resp.read())
    models = [m["name"] for m in data.get("models", [])]
    check("Ollama running", True, f"{len(models)} models loaded")
    check("avery-sovereign:latest in Ollama", "avery-sovereign:latest" in models,
          "current 7B model — will be replaced by 3B after rebuild")
except Exception as e:
    check("Ollama running", False, str(e))

# 8. Economy runner still going
try:
    resp = urllib.request.urlopen("http://localhost:8081/runner/status", timeout=3)
    data = json.loads(resp.read())
    running = data.get("running", False)
    tick = data.get("last_tick", "?")
    check("Economy runner active", running, f"tick={tick}")
except Exception as e:
    check("Economy runner", False, str(e))

print()
print("=" * 55)
result = "ALL CLEAR -- ready to upload notebook to Kaggle" if OK else "FAILURES FOUND -- fix before running"
print(f"  Result: {result}")
print("=" * 55)

if OK:
    print()
    print("Next steps:")
    print("  1. Upload kaggle_train.ipynb to https://www.kaggle.com/code")
    print("  2. Add HF_TOKEN secret in Kaggle Notebook -> Settings -> Secrets")
    print("  3. Enable GPU (T4 x1), run all cells")
    print("     Training time: ~2-4 hrs for 1 epoch on 108K records")
    print("  4. When Kaggle shows 'Done': double-click rebuild_avery.bat")
    print("     (takes 20-40 min: CPU merge + GGUF convert + ollama create)")
    print("  5. Test: ollama run avery-sovereign")
    print()

sys.exit(0 if OK else 1)
