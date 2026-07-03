"""
Free public dataset collectors for GH05T3 training pipeline.

Sources (all zero cost):
  - NVD (National Vulnerability Database) — public REST API, no key needed
  - MITRE ATT&CK — public JSON stix bundle
  - HuggingFace datasets — reasoning chains, instruction following
  - CWE (Common Weakness Enumeration) — public XML
  - HackerOne public Hacktivity feed — disclosed reports
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Generator

import httpx

LOG = logging.getLogger("ghost.training.collectors")

# Where raw collected data lands before generation pass
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# NVD CVE collector  (dataset #1 + #3 source material)
# ─────────────────────────────────────────────────────────────
async def collect_nvd_cves(max_results: int = 3000,
                            years: list[int] | None = None) -> list[dict]:
    """
    Pull CVE records from NVD public API (no key required for <5 req/30s).
    Returns list of dicts with keys: cve_id, description, cvss_score,
    weakness_ids, references, published, last_modified.
    """
    years = years or [2022, 2023, 2024]
    results: list[dict] = []
    base = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    # NVD rejects date ranges > 120 days without an API key — use 90-day windows
    windows: list[tuple[str, str]] = []
    for year in years:
        windows += [
            (f"{year}-01-01T00:00:00.000", f"{year}-03-31T23:59:59.999"),
            (f"{year}-04-01T00:00:00.000", f"{year}-06-30T23:59:59.999"),
            (f"{year}-07-01T00:00:00.000", f"{year}-09-30T23:59:59.999"),
            (f"{year}-10-01T00:00:00.000", f"{year}-12-31T23:59:59.999"),
        ]

    async with httpx.AsyncClient(timeout=30) as c:
        for start, end in windows:
            if len(results) >= max_results:
                break
            idx = 0
            while len(results) < max_results:
                try:
                    r = await c.get(base, params={
                        "pubStartDate": start,
                        "pubEndDate":   end,
                        "resultsPerPage": 100,
                        "startIndex":     idx,
                    })
                    r.raise_for_status()
                    data = r.json()
                    items = data.get("vulnerabilities", [])
                    if not items:
                        break
                    for item in items:
                        cve = item.get("cve", {})
                        desc = next(
                            (d["value"] for d in cve.get("descriptions", [])
                             if d.get("lang") == "en"), ""
                        )
                        metrics = cve.get("metrics", {})
                        cvss = 0.0
                        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                            if key in metrics:
                                cvss = metrics[key][0].get("cvssData", {}).get("baseScore", 0.0)
                                break
                        weaknesses = [
                            w["description"][0]["value"]
                            for w in cve.get("weaknesses", [])
                            if w.get("description")
                        ]
                        results.append({
                            "cve_id":        cve.get("id", ""),
                            "description":   desc,
                            "cvss_score":    cvss,
                            "weakness_ids":  weaknesses,
                            "published":     cve.get("published", ""),
                            "last_modified": cve.get("lastModified", ""),
                        })
                    idx += len(items)
                    time.sleep(0.7)  # stay under 5 req/30s rate limit
                except Exception as e:
                    LOG.warning("NVD fetch error (%s idx=%d): %s", start[:10], idx, e)
                    break

    LOG.info("Collected %d CVE records from NVD", len(results))
    out = RAW_DIR / "nvd_cves.jsonl"
    with open(out, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    return results


# ─────────────────────────────────────────────────────────────
# MITRE ATT&CK collector  (dataset #1 + #3 source material)
# ─────────────────────────────────────────────────────────────
async def collect_mitre_attack() -> list[dict]:
    """
    Download MITRE ATT&CK Enterprise STIX bundle.
    Returns techniques with name, description, tactic, detection, mitigation.
    """
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(url)
        r.raise_for_status()
        bundle = r.json()

    techniques: list[dict] = []
    mitigations: dict[str, str] = {}
    detections: dict[str, str] = {}

    for obj in bundle.get("objects", []):
        if obj.get("type") == "course-of-action":
            mitigations[obj["id"]] = obj.get("description", "")
        if obj.get("type") == "x-mitre-data-component":
            detections[obj["id"]] = obj.get("description", "")

    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("x_mitre_deprecated"):
            continue
        tactics = [p["phase_name"] for p in obj.get("kill_chain_phases", [])]
        ext = obj.get("x_mitre_detection", "")
        techniques.append({
            "technique_id":  obj.get("external_references", [{}])[0].get("external_id", ""),
            "name":          obj.get("name", ""),
            "description":   obj.get("description", ""),
            "tactics":       tactics,
            "detection":     ext,
            "platforms":     obj.get("x_mitre_platforms", []),
            "is_subtechnique": obj.get("x_mitre_is_subtechnique", False),
        })

    LOG.info("Collected %d ATT&CK techniques", len(techniques))
    out = RAW_DIR / "mitre_attack.jsonl"
    with open(out, "w") as f:
        for t in techniques:
            f.write(json.dumps(t) + "\n")
    return techniques


# ─────────────────────────────────────────────────────────────
# CWE weakness list  (dataset #1 source material)
# ─────────────────────────────────────────────────────────────
async def collect_cwe() -> list[dict]:
    """Download CWE weakness list from MITRE (JSON view)."""
    url = "https://cwe.mitre.org/data/json/cwec_latest.json.zip"
    import io, zipfile
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
        r = await c.get(url)
        if r.status_code != 200:
            LOG.warning("CWE download failed (%d) — skipping", r.status_code)
            return []
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        name = zf.namelist()[0]
        data = json.loads(zf.read(name))

    weaknesses: list[dict] = []
    for w in data.get("Weakness_Catalog", {}).get("Weaknesses", {}).get("Weakness", []):
        weaknesses.append({
            "cwe_id":        f"CWE-{w.get('@ID', '')}",
            "name":          w.get("@Name", ""),
            "description":   w.get("Description", ""),
            "extended_desc": w.get("Extended_Description", ""),
            "likelihood":    w.get("Likelihood_Of_Exploit", ""),
            "consequences":  [c.get("Impact", []) for c in
                              w.get("Common_Consequences", {}).get("Consequence", [])],
        })

    LOG.info("Collected %d CWE entries", len(weaknesses))
    out = RAW_DIR / "cwe.jsonl"
    with open(out, "w") as f:
        for w in weaknesses:
            f.write(json.dumps(w) + "\n")
    return weaknesses


# ─────────────────────────────────────────────────────────────
# HuggingFace reasoning dataset loader  (dataset #2 source)
# ─────────────────────────────────────────────────────────────
async def collect_hf_reasoning(max_examples: int = 3000) -> list[dict]:
    """
    Pull instruction/reasoning examples from HuggingFace datasets API.
    Uses 'Open-Orca/OpenOrca' (MIT license, ~1M reasoning chains) —
    no auth required for public datasets.
    """
    base = "https://datasets-server.huggingface.co/rows"
    results: list[dict] = []
    idx = 0

    async with httpx.AsyncClient(timeout=30) as c:
        while len(results) < max_examples:
            try:
                r = await c.get(base, params={
                    "dataset": "Open-Orca/OpenOrca",
                    "config":  "default",
                    "split":   "train",
                    "offset":  idx,
                    "length":  100,
                })
                if r.status_code != 200:
                    break
                rows = r.json().get("rows", [])
                if not rows:
                    break
                for row in rows:
                    row_data = row.get("row", {})
                    sys_p = row_data.get("system_prompt", "")
                    question = row_data.get("question", "")
                    response = row_data.get("response", "")
                    if question and response:
                        results.append({
                            "system_prompt": sys_p,
                            "question":      question,
                            "response":      response,
                        })
                idx += len(rows)
                time.sleep(0.3)
            except Exception as e:
                LOG.warning("HF fetch error (idx=%d): %s", idx, e)
                break

    LOG.info("Collected %d HF reasoning examples", len(results))
    out = RAW_DIR / "hf_reasoning.jsonl"
    with open(out, "w") as f:
        for ex in results:
            f.write(json.dumps(ex) + "\n")
    return results


# ─────────────────────────────────────────────────────────────
# Load cached raw data (skip network if already collected)
# ─────────────────────────────────────────────────────────────
def load_raw(name: str) -> list[dict]:
    path = RAW_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_stats() -> dict:
    stats = {}
    for p in RAW_DIR.glob("*.jsonl"):
        with open(p) as f:
            stats[p.stem] = sum(1 for line in f if line.strip())
    return stats
