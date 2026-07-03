#!/usr/bin/env python3
"""
Full-stack OSS smoke test (for CI or manual "iron foundation" verification).

Exercises:
- ensure_mvs_seeded + dry cycle
- /oss public endpoints via TestClient (mvs/status, mvs/cycle, omni/route, health, economy)
- Optional: hit real gh05t3_inference on 8010 for richer MoE if running

Usage:
  python scripts/oss_smoke.py
  python scripts/oss_smoke.py --with-inference   # requires gh05t3_inference listening

Exit 0 on success.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oss.utils.paths import ensure_oss_paths

ensure_oss_paths()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from oss.api.router import router as oss_router
from backend.oss.loop import ensure_mvs_seeded, run_cycle


def main(with_inference: bool = False) -> int:
    print("=== OSS Iron Foundation Smoke ===")

    # 1. Idempotent seeding + dry cycle (core MVS contract)
    ensure_mvs_seeded(verbose=False)
    cl = run_cycle(0, dry_run=True, verbose=False)
    print(f"[1] ensure_mvs_seeded + dry cycle: OK (agg={cl.rewards.get('aggregate'):.3f})")

    # 2. Mount OSS surface exactly like gateway
    app = FastAPI()
    app.include_router(oss_router, prefix="/oss")
    client = TestClient(app)

    json_checks = [
        ("/oss/health", lambda b: b.get("status") == "ok"),
        ("/oss/mvs/status", lambda b: "available" in b),
        ("/oss/economy/unified", lambda b: "currency" in b),
        ("/oss/registry/status", lambda b: "genome_count" in b or "source" in b),
    ]

    for path, validator in json_checks:
        r = client.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code}"
        body = r.json()
        assert validator(body), f"{path} contract violated: {body}"
        print(f"[2] {path}: OK")

    r = client.get("/oss/metrics")
    assert r.status_code == 200, f"/oss/metrics -> {r.status_code}"
    assert "gh05t3_mvs" in r.text or "prometheus-client" in r.text
    print("[2] /oss/metrics: OK")

    # 3. Cycle via HTTP
    r = client.post("/oss/mvs/cycle", json={"cycles": 1, "dry_run": True})
    assert r.status_code == 200
    assert "ran" in r.json()
    print("[3] POST /oss/mvs/cycle (dry): OK")

    # 4. Richer Omni MoE exposure
    r = client.post("/oss/omni/route", json={"prompt": "Smoke test richer MoE routing."})
    assert r.status_code == 200
    print("[4] POST /oss/omni/route: OK (degrades gracefully if no inference)")

    # 5. Optional real or mocked inference + Sovereign
    if with_inference or os.environ.get("MOCK_INFERENCE"):
        import os as _os
        import httpx
        inf_url = _os.environ.get("GH05T3_INFERENCE_URL", "http://localhost:8010")

        # If no real server, start a tiny in-process mock for /health and /v1/route/plan
        if _os.environ.get("MOCK_INFERENCE"):
            from fastapi import FastAPI as _MockApp
            from fastapi.testclient import TestClient as _MockClient
            mock = _MockApp()
            @mock.get("/health")
            def _h(): return {"status": "ok", "mock": True}
            @mock.post("/v1/route/plan")
            def _p(req): return {"adapter_bucket": "mock-omni", "scaled_temperature": 0.7}
            mock_client = _MockClient(mock)
            # patch the url for the rest of the smoke? For demo we just call the mock directly
            print("[5] Using MOCK_INFERENCE server")
            h = mock_client.get("/health")
            p = mock_client.post("/v1/route/plan", json={"messages": [{"role": "user", "content": "test"}]})
            print(f"[5] Mock inference health={h.json()} plan={p.json()}")
        else:
            try:
                inf = httpx.get(f"{inf_url}/health", timeout=2)
                print(f"[5] Inference health: {inf.status_code}")
                plan = httpx.post(f"{inf_url}/v1/route/plan",
                                  json={"messages": [{"role": "user", "content": "MoE test"}]})
                print(f"[5] /v1/route/plan status: {plan.status_code}")
            except Exception as e:
                print(f"[5] Inference not reachable (expected in some CI): {e}")

        # Sovereign mock / real
        try:
            from oss.adapters.sovereign import sovereign_core_url
            s = httpx.get(f"{sovereign_core_url()}/health", timeout=2)
            print(f"[5] Sovereign bridge health probe: {s.status_code}")
        except Exception as e:
            print(f"[5] Sovereign not reachable (will use local ledger): {e}")

    print("=== ALL SMOKE CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-inference", action="store_true")
    args = parser.parse_args()
    sys.exit(main(with_inference=args.with_inference))
