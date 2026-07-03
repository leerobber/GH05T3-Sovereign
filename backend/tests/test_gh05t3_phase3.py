"""GH05T3 phase-3 backend API tests.

Covers:
- /api/setup/status (first-boot LLM nudge)
- /api/embeddings/status (hybrid MiniLM)
- /api/ollama/status, /configure, /pull (LOQ gateway)
- /api/coder/repos, /api/coder/task whitelist validation
- /api/memory + /api/memory/search semantic ranking (local MiniLM)
- Regressions: /api/chat, /api/state, /api/llm/config
"""
import os
import time
import uuid
import pytest
import requests

BASE_URL = (os.environ.get("REACT_APP_BACKEND_URL")
            or "https://tatorot-dashboard.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"


@pytest.fixture(scope="module")
def sess():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


# --- /api/setup/status -------------------------------------------------------
def test_setup_status_shape(sess):
    r = sess.get(f"{API}/setup/status", timeout=20)
    assert r.status_code == 200, r.text
    d = r.json()
    for k in ("needs_setup", "has_google_key", "has_groq_key",
              "ollama_reachable", "emergent_available"):
        assert k in d, f"missing key {k}"
    assert isinstance(d["needs_setup"], bool)
    assert isinstance(d["emergent_available"], bool)


# --- /api/embeddings/status --------------------------------------------------
def test_embeddings_status_local_minilm(sess):
    r = sess.get(f"{API}/embeddings/status", timeout=30)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("local_loaded") is True, d
    assert d.get("local_model") == "sentence-transformers/all-MiniLM-L6-v2", d
    assert d.get("local_dim") == 384, d


# --- /api/ollama ------------------------------------------------------------
def test_ollama_status_configured_but_unreachable(sess):
    """After phase-2 tests ran, OLLAMA_GATEWAY_URL may or may not be set. The
    key guarantee is: reachable=False because nothing is actually running."""
    r = sess.get(f"{API}/ollama/status", timeout=10)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("reachable") is False, d
    assert "preferred" in d
    pref = d["preferred"]
    assert pref.get("proposer") == "qwen2.5:7b-q4"
    assert pref.get("verifier") == "deepseek-coder:6.7b"
    assert pref.get("critic") == "llama3.1"


def test_ollama_configure_fake_url_persists(sess):
    fake = "http://nonexistent.local:11434"
    r = sess.post(f"{API}/ollama/configure", json={"gateway_url": fake}, timeout=10)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("reachable") is False, d
    assert d.get("url") == fake
    # Re-query status to verify persistence on live env
    r2 = sess.get(f"{API}/ollama/status", timeout=10).json()
    assert r2.get("url") == fake
    assert r2.get("reachable") is False


def test_ollama_pull_unreachable_gateway(sess):
    # After configure above, gateway is the fake URL — pull should fail with error
    r = sess.post(f"{API}/ollama/pull", json={"model": "qwen2.5:7b-q4"}, timeout=15)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("ok") is False, d
    assert d.get("error"), d


# --- /api/coder --------------------------------------------------------------
def test_coder_repos_returns_whitelist(sess):
    r = sess.get(f"{API}/coder/repos", timeout=30)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("has_pat") is True, "GITHUB_PAT must be set"
    wl = d.get("whitelist") or []
    assert len(wl) == 5, f"expected 5 whitelisted repos, got {len(wl)}: {wl}"
    for entry in wl:
        assert entry.startswith("leerobber/"), entry
    repos = d.get("repos") or []
    # repos list may be < 5 if PAT lacks access, but in this env we expect all 5
    assert len(repos) == 5, f"expected 5 repo metadata entries, got {len(repos)}"
    for r_meta in repos:
        assert "full_name" in r_meta
        assert "description" in r_meta


def test_coder_task_rejects_non_whitelisted(sess):
    r = sess.post(f"{API}/coder/task", json={
        "repo": "attacker/evil",
        "task": "please drop all tables",
        "max_iterations": 1,
        "open_pr": False,
    }, timeout=20)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("ok") is False, d
    err = (d.get("error") or "").lower()
    assert "not whitelisted" in err, d
    assert "attacker/evil" in (d.get("error") or "")


# --- /api/memory semantic ranking -------------------------------------------
def test_memory_store_uses_local_minilm_and_ranks_semantically(sess):
    tag = uuid.uuid4().hex[:8]
    memA = f"TEST_{tag} Robert prefers terse high-signal responses"
    memB = f"TEST_{tag} The user loves the color amber"

    rA = sess.post(f"{API}/memory",
                   json={"content": memA, "type": "preference",
                         "source": "test", "importance": 0.8}, timeout=30)
    assert rA.status_code == 200, rA.text
    dA = rA.json()
    # Expected fields from server.py memory route
    assert dA.get("embed_mode") == "local:minilm", dA
    assert dA.get("embed_dim") == 384, dA

    rB = sess.post(f"{API}/memory",
                   json={"content": memB, "type": "preference",
                         "source": "test", "importance": 0.8}, timeout=30)
    assert rB.status_code == 200, rB.text
    dB = rB.json()
    assert dB.get("embed_mode") == "local:minilm", dB
    assert dB.get("embed_dim") == 384, dB

    # Give Mongo a moment
    time.sleep(1)

    q = "what tone does Robert want"
    rs = sess.get(f"{API}/memory/search", params={"q": q, "k": 10}, timeout=30)
    assert rs.status_code == 200, rs.text
    hits = rs.json().get("hits") or rs.json().get("results") or []
    assert len(hits) >= 2, f"need at least 2 hits: {rs.json()}"

    # Filter to our tagged memories and check ranking
    tagged = [h for h in hits if tag in (h.get("content") or "")]
    assert len(tagged) >= 2, f"our 2 TEST memories not both returned: {tagged}"
    # First tagged result must be the tone/Robert one, not the amber one
    top = tagged[0]
    assert "Robert" in (top.get("content") or ""), (
        f"semantic ranking failed — top={top}"
    )
    # Score should be higher than amber one
    score_map = {h["content"]: h.get("score", 0) for h in tagged}
    rob_score = next(v for k, v in score_map.items() if "Robert" in k)
    amb_score = next(v for k, v in score_map.items() if "amber" in k)
    assert rob_score > amb_score, (
        f"Robert score {rob_score} not > amber score {amb_score}"
    )


# --- Regressions -------------------------------------------------------------
def test_chat_still_works(sess):
    r = sess.post(f"{API}/chat",
                  json={"session_id": f"test_{uuid.uuid4().hex[:6]}",
                        "message": "hello"}, timeout=90)
    assert r.status_code == 200, r.text
    d = r.json()
    assert "ghost_message" in d, d
    gm = d["ghost_message"]
    assert isinstance(gm.get("content"), str) and len(gm["content"]) > 0, gm


def test_state_still_200(sess):
    r = sess.get(f"{API}/state", timeout=20)
    assert r.status_code == 200
    d = r.json()
    assert "hcm" in d
    assert "gateway" in d


def test_llm_config_still_200(sess):
    r = sess.get(f"{API}/llm/config", timeout=20)
    assert r.status_code == 200
