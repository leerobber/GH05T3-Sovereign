"""GH05T3 backend API tests."""
import os
import time
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://tatorot-dashboard.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"


@pytest.fixture(scope="module")
def sess():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    # reset state for deterministic seed
    s.post(f"{API}/state/reset", timeout=30)
    return s


def test_state_seed(sess):
    r = sess.get(f"{API}/state", timeout=20)
    assert r.status_code == 200
    d = r.json()
    assert d["memory_palace"]["total"] == 103
    assert d["hcm"]["vectors"] == 146
    assert d["feynman"]["concepts"] == 59
    assert len(d["sub_agents"]) == 7
    assert len(d["ghost_protocol"]["layers"]) == 4
    assert len(d["autotelic_goals"]) == 21
    assert len(d["hardware_tatortot"]) >= 3
    assert len(d["pcl"]["palette"]) == 7
    assert d["identity"]["strange_loop_verdict"] == "OWNED"


def test_chat_short_id_engine(sess):
    r = sess.post(f"{API}/chat", json={"message": "hey"}, timeout=90)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["session_id"]
    assert d["user_message"]["engine"] == "ID"
    assert d["ghost_message"]["content"].strip() != ""
    # stash for history test
    pytest.session_id = d["session_id"]


def test_chat_long_ego_engine(sess):
    sid = pytest.session_id
    r = sess.post(f"{API}/chat", json={"session_id": sid, "message": "Can you explain what KAIROS does and how it promotes elite cycles?"}, timeout=120)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["user_message"]["engine"] == "EGO"
    assert d["session_id"] == sid


def test_chat_history(sess):
    sid = pytest.session_id
    r = sess.get(f"{API}/chat/history", params={"session_id": sid}, timeout=20)
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert len(msgs) >= 4
    assert msgs[0]["role"] == "user"
    # ordering
    ts = [m["timestamp"] for m in msgs]
    assert ts == sorted(ts)


def test_kairos_cycle(sess):
    pre = sess.get(f"{API}/state").json()["kairos"]["simulated_cycles"]
    r = sess.post(f"{API}/kairos/cycle", timeout=20)
    assert r.status_code == 200
    d = r.json()
    assert d["verdict"] in ("PASS", "PARTIAL", "FAIL")
    assert d["critic_decision"] in ("APPROVE", "REJECT", "REVISE")
    assert isinstance(d["final_score"], (int, float))
    assert isinstance(d["elite"], bool)
    post = sess.get(f"{API}/state").json()
    assert post["kairos"]["simulated_cycles"] == pre + 1
    assert len(post["kairos"]["recent"]) >= 1
    assert len(post["kairos"]["recent"]) <= 10


def test_nightly_training(sess):
    pre = sess.get(f"{API}/state").json()
    r = sess.post(f"{API}/training/nightly", timeout=20)
    assert r.status_code == 200
    d = r.json()
    delta = d["delta"]
    for k in ("memory_palace", "hcm_vectors", "feynman_concepts", "kairos_cycles", "new_goals"):
        assert k in delta
    post = sess.get(f"{API}/state").json()
    assert post["memory_palace"]["total"] > pre["memory_palace"]["total"]
    assert post["hcm"]["vectors"] > pre["hcm"]["vectors"]
    assert post["feynman"]["concepts"] > pre["feynman"]["concepts"]


def test_pcl_tick(sess):
    r = sess.post(f"{API}/pcl/tick", params={"state": "Learning"}, timeout=20)
    assert r.status_code == 200
    s = sess.get(f"{API}/state").json()["pcl"]
    assert s["state"] == "Learning"
    assert s["frequency_hz"] == 330
    assert s["color"].lower() == "#22d3ee"


def test_pcl_tick_unknown(sess):
    r = sess.post(f"{API}/pcl/tick", params={"state": "bogus"}, timeout=20)
    assert r.status_code == 404


def test_state_reset(sess):
    r = sess.post(f"{API}/state/reset", timeout=20)
    assert r.status_code == 200
    d = sess.get(f"{API}/state").json()
    assert d["memory_palace"]["total"] == 103
    assert d["kairos"]["simulated_cycles"] == 35
