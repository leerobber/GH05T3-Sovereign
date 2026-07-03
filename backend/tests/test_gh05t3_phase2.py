"""GH05T3 phase-2 backend API tests (websocket, scheduler, kairos LLM, cassandra,
ghostscript, stego, telegram, exception middleware)."""
import asyncio
import json
import os
import time
import pytest
import requests
websockets = pytest.importorskip("websockets", reason="websockets not installed")

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://tatorot-dashboard.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"
WS_URL = BASE_URL.replace("https://", "wss://").replace("http://", "ws://") + "/api/ws"


@pytest.fixture(scope="module")
def sess():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


# --- /api/state additions -----------------------------------------------------
def test_state_has_scheduler_and_gateway(sess):
    r = sess.get(f"{API}/state", timeout=20)
    assert r.status_code == 200
    d = r.json()
    assert "scheduler" in d, "scheduler key missing"
    assert isinstance(d["scheduler"]["running"], bool)
    assert "gateway" in d
    assert "ollama_configured" in d["gateway"]
    assert "ollama_reachable" in d["gateway"]
    # ollama not installed in env
    assert d["gateway"]["ollama_reachable"] is False
    assert d["hcm"]["vectors"] == 146


# --- HCM cloud projection -----------------------------------------------------
def test_hcm_cloud_projection(sess):
    r = sess.get(f"{API}/hcm/cloud", timeout=30)
    assert r.status_code == 200
    cloud = r.json()["cloud"]
    assert len(cloud) == 146, f"expected 146 points, got {len(cloud)}"
    for pt in cloud[:5]:
        for k in ("idx", "label", "room", "color", "x", "y"):
            assert k in pt, f"missing {k}"
        assert -1.0 <= pt["x"] <= 1.0
        assert -1.0 <= pt["y"] <= 1.0


# --- KAIROS real LLM cycle ----------------------------------------------------
def test_kairos_real_cycle_and_recent(sess):
    pre = sess.get(f"{API}/state").json()["kairos"]["simulated_cycles"]
    r = sess.post(f"{API}/kairos/cycle", timeout=120)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["critic_decision"] in ("APPROVE", "REJECT", "REVISE"), d
    assert d["verdict"] in ("PASS", "PARTIAL", "FAIL")
    assert isinstance(d.get("proposal"), str) and len(d["proposal"]) > 20
    assert isinstance(d.get("critic_reason"), str) and len(d["critic_reason"]) > 5
    assert isinstance(d.get("verifier_rationale"), str) and len(d["verifier_rationale"]) > 5
    assert isinstance(d.get("base_score"), (int, float))
    assert isinstance(d.get("multiplier"), (int, float))
    assert isinstance(d.get("final_score"), (int, float))

    post = sess.get(f"{API}/state").json()
    assert post["kairos"]["simulated_cycles"] == pre + 1

    rec = sess.get(f"{API}/kairos/recent", timeout=20)
    assert rec.status_code == 200
    cycles = rec.json()["cycles"]
    assert len(cycles) >= 1


# --- Cassandra ----------------------------------------------------------------
def test_cassandra_premortem(sess):
    r = sess.post(f"{API}/cassandra", json={"scenario": "launch KAIROS live tomorrow"}, timeout=120)
    assert r.status_code == 200, r.text
    d = r.json()
    assert isinstance(d.get("autopsy"), str)
    assert len(d["autopsy"]) > 100, f"autopsy too short: {len(d['autopsy'])}"


def test_cassandra_empty_400(sess):
    r = sess.post(f"{API}/cassandra", json={"scenario": "   "}, timeout=20)
    assert r.status_code == 400


# --- Steganography ------------------------------------------------------------
def test_stego_roundtrip_hi(sess):
    r = sess.post(f"{API}/stego/encode", json={"secret": "Hi"}, timeout=20)
    assert r.status_code == 200, r.text
    enc = r.json()
    assert enc["bits"] >= 16, enc
    assert enc["covertext"]
    r2 = sess.post(f"{API}/stego/decode",
                   json={"covertext": enc["covertext"], "byte_count": 2}, timeout=20)
    assert r2.status_code == 200, r2.text
    assert r2.json()["secret"] == "Hi"


# --- GhostScript --------------------------------------------------------------
def test_ghostscript_demo(sess):
    r = sess.get(f"{API}/ghostscript/demo", timeout=20)
    assert r.status_code == 200
    body = r.json()
    res = body.get("result", body)
    assert res.get("ok") is True, res
    steps_str = json.dumps(res)
    for tok in ("spawn", "think", "emit", "bind", "dispatch"):
        assert tok in steps_str, f"missing token {tok}"


def test_ghostscript_malformed(sess):
    r = sess.post(f"{API}/ghostscript/run", json={"source": "spawn @@@ broken !!!"}, timeout=20)
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is False
    assert "error" in body and body["error"]


# --- Scheduler ----------------------------------------------------------------
def test_scheduler_toggle(sess):
    r = sess.post(f"{API}/scheduler/toggle", params={"enable": "true"}, timeout=20)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["running"] is True
    job_ids = {j["id"] for j in d["jobs"]}
    assert "kairos_03" in job_ids
    assert "amplifiers_04" in job_ids


# --- Telegram -----------------------------------------------------------------
def test_telegram_lifecycle(sess):
    s0 = sess.get(f"{API}/telegram/status", timeout=20)
    assert s0.status_code == 200
    # configure with invalid token
    cfg = sess.post(f"{API}/telegram/configure",
                    json={"bot_token": "invalid:token"}, timeout=20)
    assert cfg.status_code == 200, cfg.text
    # start - should not raise; getMe will fail and surface in last_error
    start = sess.post(f"{API}/telegram/start", timeout=30)
    assert start.status_code == 200, start.text
    time.sleep(2)
    st = sess.get(f"{API}/telegram/status", timeout=20).json()
    # Either running with last_error populated, or not running at all.
    assert "last_error" in st
    stop = sess.post(f"{API}/telegram/stop", timeout=20)
    assert stop.status_code == 200


# --- WebSocket ----------------------------------------------------------------
@pytest.mark.asyncio
async def test_ws_hello_and_kairos_event():
    async with websockets.connect(WS_URL, open_timeout=15, close_timeout=5) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=15)
        first = json.loads(msg)
        assert first.get("event") == "hello"

        # trigger a kairos cycle in background, then await events
        loop = asyncio.get_running_loop()
        def _fire():
            return requests.post(f"{API}/kairos/cycle", timeout=120)
        fut = loop.run_in_executor(None, _fire)

        events = []
        # collect for up to 60s waiting for kairos_cycle and state_delta
        end = time.time() + 60
        while time.time() < end and not ({"kairos_cycle", "state_delta"} <= set(events)):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=20)
                ev = json.loads(raw).get("event")
                events.append(ev)
            except asyncio.TimeoutError:
                break
        await fut
        assert "kairos_cycle" in events, f"events={events}"
        assert "state_delta" in events, f"events={events}"


# --- Exception middleware -----------------------------------------------------
def test_pcl_unknown_state_is_404_no_seance(sess):
    """HTTPException must NOT be captured by séance middleware."""
    pre = sess.get(f"{API}/state").json().get("seance", [])
    r = sess.post(f"{API}/pcl/tick", params={"state": "BogusState"}, timeout=20)
    assert r.status_code == 404
    post = sess.get(f"{API}/state").json().get("seance", [])
    assert len(post) == len(pre), "HTTPException should not be appended to seance"


def test_cassandra_empty_does_not_seance(sess):
    pre = sess.get(f"{API}/state").json().get("seance", [])
    r = sess.post(f"{API}/cassandra", json={"scenario": ""}, timeout=20)
    assert r.status_code == 400
    post = sess.get(f"{API}/state").json().get("seance", [])
    assert len(post) == len(pre)
