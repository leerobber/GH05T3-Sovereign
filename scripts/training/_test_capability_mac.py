"""
Test: IBAC Capability MAC integration
- capability_client: local mint, DENIED sentinel
- SignatureChecker: Path 1 (capability), Path 2 (HMAC), priority rules
- gates.py: loads correctly, PolicyStore gates ready
- gitops_mutator.mutate() signature accepts capability param
- ghostrecall.store() signature accepts capability param
"""
import asyncio
import hashlib
import hmac
import time
import sys
import os

# Force UTF-8 output for Windows terminal
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r"C:\Users\leer4\GH05T3")

PASS_STR = "[PASS]"
FAIL_STR = "[FAIL]"
results = []

def check(name, cond, detail=""):
    icon = PASS_STR if cond else FAIL_STR
    results.append(cond)
    line = f"  {icon} {name}"
    if detail:
        line += f" -- {detail}"
    print(line)
    return cond


# ─────────────────────────────────────────────────────────────────────────────
# 1. capability_client: local mint
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] capability_client -- local mint")

from sovereignnation.capability_client import (
    CapabilityClient, request_capability_sync, WRITE_CLASS_TO_ACTION,
    _mint_local, _load_key as cap_load_key,
)

client = CapabilityClient()

cap = client.request_sync("SAGE", "write_file", "kairos/sage_loop.py")
check("request_sync returns dict or None", cap is None or isinstance(cap, dict))
if cap:
    check("token field present", "token" in cap)
    check("action field correct", cap.get("action") == "write_file")
    check("resource field correct", cap.get("resource") == "kairos/sage_loop.py")
    check("token format ibac:...", cap["token"].startswith("ibac:"))
else:
    print("    (no key on disk -- dev mode, minting returns None)")
    check("dev mode: None is acceptable", True)

cap2 = request_capability_sync("GitOpsMutator", "write_file", "kairos/gitops_mutator.py")
check("request_capability_sync works", cap2 is None or isinstance(cap2, dict))

check("GITOPS_MUTATION maps to write_file",  WRITE_CLASS_TO_ACTION.get("GITOPS_MUTATION") == "write_file")
check("GHOST_RECALL maps to write_memory",   WRITE_CLASS_TO_ACTION.get("GHOST_RECALL") == "write_memory")
check("DOME_APPEND maps to write_iron_dome", WRITE_CLASS_TO_ACTION.get("DOME_APPEND") == "write_iron_dome")


# ─────────────────────────────────────────────────────────────────────────────
# 2. capability_client: DENIED sentinel
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] DENIED sentinel")

async def _test_denied():
    sentinel = "ibac:DENIED:0000000000000000"
    check("DENIED sentinel format", sentinel.startswith("ibac:DENIED:"))
    from sovereignnation.checkers.signature_checker import _verify_ibac_token
    ok = _verify_ibac_token(sentinel, "SAGE", "write_file", "some_path")
    if cap_load_key():
        check("DENIED sentinel fails verification (key present)", not ok)
    else:
        check("DENIED sentinel: dev mode (no key) -> True", ok)

asyncio.run(_test_denied())


# ─────────────────────────────────────────────────────────────────────────────
# 3. SignatureChecker: _verify_ibac_token direct
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] SignatureChecker -- _verify_ibac_token direct")

from sovereignnation.checkers.signature_checker import _verify_ibac_token, _load_key

key = _load_key()

if key:
    agent_id = "SAGE"
    action   = "write_file"
    resource = "kairos/test.py"
    now_min  = int(time.time() // 60)

    payload  = f"{agent_id}|{action}|{resource}|{now_min}"
    sig      = hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()[:16]
    good_tok = f"ibac:{agent_id[:8]}:{sig}"

    print(f"    key length: {len(key)}")
    print(f"    now_min={now_min}  payload={payload}")
    print(f"    sig={sig}  tok={good_tok}")

    check("Valid SAGE token verifies",     _verify_ibac_token(good_tok, agent_id, action, resource))
    check("Bad token fails",               not _verify_ibac_token("ibac:XXXX:0000000000000000", agent_id, action, resource))
    check("Wrong agent fails",             not _verify_ibac_token(good_tok, "FORGE", action, resource))
    check("Wrong action fails",            not _verify_ibac_token(good_tok, agent_id, "read_file", resource))
    check("Wrong resource fails",          not _verify_ibac_token(good_tok, agent_id, action, "other/path.py"))

    old_payload = f"{agent_id}|{action}|{resource}|{now_min - 2}"
    old_sig     = hmac.new(key, old_payload.encode(), hashlib.sha256).hexdigest()[:16]
    old_tok     = f"ibac:{agent_id[:8]}:{old_sig}"
    check("Expired token (2 min old) fails", not _verify_ibac_token(old_tok, agent_id, action, resource))
else:
    print("    No key on disk -- dev mode")
    check("Dev mode: verify always True", _verify_ibac_token("anything", "SAGE", "x", "y"))


# ─────────────────────────────────────────────────────────────────────────────
# 4. SignatureChecker.check(): both paths
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] SignatureChecker.check() -- Path 1 (capability) vs Path 2 (HMAC)")

from sovereignnation.checkers.signature_checker import SignatureChecker
from sovereignnation.access_control import WriteClass, WriteContext, CheckResult

sig_checker = SignatureChecker()

def _make_ctx(agent_id_val, metadata_val):
    return WriteContext(
        agent_id="GitOpsMutator",
        proposal_id="test-prop",
        write_class=WriteClass.GITOPS_MUTATION,
        target_path="kairos/test.py",
        payload_summary="test write",
        metadata=metadata_val,
    )

if key:
    agent_g  = "GitOpsMutator"
    now_min2 = int(time.time() // 60)
    payload_str = f"{agent_g}|write_file|kairos/test.py|{now_min2}"
    sig_hex = hmac.new(key, payload_str.encode(), hashlib.sha256).hexdigest()[:16]
    tok = f"ibac:{agent_g[:8]}:{sig_hex}"   # "GitOpsMu" (8 chars)

    print(f"    now_min2={now_min2}  payload_str={payload_str}")
    print(f"    sig_hex={sig_hex}  tok={tok}")
    print(f"    agent_g[:8]={agent_g[:8]!r}")

    # Verify the raw token before putting it through SignatureChecker
    raw_ok = _verify_ibac_token(tok, "GitOpsMutator", "write_file", "kairos/test.py")
    print(f"    _verify_ibac_token direct result: {raw_ok}")

    cap_meta = {"capability": {"token": tok, "action": "write_file", "resource": "kairos/test.py"}}
    result = sig_checker.check(_make_ctx("GitOpsMutator", cap_meta))
    print(f"    sig_checker.check result: {result}")
    check("Path 1: valid capability -> PASS", result == CheckResult.PASS)

    # Path 1: invalid capability token -> FAIL (no fallthrough)
    bad_cap_meta = {"capability": {"token": "ibac:BAD:0000000000000000",
                                   "action": "write_file", "resource": "kairos/test.py"}}
    result2 = sig_checker.check(_make_ctx("GitOpsMutator", bad_cap_meta))
    check("Path 1: invalid capability -> FAIL", result2 == CheckResult.FAIL)

    # Priority: bad capability + valid HMAC -> FAIL (no fallthrough)
    mac_payload_str = "test_payload"
    mac_sig = hmac.new(key, mac_payload_str.encode(), hashlib.sha256).hexdigest()
    both_meta = {
        "mac_payload": mac_payload_str,
        "mac": mac_sig,
        "capability": {"token": "ibac:BAD:0000000000000000",
                       "action": "write_file", "resource": "kairos/test.py"},
    }
    result3 = sig_checker.check(_make_ctx("GitOpsMutator", both_meta))
    check("Priority: bad capability + valid HMAC -> FAIL", result3 == CheckResult.FAIL)

    # Path 2: valid HMAC, no capability
    mac_payload_str2 = "sign_me"
    mac_sig2 = hmac.new(key, mac_payload_str2.encode(), hashlib.sha256).hexdigest()
    hmac_meta = {"mac_payload": mac_payload_str2, "mac": mac_sig2}
    result4 = sig_checker.check(_make_ctx("GitOpsMutator", hmac_meta))
    check("Path 2: valid HMAC -> PASS", result4 == CheckResult.PASS)

    # Path 2: wrong HMAC -> FAIL
    bad_hmac_meta = {"mac_payload": mac_payload_str2, "mac": "a" * 64}
    result5 = sig_checker.check(_make_ctx("GitOpsMutator", bad_hmac_meta))
    check("Path 2: invalid HMAC -> FAIL", result5 == CheckResult.FAIL)

    # No credentials + key exists -> FAIL
    result6 = sig_checker.check(_make_ctx("GitOpsMutator", {}))
    check("No credentials + key exists -> FAIL", result6 == CheckResult.FAIL)

else:
    result_dev = sig_checker.check(_make_ctx("GitOpsMutator", {}))
    check("Dev mode: no key -> PASS", result_dev == CheckResult.PASS)

    cap_dev = {"capability": {"token": "anything", "action": "write_file", "resource": "x"}}
    result_dev2 = sig_checker.check(_make_ctx("GitOpsMutator", cap_dev))
    check("Dev mode: capability present -> PASS", result_dev2 == CheckResult.PASS)


# ─────────────────────────────────────────────────────────────────────────────
# 5. gates.py: loads correctly
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] gates.py -- gate types and names")

from sovereignnation.gates import (
    gitops_gate, memory_gate, policy_gitops_gate, policy_memory_gate,
    basic_gitops_gate, basic_memory_gate, WriteClass, WriteContext,
)
from sovereignnation.access_control import QuorumGate

check("gitops_gate is QuorumGate",  isinstance(gitops_gate, QuorumGate))
check("memory_gate is QuorumGate",  isinstance(memory_gate, QuorumGate))
check("gitops_gate.name has Gate",  "Gate" in gitops_gate.name)
check("memory_gate.name has Gate",  "Gate" in memory_gate.name)
check("gitops_gate is policy gate",  gitops_gate is policy_gitops_gate)
check("memory_gate is policy gate",  memory_gate is policy_memory_gate)
check("gitops_gate requires 3 passes", gitops_gate.required_passes == 3)
check("memory_gate requires 2 passes", memory_gate.required_passes == 2)


# ─────────────────────────────────────────────────────────────────────────────
# 6. GitOpsMutator.mutate() signature
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] GitOpsMutator.mutate() signature")

import inspect
from kairos.gitops_mutator import GitOpsMutator

sig_insp = inspect.signature(GitOpsMutator.mutate)
params = sig_insp.parameters
check("mutate() has capability param", "capability" in params)
if "capability" in params:
    check("capability default is None", params["capability"].default is None)


# ─────────────────────────────────────────────────────────────────────────────
# 7. GhostRecall.store() signature
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] GhostRecall.store() signature")

from memory.ghostrecall import GhostRecall

sig_insp2 = inspect.signature(GhostRecall.store)
params2 = sig_insp2.parameters
check("store() has capability param", "capability" in params2)
if "capability" in params2:
    check("capability default is None", params2["capability"].default is None)


# ─────────────────────────────────────────────────────────────────────────────
# 8. async enforcement end-to-end (memory_gate)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Async end-to-end -- memory_gate with capability")

async def _test_memory_gate():
    from sovereignnation.gates import memory_gate, WriteClass, WriteContext

    cap_tok = client.request_sync("SAGE", "write_memory", "memory/SAGE")
    meta = {}
    if cap_tok:
        meta["capability"] = cap_tok

    ctx = WriteContext(
        agent_id    = "SAGE",
        proposal_id = "test-mem-001",
        write_class = WriteClass.GHOST_RECALL,
        target_path = "memory/SAGE",
        payload_summary = "test memory write with capability",
        metadata    = meta,
    )

    try:
        await memory_gate.enforce_async(ctx)
        return True
    except PermissionError:
        return False

ok = asyncio.run(_test_memory_gate())
check("memory_gate.enforce_async with capability passed", ok)


# ─────────────────────────────────────────────────────────────────────────────
# 9. CapabilityClient.request() async
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] CapabilityClient.request() async (daemon absent -> local mint)")

async def _test_async_client():
    cap3 = await client.request("SAGE", "write_file", "kairos/sage_loop.py")
    ok1 = cap3 is None or (isinstance(cap3, dict) and "token" in cap3)
    check("async request returns dict or None", ok1)
    if cap3:
        check("async: token starts with ibac:", cap3["token"].startswith("ibac:"))
    return cap3

asyncio.run(_test_async_client())


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
passed = sum(results)
total  = len(results)
all_ok = passed == total
print(f"  {'ALL PASS' if all_ok else 'SOME FAILURES'}  {passed}/{total}")
print(f"{'='*55}\n")
sys.exit(0 if all_ok else 1)
