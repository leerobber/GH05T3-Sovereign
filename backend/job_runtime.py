"""Autotelic job runtime for GH05T3.

Creates policy-classified job records and executes allowed GhostScript jobs.
The persistence layer is optional so pure policy/runtime behavior is easy to
test and chat tools can use the same entrypoint.
"""
from __future__ import annotations

import inspect
import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable, Any

from autonomy_policy import classify_action


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job_document(
    *,
    title: str,
    description: str,
    source: str,
    paths: list[str] | None = None,
    goal_id: str | None = None,
    emergency: bool = False,
) -> dict:
    policy = classify_action(description, paths=paths or [], emergency=emergency)
    now = _now_iso()
    return {
        "_id": str(uuid.uuid4()),
        "title": title[:160],
        "description": description[:1000],
        "kind": "ghostscript",
        "source": source,
        "paths": paths or [],
        "goal_id": goal_id,
        "status": "queued",
        "policy": policy,
        "events": [{
            "ts": now,
            "type": "created",
            "message": policy["reason"],
            "policy_level": policy["level"],
        }],
        "result": None,
        "created_at": now,
        "updated_at": now,
    }


async def _maybe_store(db, job: dict) -> None:
    if db is not None:
        await db.jobs.update_one({"_id": job["_id"]}, {"$set": job}, upsert=True)


async def _run_runner(runner: Callable[[str], Any], source: str) -> Any:
    result = runner(source)
    if inspect.isawaitable(result):
        return await result
    return result


async def run_ghostscript_job(
    *,
    title: str,
    description: str,
    source: str,
    runner: Callable[[str], Awaitable[dict] | dict],
    paths: list[str] | None = None,
    goal_id: str | None = None,
    emergency: bool = False,
    db=None,
) -> dict:
    job = create_job_document(
        title=title,
        description=description,
        source=source,
        paths=paths,
        goal_id=goal_id,
        emergency=emergency,
    )
    await _maybe_store(db, job)

    if not job["policy"]["allowed"]:
        job["status"] = "blocked"
        job["updated_at"] = _now_iso()
        job["events"].append({
            "ts": job["updated_at"],
            "type": "blocked",
            "message": job["policy"]["reason"],
        })
        await _maybe_store(db, job)
        return job

    job["status"] = "running"
    job["updated_at"] = _now_iso()
    job["events"].append({
        "ts": job["updated_at"],
        "type": "started",
        "message": f"Running under policy level: {job['policy']['level']}",
    })
    await _maybe_store(db, job)

    try:
        job["result"] = await _run_runner(runner, source)
        job["status"] = "complete" if job["result"].get("ok", False) else "failed"
        message = "GhostScript job completed" if job["status"] == "complete" else "GhostScript job failed"
    except Exception as exc:  # noqa: BLE001
        job["result"] = {"ok": False, "error": str(exc)}
        job["status"] = "failed"
        message = str(exc)

    job["updated_at"] = _now_iso()
    job["events"].append({
        "ts": job["updated_at"],
        "type": job["status"],
        "message": message,
    })
    await _maybe_store(db, job)
    return job
