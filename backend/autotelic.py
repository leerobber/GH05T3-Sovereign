"""
AutotelicEngine — CRUD + lifecycle management for autotelic goals.
Goals are stored in MongoDB inside system_state.autotelic_goals.
Each goal has: id, title, detail, progress, status, priority, category, timestamps.
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Optional

VALID_STATUSES  = {"active", "complete", "paused", "abandoned"}
VALID_PRIORITIES = {0, 1, 2, 3}          # 0=critical, 1=high, 2=medium, 3=low
VALID_CATEGORIES = {
    "training", "security", "memory", "integration", "meta", "general",
}
PRIORITY_LABELS = {0: "critical", 1: "high", 2: "medium", 3: "low"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(g: dict) -> dict:
    """Backfill any fields missing on legacy goals that only had title/detail/progress."""
    return {
        "id":         g.get("id") or str(uuid.uuid4()),
        "title":      g.get("title", "Untitled"),
        "detail":     g.get("detail", ""),
        "progress":   min(1.0, max(0.0, float(g.get("progress", 0.0)))),
        "status":     g.get("status", "active") if g.get("status") in VALID_STATUSES else "active",
        "priority":   int(g.get("priority", 2)),
        "category":   g.get("category", "general") if g.get("category") in VALID_CATEGORIES else "general",
        "created_at": g.get("created_at", _now_iso()),
        "updated_at": g.get("updated_at", _now_iso()),
    }


async def _load(db) -> list[dict]:
    doc = await db.system_state.find_one({"_id": "singleton"}, {"autotelic_goals": 1})
    raw = (doc or {}).get("autotelic_goals", [])
    return [_normalize(g) for g in raw]


async def _save(db, goals: list[dict]):
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"autotelic_goals": goals}},
    )


class AutotelicEngine:
    def __init__(self, db, ws):
        self.db = db
        self.ws = ws

    async def list_goals(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[dict]:
        goals = await _load(self.db)
        if status:
            goals = [g for g in goals if g["status"] == status]
        if category:
            goals = [g for g in goals if g["category"] == category]
        goals.sort(key=lambda g: (g["priority"], g.get("created_at", "")))
        return goals

    async def get_goal(self, goal_id: str) -> Optional[dict]:
        goals = await _load(self.db)
        return next((g for g in goals if g["id"] == goal_id), None)

    async def create_goal(
        self,
        title: str,
        detail: str = "",
        priority: int = 2,
        category: str = "general",
    ) -> dict:
        goal = _normalize({
            "id":       str(uuid.uuid4()),
            "title":    title[:120],
            "detail":   detail[:300],
            "progress": 0.0,
            "status":   "active",
            "priority": max(0, min(3, int(priority))),
            "category": category if category in VALID_CATEGORIES else "general",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        })
        await self.db.system_state.update_one(
            {"_id": "singleton"},
            {"$push": {"autotelic_goals": {"$each": [goal], "$slice": -80}}},
        )
        await self.ws.broadcast("goal_added", goal)
        return goal

    async def update_goal(self, goal_id: str, **fields) -> Optional[dict]:
        goals = await _load(self.db)
        updated = None
        for g in goals:
            if g["id"] == goal_id:
                allowed = {"title", "detail", "progress", "status", "priority", "category"}
                for k, v in fields.items():
                    if k not in allowed or v is None:
                        continue
                    if k == "progress":
                        g[k] = min(1.0, max(0.0, float(v)))
                    elif k == "status" and v in VALID_STATUSES:
                        g[k] = v
                    elif k == "priority" and int(v) in VALID_PRIORITIES:
                        g[k] = int(v)
                    elif k == "category" and v in VALID_CATEGORIES:
                        g[k] = v
                    elif k in ("title", "detail"):
                        g[k] = str(v)[:120 if k == "title" else 300]
                if g["progress"] >= 1.0 and g["status"] == "active":
                    g["status"] = "complete"
                g["updated_at"] = _now_iso()
                updated = g
                break
        if updated is None:
            return None
        await _save(self.db, goals)
        await self.ws.broadcast("goal_updated", updated)
        return updated

    async def delete_goal(self, goal_id: str) -> bool:
        goals = await _load(self.db)
        new_goals = [g for g in goals if g["id"] != goal_id]
        if len(new_goals) == len(goals):
            return False
        await _save(self.db, new_goals)
        await self.ws.broadcast("goal_deleted", {"id": goal_id})
        return True

    async def complete_goal(self, goal_id: str) -> Optional[dict]:
        return await self.update_goal(goal_id, status="complete", progress=1.0)

    async def suggest_goals(self, state_snapshot: dict, count: int = 3) -> list[dict]:
        """Generate goal suggestions based on current system state."""
        suggestions: list[dict] = []

        kairos = state_snapshot.get("kairos", {})
        last_score = kairos.get("last_score", 0.5)
        if last_score < 0.70:
            suggestions.append({
                "title": "Raise KAIROS cycle quality score",
                "detail": f"Last score {last_score:.2f} is below 0.70 — deepen proposal reasoning and adversarial probing",
                "priority": 0, "category": "training",
            })
        elif last_score < 0.85:
            suggestions.append({
                "title": "Push KAIROS toward elite threshold",
                "detail": f"Score {last_score:.2f} — need 0.85+ for elite promotion",
                "priority": 1, "category": "training",
            })

        seance = state_snapshot.get("seance", [])
        if seance:
            last = seance[-1]
            suggestions.append({
                "title": f"Integrate lesson: {last['domain']}",
                "detail": last.get("lesson", "")[:180],
                "priority": 1, "category": "training",
            })

        mem = state_snapshot.get("memory_palace", {})
        total_mem = mem.get("total", 0)
        if total_mem > 80:
            suggestions.append({
                "title": "Prune low-confidence memories",
                "detail": f"{total_mem} entries — archive anything below 0.4 confidence to cold tier",
                "priority": 2, "category": "memory",
            })

        kairos_cycles = kairos.get("live_cycles", 0)
        if kairos_cycles > 0 and kairos.get("meta_rewrites", 0) == 0:
            suggestions.append({
                "title": "Trigger first meta-rewrite",
                "detail": f"{kairos_cycles} live cycles with no meta-rewrite yet — extract architectural rules",
                "priority": 1, "category": "meta",
            })

        suggestions.append({
            "title": "GH05T3 defines her own agenda",
            "detail": "Month-3 stretch: autonomous goal proposal without human prompting",
            "priority": 3, "category": "meta",
        })

        existing = await _load(self.db)
        existing_titles = {g["title"].lower() for g in existing}
        unique = [s for s in suggestions if s["title"].lower() not in existing_titles]
        return unique[:count]
