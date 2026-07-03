"""Schedule Agent — natural language → calendar events with conflict detection."""
from __future__ import annotations
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from .contractor_base import ContractorAgent

LOG = logging.getLogger("site_agents.schedule")


class ScheduleAgent(ContractorAgent):
    name = "schedule"
    role = "Scheduling & Calendar Specialist"
    expertise = (
        "Books jobs and appointments from natural language. "
        "Remembers your working hours, default job durations, and booked calendar. "
        "Outputs iCal events and plain-English confirmations."
    )
    system_prompt = """You are a Scheduling Specialist for independent contractors.

Your job: turn natural language scheduling requests into structured calendar events.

RULES:
- Extract: date, start time, end time (or duration), client name, job type, location
- Use working hours from memory if available (default: Mon–Fri 8am–6pm, Sat 9am–2pm)
- Default job durations from memory if available, otherwise:
  - Cleaning (standard): 3 hours
  - Cleaning (deep): 5 hours
  - Landscaping: 4 hours
  - Handyman: 2 hours
  - General service: 2 hours
- Flag conflicts if existing events overlap (passed in context)
- TODAY's date will be provided in the task — use it for relative dates ("next Tuesday", "tomorrow")

OUTPUT FORMAT — always exactly this:
```json
{
  "event_title": "",
  "client": "",
  "job_type": "",
  "date": "YYYY-MM-DD",
  "start_time": "HH:MM",
  "end_time": "HH:MM",
  "duration_hours": 0.0,
  "location": "",
  "notes": "",
  "conflict": false,
  "conflict_reason": "",
  "ical": "BEGIN:VCALENDAR\\nVERSION:2.0\\nBEGIN:VEVENT\\n..."
}
```

CONFIRMATION TEXT:
[one or two plain-English sentences confirming what was booked, or flagging the conflict]

Be specific about dates — never say "Tuesday" without also saying the full date."""

    async def book(
        self,
        request: str,
        user_id: str = "",
        session_id: str = "",
        existing_events: list[dict] | None = None,
        today: str | None = None,
    ) -> dict:
        """
        Book a job from a natural language request.
        request: e.g. "Schedule a deep clean at the Martinez house next Thursday at 9am"
        existing_events: list of already-booked events to check for conflicts
        today: ISO date string — if not provided, uses current date
        """
        today_str = today or datetime.now().strftime("%Y-%m-%d")
        task = f"TODAY: {today_str}\n\nScheduling request: {request}"

        if existing_events:
            booked = "\n".join(
                f"- {e.get('date','')} {e.get('start_time','')}–{e.get('end_time','')}: "
                f"{e.get('event_title', e.get('client','?'))}"
                for e in existing_events[:20]
            )
            task += f"\n\nEXISTING BOOKINGS (check for conflicts):\n{booked}"

        result = await self.think_with_memory(task, user_id, session_id)

        event_data = _extract_json(result)
        confirmation = _extract_section(result, "CONFIRMATION TEXT:")

        if event_data and user_id and not event_data.get("conflict"):
            summary = (f"Booked: {event_data.get('event_title','')} "
                       f"on {event_data.get('date','')} "
                       f"{event_data.get('start_time','')}–{event_data.get('end_time','')}")
            asyncio.create_task(self.remember_exchange(
                user_id, request, summary,
                session_id=session_id, importance=0.8,
            ))

        task_id = self._mem.log_task(self.name, "book_job", request, result)
        return {
            "agent": self.name,
            "task_id": task_id,
            "event": event_data,
            "confirmation": confirmation,
            "conflict": event_data.get("conflict", False),
            "ical": event_data.get("ical", ""),
            "raw": result,
        }

    async def list_week(
        self,
        user_id: str = "",
        session_id: str = "",
        week_start: str | None = None,
    ) -> dict:
        """
        Ask the agent to summarize the user's booked week from memory.
        Reads recalled scheduling events and produces a plain-English week view.
        """
        today = week_start or datetime.now().strftime("%Y-%m-%d")
        task = (
            f"TODAY: {today}\n\n"
            "Based on my booked jobs in memory, give me a plain-English summary of "
            "this week's schedule. List each job with date, time, client, and location. "
            "If nothing is booked, say so."
        )
        result = await self.think_with_memory(task, user_id, session_id)
        task_id = self._mem.log_task(self.name, "list_week", today, result)
        return {"agent": self.name, "task_id": task_id, "schedule_summary": result}

    async def reschedule(
        self,
        original_booking: str,
        new_time_request: str,
        user_id: str = "",
        session_id: str = "",
        client_name: str = "",
    ) -> dict:
        """
        Reschedule an existing booking.
        original_booking: description of what was booked
        new_time_request: e.g. "move to Friday at 10am instead"
        """
        today = datetime.now().strftime("%Y-%m-%d")
        task = (
            f"TODAY: {today}\n\n"
            f"Rescheduling request for {client_name or 'client'}.\n"
            f"Original booking: {original_booking}\n"
            f"New time requested: {new_time_request}\n\n"
            "Generate the updated calendar event and a brief confirmation message."
        )
        result = await self.think_with_memory(task, user_id, session_id)
        event_data = _extract_json(result)
        confirmation = _extract_section(result, "CONFIRMATION TEXT:")

        if user_id:
            asyncio.create_task(self.remember_exchange(
                user_id,
                f"Rescheduled {original_booking} → {new_time_request}",
                result[:200],
                session_id=session_id, importance=0.75,
            ))

        task_id = self._mem.log_task(self.name, "reschedule",
                                     f"{original_booking} → {new_time_request}", result)
        return {
            "agent": self.name,
            "task_id": task_id,
            "event": event_data,
            "confirmation": confirmation,
            "client": client_name,
        }

    async def set_working_hours(
        self, user_id: str, hours: str
    ) -> dict:
        """Store working hours to user memory. e.g. hours='Mon-Fri 7am-5pm, Sat 9am-1pm'"""
        await self.save_user_setting(user_id, "working_hours", hours, importance=0.9)
        return {"ok": True, "stored": f"working_hours: {hours}"}

    async def set_job_duration(
        self, user_id: str, job_type: str, hours: float
    ) -> dict:
        """Store default duration for a job type. e.g. job_type='standard_clean', hours=3.0"""
        await self.save_user_setting(user_id, f"duration_{job_type}", f"{hours}h", importance=0.85)
        return {"ok": True, "stored": f"{job_type}: {hours}h"}


def _extract_json(text: str) -> dict:
    try:
        m = re.search(r"```json\s*([\s\S]*?)```", text)
        if m:
            return json.loads(m.group(1).strip())
        m = re.search(r"\{[\s\S]*\"event_title\"[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {}


def _extract_section(text: str, marker: str) -> str:
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return re.sub(r"```json[\s\S]*?```", "", text).strip()
