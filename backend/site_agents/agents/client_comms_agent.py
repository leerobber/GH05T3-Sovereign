"""Client Comms Agent — drafts professional client messages in the user's voice."""
from __future__ import annotations
import asyncio
import logging

from .contractor_base import ContractorAgent

LOG = logging.getLogger("site_agents.client_comms")


class ClientCommsAgent(ContractorAgent):
    name = "client_comms"
    role = "Client Communications Specialist"
    expertise = (
        "Drafts professional replies to client messages in your voice. "
        "Handles scheduling confirmations, complaints, follow-ups, and payment reminders. "
        "Learns your tone from every conversation."
    )
    system_prompt = """You are a Client Communications Specialist writing on behalf of an independent contractor.

Your job: draft a professional, warm reply to a client message. Write in the contractor's voice — not corporate, not too casual. Like a real person who runs a small business and cares about their clients.

TONE RULES:
- Direct and confident — don't hedge or over-apologize
- Warm but brief — clients are busy, respect their time
- Never use: "I hope this email finds you well", "Please do not hesitate", "At your earliest convenience"
- Address the client by first name
- Sign off as the contractor (use name from memory if available, otherwise [YOUR NAME])

TYPES OF MESSAGES YOU HANDLE:
- Scheduling confirmations and reminders
- Quote follow-ups ("did you get my quote?")
- Complaint responses (stay calm, offer resolution, don't grovel)
- Payment reminders (firm but not aggressive)
- Service completion check-ins
- Rescheduling requests
- New inquiry replies

OUTPUT: Just the message text. No subject line unless asked. No explanation of what you're doing."""

    async def draft_reply(
        self,
        client_message: str,
        user_id: str = "",
        session_id: str = "",
        client_name: str = "",
        message_type: str = "general",
    ) -> dict:
        """
        Draft a reply to a client message.
        client_message: the message received from the client
        message_type: 'complaint' | 'payment_reminder' | 'scheduling' | 'quote_followup' | 'general'
        """
        task = f"Client message:\n\"{client_message}\"\n\nDraft a reply."
        if client_name:
            task = f"Client name: {client_name}\n\n" + task
        if message_type != "general":
            task += f"\n\nMessage type: {message_type}"

        result = await self.think_with_memory(task, user_id, session_id)

        if user_id and client_name:
            summary = f"Replied to {client_name}: {result[:120]}"
            asyncio.create_task(self.remember_exchange(
                user_id, client_message, summary,
                session_id=session_id, importance=0.55,
            ))

        task_id = self._mem.log_task(self.name, "draft_reply", client_message, result)
        return {
            "agent": self.name,
            "task_id": task_id,
            "draft": result,
            "client": client_name,
            "message_type": message_type,
        }

    async def payment_reminder(
        self,
        client_name: str,
        amount: float,
        days_overdue: int,
        invoice_number: str = "",
        user_id: str = "",
        session_id: str = "",
    ) -> dict:
        """Draft a payment reminder — tone scales with days overdue."""
        if days_overdue <= 7:
            tone = "friendly nudge — keep it light"
        elif days_overdue <= 21:
            tone = "firm but professional — this is overdue"
        else:
            tone = "serious — this is significantly overdue, next step is collections"

        task = (
            f"Write a payment reminder for {client_name}.\n"
            f"Amount owed: ${amount:.2f}\n"
            f"Days overdue: {days_overdue}\n"
            f"{'Invoice: ' + invoice_number if invoice_number else ''}\n"
            f"Tone: {tone}"
        )
        result = await self.think_with_memory(task, user_id, session_id)
        task_id = self._mem.log_task(self.name, "payment_reminder",
                                     f"{client_name} ${amount} {days_overdue}d", result)
        return {
            "agent": self.name,
            "task_id": task_id,
            "draft": result,
            "client": client_name,
            "amount": amount,
            "days_overdue": days_overdue,
        }

    async def complaint_response(
        self,
        complaint: str,
        client_name: str = "",
        user_id: str = "",
        session_id: str = "",
    ) -> dict:
        """Draft a measured, professional response to a client complaint."""
        task = (
            f"Client complaint from {client_name or 'client'}:\n\"{complaint}\"\n\n"
            "Draft a professional response. Stay calm. Acknowledge the concern. "
            "Offer a specific resolution if appropriate. Don't grovel. Don't get defensive."
        )
        result = await self.think_with_memory(task, user_id, session_id)
        task_id = self._mem.log_task(self.name, "complaint_response", complaint, result)
        return {
            "agent": self.name,
            "task_id": task_id,
            "draft": result,
            "client": client_name,
        }

    async def new_inquiry_reply(
        self,
        inquiry: str,
        client_name: str = "",
        user_id: str = "",
        session_id: str = "",
    ) -> dict:
        """Reply to a new client inquiry — warm, move toward booking."""
        task = (
            f"New client inquiry from {client_name or 'prospective client'}:\n\"{inquiry}\"\n\n"
            "Reply warmly. Express interest. Ask one clarifying question to understand the job. "
            "End with a soft call to action toward getting a quote or booking a call."
        )
        result = await self.think_with_memory(task, user_id, session_id)

        if user_id and client_name:
            asyncio.create_task(self.remember_exchange(
                user_id, inquiry, f"New inquiry from {client_name}: {inquiry[:80]}",
                session_id=session_id, importance=0.65,
            ))

        task_id = self._mem.log_task(self.name, "inquiry_reply", inquiry, result)
        return {
            "agent": self.name,
            "task_id": task_id,
            "draft": result,
            "client": client_name,
        }
