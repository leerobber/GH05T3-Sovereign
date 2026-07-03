"""Invoice Agent — drafts professional invoices from plain-language descriptions."""
from __future__ import annotations
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional

from .contractor_base import ContractorAgent

LOG = logging.getLogger("site_agents.invoice")


class InvoiceAgent(ContractorAgent):
    name = "invoice"
    role = "Invoice & Billing Specialist"
    expertise = (
        "Drafts professional invoices from plain-language job descriptions. "
        "Remembers your rates, payment terms, and business name. "
        "Outputs invoice-ready HTML and structured JSON."
    )
    system_prompt = """You are an Invoice Specialist for independent contractors — cleaners, handypeople, landscapers, and small service businesses.

Your job: turn a plain-language description of work done into a clean, professional invoice.

RULES:
- Extract: client name, services performed, quantities, rates, subtotal, tax (if mentioned), total
- If rates aren't mentioned, use rates from memory if available, otherwise flag as [RATE NEEDED]
- Payment terms default to NET 15 unless user specifies otherwise
- Invoice number format: INV-{YYYY}-{NNN} (auto-increment from last known invoice)
- Always output TWO sections: (1) JSON invoice data, (2) human-readable text invoice

OUTPUT FORMAT — always exactly this:
```json
{
  "invoice_number": "INV-2026-001",
  "date": "2026-06-07",
  "due_date": "2026-06-22",
  "client": {"name": "", "email": "", "address": ""},
  "business": {"name": "", "phone": "", "email": ""},
  "line_items": [{"description": "", "qty": 1, "rate": 0.00, "amount": 0.00}],
  "subtotal": 0.00,
  "tax_rate": 0.0,
  "tax_amount": 0.00,
  "total": 0.00,
  "payment_terms": "NET 15",
  "notes": ""
}
```

TEXT INVOICE:
[clean plain-text version ready to paste into an email or print]

Never make up rates. Never make up client details. If something is missing, mark it [NEEDS INFO]."""

    async def draft(
        self,
        description: str,
        user_id: str = "",
        session_id: str = "",
        client_name: str = "",
        client_email: str = "",
    ) -> dict:
        """
        Draft an invoice from a plain-language description.
        description: e.g. "Cleaned the Johnson house on Monday, 3 hours, normally charge $45/hr"
        """
        task = description
        if client_name:
            task = f"Client: {client_name}" + (f" <{client_email}>" if client_email else "") + f"\n\n{description}"

        result = await self.think_with_memory(task, user_id, session_id)

        # Parse the JSON block out of the response
        invoice_data = _extract_json(result)
        text_invoice = _extract_text_invoice(result)

        # Store invoice reference to memory
        if invoice_data and user_id:
            summary = (f"Invoice {invoice_data.get('invoice_number','?')} "
                       f"to {invoice_data.get('client',{}).get('name','?')} "
                       f"for ${invoice_data.get('total',0):.2f}")
            asyncio.create_task(self.remember_exchange(
                user_id, description, summary,
                session_id=session_id, importance=0.75,
            ))

        task_id = self._mem.log_task(self.name, "draft_invoice", description, result)
        return {
            "agent": self.name,
            "task_id": task_id,
            "invoice_json": invoice_data,
            "invoice_text": text_invoice,
            "raw": result,
        }

    async def set_rate(self, user_id: str, service: str, rate: float, unit: str = "hr") -> dict:
        """Store a rate to the user's memory. e.g. service='cleaning', rate=45, unit='hr'"""
        await self.save_user_setting(user_id, f"rate_{service}", f"${rate}/{unit}")
        return {"ok": True, "stored": f"{service}: ${rate}/{unit}"}

    async def set_business(self, user_id: str, name: str, phone: str = "", email: str = "") -> dict:
        """Store business identity to core memory."""
        await self.save_user_setting(user_id, "business_name", name)
        if phone:
            await self.save_user_setting(user_id, "business_phone", phone)
        if email:
            await self.save_user_setting(user_id, "business_email", email)
        return {"ok": True, "stored": {"name": name, "phone": phone, "email": email}}


import asyncio


def _extract_json(text: str) -> dict:
    try:
        m = re.search(r"```json\s*([\s\S]*?)```", text)
        if m:
            return json.loads(m.group(1).strip())
        m = re.search(r"\{[\s\S]*\"invoice_number\"[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {}


def _extract_text_invoice(text: str) -> str:
    marker = "TEXT INVOICE:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    # If no marker, strip the JSON block and return the rest
    cleaned = re.sub(r"```json[\s\S]*?```", "", text).strip()
    return cleaned
