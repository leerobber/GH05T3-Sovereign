"""Quote Agent — generates service quotes from job descriptions using stored rates."""
from __future__ import annotations
import asyncio
import json
import logging
import re

from .contractor_base import ContractorAgent

LOG = logging.getLogger("site_agents.quote")


class QuoteAgent(ContractorAgent):
    name = "quote"
    role = "Quote & Estimation Specialist"
    expertise = (
        "Generates professional service quotes from job descriptions. "
        "Remembers your standard rates and packages. "
        "Outputs client-ready quotes with line items, total, and validity period."
    )
    system_prompt = """You are a Quote Specialist for independent contractors and small service businesses.

Your job: turn a job description into a clear, professional service quote a client would accept.

RULES:
- Be specific — break work into line items, not one lump sum
- If user rates are in memory, use them. If not, flag [RATE NEEDED]
- Quotes are valid for 30 days by default
- Include a clear "What's included" and "What's NOT included" section
- Keep language warm but professional — not corporate, not casual
- If the job description is vague, list your assumptions explicitly

OUTPUT FORMAT — always exactly this:
```json
{
  "quote_number": "QTE-2026-001",
  "date": "",
  "valid_until": "",
  "client": {"name": "", "email": ""},
  "job_summary": "",
  "line_items": [{"service": "", "description": "", "qty": 1, "rate": 0.00, "amount": 0.00}],
  "subtotal": 0.00,
  "tax_rate": 0.0,
  "total": 0.00,
  "included": [],
  "not_included": [],
  "assumptions": [],
  "terms": "Quote valid 30 days. 50% deposit required to book."
}
```

CLIENT-FACING TEXT:
[clean professional quote text to paste into email or print]

Be honest about what you don't know. Never invent specifics."""

    async def draft(
        self,
        job_description: str,
        user_id: str = "",
        session_id: str = "",
        client_name: str = "",
        client_email: str = "",
    ) -> dict:
        """
        Generate a quote from a job description.
        job_description: e.g. "2-story house, deep clean before move-out, ~2500 sqft, appliances included"
        """
        task = job_description
        if client_name:
            task = f"Client: {client_name}" + (f" <{client_email}>" if client_email else "") + f"\n\n{job_description}"

        # Pull past quotes for this client from memory for consistency
        context = ""
        if user_id:
            rag_hits = await self.recall_context(f"quote {client_name or ''} {job_description[:60]}")
            context = rag_hits

        result = await self.think_with_memory(task, user_id, session_id, extra_context=context)

        quote_data = _extract_json(result)
        quote_text = _extract_section(result, "CLIENT-FACING TEXT:")

        if quote_data and user_id:
            summary = (f"Quote {quote_data.get('quote_number','?')} "
                       f"to {client_name or '?'} "
                       f"for ${quote_data.get('total',0):.2f}: {quote_data.get('job_summary','')[:60]}")
            asyncio.create_task(self.remember_exchange(
                user_id, job_description, summary,
                session_id=session_id, importance=0.7,
            ))

        task_id = self._mem.log_task(self.name, "draft_quote", job_description, result)
        return {
            "agent": self.name,
            "task_id": task_id,
            "quote_json": quote_data,
            "quote_text": quote_text,
            "raw": result,
        }

    async def estimate_range(
        self,
        job_type: str,
        user_id: str = "",
        session_id: str = "",
    ) -> dict:
        """
        Quick ballpark estimate for a job type.
        e.g. "how much do I normally charge for a 3-bed house clean?"
        """
        task = f"Based on my rates and past quotes, give me a quick price range for: {job_type}"
        result = await self.think_with_memory(task, user_id, session_id)
        task_id = self._mem.log_task(self.name, "estimate_range", job_type, result)
        return {"agent": self.name, "task_id": task_id, "estimate": result}


def _extract_json(text: str) -> dict:
    try:
        m = re.search(r"```json\s*([\s\S]*?)```", text)
        if m:
            return json.loads(m.group(1).strip())
        m = re.search(r"\{[\s\S]*\"quote_number\"[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {}


def _extract_section(text: str, marker: str) -> str:
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return re.sub(r"```json[\s\S]*?```", "", text).strip()
