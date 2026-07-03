"""
Compatibility shim for emergentintegrations.llm.chat.
Routes calls to native provider SDKs using env-var API keys.
"""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class UserMessage:
    text: str


class LlmChat:
    def __init__(self, api_key: str = "", session_id: str = "", system_message: str = ""):
        self._system_message = system_message
        self._provider = "anthropic"
        self._model = "claude-sonnet-4-5-20250929"

    def with_model(self, provider: str, model: str) -> "LlmChat":
        self._provider = provider.lower()
        self._model = model
        return self

    async def send_message(self, message: UserMessage) -> str:
        if self._provider == "anthropic":
            return await self._call_anthropic(message.text)
        if self._provider == "openai":
            return await self._call_openai(message.text)
        raise ValueError(f"Unsupported provider: {self._provider!r}")

    async def _call_anthropic(self, text: str) -> str:
        import anthropic  # already in requirements.txt
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = anthropic.AsyncAnthropic(api_key=key)
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": text}],
        }
        if self._system_message:
            kwargs["system"] = self._system_message
        response = await client.messages.create(**kwargs)
        return response.content[0].text

    async def _call_openai(self, text: str) -> str:
        from openai import AsyncOpenAI  # already in requirements.txt
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = AsyncOpenAI(api_key=key)
        messages = []
        if self._system_message:
            messages.append({"role": "system", "content": self._system_message})
        messages.append({"role": "user", "content": text})
        response = await client.chat.completions.create(
            model=self._model, messages=messages
        )
        return response.choices[0].message.content
