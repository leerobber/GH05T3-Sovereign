# claude_client.py — thin Anthropic wrapper shared by the bridge tools.
from _common import anthropic_key

CLAUDE_MODEL = "claude-opus-4-8"


class ClaudeClient:
    def __init__(self, model: str = CLAUDE_MODEL, max_tokens: int = 2048):
        import anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=anthropic_key())

    def ask(self, system: str, user: str, max_tokens: int | None = None) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
