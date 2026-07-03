# providers.py — uniform "agent" seat for the mesh, so the proposer/second seat
# can be Gemini (vision), Groq (fast, free tier works), or local Ollama (free,
# offline). Claude holds the reviewer seat. One interface: .name + .ask().
import json
import urllib.request

from _common import get_key

DEFAULTS = {
    "gemini":     "gemini-2.0-flash",
    "groq":       "llama-3.3-70b-versatile",
    "ollama":     "llama3.2",
    "mistral":    "mistral-large-latest",
    "openrouter": "openai/gpt-4o-mini",
    "nvidia":     "meta/llama-3.1-70b-instruct",   # verified working
}
VISION = {"gemini"}  # which providers can actually see images (native path)

# OpenAI-compatible base URLs for providers that speak the OpenAI schema.
_OPENAI_COMPAT = {
    "mistral":    "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia":     "https://integrate.api.nvidia.com/v1",
}
_KEY_ENV = {
    "mistral":    "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia":     "NVIDIA_API_KEY",
}


class Agent:
    name = "AGENT"
    supports_vision = False

    def ask(self, system: str, user: str, images: list[str] | None = None) -> str:
        raise NotImplementedError


class GeminiAgent(Agent):
    name = "GEMINI"
    supports_vision = True

    def __init__(self, model: str | None = None):
        from gemini_bridge import GeminiBridge
        self.bridge = GeminiBridge(model=model or DEFAULTS["gemini"])

    def ask(self, system, user, images=None):
        return self.bridge.ask(user, images=images, system=system)


class GroqAgent(Agent):
    name = "GROQ"
    supports_vision = False

    def __init__(self, model: str | None = None):
        from groq import Groq
        self.model = model or DEFAULTS["groq"]
        self.client = Groq(api_key=get_key("GROQ_API_KEY"))

    def ask(self, system, user, images=None):
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=4096,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()


class OllamaAgent(Agent):
    name = "OLLAMA"
    supports_vision = False

    def __init__(self, model: str | None = None, host: str = "http://localhost:11434"):
        self.model = model or DEFAULTS["ollama"]
        self.host = host

    def ask(self, system, user, images=None):
        payload = {
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        return (data.get("message", {}).get("content", "") or "").strip()


class ClaudeAgent(Agent):
    name = "CLAUDE"
    supports_vision = True

    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        from claude_client import ClaudeClient
        self.client = ClaudeClient(model=model, max_tokens=max_tokens) if model else ClaudeClient(max_tokens=max_tokens)

    def ask(self, system, user, images=None):
        return self.client.ask(system, user)


class _OpenAICompatAgent(Agent):
    """Any OpenAI-schema endpoint (Mistral, OpenRouter). Subclasses set `key`."""
    key = ""  # provider key into _OPENAI_COMPAT / _KEY_ENV / DEFAULTS

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.model = model or DEFAULTS[self.key]
        self.client = OpenAI(api_key=get_key(_KEY_ENV[self.key]), base_url=_OPENAI_COMPAT[self.key])

    def ask(self, system, user, images=None):
        import time
        last_err = None
        for attempt in range(3):           # 3 retries on connection errors
            try:
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=4096,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                if "Connection" in type(e).__name__ or "connection" in str(e).lower():
                    time.sleep(3 * (attempt + 1))   # 3s, 6s, 9s back-off
                    continue
                raise   # non-connection errors: fail immediately
        raise last_err


class MistralAgent(_OpenAICompatAgent):
    name = "MISTRAL"
    key = "mistral"


class OpenRouterAgent(_OpenAICompatAgent):
    key = "openrouter"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        self.name = "OR:" + self.model.split("/")[-1][:14]


class NvidiaAgent(_OpenAICompatAgent):
    key = "nvidia"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        self.name = "NV:" + self.model.split("/")[-1][:16]


_REGISTRY = {"gemini": GeminiAgent, "groq": GroqAgent, "ollama": OllamaAgent,
             "claude": ClaudeAgent, "mistral": MistralAgent,
             "openrouter": OpenRouterAgent, "nvidia": NvidiaAgent}


def make_agent(provider: str, model: str | None = None) -> Agent:
    provider = provider.lower()
    if provider not in _REGISTRY:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[provider](model=model)
