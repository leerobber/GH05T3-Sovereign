#!/usr/bin/env python
# gemini_bridge.py â€” Phase 1 of the Claude <-> Gemini bridge.
#
# An on-demand tool so Claude Code (or any caller) can hand work to Gemini:
# text, images, or file attachments in, Gemini's answer out.
#
# CLI:
#   python bridge/gemini_bridge.py ask "what's wrong with this layout?" --image shot.png
#   python bridge/gemini_bridge.py ask "review for bugs" --file app.py --file utils.py
#   python bridge/gemini_bridge.py ask "summarize" --system "You are terse." --json
#   echo "long prompt on stdin" | python bridge/gemini_bridge.py ask -
#
# Importable:
#   from bridge.gemini_bridge import GeminiBridge
#   g = GeminiBridge(); print(g.ask("hi", images=["shot.png"]))
import argparse
from mimetypes import mimetypes
import sys
from pathlib import Path

from _common import gemini_key

DEFAULT_MODEL = "gemini-2.0-flash"  # vision-capable, has a free tier


class GeminiBridge:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        from google import genai  # imported lazily so --help works without the SDK
        self.model = model
        self._genai = genai
        self.client = genai.Client(api_key=api_key or gemini_key())

    def _part_from_path(self, path: str):
        from google.genai import types
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"attachment not found: {path}")
        mime = mimetypes.guess_type(p.name)[0]
        if mime is None:
            # treat unknown types as text if decodable, else octet-stream
            try:
                return p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                mime = "application/octet-stream"
        return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime)

    def ask(
        self,
        prompt: str,
        images: list[str] | None = None,
        files: list[str] | None = None,
        system: str | None = None,
        as_json: bool = False,
    ) -> str:
        from google.genai import types
        contents: list = []
        for path in (images or []) + (files or []):
            contents.append(self._part_from_path(path))
        contents.append(prompt)

        config = None
        kwargs = {}
        if system:
            kwargs["system_instruction"] = system
        if as_json:
            kwargs["response_mime_type"] = "application/json"
        if kwargs:
            config = types.GenerateContentConfig(**kwargs)

        resp = self.client.models.generate_content(
            model=self.model, contents=contents,
            **({"config": config} if config else {}),
        )
        return (resp.text or "").strip()


def _read_prompt(value: str) -> str:
    return sys.stdin.read() if value == "-" else value


def main():
    ap = argparse.ArgumentParser(description="Ask Gemini (text + vision) from the CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="Send a prompt to Gemini.")
    a.add_argument("prompt", help="Prompt text, or '-' to read from stdin.")
    a.add_argument("--image", action="append", default=[], metavar="PATH",
                   help="Image attachment (repeatable).")
    a.add_argument("--file", action="append", default=[], metavar="PATH",
                   help="File attachment: code, pdf, etc. (repeatable).")
    a.add_argument("--system", help="System instruction.")
    a.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default {DEFAULT_MODEL}).")
    a.add_argument("--json", action="store_true", help="Request JSON output.")

    args = ap.parse_args()
    if args.cmd == "ask":
        bridge = GeminiBridge(model=args.model)
        try:
            out = bridge.ask(
                _read_prompt(args.prompt),
                images=args.image, files=args.file,
                system=args.system, as_json=args.json,
            )
        except Exception as e:  # surface API/billing errors cleanly to stdout
            print(f"[gemini error] {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(2)
        print(out)


if __name__ == "__main__":
    main()
