from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_RE = re.compile(
    r"""
    (?P<STRING>"[^"]*"|'[^']*')         |
    (?P<NUMBER>-?\d+(?:\.\d+)?)         |
    (?P<BOOL>true|false)                |
    (?P<PIPE_OP>\|>)                    |
    (?P<ARROW>->|→)                     |
    (?P<COLON>:)                        |
    (?P<LBRACE>\{)                      |
    (?P<RBRACE>\})                      |
    (?P<LBRACKET>\[)                    |
    (?P<RBRACKET>\])                    |
    (?P<LPAREN>\()                      |
    (?P<RPAREN>\))                      |
    (?P<COMMA>,)                        |
    (?P<DOT>\.)                         |
    (?P<EQ>=)                           |
    (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)  |
    (?P<COMMENT>\#[^\n]*)               |
    (?P<WS>\s+)                         |
    (?P<OTHER>.)
    """,
    re.VERBOSE,
)

KEYWORDS = {
    "agent", "let", "async", "await", "think", "emit", "on",
    "if", "else", "for", "in",
}

_BUILTIN_NS  = {"llm", "memory", "kairos", "Archive"}
_KEYWORD_FNS = {"evolve", "print", "reply_from"}


@dataclass
class Token:
    kind: str
    value: str
    pos: int


def lex(src: str) -> list[Token]:
    toks = []
    for m in TOKEN_RE.finditer(src):
        kind = m.lastgroup
        if kind in ("WS", "COMMENT"):
            continue
        if kind == "IDENT" and m.group() in KEYWORDS:
            kind = m.group().upper()
        toks.append(Token(kind, m.group(), m.start()))
    return toks
