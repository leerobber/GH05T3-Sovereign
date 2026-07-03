"""GhostScript — AI/ML orchestration language for GH05T3.

Grammar:
    program     ::= stmt*
    stmt        ::= let | if | for | agent | async | await | think | emit | on | expr_stmt
    let_stmt    ::= "let" IDENT "=" expr
    if_stmt     ::= "if" "(" expr ")" block ("else" block)?
    for_stmt    ::= "for" IDENT "in" expr block
    agent_stmt  ::= "agent" IDENT "{" stmt* "}"
    async_stmt  ::= "async" block
    await_stmt  ::= "await" expr
    think_stmt  ::= "think" ":" STRING
    emit_stmt   ::= "emit" IDENT "->" IDENT
    on_stmt     ::= "on" IDENT "->" expr
    expr_stmt   ::= expr
    expr        ::= pipeline
    pipeline    ::= call ("|>" call)*
    call        ::= atom ("." IDENT "(" arglist ")")* ("(" arglist ")")?
    atom        ::= STRING | NUMBER | BOOL | list | IDENT | "(" expr ")"
    list        ::= "[" (expr ("," expr)*)? "]"
    arglist     ::= (expr ("," expr)*)?

Built-in namespaces:
    llm.chat(prompt)          -- call the active LLM provider chain
    llm.embed(text)           -- embed text
    memory.store(key, value)  -- store in MemoryPalace
    memory.search(query)      -- search MemoryPalace
    kairos.propose(idea)      -- archive proposal for SAGE cycle
    evolve(strategy)          -- request self-modification
    reply_from(AGENT)         -- await a RESULT from a named SwarmBus agent
    think: "..."              -- log a reasoning step
    emit VAR -> TARGET        -- publish TASK to SwarmBus channel #swarm/TARGET
    on EVENT -> expr          -- fire expr when EVENT reply arrives
    if (cond) { ... } else { ... }
    for item in list { ... }
"""
from __future__ import annotations

import pathlib

from .lexer import TOKEN_RE, KEYWORDS, _BUILTIN_NS, _KEYWORD_FNS, Token, lex
from .parser import Node, ParseError, Parser, parse
from .runtime import GhostRuntimeError, Env, _truthy, GhostRuntime

__all__ = [
    # Lexer
    "TOKEN_RE", "KEYWORDS", "_BUILTIN_NS", "_KEYWORD_FNS", "Token", "lex",
    # Parser
    "Node", "ParseError", "Parser", "parse",
    # Runtime
    "GhostRuntimeError", "Env", "_truthy", "GhostRuntime",
    # Public API
    "run", "run_async", "run_file", "run_file_async",
    # Demos
    "DEMO_AGENT", "DEMO_PIPELINE", "DEMO_ASYNC", "DEMO_IF_FOR", "DEMO_MULTI_AGENT", "DEMO",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run(src: str, llm_fn=None, memory_engine=None,
        agent_id: str | None = None) -> dict:
    """Execute GhostScript synchronously. Returns trace log + archive."""
    rt = GhostRuntime(llm_fn=llm_fn, memory_engine=memory_engine, agent_id=agent_id)
    return rt.run(src)


async def run_async(src: str, llm_fn=None, memory_engine=None,
                    agent_id: str | None = None,
                    reply_timeout: float = 30.0) -> dict:
    """Execute GhostScript asynchronously with real SwarmBus + LLM wiring."""
    rt = GhostRuntime(llm_fn=llm_fn, memory_engine=memory_engine,
                      agent_id=agent_id, reply_timeout=reply_timeout)
    return await rt.run_async(src)


def run_file(path: str, llm_fn=None, memory_engine=None) -> dict:
    """Load and execute a .gs file synchronously."""
    src = pathlib.Path(path).read_text(encoding="utf-8")
    return run(src, llm_fn=llm_fn, memory_engine=memory_engine,
               agent_id=pathlib.Path(path).stem)


async def run_file_async(path: str, llm_fn=None, memory_engine=None,
                         reply_timeout: float = 30.0) -> dict:
    """Load and execute a .gs file asynchronously."""
    src = pathlib.Path(path).read_text(encoding="utf-8")
    return await run_async(src, llm_fn=llm_fn, memory_engine=memory_engine,
                           agent_id=pathlib.Path(path).stem,
                           reply_timeout=reply_timeout)


# ---------------------------------------------------------------------------
# Demo programs
# ---------------------------------------------------------------------------
DEMO_AGENT = '''# Classic SAGE cycle: Proposer -> Critic with real emit
agent Proposer {
    think: "Optimizing VRAM allocation for Qwen2.5"
    let proposal = llm.chat("Propose one concrete VRAM optimization. Under 20 words.")
    emit proposal -> Critic
    on APPROVE -> Archive.store(proposal)
    on REJECT  -> evolve("diversity_boost")
    on RESULT  -> memory.store("last_proposal", proposal)
}'''

DEMO_PIPELINE = '''# Pipeline: LLM output piped into memory
let query = "What is KAIROS?"
let result = llm.chat(query) |> memory.store("last_answer")
print(result)
'''

DEMO_ASYNC = '''# Async parallel proposals
async {
    let a = llm.chat("Propose optimization A")
    let b = llm.chat("Propose optimization B")
    kairos.propose(a)
    kairos.propose(b)
}
'''

DEMO_IF_FOR = '''# if/else + for loop
let score = 0.9
if (score) {
    let ideas = ["FAISS archive", "cold-tier pruning", "RLVR rewards"]
    for idea in ideas {
        kairos.propose(idea)
    }
} else {
    evolve("plateau_recovery")
}
'''

DEMO_MULTI_AGENT = '''# Multi-agent: Proposer -> ORACLE, then FORGE based on reply
agent Researcher {
    think: "Finding best VRAM optimization strategy"
    let query = "Best VRAM optimization for 8GB GPU running Qwen2.5?"
    emit query -> ORACLE
    on RESULT -> memory.store("research_result", RESULT)
    on REJECT -> evolve("search_wider")
}

agent Builder {
    think: "Implementing the top research finding"
    let spec = memory.search("research_result")
    emit spec -> FORGE
    on RESULT -> Archive.store(RESULT)
    on REJECT  -> evolve("simplify_spec")
}
'''

DEMO = DEMO_AGENT  # backward-compat alias used by server.py


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("=== Agent + SAGE cycle ===")
    print(json.dumps(run(DEMO_AGENT), indent=2))

    print("\n=== Pipeline ===")
    print(json.dumps(run(DEMO_PIPELINE), indent=2))

    print("\n=== Async ===")
    print(json.dumps(run(DEMO_ASYNC), indent=2))

    print("\n=== if/else + for loop ===")
    print(json.dumps(run(DEMO_IF_FOR), indent=2))

    print("\n=== Multi-agent ===")
    print(json.dumps(run(DEMO_MULTI_AGENT), indent=2))
