from __future__ import annotations

import asyncio
import uuid
from typing import Any

from .lexer import _BUILTIN_NS, _KEYWORD_FNS
from .parser import Node, ParseError, parse

try:
    from swarm.bus import SwarmBus as _SwarmBus, MsgType as _BusMsgType
    _BUS_OK = True
except ImportError:
    _BUS_OK = False


class GhostRuntimeError(Exception):
    pass


class Env:
    """Lexical environment for variable binding. Child scopes inherit parent."""
    def __init__(self, parent: "Env | None" = None):
        self._vars: dict[str, Any] = {}
        self._parent = parent

    def get(self, name: str) -> Any:
        if name in self._vars:
            return self._vars[name]
        if self._parent:
            return self._parent.get(name)
        raise GhostRuntimeError(f"undefined variable: {name!r}")

    def set(self, name: str, val: Any):
        self._vars[name] = val

    def child(self) -> "Env":
        return Env(parent=self)


def _truthy(val: Any) -> bool:
    if val is None or val is False or val == 0 or val == "" or val == []:
        return False
    if isinstance(val, str) and val.lower() in ("false", "none", "null", "0"):
        return False
    return True


class GhostRuntime:
    """
    Executes a GhostScript AST.

    llm_fn(prompt: str) -> str      — sync or async; enables real LLM calls
    memory_engine                   — MemoryPalace instance for real storage
    agent_id                        — identity on SwarmBus (default: random ghost-*)
    reply_timeout                   — seconds to wait for SwarmBus replies (default 30)
    """

    def __init__(self, llm_fn=None, memory_engine=None,
                 agent_id: str | None = None, reply_timeout: float = 30.0):
        self._llm      = llm_fn
        self._mem      = memory_engine
        self._id       = agent_id or f"ghost-{uuid.uuid4().hex[:6]}"
        self._timeout  = reply_timeout
        self.log: list[dict] = []
        self.archive: list[str] = []

    def _log(self, step: str, agent: str = "runtime", note: str = "", value: Any = None):
        entry: dict = {"step": step, "agent": agent, "note": note}
        if value is not None:
            entry["value"] = str(value)[:200]
        self.log.append(entry)

    # ── sync entry ─────────────────────────────────────────────────────────
    def run(self, src: str) -> dict:
        try:
            ast = parse(src)
        except ParseError as e:
            return {"ok": False, "error": str(e), "log": [], "archive": []}
        try:
            env = Env()
            self._exec_block(ast.children, env, agent="runtime")
        except GhostRuntimeError as e:
            return {"ok": False, "error": str(e), "log": self.log, "archive": self.archive}
        return {"ok": True, "log": self.log, "archive": self.archive}

    # ── async entry ────────────────────────────────────────────────────────
    async def run_async(self, src: str) -> dict:
        try:
            ast = parse(src)
        except ParseError as e:
            return {"ok": False, "error": str(e), "log": [], "archive": []}
        try:
            env = Env()
            await self._exec_block_async(ast.children, env, agent="runtime")
        except GhostRuntimeError as e:
            return {"ok": False, "error": str(e), "log": self.log, "archive": self.archive}
        return {"ok": True, "log": self.log, "archive": self.archive}

    # ── sync block/statement execution ────────────────────────────────────
    def _exec_block(self, stmts: list[Node], env: Env, agent: str):
        for stmt in stmts:
            self._exec_stmt(stmt, env, agent)

    def _exec_stmt(self, node: Node, env: Env, agent: str):
        if node.kind == "let":
            val = self._eval(node.children[0], env, agent)
            env.set(node.data["name"], val)
            self._log("let", agent, f"{node.data['name']} = {val!r}")

        elif node.kind == "if":
            cond = self._eval(node.children[0], env, agent)
            if _truthy(cond):
                self._exec_block(node.children[1].children, env.child(), agent)
            elif len(node.children) > 2:
                self._exec_block(node.children[2].children, env.child(), agent)

        elif node.kind == "for":
            iterable = self._eval(node.children[0], env, agent)
            var = node.data["var"]
            if isinstance(iterable, str):
                iterable = iterable.split()
            for item in (iterable if hasattr(iterable, "__iter__") else []):
                child = env.child()
                child.set(var, item)
                self._exec_block(node.children[1].children, child, agent)

        elif node.kind == "agent":
            self._exec_agent_sync(node, env)

        elif node.kind == "think":
            self._log("think", agent, node.data["text"])

        elif node.kind == "emit":
            what = node.data["what"]
            to   = node.data["to"]
            val  = env.get(what) if what in env._vars else what
            self.archive.append(str(val))
            self._log("emit", agent, f"{what} -> {to} [sync: archived only]", val)

        elif node.kind == "on":
            key = f"__on_{node.data['event']}"
            env.set(key, node.children[0])
            self._log("bind", agent, f"on {node.data['event']} bound")

        elif node.kind == "async":
            self._log("async", agent, "async block (sync degradation -- sequential)")
            self._exec_block(node.children[0].children, env.child(), agent)

        elif node.kind in ("expr_stmt",):
            self._eval(node.children[0], env, agent)

        else:
            self._eval(node, env, agent)

    def _exec_agent_sync(self, node: Node, parent_env: Env):
        name = node.data["name"]
        self._log("spawn", name, f"{name} agent (sync mode)")
        env = parent_env.child()
        env.set("self", name)
        handlers: dict[str, Node] = {}
        proposal = None

        for stmt in node.children[0].children:
            if stmt.kind == "on":
                handlers[stmt.data["event"]] = stmt.children[0]
                self._log("bind", name, f"on {stmt.data['event']} bound")
            elif stmt.kind == "emit":
                what = stmt.data["what"]
                to   = stmt.data["to"]
                val  = env.get(what) if what in env._vars else what
                proposal = str(val)
                self.archive.append(proposal)
                self._log("emit", name, f"{what} -> {to}", val)
            else:
                self._exec_stmt(stmt, env, name)

        if "APPROVE" in handlers and proposal:
            result = self._eval(handlers["APPROVE"], env, name)
            self._log("dispatch", name, f"APPROVE -> {result!r}")
        elif "REJECT" in handlers:
            result = self._eval(handlers["REJECT"], env, name)
            self._log("dispatch", name, f"REJECT -> {result!r}")

    # ── async block/statement execution ───────────────────────────────────
    async def _exec_block_async(self, stmts: list[Node], env: Env, agent: str):
        for stmt in stmts:
            await self._exec_stmt_async(stmt, env, agent)

    async def _exec_stmt_async(self, node: Node, env: Env, agent: str):
        if node.kind == "let":
            val = await self._eval_async(node.children[0], env, agent)
            env.set(node.data["name"], val)
            self._log("let", agent, f"{node.data['name']} = {val!r}")

        elif node.kind == "if":
            cond = await self._eval_async(node.children[0], env, agent)
            if _truthy(cond):
                await self._exec_block_async(node.children[1].children, env.child(), agent)
            elif len(node.children) > 2:
                await self._exec_block_async(node.children[2].children, env.child(), agent)

        elif node.kind == "for":
            iterable = await self._eval_async(node.children[0], env, agent)
            var = node.data["var"]
            if isinstance(iterable, str):
                iterable = iterable.split()
            for item in (iterable if hasattr(iterable, "__iter__") else []):
                child = env.child()
                child.set(var, item)
                await self._exec_block_async(node.children[1].children, child, agent)

        elif node.kind == "agent":
            await self._exec_agent_async(node, env)

        elif node.kind == "async":
            self._log("async", agent, "async block started")
            await self._exec_block_async(node.children[0].children, env.child(), agent)

        elif node.kind == "await":
            val = await self._eval_async(node.children[0], env, agent)
            self._log("await", agent, f"resolved: {val!r}")

        elif node.kind == "think":
            self._log("think", agent, node.data["text"])
            if _BUS_OK:
                try:
                    bus = _SwarmBus.instance()
                    await bus.emit(src=agent, content=node.data["text"],
                                   channel=f"#swarm/{agent}",
                                   msg_type=_BusMsgType.THOUGHT)
                except Exception:
                    pass

        elif node.kind == "emit":
            await self._emit_async(node, env, agent)

        elif node.kind == "on":
            key = f"__on_{node.data['event']}"
            env.set(key, node.children[0])
            self._log("bind", agent, f"on {node.data['event']} bound")

        elif node.kind in ("expr_stmt",):
            await self._eval_async(node.children[0], env, agent)

        else:
            await self._eval_async(node, env, agent)

    async def _emit_async(self, node: Node, env: Env, agent: str):
        what = node.data["what"]
        to   = node.data["to"]
        val  = env.get(what) if what in env._vars else what
        self.archive.append(str(val))
        self._log("emit", agent, f"{what} -> {to}", val)

        if _BUS_OK:
            try:
                bus = _SwarmBus.instance()
                await bus.emit(
                    src=agent,
                    content=str(val),
                    channel=f"#swarm/{to}",
                    msg_type=_BusMsgType.TASK,
                    dst=to,
                    task_id=str(uuid.uuid4())[:8],
                    ghostscript=True,
                )
                self._log("bus_emit", agent, f"published to #swarm/{to}")
            except Exception as e:
                self._log("bus_error", agent, str(e))

    async def _exec_agent_async(self, node: Node, parent_env: Env):
        """
        Execute an agent {} block with real SwarmBus wiring.

        Order: register identity -> run non-emit/on stmts -> collect handlers
               and emit targets -> publish TASK messages -> await replies
               -> fire matching on-handlers.
        """
        name = node.data["name"]
        self._log("spawn", name, f"{name} agent instantiated")

        env = parent_env.child()
        env.set("self", name)

        handlers:  dict[str, Node] = {}
        emit_jobs: list[tuple[str, str]] = []  # (val, target)

        for stmt in node.children[0].children:
            if stmt.kind == "on":
                handlers[stmt.data["event"]] = stmt.children[0]
                self._log("bind", name, f"on {stmt.data['event']} bound")
            elif stmt.kind == "emit":
                what = stmt.data["what"]
                to   = stmt.data["to"]
                val  = env.get(what) if what in env._vars else what
                emit_jobs.append((str(val), to))
                self.archive.append(str(val))
                self._log("emit", name, f"{what} -> {to}", val)
            else:
                await self._exec_stmt_async(stmt, env, name)

        for val, to in emit_jobs:
            if _BUS_OK:
                try:
                    bus = _SwarmBus.instance()
                    await bus.emit(
                        src=name,
                        content=val,
                        channel=f"#swarm/{to}",
                        msg_type=_BusMsgType.TASK,
                        dst=to,
                        task_id=str(uuid.uuid4())[:8],
                        ghostscript=True,
                    )
                except Exception as e:
                    self._log("bus_error", name, str(e))

        if emit_jobs and handlers and _BUS_OK:
            for val, to in emit_jobs:
                reply = await self._await_reply(name, to)
                if reply is not None:
                    env.set("RESULT", reply)
                    evt = self._classify_reply(reply)
                    if evt in handlers:
                        result = await self._eval_async(handlers[evt], env, name)
                        self._log("dispatch", name, f"{evt}({to}) -> {result!r}")
                    elif "RESULT" in handlers:
                        result = await self._eval_async(handlers["RESULT"], env, name)
                        self._log("dispatch", name, f"RESULT({to}) -> {result!r}")
                else:
                    self._log("timeout", name, f"no reply from {to} within {self._timeout}s")
                    if "REJECT" in handlers:
                        await self._eval_async(handlers["REJECT"], env, name)
        elif emit_jobs and handlers:
            if "APPROVE" in handlers:
                result = await self._eval_async(handlers["APPROVE"], env, name)
                self._log("dispatch", name, f"APPROVE (simulated) -> {result!r}")

    def _classify_reply(self, content: str) -> str:
        upper = content.upper()
        if "APPROVE" in upper: return "APPROVE"
        if "REJECT"  in upper: return "REJECT"
        return "RESULT"

    async def _await_reply(self, from_agent: str, to_agent: str) -> str | None:
        if not _BUS_OK:
            return None
        bus = _SwarmBus.instance()
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        consumed = [False]

        async def _handler(msg):
            if consumed[0] or fut.done():
                return
            if (msg.src == to_agent and
                    msg.msg_type in (_BusMsgType.RESULT, _BusMsgType.CRITIQUE,
                                     _BusMsgType.VERDICT, _BusMsgType.CHAT)):
                consumed[0] = True
                fut.set_result(msg.content)

        bus.subscribe(f"#swarm/{from_agent}", _handler)
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=self._timeout)
        except asyncio.TimeoutError:
            consumed[0] = True
            return None
        finally:
            bus.unsubscribe(f"#swarm/{from_agent}", _handler)

    # ── expression evaluation (sync) ──────────────────────────────────────
    def _eval(self, node: Node, env: Env, agent: str = "runtime") -> Any:
        if node.kind == "string":       return node.data["value"]
        if node.kind == "number":       return node.data["value"]
        if node.kind == "bool":         return node.data["value"]
        if node.kind == "list_literal": return [self._eval(c, env, agent) for c in node.children]

        if node.kind == "ident":
            name = node.data["name"]
            if name in _BUILTIN_NS or name in _KEYWORD_FNS:
                return name
            return env.get(name)

        if node.kind == "let":
            val = self._eval(node.children[0], env, agent)
            env.set(node.data["name"], val)
            return val

        if node.kind == "pipe":
            left_val = self._eval(node.children[0], env, agent)
            return self._call_with_pipe(node.children[1], left_val, env, agent)

        if node.kind in ("func_call", "method_call"):
            return self._dispatch_call(node, env, agent)

        if node.kind == "expr_stmt":
            return self._eval(node.children[0], env, agent)

        raise GhostRuntimeError(f"cannot evaluate node: {node.kind}")

    # ── expression evaluation (async) ─────────────────────────────────────
    async def _eval_async(self, node: Node, env: Env, agent: str) -> Any:
        if node.kind == "list_literal":
            return [await self._eval_async(c, env, agent) for c in node.children]
        if node.kind in ("func_call", "method_call"):
            return await self._dispatch_call_async(node, env, agent)
        if node.kind == "pipe":
            left_val = await self._eval_async(node.children[0], env, agent)
            return await self._call_with_pipe_async(node.children[1], left_val, env, agent)
        if node.kind == "let":
            val = await self._eval_async(node.children[0], env, agent)
            env.set(node.data["name"], val)
            return val
        if node.kind == "expr_stmt":
            return await self._eval_async(node.children[0], env, agent)
        return self._eval(node, env, agent)

    # ── pipe helpers ───────────────────────────────────────────────────────
    def _call_with_pipe(self, node: Node, piped: Any, env: Env, agent: str) -> Any:
        if node.kind == "func_call":
            args = [piped] + [self._eval(c, env, agent) for c in node.children]
            return self._builtin(node.data["name"], args, agent)
        if node.kind == "method_call":
            ns_val = self._eval(node.children[0], env, agent)
            extra  = [self._eval(c, env, agent) for c in node.children[1:]]
            return self._ns_call(str(ns_val), node.data["method"], [piped] + extra, agent)
        return self._eval(node, env, agent)

    async def _call_with_pipe_async(self, node: Node, piped: Any, env: Env, agent: str) -> Any:
        if node.kind == "func_call":
            args = [piped] + [await self._eval_async(c, env, agent) for c in node.children]
            return await self._builtin_async(node.data["name"], args, agent)
        if node.kind == "method_call":
            ns_val = await self._eval_async(node.children[0], env, agent)
            extra  = [await self._eval_async(c, env, agent) for c in node.children[1:]]
            return await self._ns_call_async(str(ns_val), node.data["method"], [piped] + extra, agent)
        return await self._eval_async(node, env, agent)

    # ── call dispatch ──────────────────────────────────────────────────────
    def _dispatch_call(self, node: Node, env: Env, agent: str) -> Any:
        if node.kind == "func_call":
            args = [self._eval(c, env, agent) for c in node.children]
            return self._builtin(node.data["name"], args, agent)
        if node.kind == "method_call":
            ns_val = self._eval(node.children[0], env, agent)
            args   = [self._eval(c, env, agent) for c in node.children[1:]]
            return self._ns_call(str(ns_val), node.data["method"], args, agent)
        raise GhostRuntimeError(f"unknown call kind: {node.kind}")

    async def _dispatch_call_async(self, node: Node, env: Env, agent: str) -> Any:
        if node.kind == "func_call":
            args = [await self._eval_async(c, env, agent) for c in node.children]
            return await self._builtin_async(node.data["name"], args, agent)
        if node.kind == "method_call":
            ns_val = await self._eval_async(node.children[0], env, agent)
            args   = [await self._eval_async(c, env, agent) for c in node.children[1:]]
            return await self._ns_call_async(str(ns_val), node.data["method"], args, agent)
        raise GhostRuntimeError(f"unknown call kind: {node.kind}")

    # ── sync built-ins ─────────────────────────────────────────────────────
    def _builtin(self, name: str, args: list, agent: str) -> Any:
        if name == "evolve":
            strategy = args[0] if args else "default"
            self._log("evolve", agent, f"strategy: {strategy}")
            return f"evolve({strategy})"
        if name == "print":
            val = " ".join(str(a) for a in args)
            self._log("print", agent, val)
            return val
        if name == "reply_from":
            self._log("reply_from", agent, "[sync mode -- no-op, use async {}]")
            return "[reply_from requires async mode]"
        if name == "llm":
            prompt = args[0] if args else ""
            if self._llm:
                try:
                    result = self._llm(prompt)
                    self._log("llm", agent, f"prompt={prompt[:60]!r}", result)
                    return result
                except Exception as e:
                    self._log("llm_error", agent, str(e))
                    return f"[llm error: {e}]"
            self._log("llm", agent, f"[simulated] {prompt[:60]!r}")
            return f"[LLM: {prompt[:60]}]"
        raise GhostRuntimeError(f"unknown function: {name!r}")

    # ── async built-ins ────────────────────────────────────────────────────
    async def _builtin_async(self, name: str, args: list, agent: str) -> Any:
        if name == "llm":
            prompt = args[0] if args else ""
            if self._llm:
                try:
                    if asyncio.iscoroutinefunction(self._llm):
                        result = await self._llm(prompt)
                    else:
                        result = self._llm(prompt)
                    self._log("llm", agent, f"prompt={prompt[:60]!r}", result)
                    return result
                except Exception as e:
                    self._log("llm_error", agent, str(e))
                    return f"[llm error: {e}]"
            self._log("llm", agent, f"[simulated] {prompt[:60]!r}")
            return f"[LLM: {prompt[:60]}]"
        if name == "reply_from":
            target = str(args[0]) if args else ""
            if not target:
                return None
            self._log("reply_from", agent, f"awaiting reply from {target}")
            return await self._await_reply(agent, target)
        return self._builtin(name, args, agent)

    # ── namespace calls ────────────────────────────────────────────────────
    def _ns_call(self, ns: str, method: str, args: list, agent: str) -> Any:
        if ns == "llm":
            if method == "chat":
                prompt = args[0] if args else ""
                if self._llm:
                    try:
                        result = self._llm(prompt)
                        self._log("llm.chat", agent, f"{prompt[:60]!r}", result)
                        return result
                    except Exception as e:
                        self._log("llm_error", agent, str(e))
                        return f"[llm error: {e}]"
                self._log("llm.chat", agent, f"[simulated] {prompt[:60]!r}")
                return f"[LLM: {prompt[:80]}]"
            if method == "embed":
                text = args[0] if args else ""
                self._log("llm.embed", agent, f"embedding {len(text)} chars")
                return f"[embedding:{len(text)}dims]"
            raise GhostRuntimeError(f"llm.{method} not implemented")

        if ns == "memory":
            if method == "store":
                key = str(args[0]) if args else "unnamed"
                val = args[1] if len(args) > 1 else ""
                if self._mem:
                    try:
                        self._mem.store(content=f"{key}: {val}", room="ghostscript")
                    except Exception as e:
                        self._log("memory_error", agent, str(e))
                self._log("memory.store", agent, f"{key!r} = {str(val)[:60]!r}")
                return val
            if method == "search":
                query = str(args[0]) if args else ""
                if self._mem:
                    try:
                        results = self._mem.search(query)
                        hits = [r.get("content", "") for r in results[:5]]
                        self._log("memory.search", agent, f"query={query!r} -> {len(hits)} hits")
                        return hits
                    except Exception as e:
                        self._log("memory_error", agent, str(e))
                self._log("memory.search", agent, f"[simulated] query={query!r}")
                return [f"[memory result for: {query}]"]
            raise GhostRuntimeError(f"memory.{method} not implemented")

        if ns == "kairos":
            if method == "propose":
                idea = str(args[0]) if args else ""
                self._log("kairos.propose", agent, f"proposal: {idea!r}")
                self.archive.append(idea)
                return idea
            if method == "score":
                result = args[0] if args else ""
                self._log("kairos.score", agent, f"scoring: {result!r}")
                return 0.85
            raise GhostRuntimeError(f"kairos.{method} not implemented")

        if ns == "Archive":
            if method == "store":
                val = args[0] if args else ""
                self.archive.append(str(val))
                self._log("archive.store", agent, str(val)[:80])
                return val
            raise GhostRuntimeError(f"Archive.{method} not implemented")

        raise GhostRuntimeError(f"unknown namespace: {ns!r}")

    async def _ns_call_async(self, ns: str, method: str, args: list, agent: str) -> Any:
        # Only llm.chat needs real async (may await the LLM coroutine).
        # Everything else is I/O-free and delegates to the sync implementation.
        if ns == "llm" and method == "chat":
            prompt = args[0] if args else ""
            if self._llm:
                try:
                    if asyncio.iscoroutinefunction(self._llm):
                        result = await self._llm(prompt)
                    else:
                        result = self._llm(prompt)
                    self._log("llm.chat", agent, f"{prompt[:60]!r}", result)
                    return result
                except Exception as e:
                    self._log("llm_error", agent, str(e))
                    return f"[llm error: {e}]"
            self._log("llm.chat", agent, f"[simulated] {prompt[:60]!r}")
            return f"[LLM: {prompt[:80]}]"
        return self._ns_call(ns, method, args, agent)
