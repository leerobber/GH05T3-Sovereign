from __future__ import annotations

from dataclasses import dataclass, field

from .lexer import Token, lex, _BUILTIN_NS, _KEYWORD_FNS


@dataclass
class Node:
    kind: str
    data: dict = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, toks: list[Token]):
        self.toks = toks
        self.i = 0

    def peek(self, offset: int = 0) -> Token | None:
        idx = self.i + offset
        return self.toks[idx] if idx < len(self.toks) else None

    def eat(self, kind: str, value: str | None = None) -> Token:
        t = self.peek()
        if not t:
            raise ParseError(f"unexpected end of input, expected {kind}")
        if t.kind != kind:
            raise ParseError(f"pos {t.pos}: expected {kind}, got {t.kind}={t.value!r}")
        if value is not None and t.value != value:
            raise ParseError(f"pos {t.pos}: expected {value!r}, got {t.value!r}")
        self.i += 1
        return t

    def maybe(self, kind: str, value: str | None = None) -> Token | None:
        t = self.peek()
        if t and t.kind == kind and (value is None or t.value == value):
            self.i += 1
            return t
        return None

    # ── top level ──────────────────────────────────────────────────────────
    def parse_program(self) -> Node:
        prog = Node("program")
        while self.peek():
            prog.children.append(self.parse_stmt())
        return prog

    def parse_block(self) -> Node:
        self.eat("LBRACE")
        block = Node("block")
        while self.peek() and self.peek().kind != "RBRACE":
            block.children.append(self.parse_stmt())
        self.eat("RBRACE")
        return block

    def parse_stmt(self) -> Node:
        t = self.peek()
        if not t:
            raise ParseError("unexpected end")
        if t.kind == "LET":   return self.parse_let()
        if t.kind == "IF":    return self.parse_if()
        if t.kind == "FOR":   return self.parse_for()
        if t.kind == "AGENT": return self.parse_agent()
        if t.kind == "ASYNC": return self.parse_async()
        if t.kind == "AWAIT": return self.parse_await()
        if t.kind == "THINK": return self.parse_think()
        if t.kind == "EMIT":  return self.parse_emit()
        if t.kind == "ON":    return self.parse_on()
        return Node("expr_stmt", {}, [self.parse_expr()])

    def parse_let(self) -> Node:
        self.eat("LET")
        name = self.eat("IDENT").value
        self.eat("EQ")
        val = self.parse_expr()
        return Node("let", {"name": name}, [val])

    def parse_if(self) -> Node:
        self.eat("IF")
        self.eat("LPAREN")
        cond = self.parse_expr()
        self.eat("RPAREN")
        then_block = self.parse_block()
        else_block = None
        if self.maybe("ELSE"):
            else_block = self.parse_block()
        children = [cond, then_block] + ([else_block] if else_block else [])
        return Node("if", {}, children)

    def parse_for(self) -> Node:
        self.eat("FOR")
        var = self.eat("IDENT").value
        self.eat("IN")
        iterable = self.parse_expr()
        body = self.parse_block()
        return Node("for", {"var": var}, [iterable, body])

    def parse_agent(self) -> Node:
        self.eat("AGENT")
        name = self.eat("IDENT").value
        block = self.parse_block()
        return Node("agent", {"name": name}, [block])

    def parse_async(self) -> Node:
        self.eat("ASYNC")
        block = self.parse_block()
        return Node("async", {}, [block])

    def parse_await(self) -> Node:
        self.eat("AWAIT")
        expr = self.parse_expr()
        return Node("await", {}, [expr])

    def parse_think(self) -> Node:
        self.eat("THINK")
        self.eat("COLON")
        s = self.eat("STRING").value.strip("\"'")
        return Node("think", {"text": s})

    def parse_emit(self) -> Node:
        self.eat("EMIT")
        what = self.eat("IDENT").value
        self.eat("ARROW")
        to = self.eat("IDENT").value
        return Node("emit", {"what": what, "to": to})

    def parse_on(self) -> Node:
        self.eat("ON")
        event = self.eat("IDENT").value
        self.eat("ARROW")
        call = self.parse_expr()
        return Node("on", {"event": event}, [call])

    # ── expressions ────────────────────────────────────────────────────────
    def parse_expr(self) -> Node:
        return self.parse_pipeline()

    def parse_pipeline(self) -> Node:
        left = self.parse_call()
        while self.peek() and self.peek().kind == "PIPE_OP":
            self.eat("PIPE_OP")
            right = self.parse_call()
            left = Node("pipe", {}, [left, right])
        return left

    def parse_call(self) -> Node:
        node = self.parse_atom()
        while True:
            if self.peek() and self.peek().kind == "DOT":
                self.eat("DOT")
                method = self.eat("IDENT").value
                self.eat("LPAREN")
                args = self.parse_arglist()
                self.eat("RPAREN")
                node = Node("method_call", {"method": method}, [node] + args)
            elif self.peek() and self.peek().kind == "LPAREN" and node.kind == "ident":
                self.eat("LPAREN")
                args = self.parse_arglist()
                self.eat("RPAREN")
                node = Node("func_call", {"name": node.data["name"]}, args)
            else:
                break
        return node

    def parse_arglist(self) -> list[Node]:
        args = []
        if self.peek() and self.peek().kind not in ("RPAREN",):
            args.append(self.parse_expr())
            while self.maybe("COMMA"):
                args.append(self.parse_expr())
        return args

    def parse_atom(self) -> Node:
        t = self.peek()
        if not t:
            raise ParseError("unexpected end in expression")
        if t.kind == "STRING":
            self.i += 1
            return Node("string", {"value": t.value.strip("\"'")})
        if t.kind == "NUMBER":
            self.i += 1
            v = float(t.value) if "." in t.value else int(t.value)
            return Node("number", {"value": v})
        if t.kind == "BOOL":
            self.i += 1
            return Node("bool", {"value": t.value == "true"})
        if t.kind == "LBRACKET":
            return self.parse_list_literal()
        if t.kind == "IDENT":
            self.i += 1
            return Node("ident", {"name": t.value})
        if t.kind == "LPAREN":
            self.eat("LPAREN")
            e = self.parse_expr()
            self.eat("RPAREN")
            return e
        raise ParseError(f"unexpected token {t.kind}={t.value!r}")

    def parse_list_literal(self) -> Node:
        self.eat("LBRACKET")
        items = []
        if self.peek() and self.peek().kind != "RBRACKET":
            items.append(self.parse_expr())
            while self.maybe("COMMA"):
                items.append(self.parse_expr())
        self.eat("RBRACKET")
        return Node("list_literal", {}, items)


def parse(src: str) -> Node:
    return Parser(lex(src)).parse_program()
