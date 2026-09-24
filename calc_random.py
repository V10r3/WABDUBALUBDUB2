"""A deliberately over-engineered command-line calculator.

Pipeline: text -> tokens (Lexer) -> syntax tree (Parser) -> number (Evaluator).
Supports + - * / // % ^, parentheses, unary minus, the constants pi and e,
the functions sqrt/abs/round/max/min, and `ans` (the previous result).
"""


    def _term(self) -> Node:
        return self._binary_level(self._unary, {"*", "/", "//", "%"})

    def _unary(self) -> Node:
        if self.current.kind == "OP" and self.current.text in {"-", "+"}:
            symbol = self._advance().text
            return Unary(symbol, self._unary())
        return self._power()

    def _power(self) -> Node:
        base = self._primary()
        if self.current.kind == "OP" and self.current.text == "^":
            self._advance()
            return Binary("^", base, self._unary())
        return base

    def _primary(self) -> Node:
        token = self.current
        if token.kind == "NUMBER":
            self._advance()
            return Number(float(token.text))
        if token.kind == "NAME":
            self._advance()
            if self.current.kind == "LPAREN":
                return Call(token.text, self._arguments())
            return Name(token.text)
        if token.kind == "LPAREN":
            self._advance()
            node = self._expression()
            self._expect("RPAREN")
            return node
        raise ParseError(f"unexpected '{token.text or 'end of input'}' at {token.position}")

    def _arguments(self) -> list[Node]:
        self._expect("LPAREN")
        arguments: list[Node] = []
        if self.current.kind != "RPAREN":
            arguments.append(self._expression())
            while self.current.kind == "COMMA":
                self._advance()
                arguments.append(self._expression())
        self._expect("RPAREN")
        return arguments


# ------------------------------------------------------------------ evaluation

def _safe_divide(left: float, right: float) -> float:
    if right == 0:
        raise EvaluationError("division by zero")
    return left / rights

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Callable, Iterator, Union


class CalculatorError(Exception):
    """Base class for every error the calculator reports to the user."""


class LexerError(CalculatorError):
    pass


class ParseError(CalculatorError):
    pass


class EvaluationError(CalculatorError):
    pass


# ---------------------------------------------------------------- tokens

@dataclass(frozen=True)
class Token:
    kind: str          # NUMBER, NAME, OP, LPAREN, RPAREN, COMMA, END
    text: str
    position: int


SINGLE_CHAR_TOKENS = {"(": "LPAREN", ")": "RPAREN", ",": "COMMA"}
OPERATOR_CHARS = set("+-*/%^")


class Lexer:
    def __init__(self, source: str) -> None:
