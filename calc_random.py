"""A deliberately over-engineered command-line calculator.

Pipeline: text -> tokens (Lexer) -> syntax tree (Parser) -> number (Evaluator).
Supports + - * / // % ^, parentheses, unary minus, the constants pi and e,
the functions sqrt/abs/round/max/min, and `ans` (the previous result).
"""

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
        self.source = source
        self.index = 0

    def _peek(self) -> str:
        return self.source[self.index] if self.index < len(self.source) else ""

    def _read_while(self, predicate: Callable[[str], bool]) -> str:
        start = self.index
        while self.index < len(self.source) and predicate(self.source[self.index]):
            self.index += 1
        return self.source[start:self.index]

    def tokens(self) -> Iterator[Token]:
        while self.index < len(self.source):
            char = self._peek()
            start = self.index
            if char.isspace():
                self.index += 1
            elif char.isdigit() or char == ".":
                text = self._read_while(lambda c: c.isdigit() or c == ".")
                if text.count(".") > 1:
                    raise LexerError(f"malformed number '{text}' at {start}")
                yield Token("NUMBER", text, start)
            elif char.isalpha() or char == "_":
                yield Token("NAME", self._read_while(lambda c: c.isalnum() or c == "_"), start)
            elif char in OPERATOR_CHARS:
                self.index += 1
                if char == "/" and self._peek() == "/":
                    self.index += 1
                    yield Token("OP", "//", start)
                else:
                    yield Token("OP", char, start)
            elif char in SINGLE_CHAR_TOKENS:
                self.index += 1
                yield Token(SINGLE_CHAR_TOKENS[char], char, start)
            else:
                raise LexerError(f"unexpected character '{char}' at {start}")
        yield Token("END", "", self.index)


# ----------------------------------------------------------------- syntax tree

@dataclass
class Number:
    value: float


@dataclass
class Name:
    identifier: str


@dataclass
class Unary:
    operator: str
    operand: "Node"



@dataclass
class Call:
    function: str
    arguments: list["Node"] = field(default_factory=list)


Node = Union[Number, Name, Unary, Binary, Call]


class Parser:
    """Recursive descent with one function per precedence level.

    expression := term (('+' | '-') term)*
    term       := unary (('*' | '/' | '//' | '%') unary)*
    unary      := ('-' | '+') unary | power   # so -2 ^ 2 == -(2 ^ 2)
    power      := primary ('^' unary)?        # right-associative, allows 2 ^ -1
    primary    := NUMBER | NAME | NAME '(' args ')' | '(' expression ')'
    """

    def __init__(self, tokens: Iterator[Token]) -> None:
        self.tokens = list(tokens)
        self.position = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.position]

    def _advance(self) -> Token:
        token = self.current
        self.position += 1
        return token

    def _expect(self, kind: str) -> Token:
        if self.current.kind != kind:
            raise ParseError(f"expected {kind} at {self.current.position}, found '{self.current.text}'")
        return self._advance()

    def parse(self) -> Node:
        node = self._expression()
        self._expect("END")
        return node

    def _binary_level(self, next_level: Callable[[], Node], operators: set[str]) -> Node:
        node = next_level()
        while self.current.kind == "OP" and self.current.text in operators:
            symbol = self._advance().text
            node = Binary(symbol, node, next_level())
        return node

    def _expression(self) -> Node:
        return self._binary_level(self._term, {"+", "-"})

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
    return left / right


def _safe_floor_divide(left: float, right: float) -> float:
    if right == 0:
        raise EvaluationError("division by zero")
    return left // right


def _safe_modulo(left: float, right: float) -> float:
    if right == 0:
        raise EvaluationError("modulo by zero")
    return left % right


BINARY_OPERATIONS: dict[str, Callable[[float, float], float]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": _safe_divide,
    "//": _safe_floor_divide,
    "%": _safe_modulo,
    "^": operator.pow,
}

FUNCTIONS: dict[str, tuple[int, Callable[..., float]]] = {
    # name: (minimum argument count, implementation)
    "sqrt": (1, math.sqrt),
    "abs": (1, abs),
    "round": (1, round),
    "max": (2, max),
    "min": (2, min),
}

CONSTANTS = {"pi": math.pi, "e": math.e}


@dataclass
class Evaluator:
    memory: dict[str, float] = field(default_factory=lambda: {"ans": 0.0})

    def evaluate(self, node: Node) -> float:
        handler = getattr(self, f"_eval_{type(node).__name__.lower()}")
        return handler(node)

    def _eval_number(self, node: Number) -> float:
        return node.value

    def _eval_name(self, node: Name) -> float:
        if node.identifier in CONSTANTS:
            return CONSTANTS[node.identifier]
        if node.identifier in self.memory:
            return self.memory[node.identifier]
        raise EvaluationError(f"unknown name '{node.identifier}'")

    def _eval_unary(self, node: Unary) -> float:
        value = self.evaluate(node.operand)
        return -value if node.operator == "-" else value

    def _eval_binary(self, node: Binary) -> float:
        left, right = self.evaluate(node.left), self.evaluate(node.right)
        try:
            return BINARY_OPERATIONS[node.operator](left, right)
        except OverflowError as error:
            raise EvaluationError("result too large") from error

    def _eval_call(self, node: Call) -> float:
        if node.function not in FUNCTIONS:
            raise EvaluationError(f"unknown function '{node.function}'")
        minimum, implementation = FUNCTIONS[node.function]
        if len(node.arguments) < minimum:
            raise EvaluationError(f"{node.function} needs at least {minimum} argument(s)")
        values = [self.evaluate(argument) for argument in node.arguments]
        try:
            return float(implementation(*values))
        except ValueError as error:
            raise EvaluationError(f"{node.function}: {error}") from error


# ---------------------------------------------------------------- front end

@dataclass
class Calculator:
    evaluator: Evaluator = field(default_factory=Evaluator)
    history: list[tuple[str, float]] = field(default_factory=list)

    def calculate(self, expression: str) -> float:
        tree = Parser(Lexer(expression).tokens()).parse()
        result = self.evaluator.evaluate(tree)
        if isinstance(result, float) and result.is_integer():
            result = float(int(result))
        self.evaluator.memory["ans"] = result
        self.history.append((expression, result))
        return result

    def describe_history(self) -> str:
        if not self.history:
            return "(no calculations yet)"
        width = max(len(expression) for expression, _ in self.history)
        return "\n".join(
            f"{index:>3}. {expression:<{width}} = {format_number(result)}"
            for index, (expression, result) in enumerate(self.history, start=1)
        )


def format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.10g}"


COMMANDS = {
    "help": "show this message",
    "history": "list previous calculations",
    "clear": "forget history and reset ans",
    "quit": "leave the calculator",
}


def main() -> None:
    calculator = Calculator()
    print("Calculator ready. Type 'help' for commands.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "quit":
            break
        if line == "help":
            for name, description in COMMANDS.items():
                print(f"  {name:<8} {description}")
            continue
        if line == "history":
            print(calculator.describe_history())
            continue
        if line == "clear":
            calculator = Calculator()
            print("cleared")
            continue
        try:
            print(format_number(calculator.calculate(line)))
        except CalculatorError as error:
            print(f"error: {error}")


if __name__ == "__main__":
    main()