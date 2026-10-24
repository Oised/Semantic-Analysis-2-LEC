"""Verificações realizadas após a convergência dos estados de inicialização."""

from __future__ import annotations

import load_previous  # Disponibiliza os módulos das etapas anteriores.

from ast_nodes import (
    Assignment, BinaryExpr, BoolLiteral, CallExpr, CallStmt, Expr,
    IdentifierExpr, IntLiteral, PrintStmt, TypeName, UnaryExpr, VarDecl,
)
from cfg import Branch, CFG, Return, ScopeExit
from initialization import InitializationFacts, symbol_of
from semantic_errors import SemanticDiagnostic, SemanticErrorKind
from symbols import Symbol


class ReadChecker:
    def __init__(self) -> None:
        self.diagnostics: list[SemanticDiagnostic] = []

    def check(
        self, cfg: CFG, reachable: set[int], facts: InitializationFacts,
    ) -> list[SemanticDiagnostic]:
        self.diagnostics = []
        for block_id in sorted(reachable):
            block = cfg.blocks[block_id]
            state = set(facts.initialized_in[block_id])
            for operation in block.operations:
                self._check_operation(operation, state)

            if isinstance(block.terminator, Branch):
                self._check_expression(block.terminator.condition, state)
            elif isinstance(block.terminator, Return):
                value = block.terminator.statement.value
                if value is not None:
                    self._check_expression(value, state)
        return self.diagnostics

    def _check_operation(self, operation, state: set[Symbol]) -> None:
        if isinstance(operation, VarDecl):
            symbol = symbol_of(operation)
            state.discard(symbol)
            if operation.initializer is not None:
                self._check_expression(operation.initializer, state)
                state.add(symbol)
        elif isinstance(operation, Assignment):
            self._check_expression(operation.value, state)
            state.add(symbol_of(operation.target))
        elif isinstance(operation, PrintStmt):
            for item in operation.items:
                if isinstance(item, Expr):
                    self._check_expression(item, state)
        elif isinstance(operation, CallStmt):
            self._check_expression(operation.call, state)
        elif isinstance(operation, ScopeExit):
            state.difference_update(operation.scope.symbols.values())
        else:
            raise TypeError(f"operação inesperada no CFG: {type(operation).__name__}")

    def _check_expression(self, expression: Expr, state: set[Symbol]) -> None:
        if isinstance(expression, IdentifierExpr):
            symbol = symbol_of(expression)
            if symbol not in state: # testa se a variável foi inicializada antes de ser lida
                self.diagnostics.append( # adiciona um diagnóstico de erro semântico à lista de diagnósticos
                    SemanticDiagnostic(
                        kind=SemanticErrorKind.UNINITIALIZED_READ,
                        message=f"variável {expression.name!r} pode não ter sido inicializada",
                        span=expression.span,
                    )
                )
            return
        if isinstance(expression, (IntLiteral, BoolLiteral)):
            return
        if isinstance(expression, UnaryExpr):
            self._check_expression(expression.operand, state)
            return
        if isinstance(expression, BinaryExpr):
            # Escolha estática conservadora: confira ambos os operandos,
            # inclusive && e ||. O backend preservará o curto-circuito.
            self._check_expression(expression.left, state)
            self._check_expression(expression.right, state)
            return
        if isinstance(expression, CallExpr):
            for argument in expression.arguments:
                self._check_expression(argument, state)
            return
        raise TypeError(f"expressão desconhecida: {type(expression).__name__}")


def check_function_exit(cfg: CFG, reachable: set[int]) -> list[SemanticDiagnostic]:
    if cfg.function.return_type is TypeName.VOID:
        return []
    if cfg.fallthrough_exit not in reachable:
        return []
    return [ # adiciona um diagnóstico de erro semântico à lista de diagnósticos para indicar que a função pode terminar sem retornar valor
        SemanticDiagnostic(
            kind=SemanticErrorKind.MISSING_RETURN,
            message=f"função {cfg.function.name!r} pode terminar sem retornar valor",
            span=cfg.function.span,
        )
    ]
