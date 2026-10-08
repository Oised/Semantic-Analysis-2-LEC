from __future__ import annotations

from ast_nodes import (
    Assignment,
    BinaryExpr,
    Block,
    BoolLiteral,
    CallExpr,
    CallStmt,
    Expr,
    FunctionDecl,
    IdentifierExpr,
    IfStmt,
    IntLiteral,
    Node,
    Program,
    PrintStmt,
    ReturnStmt,
    Stmt,
    TypeName,
    UnaryExpr,
    VarDecl,
    WhileStmt,
)
from semantic_errors import SemanticDiagnostic, SemanticError, SemanticErrorKind
from symbols import FunctionSymbol, Scope, Symbol, SymbolKind


class NameResolver:
    """Associa identificadores do programa às suas declarações."""

    def __init__(self, program: Program) -> None:
        self.program = program
        self.functions: dict[str, FunctionSymbol] = {}
        self.diagnostics: list[SemanticDiagnostic] = []

    def error(
        self,
        kind: SemanticErrorKind,
        message: str,
        node: Node,
    ) -> None:
        """Registra um erro semântico ligado ao trecho de código informado."""
        self.diagnostics.append(
            SemanticDiagnostic(kind=kind, message=message, span=node.span)
        )

    def resolve(self) -> None:
        """Declara funções, prepara escopos e resolve os corpos do programa."""
        # Registrar todas as funções primeiro permite chamadas antes da definição.
        for function in self.program.functions:
            self.declare_function(function)

        self.validate_main()

        for function in self.program.functions:
            # Parâmetros pertencem ao escopo externo do corpo da função.
            # Tipos void são validados na verificação de tipos (Seção 5.1).
            scope = Scope(parent=None)
            for parameter in function.parameters:
                symbol = Symbol(
                    name=parameter.name,
                    kind=SymbolKind.PARAMETER,
                    type=parameter.type,
                    declaration=parameter,
                )
                if self.declare(symbol, scope):
                    parameter.metadata["symbol"] = symbol

            self.resolve_block(function.body, scope)

        if self.diagnostics:
            raise SemanticError(self.diagnostics)

    def declare_function(self, function: FunctionDecl) -> None:
        """Cria o símbolo global da função ou diagnostica uma duplicata."""
        symbol = FunctionSymbol(
            name=function.name,
            kind=SymbolKind.FUNCTION,
            type=function.return_type,
            declaration=function,
            parameter_types=tuple(parameter.type for parameter in function.parameters),
        )

        if function.name in self.functions:
            self.error(
                SemanticErrorKind.DUPLICATE_FUNCTION,
                f"função {function.name!r} já declarada",
                function,
            )
            return

        self.functions[function.name] = symbol
        function.metadata["symbol"] = symbol

    def validate_main(self) -> None:
        """Verifica se existe uma função main sem parâmetros que retorna int."""
        main = self.functions.get("main")
        if main is None:
            self.error(
                SemanticErrorKind.INVALID_MAIN,
                "o programa deve declarar int main()",
                self.program,
            )
        elif main.type is not TypeName.INT or main.parameter_types:
            self.error(
                SemanticErrorKind.INVALID_MAIN,
                "main deve ter retorno int e não receber parâmetros",
                main.declaration,
            )

    def declare(self, symbol: Symbol, scope: Scope) -> bool:
        """Insere um símbolo no escopo, rejeitando nomes já declarados nele."""
        if symbol.name in scope.symbols:
            self.error(
                SemanticErrorKind.DUPLICATE_DECLARATION,
                f"declaração duplicada de {symbol.name!r}",
                symbol.declaration,
            )
            return False

        scope.symbols[symbol.name] = symbol
        return True

    def resolve_block(self, block: Block, scope: Scope) -> None:
        """Registra o escopo do bloco e resolve suas instruções em ordem."""
        block.metadata["scope"] = scope
        for statement in block.statements:
            self.resolve_statement(statement, scope)

    def resolve_statement(self, statement: Stmt, scope: Scope) -> None:
        """Resolve nomes da instrução e cria escopos para blocos aninhados."""
        if isinstance(statement, VarDecl):
            # Tipos void são validados na verificação de tipos (Seção 5.1).
            symbol = Symbol(
                name=statement.name,
                kind=SymbolKind.VARIABLE,
                type=statement.type,
                declaration=statement,
            )
            if self.declare(symbol, scope):
                statement.metadata["symbol"] = symbol

            if statement.initializer is not None:
                self.resolve_expr(statement.initializer, scope)

        elif isinstance(statement, Assignment):
            self.resolve_identifier(statement.target, scope)
            self.resolve_expr(statement.value, scope)

        elif isinstance(statement, CallStmt):
            self.resolve_expr(statement.call, scope)

        elif isinstance(statement, IfStmt):
            self.resolve_expr(statement.condition, scope)
            self.resolve_block(statement.then_block, Scope(parent=scope))
            if statement.else_block is not None:
                self.resolve_block(statement.else_block, Scope(parent=scope))

        elif isinstance(statement, WhileStmt):
            self.resolve_expr(statement.condition, scope)
            self.resolve_block(statement.body, Scope(parent=scope))

        elif isinstance(statement, PrintStmt):
            for item in statement.items:
                if isinstance(item, Expr):
                    self.resolve_expr(item, scope)

        elif isinstance(statement, ReturnStmt):
            if statement.value is not None:
                self.resolve_expr(statement.value, scope)

        elif isinstance(statement, Block):
            self.resolve_block(statement, Scope(parent=scope))

        else:
            raise NotImplementedError(
                f"resolução de nomes não implementada para {type(statement).__name__}"
            )

    def resolve_identifier(self, expression: IdentifierExpr, scope: Scope) -> None:
        """Procura a variável no escopo atual e, depois, nos escopos externos."""
        current: Scope | None = scope
        while current is not None:
            symbol = current.symbols.get(expression.name)
            if symbol is not None:
                expression.metadata["symbol"] = symbol
                return
            current = current.parent

        self.error(
            SemanticErrorKind.UNDECLARED_VARIABLE,
            f"variável não declarada {expression.name!r}",
            expression,
        )

    def resolve_expr(self, expression: Expr, scope: Scope) -> None:
        """Resolve identificadores e chamadas, percorrendo expressões compostas."""
        if isinstance(expression, IdentifierExpr):
            self.resolve_identifier(expression, scope)
        elif isinstance(expression, CallExpr):
            symbol = self.functions.get(expression.name)
            if symbol is None:
                self.error(
                    SemanticErrorKind.UNDECLARED_FUNCTION,
                    f"função não declarada {expression.name!r}",
                    expression,
                )
            else:
                expression.metadata["symbol"] = symbol

            for argument in expression.arguments:
                self.resolve_expr(argument, scope)
        elif isinstance(expression, BinaryExpr):
            self.resolve_expr(expression.left, scope)
            self.resolve_expr(expression.right, scope)
        elif isinstance(expression, UnaryExpr):
            self.resolve_expr(expression.operand, scope)
        elif isinstance(expression, (IntLiteral, BoolLiteral)):
            return


def resolve_names(program: Program) -> None:
    NameResolver(program).resolve()