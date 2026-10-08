from __future__ import annotations

from collections.abc import Sequence

from Lexer import Token, TokenKind
from ast_nodes import (
    Block,
    BoolLiteral,
    Expr,
    FunctionDecl,
    Node,
    Parameter,
    PrintItem,
    Program,
    SourceSpan,
    Stmt,
    StringLiteral,
    TypeName,
    Assignment,
    CallExpr,
    CallStmt,
    IdentifierExpr,
    VarDecl,
    IfStmt,
    WhileStmt,
    ReturnStmt,
    BinaryExpr,
    BinaryOperator,
    PrintStmt,
    UnaryOperator,
    UnaryExpr,
    IntLiteral
)


TYPE_START = {TokenKind.KW_INT, TokenKind.KW_BOOL, TokenKind.KW_VOID}
EXPRESSION_START = {
    TokenKind.IDENTIFIER,
    TokenKind.INT_LITERAL,
    TokenKind.KW_FALSE,
    TokenKind.KW_TRUE,
    TokenKind.LEFT_PAREN,
    TokenKind.LOGICAL_NOT,
    TokenKind.MINUS,
}
STATEMENT_START = TYPE_START | {
    TokenKind.IDENTIFIER,
    TokenKind.KW_IF,
    TokenKind.KW_WHILE,
    TokenKind.KW_RETURN,
    TokenKind.KW_PRINT,
    TokenKind.LEFT_BRACE,
}


TYPE_BY_TOKEN = {
    TokenKind.KW_INT: TypeName.INT,
    TokenKind.KW_BOOL: TypeName.BOOL,
    TokenKind.KW_VOID: TypeName.VOID,
}


class ParserError(Exception):
    def __init__(self, token: Token, expected: set[TokenKind]):
        self.token = token
        self.expected = frozenset(expected)
        super().__init__()

    @property
    def line(self) -> int:
        return self.token.line

    @property
    def column(self) -> int:
        return self.token.column

    def __str__(self) -> str:
        names = ", ".join(kind.name for kind in sorted(
            self.expected,
            key=lambda kind: kind.value,
        ))
        return (
            f"erro sintático em {self.line}:{self.column}: esperado {{{names}}}, "
            f"encontrado {self.token.kind.name} ({self.token.lexeme!r})"
        )


class Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        if not self.tokens:
            raise ValueError("a sequência de tokens deve terminar em EOF")
        if self.tokens[-1].kind is not TokenKind.EOF:
            raise ValueError("o último token deve ser EOF")
        if any(token.kind is TokenKind.EOF for token in self.tokens[:-1]):
            raise ValueError("EOF deve aparecer uma única vez, no final")
        self.current = 0

    def peek(self, offset: int = 0) -> Token:
        index = min(self.current + offset, len(self.tokens) - 1)
        return self.tokens[index]

    def check(self, kind: TokenKind) -> bool:
        return self.peek().kind is kind

    def advance(self) -> Token:
        token = self.peek()
        if self.current < len(self.tokens) - 1:
            self.current += 1
        return token

    def match(self, *kinds: TokenKind) -> Token | None:
        if self.peek().kind in kinds:
            return self.advance()
        return None

    def expect(self, kinds: TokenKind | set[TokenKind]) -> Token:
        expected = kinds if isinstance(kinds, set) else {kinds}
        token = self.peek()
        if token.kind not in expected:
            raise ParserError(token, set(expected))
        return self.advance()

    @staticmethod
    def _token_span(token: Token) -> SourceSpan:
        return SourceSpan(
            token.line,
            token.column,
            token.line,
            token.column + len(token.lexeme),
        )

    @staticmethod
    def _start(value: Token | Node) -> tuple[int, int]:
        if isinstance(value, Node):
            return value.span.start_line, value.span.start_column
        return value.line, value.column

    @staticmethod
    def _end(value: Token | Node) -> tuple[int, int]:
        if isinstance(value, Node):
            return value.span.end_line, value.span.end_column
        return value.line, value.column + len(value.lexeme)

    @classmethod
    def _span(cls, first: Token | Node, last: Token | Node) -> SourceSpan:
        start_line, start_column = cls._start(first)
        end_line, end_column = cls._end(last)
        return SourceSpan(start_line, start_column, end_line, end_column)

    def parse(self) -> Program:
        return self.parse_program()

    # program ::= function* EOF
    def parse_program(self) -> Program:
        start = self.peek()
        functions: list[FunctionDecl] = []
        while self.peek().kind in TYPE_START:
            functions.append(self.parse_function())
        eof = self.expect(TokenKind.EOF)
        return Program(functions, span=self._span(start, eof))

    # function ::= type IDENTIFIER ... block
    def parse_function(self) -> FunctionDecl:
        start = self.peek()
        return_type = self.parse_type()
        name = self.expect(TokenKind.IDENTIFIER)
        self.expect(TokenKind.LEFT_PAREN)
        parameters = (
            self.parse_parameter_list()
            if self.peek().kind in TYPE_START
            else []
        )
        self.expect(TokenKind.RIGHT_PAREN)
        body = self.parse_block()
        return FunctionDecl(
            return_type,
            name.lexeme,
            parameters,
            body,
            span=self._span(start, body),
        )

    # type ::= KW_INT | KW_BOOL | KW_VOID
    def parse_type(self) -> TypeName:
        token = self.expect(TYPE_START)
        return TYPE_BY_TOKEN[token.kind]

    def parse_parameter_list(self) -> list[Parameter]:
        parameters: list[Parameter] = [self.parse_parameter()]

        while self.match(TokenKind.COMMA) is not None:
            parameters.append(self.parse_parameter())

        return parameters

    def parse_parameter(self) -> Parameter:
        start = self.peek()
        param_type = self.parse_type()
        name = self.expect(TokenKind.IDENTIFIER)
        return Parameter(
            param_type,
            name.lexeme,
            span=self._span(start, name)
        )

    def parse_block(self) -> Block:
        start = self.expect(TokenKind.LEFT_BRACE)
        statements: list[Stmt] = []

        while self.peek().kind in STATEMENT_START:
            statements.append(self.parse_statement())

        end = self.expect(TokenKind.RIGHT_BRACE)
        return Block(statements, span=self._span(start, end))


    def parse_statement(self) -> Stmt:
        kind = self.peek().kind
        if kind in TYPE_START:
            return self.parse_declaration()
        if kind is TokenKind.IDENTIFIER:
            return self.parse_id_or_call_statement()
        if kind is TokenKind.KW_IF:
            return self.parse_if_statement()
        if kind is TokenKind.KW_WHILE:
            return self.parse_while_statement()
        if kind is TokenKind.KW_RETURN:
            return self.parse_return_statement()
        if kind is TokenKind.KW_PRINT:
            return self.parse_print_statement()
        if kind is TokenKind.LEFT_BRACE:
            return self.parse_block()
        
        raise ParserError(self.peek(), STATEMENT_START)

    def parse_id_or_call_statement(self) -> Stmt:
        name_token = self.expect(TokenKind.IDENTIFIER)

        next_token = self.expect({TokenKind.ASSIGN, TokenKind.LEFT_PAREN})
        if next_token.kind is TokenKind.ASSIGN:
            value_expr = self.parse_expression()
            semicolon_token = self.expect(TokenKind.SEMICOLON)
            target_expr = IdentifierExpr(name_token.lexeme, span=self._span(name_token, name_token))

            return Assignment(target_expr, value_expr, span=self._span(name_token, semicolon_token))

        arguments = self.parse_arguments()
        right_paren = self.expect(TokenKind.RIGHT_PAREN)
        semicolon_token = self.expect(TokenKind.SEMICOLON)

        call_expr = CallExpr(
            name_token.lexeme,
            arguments, 
            span=self._span(name_token, right_paren)
            )

        return CallStmt(call_expr, span=self._span(name_token, semicolon_token))

    def parse_declaration(self) -> Stmt:
        start = self.peek()
        decl_type = self.parse_type()
        name = self.expect(TokenKind.IDENTIFIER)
        initializer: Expr | None = None

        if self.match(TokenKind.ASSIGN) is not None:
            initializer = self.parse_expression()

        semicolon_token = self.expect(TokenKind.SEMICOLON)
        return VarDecl(
            decl_type,
            name.lexeme,
            initializer,
            span=self._span(start, semicolon_token),
        )

    def parse_if_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_IF)
        self.expect(TokenKind.LEFT_PAREN)
        condition = self.parse_expression()
        self.expect(TokenKind.RIGHT_PAREN)
        then_block = self.parse_block()
        else_block: Block | None = None

        if self.match(TokenKind.KW_ELSE) is not None:
            else_block = self.parse_block()
            
        return IfStmt(
            condition,
            then_block,
            else_block,
            span=self._span(start, else_block or then_block),
        )

    def parse_while_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_WHILE)
        self.expect(TokenKind.LEFT_PAREN)
        condition = self.parse_expression()
        self.expect(TokenKind.RIGHT_PAREN)
        body = self.parse_block()
        return WhileStmt(
            condition,
            body,
            span=self._span(start, body),
        )

    def parse_return_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_RETURN)
        value: Expr | None = None

        if self.peek().kind in EXPRESSION_START:
            value = self.parse_expression()

        semicolon_token = self.expect(TokenKind.SEMICOLON)
        return ReturnStmt(
            value,
            span=self._span(start, semicolon_token),
        )

    # KW_PRINT LEFT_PAREN print_item (COMMA print_item)* RIGHT_PAREN SEMICOLON
    def parse_print_statement(self) -> Stmt:
        # KW_PRINT
        start = self.expect(TokenKind.KW_PRINT)
        # LEFT_PAREN
        self.expect(TokenKind.LEFT_PAREN)
        #print_item
        items: list[PrintItem] = [self.parse_print_item()]
        # (COMMA print_item)*
        while self.match(TokenKind.COMMA) is not None:
            items.append(self.parse_print_item())
        # RIGHT_PAREN 
        self.expect(TokenKind.RIGHT_PAREN)
        # SEMICOLON
        semicolon_token = self.expect(TokenKind.SEMICOLON)
        
        return PrintStmt(
            items,
            span=self._span(start, semicolon_token)
        )

    # expression | string_literals
    def parse_print_item(self) -> PrintItem:
        if self.check(TokenKind.STRING_LITERAL):
            return self.parse_string_literals()
        return self.parse_expression()

    # STRING_LITERAL+
    def parse_string_literals(self) -> StringLiteral:
        first = self.expect(TokenKind.STRING_LITERAL)
        value = first.value
        last = first
        while self.check(TokenKind.STRING_LITERAL):
            token = self.advance()
            value += token.value
            last = token
        return StringLiteral(value, span=self._span(first, last))

    # logical_or
    def parse_expression(self) -> Expr:
        return self.parse_logical_or()

    # logical_and (LOGICAL_OR logical_and)*
    def parse_logical_or(self) -> Expr:
        # logical_and
        left = self.parse_logical_and()
        # (LOGICAL_OR logical_and)*
        while self.match(TokenKind.LOGICAL_OR) is not None:
            right = self.parse_logical_and()
            left = BinaryExpr(
                BinaryOperator.LOGICAL_OR,
                left,
                right,
                span=self._span(left, right),
            )
        return left

    # equality (LOGICAL_AND equality)*
    def parse_logical_and(self) -> Expr:
        left = self.parse_equality()
        while self.match(TokenKind.LOGICAL_AND) is not None:
            right = self.parse_equality()
            left = BinaryExpr(
                BinaryOperator.LOGICAL_AND,
                left,
                right,
                span=self._span(left, right),
            )
        return left

    # relational ((EQUAL_EQUAL | NOT_EQUAL) relational)*
    def parse_equality(self) -> Expr:
        # relational
        left = self.parse_relational()
        #((EQUAL_EQUAL | NOT_EQUAL) relational)*
        while (op := self.match(TokenKind.EQUAL_EQUAL, TokenKind.NOT_EQUAL)) is not None:
            # relational
            right = self.parse_relational()
            # (EQUAL_EQUAL | NOT_EQUAL)
            if op.kind is TokenKind.EQUAL_EQUAL:
                operador = BinaryOperator.EQUAL
            else:
                operador = BinaryOperator.NOT_EQUAL
            left = BinaryExpr(operador, left, right, span=self._span(left, right))
        return left

    # additive ((LESS | LESS_EQUAL | GREATER | GREATER_EQUAL) additive)*
    def parse_relational(self) -> Expr:
        # additive
        left = self.parse_additive()
        # ((LESS | LESS_EQUAL | GREATER | GREATER_EQUAL) additive)*
        while (op := self.match(TokenKind.LESS, TokenKind.LESS_EQUAL, TokenKind.GREATER, TokenKind.GREATER_EQUAL)) is not None:
            # additive
            right = self.parse_additive()
            # (LESS | LESS_EQUAL | GREATER | GREATER_EQUAL)
            if op.kind is TokenKind.LESS:
                operador = BinaryOperator.LESS
            elif op.kind is TokenKind.LESS_EQUAL:
                operador = BinaryOperator.LESS_EQUAL
            elif op.kind is TokenKind.GREATER:
                operador = BinaryOperator.GREATER
            else:
                operador = BinaryOperator.GREATER_EQUAL
            left = BinaryExpr(operador, left, right, span=self._span(left, right))
        return left

    # multiplicative ((PLUS | MINUS) multiplicative)*
    def parse_additive(self) -> Expr:
        # multiplicative
        left = self.parse_multiplicative()
        #((PLUS | MINUS) multiplicative)*
        while (op := self.match(TokenKind.PLUS, TokenKind.MINUS)) is not None:
            # multiplicative
            right = self.parse_multiplicative()
            # (PLUS | MINUS)
            if op.kind is TokenKind.PLUS:
                operador = BinaryOperator.ADD
            else:
                operador = BinaryOperator.SUBTRACT
            left = BinaryExpr(operador, left, right, span=self._span(left, right))
        return left

    # unary ((STAR | SLASH | PERCENT) unary)*
    def parse_multiplicative(self) -> Expr:
        # unary
        left = self.parse_unary()
        # ((STAR | SLASH | PERCENT) unary)*
        while (op := self.match(TokenKind.STAR, TokenKind.SLASH, TokenKind.PERCENT)) is not None:
            # unary
            right = self.parse_unary()
            # (STAR | SLASH | PERCENT)
            if op.kind is TokenKind.STAR:
                operador = BinaryOperator.MULTIPLY
            elif op.kind is TokenKind.SLASH:
                operador = BinaryOperator.DIVIDE
            else:
                operador = BinaryOperator.REMAINDER
            left = BinaryExpr(operador, left, right, span=self._span(left, right))
        return left

    # unary ::= (LOGICAL_NOT | MINUS) unary | primary
    def parse_unary(self) -> Expr:
        start = self.peek() # token atual

        if self.match(TokenKind.LOGICAL_NOT) is not None: #caso seja um operador unário de negação lógica
            operand = self.parse_unary() # Parseia o operando da expressão unária
            return UnaryExpr( #retorna uma expressão unária com o operador de negação lógica e o operando
                UnaryOperator.NOT,
                operand,
                span=self._span(start, operand),
            )

        if self.match(TokenKind.MINUS) is not None: #caso seja um operador unário de negação aritmética
            operand = self.parse_unary() # Parseia o operando da expressão unária
            return UnaryExpr( #retorna uma expressão unária com o operador de negação aritmética e o operando
                UnaryOperator.NEGATE,
                operand,
                span=self._span(start, operand),
            )

        return self.parse_primary() #caso não seja um operador unário, chama o método parse_primary para parsear a expressão primária

    # primary ::= IDENTIFIER | INT_LITERAL | KW_TRUE | KW_FALSE | LEFT_PAREN expression RIGHT_PAREN
    def parse_primary(self) -> Expr:
        if self.match(TokenKind.LEFT_PAREN) is not None: #parenteses de abertura
            expr = self.parse_expression()
            self.expect(TokenKind.RIGHT_PAREN)
            return expr # retorna a expressão entre parênteses
        
        if self.check(TokenKind.IDENTIFIER): # verifica se o token atual é um identificador
            name_token = self.expect(TokenKind.IDENTIFIER)
            if self.match(TokenKind.LEFT_PAREN) is not None: # Paresenteses de abertura, indicando uma chamada de função
                arguments = self.parse_arguments()
                right_paren = self.expect(TokenKind.RIGHT_PAREN) # Parseia consumindo argumentos da func
                return CallExpr(
                    name_token.lexeme,
                    arguments,
                    span=self._span(name_token, right_paren),
                ) # retorna uma expressão de chamada de função com o nome da função, os argumentos e o intervalo de origem
            
            return IdentifierExpr( # Caso não é uma chamada de func
                name_token.lexeme,
                span=self._span(name_token, name_token),
            ) # retorna uma expr de identificador

        if self.check(TokenKind.INT_LITERAL): # token atual é um literal inteiro
            token = self.expect(TokenKind.INT_LITERAL)
            return IntLiteral(token.value, span=self._span(token, token)) # retorna uma expressão de literal 

        if self.check(TokenKind.KW_TRUE): # token atual é o literal booleano verdadeiro
            token = self.expect(TokenKind.KW_TRUE)
            return BoolLiteral(True, span=self._span(token, token)) # retorna uma expressão de literal booleano verdadeiro

        if self.check(TokenKind.KW_FALSE): # token atual é o literal booleano falso
            token = self.expect(TokenKind.KW_FALSE)
            return BoolLiteral(False, span=self._span(token, token)) # retorna uma expressão de literal booleano falso

        raise ParserError( # caso o token atual não seja nenhum dos tipos esperados, lança um erro de análise sintática
            self.peek(),
            {
                TokenKind.LEFT_PAREN,
                TokenKind.IDENTIFIER,
                TokenKind.KW_TRUE,
                TokenKind.KW_FALSE,
                TokenKind.INT_LITERAL
            }
        )

    # arguments ::= expression (COMMA expression)*
    def parse_arguments(self) -> list[Expr]:
        if self.check(TokenKind.RIGHT_PAREN): # Consome Parentese fechado, não possui arg
            return []

        arguments: list[Expr] = [self.parse_expression()] # Parseia o primeiro argumento da expressão

        while self.match(TokenKind.COMMA) is not None: # Consome a vírgula e parseia o próximo argumento da expressão
            arguments.append(self.parse_expression())

        return arguments 

