from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterator


class TokenKind(enum.Enum):
    """Classe já implementada: nomes e números não devem ser alterados."""

    EOF = -1

    IDENTIFIER = 1
    INT_LITERAL = 2
    STRING_LITERAL = 3

    KW_INT = 10
    KW_BOOL = 11
    KW_VOID = 12
    KW_TRUE = 13
    KW_FALSE = 14
    KW_IF = 15
    KW_ELSE = 16
    KW_WHILE = 17
    KW_RETURN = 18
    KW_PRINT = 19

    PLUS = 20
    MINUS = 21
    STAR = 22
    SLASH = 23
    PERCENT = 24
    LESS = 25
    LESS_EQUAL = 26
    GREATER = 27
    GREATER_EQUAL = 28
    EQUAL_EQUAL = 29
    NOT_EQUAL = 30
    LOGICAL_AND = 31
    LOGICAL_OR = 32
    LOGICAL_NOT = 33
    ASSIGN = 34

    LEFT_PAREN = 40
    RIGHT_PAREN = 41
    LEFT_BRACE = 42
    RIGHT_BRACE = 43
    COMMA = 44
    SEMICOLON = 45


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    lexeme: str
    value: int | str | bool | None
    line: int
    column: int

    def __str__(self) -> str:
        return (
            f"<{self.kind.value}, {self.kind.name}, {self.lexeme!r}, "
            f"{self.value!r}, {self.line}, {self.column}>"
        )


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return f"erro léxico em {self.line}:{self.column}: {self.message}"


class Lexer:
    """Converte texto-fonte MicroC em uma sequência de tokens."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1

    def _advance(self) -> None: # avança 1 caractere e atualiza linha/coluna

        # le o caracter atual
        ch = self.source[self.pos]

        # atualiza a linha e a coluna
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        # atualiza a posição (ponteiro)
        self.pos += 1

    def _next_char(self) -> str: # Retorna o valor do próximo char, sem consumir ou gerar token
        if self.pos + 1 < len(self.source):
            return self.source[self.pos+1]
        return None

    def _auto_IDENTIFIER(self) -> Token:
        
        start_line, start_col = self.line, self.column
        lexeme = ''

        while self.pos < len(self.source) and (self.source[self.pos].isalpha() or self.source[self.pos].isdigit() or self.source[self.pos] == '_'):

            lexeme += self.source[self.pos]    # guarda o caracter na string lexeme

            self._advance() # avança uma posição (pos+1)

        # identificando palavras reservadas
        if lexeme == 'int':
            return Token(TokenKind.KW_INT, lexeme, None, start_line, start_col)

        elif lexeme == 'bool':
            return Token(TokenKind.KW_BOOL, lexeme, None, start_line, start_col)

        elif lexeme == 'void':
            return Token(TokenKind.KW_VOID, lexeme, None, start_line, start_col)

        elif lexeme == 'true':
            return Token(TokenKind.KW_TRUE, lexeme, True, start_line, start_col)

        elif lexeme == 'false':
            return Token(TokenKind.KW_FALSE, lexeme, False, start_line, start_col)

        elif lexeme == 'if':
            return Token(TokenKind.KW_IF, lexeme, None, start_line, start_col)

        elif lexeme == 'else':
            return Token(TokenKind.KW_ELSE, lexeme, None, start_line, start_col)

        elif lexeme == 'while':
            return Token(TokenKind.KW_WHILE, lexeme, None, start_line, start_col)

        elif lexeme == 'return':
            return Token(TokenKind.KW_RETURN, lexeme, None, start_line, start_col)

        elif lexeme == 'print':
            return Token(TokenKind.KW_PRINT, lexeme, None, start_line, start_col)

        # nao bateu com nenhuma KW entao e indentificador
        else:
            return Token(TokenKind.IDENTIFIER, lexeme, lexeme, start_line, start_col)

    def _auto_INT_LITERAL(self) -> Token:

        start_line, start_col = self.line, self.column
        lexeme = ''
        valor = 0

        while self.pos < len(self.source) and self.source[self.pos].isdigit():

            lexeme += self.source[self.pos]    # guarda o caracter na string lexeme

            self._advance() # avança uma posição (pos+1)

        valor = int(lexeme)

        return Token(TokenKind.INT_LITERAL, lexeme, valor, start_line, start_col)
    

    def _auto_STRING_LITERAL(self) -> Token:

        start_line, start_col = self.line, self.column
        lexeme = '"'
        value = ""

        self._advance()  # Avança aspa de abertura

        while self.pos < len(self.source):
            ch = self.source[self.pos]

            if ch == "\n" or ch == "\r":
                raise LexerError("String literal não pode quebrar linha", self.line, self.column)

            if ch == '"':
                lexeme += '"'
                self._advance()  # Avança aspa de fechamento
                return Token(TokenKind.STRING_LITERAL, lexeme, value, start_line, start_col)

            if ch == '\\':
                if self.pos + 1 >= len(self.source):
                    raise LexerError("String literal não fechada", start_line, start_col)

                nxt = self.source[self.pos + 1]
                map_escapes = {
                    'n': '\n',
                    't': '\t',
                    '"': '"',
                    '\\': '\\'
                }
                if nxt not in map_escapes:
                    raise LexerError(f"Escape sequence inválida: '\\{nxt}'", self.line, self.column)

                lexeme += '\\' + nxt
                value += map_escapes[nxt]
                self._advance()  # Avança escape
                self._advance()  # Avança escape
                continue

            lexeme += ch
            value += ch
            self._advance()  # Avança para o prox caractere

        raise LexerError("String literal não fechada", start_line, start_col)
    

        

    def tokens(self) -> Iterator[Token]:
        """Produza todos os tokens significativos e um único EOF ao final."""

        while self.pos < len(self.source):

            ch = self.source[self.pos]

            # Ignora espaços
            if ch.isspace():
                self._advance()
                continue

            # Reconhece identificadores/palavras reservadas
            elif ch.isascii() and (ch.isalpha() or ch == '_'):
                yield self._auto_IDENTIFIER()
                continue

            # Reconhece inteiros_literais
            elif ch.isdigit():
                yield self._auto_INT_LITERAL()
                continue
            
            # Reconhece strings_literais
            elif ch == '"':
                yield self._auto_STRING_LITERAL()
                continue

            # Comentarios de linha
            elif ch == '/' and self._next_char() == '/':
                self._advance() # Avança '/'
                self._advance()  # Avança '/'
                while self.pos < len(self.source):
                    if self.source[self.pos] == '\n' or self.source[self.pos] == '\r':
                        self._advance()
                        break
                    self._advance()
                continue

            # Comentarios de bloco
            elif ch == '/' and self._next_char() == '*':
                start_line, start_col = self.line, self.column
                self._advance()  # Avança '/'
                self._advance()  # Avança '*'
                while self.pos < len(self.source):
                    if self.source[self.pos] == '*' and self._next_char() == '/':
                        self._advance()  # Avança '*'
                        self._advance()  # Avança '/'
                        break
                    self._advance()
                else:
                    raise LexerError("Comentário de bloco não fechado", start_line, start_col)
                continue

            # Operadores matematicos únicos:
            elif ch == '+':
                yield Token(TokenKind.PLUS, ch, None, self.line, self.column)
                self._advance()
                continue
            elif ch == '-':
                yield Token(TokenKind.MINUS, ch, None, self.line, self.column)
                self._advance()
                continue
            elif ch == '*':
                yield Token(TokenKind.STAR, ch, None, self.line, self.column)
                self._advance()
                continue
            elif ch == '/':
                yield Token(TokenKind.SLASH, ch, None, self.line, self.column)
                self._advance()
                continue
            elif ch == '%':
                yield Token(TokenKind.PERCENT, ch, None, self.line, self.column)
                self._advance()
                continue

            # Operadores matematicos que podem ter mais de um char:

            elif ch == '<':
                if self._next_char() == '=':
                    yield Token(TokenKind.LESS_EQUAL, '<=', None, self.line, self.column)
                    self._advance()
                    self._advance()
                else:
                    yield Token(TokenKind.LESS, '<', None, self.line, self.column)
                    self._advance()
                continue

            elif ch == '>':
                if self._next_char() == '=':
                    yield Token(TokenKind.GREATER_EQUAL, '>=', None, self.line, self.column)
                    self._advance()
                    self._advance()
                else:
                    yield Token(TokenKind.GREATER, '>', None, self.line, self.column)
                    self._advance()
                continue

            elif ch == '=':
                if self._next_char() == '=':
                    yield Token(TokenKind.EQUAL_EQUAL, '==', None, self.line, self.column)
                    self._advance()
                    self._advance()
                else:
                    yield Token(TokenKind.ASSIGN, '=', None, self.line, self.column)
                    self._advance()
                continue

            elif ch == '!':
                if self._next_char() == '=':
                    yield Token(TokenKind.NOT_EQUAL, '!=', None, self.line, self.column)
                    self._advance()
                    self._advance()
                else:
                    yield Token(TokenKind.LOGICAL_NOT, '!', None, self.line, self.column)
                    self._advance()
                continue

            elif ch == '&':
                if self._next_char() == '&':
                    yield Token(TokenKind.LOGICAL_AND, '&&', None, self.line, self.column)
                    self._advance()
                    self._advance()
                else:
                    raise LexerError("'&' isolado não é permitido", self.line, self.column)
                continue

            elif ch == '|':
                if self._next_char() == '|':
                    yield Token(TokenKind.LOGICAL_OR, '||', None, self.line, self.column)
                    self._advance()
                    self._advance()
                else:
                    raise LexerError("'|' isolado não é permitido", self.line, self.column)
                continue

            # Outros operadores:
            elif ch == '(':
                yield Token(TokenKind.LEFT_PAREN, ch, None, self.line, self.column)
                self._advance()
                continue
            elif ch == ')':
                yield Token(TokenKind.RIGHT_PAREN, ch, None, self.line, self.column)
                self._advance()
                continue
            elif ch == '{':
                yield Token(TokenKind.LEFT_BRACE, ch, None, self.line, self.column)
                self._advance()
                continue
            elif ch == '}':
                yield Token(TokenKind.RIGHT_BRACE, ch, None, self.line, self.column)
                self._advance()
                continue
            elif ch == ',':
                yield Token(TokenKind.COMMA, ch, None, self.line, self.column)
                self._advance()
                continue
            elif ch == ';':
                yield Token(TokenKind.SEMICOLON, ch, None, self.line, self.column)
                self._advance()
                continue
            else:
                raise LexerError(f"Caractere inesperado: '{ch}'", self.line, self.column)

        yield Token(TokenKind.EOF, '', None, self.line, self.column)
            
    def scan(self) -> list[Token]:
        return list(self.tokens())

