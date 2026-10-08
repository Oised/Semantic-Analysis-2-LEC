from __future__ import annotations
from dataclasses import fields
from typing import cast
from ast_nodes import (
    Assignment,
    BinaryExpr,
    BinaryOperator,
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
    PrintStmt,
    Program,
    ReturnStmt,
    Stmt,
    TypeName,
    UnaryExpr,
    UnaryOperator,
    VarDecl,
    WhileStmt,
)
from semantic_errors import SemanticDiagnostic, SemanticError, SemanticErrorKind
from symbols import FunctionSymbol, Symbol

MAX_INTEGER = 2**63 - 1

def infer_type(expression: Expr, diagnostics: list[SemanticDiagnostic] | None = None,
               *,
               value_required: bool = True,) -> TypeName:
    """Inferir o tipo de uma expressão.

    Retorna TypeName.VOID se a expressão não for válida.
    """
    def error(kind: SemanticErrorKind, message: str, node: Node) -> None:
        if diagnostics is not None:
            diagnostics.append(SemanticDiagnostic(kind, message, span=node.span))

    if isinstance(expression, IntLiteral): # Verifica se a expressão é um literal inteiro
        if expression.value > MAX_INTEGER: # Verifica se o valor do literal inteiro está fora do intervalo permitido
            error(
                SemanticErrorKind.INTEGER_LITERAL_OUT_OF_RANGE,
                "Literal inteiro fora do intervalo permitido",
                expression,
            )
            result = TypeName.VOID
        else:
            result = TypeName.INT
    
    elif isinstance(expression, BoolLiteral): # Verifica se a expressão é um literal booleano
        result = TypeName.BOOL
    
    elif isinstance(expression, IdentifierExpr): # Verifica se a expressão é um identificador
        symbol = cast(Symbol, expression.metadata["symbol"]) # Obtém o símbolo associado ao identificador
        result = symbol.type # Retorna o tipo do símbolo associado ao identificador

    elif isinstance(expression, CallExpr): # Verifica se a expressão é uma chamada de função 
        function = cast(FunctionSymbol, expression.metadata["symbol"]) # Obtém o símbolo da função associada à chamada
        arguments_types = [
            infer_type(arguments, diagnostics) 
            for arguments in expression.arguments
        ]

        if len(expression.arguments) != len(function.parameter_types): # Verifica se a quantidade de argumentos fornecidos na chamada é diferente da quantidade de parâmetros esperados pela função
            error(
                SemanticErrorKind.ARITY_MISMATCH,
                f"quantidade incorreta de argumentos para {expression.name!r}",
                expression,
            )

        for argument, actual_type, expected_type in zip( # Zip percorre os 3 parametros ao mesmo tempo........
            expression.arguments, 
            arguments_types, 
            function.parameter_types
        ):
            if actual_type is not TypeName.VOID and actual_type is not expected_type: # Verifica se o tipo do argumento fornecido é diferente do tipo do parâmetro esperado
                error(
                    SemanticErrorKind.ARGUMENT_TYPE_MISMATCH,
                    f"tipo incorreto de argumento para {expression.name!r}",
                    argument,
                )

        result = function.type # Retorna o tipo de retorno da função associada à chamada
        if result is TypeName.VOID and value_required: # Verifica se o tipo de retorno da função é void e se um valor é necessário
            error(
                SemanticErrorKind.VOID_VALUE_USED,
                f"função void {expression.name!r} usada como valor",
                expression,
            )
            

    elif isinstance(expression, UnaryExpr): # Verifica se a expressão é uma expressão unária
        operand_type = infer_type(expression.operand, diagnostics) # Infere o tipo do operando da expressão unária

        if operand_type is TypeName.VOID: # Verifica se o tipo do arg é void, caso seja, o tipo da expressão unária é void
            result = TypeName.VOID
        elif ( 
            expression.operator is UnaryOperator.NEGATE # Verifica se o operador unário é de negação (-) e se o tipo do operando é inteiro, 
            and operand_type is TypeName.INT            # caso seja, o tipo da expressão unária é inteiro
        ):
            result = TypeName.INT

        elif (
            expression.operator is UnaryOperator.NOT # Verifica se o operador unário é de negação lógica (!)
            and operand_type is TypeName.BOOL
        ):
            result = TypeName.BOOL

        else: # Caso o operador unário seja inválido para o tipo do operando
            error(
                SemanticErrorKind.INVALID_UNARY_OPERAND,
                f"operando inválido para {expression.operator.value}",
                expression,
            )
            result = TypeName.VOID

    elif isinstance(expression, BinaryExpr): # Verifica se a expressão é uma expressão binária
        left_type = infer_type(expression.left, diagnostics) # Infere os tipos dos operandos esquerdo e direito da expressão binária
        right_type = infer_type(expression.right, diagnostics)

        if left_type is TypeName.VOID or right_type is TypeName.VOID: # Verifica se algum dos operandos é do tipo void
            result = TypeName.VOID

        elif expression.operator in { # Operadores aritméticos
            BinaryOperator.ADD,
            BinaryOperator.SUBTRACT,
            BinaryOperator.MULTIPLY,
            BinaryOperator.DIVIDE,
            BinaryOperator.REMAINDER,
        }:
            if left_type is TypeName.INT and right_type is TypeName.INT: # Verifica se ambos os operandos são do tipo inteiro
                result = TypeName.INT
            else:
                result = TypeName.VOID

        elif expression.operator in { # Operadores de Comparação
            BinaryOperator.LESS,
            BinaryOperator.LESS_EQUAL,
            BinaryOperator.GREATER,
            BinaryOperator.GREATER_EQUAL,
        }:
            if left_type is TypeName.INT and right_type is TypeName.INT: # Verifica se ambos os operandos são do tipo inteiro
                result = TypeName.BOOL
            else:
                result = TypeName.VOID

        elif expression.operator in { # Operadores Lógicos
            BinaryOperator.LOGICAL_AND,
            BinaryOperator.LOGICAL_OR,
        }:
            if left_type is TypeName.BOOL and right_type is TypeName.BOOL: # Verifica se ambos os operandos são do tipo booleano
                result = TypeName.BOOL
            else:
                result = TypeName.VOID

        else:
            # == e != aceitam dois operandos do mesmo tipo não-void.
            if left_type is right_type: # Verifica se ambos os operandos são do mesmo tipo
                result = TypeName.BOOL
            else:
                result = TypeName.VOID

        if result is TypeName.VOID: # Verifica se o tipo da expressão binária é void
            error(
                SemanticErrorKind.INVALID_BINARY_OPERANDS,
                f"operandos inválidos para {expression.operator.value}",
                expression,
            )
    else:
        raise TypeError(f"tipo de expressão inesperado: {type(expression).__name__}")

    expression.metadata["type"] = result # Armazena o tipo inferido da expressão no campo metadata do nó da AST
    return result

def check_types(program: Program) -> None:
    """Determine tipos de expressões e valide seus contextos."""

    diagnostics: list[SemanticDiagnostic] = [] # Lista para armazenar os diagnósticos de erros semânticos encontrados durante a verificação de tipos

    def error(
        kind: SemanticErrorKind,
        message: str,
        node: Node,
    ) -> None:
        diagnostics.append(
            SemanticDiagnostic(kind=kind, message=message, span=node.span)
        )

    def check_block(block: Block, return_type: TypeName) -> None: # Verifica o tipo de cada declaração em um bloco
        for statement in block.statements:
            check_statement(statement, return_type)

    def check_statement(statement: Stmt, return_type: TypeName) -> None: # Verifica o tipo de cada declaração em um bloco
        if isinstance(statement, Block): # Verifica se a declaração é um bloco de código
            check_block(statement, return_type)

        elif isinstance(statement, VarDecl): # Verifica se a declaração é uma declaração de variável
            if statement.type is TypeName.VOID: # se for void, add ao erro ao diagnóstico
                error(
                    SemanticErrorKind.VOID_VARIABLE,
                    f"declaração de variável {statement.name!r} com tipo void",
                    statement,
                )

            if statement.initializer is not None: # Verifica se a declaração de variável possui um inicializador
                actual_type = infer_type(statement.initializer, diagnostics) # Infere o tipo do inicializador da variável
                if (
                    statement.type is not TypeName.VOID
                    and actual_type is not TypeName.VOID
                    and actual_type is not statement.type
                ):
                    error( # caso o tipo do inicializador seja diferente do tipo da variável, add ao erro ao diagnóstico
                        SemanticErrorKind.INITIALIZER_TYPE_MISMATCH,
                        "tipo do inicializador incompatível com a variável",
                        statement.initializer,
                    )

        elif isinstance(statement, Assignment): # Verifica se a declaração é uma atribuição
            target_type = infer_type(statement.target, diagnostics) # Infere o tipo do alvo e do valor da atribuição
            value_type = infer_type(statement.value, diagnostics)

            if (
                target_type is not TypeName.VOID
                and value_type is not TypeName.VOID
                and target_type is not value_type
            ):
                error( # caso o tipo do valor seja diferente do tipo do alvo da atribuição, add ao erro ao diagnóstico
                    SemanticErrorKind.ASSIGNMENT_TYPE_MISMATCH,
                    "tipo do valor incompatível com o alvo da atribuição",
                    statement.value,
            )

        elif isinstance(statement, CallStmt): # Verifica se a declaração é uma chamada de função
            # Uma chamada void é válida como comando, não como valor.
            infer_type(statement.call, diagnostics, value_required=False)

        elif isinstance(statement, IfStmt): # Verifica se a declaração é uma declaração condicional (if)
            condition_type = infer_type(statement.condition, diagnostics) 
            if ( # Verifica se o tipo da condição do if é diferente de void e diferente de bool
                condition_type is not TypeName.VOID
                and condition_type is not TypeName.BOOL
            ):
                error(
                    SemanticErrorKind.CONDITION_TYPE_MISMATCH, # caso o tipo da condição do if seja diferente de void e diferente de bool, add ao erro ao diagnóstico
                    "condição de if não é do tipo bool",
                    statement.condition,
                )
            check_block(statement.then_block, return_type)  # Verifica o bloco de código do ramo "then" do if
            if statement.else_block is not None: # Verifica se o if possui um ramo "else"
                check_block(statement.else_block, return_type)

        elif isinstance(statement, WhileStmt): # Verifica se a declaração é uma declaração de loop (while)
            condition_type = infer_type(statement.condition, diagnostics)
            if (
                condition_type is not TypeName.VOID
                and condition_type is not TypeName.BOOL
            ):
                error( # caso o tipo da condição do while seja diferente de void e diferente de bool, add ao erro ao diagnóstico    
                    SemanticErrorKind.CONDITION_TYPE_MISMATCH,
                    "condição de while não é do tipo bool",
                    statement.condition,
                )
            check_block(statement.body, return_type) # Verifica o bloco de código do corpo do loop while

        elif isinstance(statement, ReturnStmt): # Verifica se a declaração é uma declaração de retorno (return)
            if statement.value is None:
                if return_type is not TypeName.VOID:
                    error( # caso a função não seja do tipo void e o return não possua um valor, add ao erro ao diagnóstico
                        SemanticErrorKind.RETURN_MISMATCH,
                        "funcao não-void deve retornar um valor",
                        statement,
                    )
            else:
                actual_type = infer_type(statement.value, diagnostics)
                if (
                    actual_type is not TypeName.VOID
                    and actual_type is not return_type
                ):
                    error(
                        SemanticErrorKind.RETURN_MISMATCH,
                        "tipo do valor de retorno incompatível com a função",
                        statement.value,
                    )

        elif isinstance(statement, PrintStmt): # Verifica se a declaração é uma declaração de impressão (print)
            for item in statement.items: # itera sobre os itens a serem impressos e infere o tipo de cada item
                if isinstance(item, Expr):  # Verifica se o item é uma expressão
                    infer_type(item, diagnostics)

        else: # Caso a declaração seja de um tipo não suportado, lança um erro de tipo
            raise TypeError(f"tipo de comando não suportado: {type(statement).__name__}")

    for function in program.functions: # Verifica cada função declarada no programa
        for parameter in function.parameters: # Verifica cada parâmetro da função
            if parameter.type is TypeName.VOID:
                error( # caso o tipo do parâmetro seja void, add ao erro ao diagnóstico
                    SemanticErrorKind.VOID_PARAMETER,
                    f"parâmetro {parameter.name!r} com tipo void",
                    parameter,
                )

        check_block(function.body, function.return_type) # Verifica o bloco de código do corpo da função, passando o tipo de retorno da função como parâmetro

    if diagnostics:
        raise SemanticError(diagnostics)