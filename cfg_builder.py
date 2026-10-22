"""Construa um CFG auxiliar por função, preservando os nós da AST."""

from __future__ import annotations

import load_previous  # Disponibiliza os módulos das etapas anteriores.

from ast_nodes import (
    Assignment,
    Block,
    BoolLiteral,
    CallStmt,
    FunctionDecl,
    IfStmt,
    PrintStmt,
    ReturnStmt,
    Stmt,
    VarDecl,
    WhileStmt,
)
from cfg import BlockId, Branch, CFG, Jump, Return, ScopeExit


class CFGBuilder:
    def build(self, function: FunctionDecl) -> CFG:
        self.cfg = CFG(function)
        body_entry = self.cfg.new_block()
        self.cfg.terminate(self.cfg.entry, Jump(body_entry.id))
        continuation = self._build_block(function.body, body_entry.id)
        if continuation is not None:
            self.cfg.terminate(continuation, Jump(self.cfg.fallthrough_exit))
        return self.cfg

    def _build_block(self, block: Block, current: BlockId) -> BlockId | None:
        """Percorra um bloco da AST e devolva sua continuação normal.

        `current` identifica um bloco básico aberto. Comandos sequenciais
        permanecem nele. Comandos de controle podem mudar a continuação.
        None indica que os caminhos deste bloco não alcançam o próximo comando.
        """

        # Percorre a AST 
        for statement in block.statements:
            next_block = self._build_statement(statement, current) # Verifica o nó atual da AST, incorporando ao CFG e retornando o próx. bloco
            if next_block is None:
                # Nenhum caminho continua: os comandos restantes são
                # inalcançáveis e não entram no CFG.
                return None
            current = next_block

        # Fim normal do bloco: registra a fronteira do escopo.
        scope = block.metadata["scope"]
        self.cfg.blocks[current].operations.append(ScopeExit(scope))
        return current

    def _build_statement(self, statement: Stmt, current: BlockId) -> BlockId | None:
        """Incorpore um comando ao CFG e devolva sua continuação normal."""

        # Sequencia de ifs que verifica se:
        # - O nó é um bloco aninhado ("{}" dentro do bloco atual). Novo escopo no mesmo bloco básico
        if isinstance(statement, Block):
            return self._build_block(statement, current)
        # - O nó é uma comparação if, então chama a função que cria os novos blocos necessários
        if isinstance(statement, IfStmt):
            return self._build_if(statement, current)
        # - O nó é uma comparação while, então chama a função que cria os novos blocos necessários
        if isinstance(statement, WhileStmt):
            return self._build_while(statement, current)
        # - O nó é um return, então chama a função que liga o fluxo atual ao bloco padrão de fim
        if isinstance(statement, ReturnStmt):
            self.cfg.terminate(current, Return(statement))
            # Retorna none, fechando/terminando o fluxo do bloco atual
            return None
        # - O nó é uma operação comum (declaração, atribuição, print, chamada), anexa ao bloco atual e continua nele
        if isinstance(statement, (VarDecl, Assignment, PrintStmt, CallStmt)):
            self.cfg.blocks[current].operations.append(statement)
            return current

        # Se nenhuma das verificações for verdadeira, a operação do nó é inesperada, portanto um erro
        raise TypeError(f"comando desconhecido: {type(statement).__name__}")

    def _build_if(self, statement: IfStmt, current: BlockId) -> BlockId | None:
        """Construa o fluxo da condicional e devolva sua continuação."""

        # Cria o novo bloco padrão de um if (entrada do ramo verdadeiro)
        then_entry = self.cfg.new_block().id

        # Verifica se a operação tem else
        if statement.else_block is None:
            # Sem else, o caminho falso vai direto para a continuação.
            join = self.cfg.new_block().id
            self.cfg.terminate(
                current, Branch(statement.condition, then_entry, join)
            )
            # Chama a função padrão para começar preencher/contruir o bloco then_entry, reinicia a lógica de construção do bloco
            then_end = self._build_block(statement.then_block, then_entry)
            if then_end is not None:
                self.cfg.terminate(then_end, Jump(join))
            return join

        else_entry = self.cfg.new_block().id
        self.cfg.terminate(
            current, Branch(statement.condition, then_entry, else_entry)
        )

        # Se tratando de um operação com if e else, criamos dois novos blocos diferentes que serão construidos de forma independente
        then_end = self._build_block(statement.then_block, then_entry)
        else_end = self._build_block(statement.else_block, else_entry)

        if then_end is None and else_end is None:
            return None  # os dois ramos terminam: não há continuação

        # Criamos continuação (bloco) padrão para o if e o else se esses não retornarem None (terminarem)
        join = self.cfg.new_block().id
        if then_end is not None:
            self.cfg.terminate(then_end, Jump(join))
        if else_end is not None:
            self.cfg.terminate(else_end, Jump(join))

        # Retornamos o novo bloco, a continuação
        # Assim voltamos para _build_statement e então para o for que percorre a AST dentro de _build_block
        return join

    def _build_while(self, statement: WhileStmt, current: BlockId) -> BlockId | None:
        """Construa o fluxo do laço e devolva sua continuação."""

        # Por padrão criamos dois blocos, header e body_entry, que são respectivamente o bloco que decide o fluxo (entra ou não no laço) e o bloco que tem o corpo do while
        header = self.cfg.new_block().id
        body_entry = self.cfg.new_block().id
        self.cfg.terminate(current, Jump(header))
        
        # Percorre o corpo do while, a partir de body_entry, retorna o último bloco em end antes do while terminar (ou None, se todos os comainhos do corpo terminam em return)
        # o bloco body_end tem fluxo incondicional para o header, para reiniciar o loop se não terminar (return)
        body_end = self._build_block(statement.body, body_entry)
        if body_end is not None:
            self.cfg.terminate(body_end, Jump(header))  # aresta de retorno

        # Define os terminadores do header (corpo do while ou saída do laço).
        condition = statement.condition
        if isinstance(condition, BoolLiteral) and condition.value is True:
            # while (true): sem saída normal. Só o literal é reconhecido.
            self.cfg.terminate(header, Jump(body_entry))
            exit_block = None
        else:
            # exit_block: continuação depois do while (ramo falso do header).
            exit_block = self.cfg.new_block().id
            self.cfg.terminate(header, Branch(condition, body_entry, exit_block))

        return exit_block
