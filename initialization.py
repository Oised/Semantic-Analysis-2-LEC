"""Atribuição definida: estado por símbolo e propagação para frente."""

from __future__ import annotations

import load_previous  # Disponibiliza os módulos das etapas anteriores.

from dataclasses import dataclass, field

from ast_nodes import Assignment, CallStmt, Node, PrintStmt, VarDecl
from cfg import BasicBlock, BlockId, CFG, ScopeExit
from symbols import Symbol


@dataclass(slots=True)
class InitializationFacts:
    initialized_in: dict[BlockId, set[Symbol]] = field(default_factory=dict)
    initialized_out: dict[BlockId, set[Symbol]] = field(default_factory=dict)


def symbol_of(node: Node) -> Symbol:
    """Reutilize a identidade estabelecida pela resolução de nomes."""

    symbol = node.metadata["symbol"]
    if not isinstance(symbol, Symbol):
        raise TypeError("a AST deve possuir símbolos da Análise Semântica 1")
    return symbol


def transfer(block: BasicBlock, incoming: set[Symbol]) -> set[Symbol]:
    """Efeito sobre inicialização. A conferência das leituras vem depois.

    Esta função não emite diagnósticos, pois será chamada várias vezes pelo
    solver. Retornos e condições estão nos terminadores e não escrevem locais.
    """

    state = set(incoming)
    for operation in block.operations:
        if isinstance(operation, VarDecl):
            symbol = symbol_of(operation)
            state.discard(symbol)
            if operation.initializer is not None:
                state.add(symbol)
        elif isinstance(operation, Assignment):
            state.add(symbol_of(operation.target))
        elif isinstance(operation, ScopeExit):
            state.difference_update(operation.scope.symbols.values())
        elif isinstance(operation, (PrintStmt, CallStmt)):
            continue
        else:
            raise TypeError(f"operação inesperada no CFG: {type(operation).__name__}")
    return state


class InitializationAnalyzer:
    def __init__(self, cfg: CFG, reachable: set[BlockId]) -> None:
        self.cfg = cfg
        self.reachable = reachable
        self.parameters = {symbol_of(p) for p in cfg.function.parameters}
        self.universe = self.parameters | {
            symbol_of(operation)
            for block in cfg.blocks.values()
            for operation in block.operations
            if isinstance(operation, VarDecl)
        }

    def solve(self) -> InitializationFacts:
        # Análise must: comece por uma aproximação superior, não pelo vazio.
        # A transferência reinicia declarações e remove símbolos ao sair do
        # escopo. O universo da função é um limite superior para os estados.
        facts = InitializationFacts(
            initialized_in={b: set(self.universe) for b in self.reachable},
            initialized_out={b: set(self.universe) for b in self.reachable},
        )
        facts.initialized_in[self.cfg.entry] = set(self.parameters)

        changed = True
        while changed:
            changed = False
            for block_id in sorted(self.reachable):
                if block_id == self.cfg.entry:
                    incoming = set(self.parameters)
                else:
                    incoming = self._meet_predecessors(block_id, facts)
                outgoing = transfer(self.cfg.blocks[block_id], incoming)
                
                # Faz a verificação se o estado de entrada ou saída mudou em relação ao estado anterior. Se houver mudança,
                # atualiza o estado e marca 'changed' como True para indicar que a análise precisa continuar.

                if incoming != facts.initialized_in[block_id]:
                    facts.initialized_in[block_id] = incoming
                    changed = True
                if outgoing != facts.initialized_out[block_id]:
                    facts.initialized_out[block_id] = outgoing
                    changed = True

        return facts

    def _meet_predecessors(
        self, block_id: BlockId, facts: InitializationFacts,
    ) -> set[Symbol]:
        predecessors = self.cfg.blocks[block_id].predecessors & self.reachable

        if not predecessors:
            return set(self.universe)

        # Mantém apenas símbolos inicializados em todos os predecessores.
        
        result = facts.initialized_out[next(iter(predecessors))].copy()
        for pred in predecessors:
            result.intersection_update(facts.initialized_out[pred])

        return result
