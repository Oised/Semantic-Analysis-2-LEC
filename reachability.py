"""Alcance no CFG. O percurso da AST já foi realizado pelo builder."""

from __future__ import annotations

from cfg import BlockId, CFG


def compute_reachable(cfg: CFG) -> set[BlockId]:
    """Devolva os IDs alcançáveis a partir da entrada, incluindo a entrada."""

    #Percorre o grafo de blocos da CFG a partir do bloco de entrada, registrando os blocos alcançáveis. O resultado é um conjunto de IDs de blocos.
    reachable: set[BlockId] = set()
    pending = [cfg.entry]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue

    #Marca o bloco que está sendo percorrido como alcançável e adiciona o sucessor a lista de pendentes.
        reachable.add(current)
        block = cfg.blocks[current]
        for successor in block.successors:
            if successor not in reachable:
                pending.append(successor)

    return reachable
