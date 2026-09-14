"""
Path Numba ad alto throughput.

`core` contiene le regole numeriche e le baseline JIT, `observation` l'encoder
e i kernel condivisi, `value_lookahead` espone il core depth-1 su stati determinizzati,
`mlp` i wrapper Python per modelli/A2C, `types` i DTO.
Il solver endgame JIT vive in `ai.endgame.numba_solver`, vicino agli altri solver
del finale, perché è condiviso da agenti runtime e futuri training.
"""

from .value_lookahead import warm_up_numba_value_lookahead

__all__ = ["warm_up_numba_value_lookahead"]
